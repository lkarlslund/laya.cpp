#include "laya/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace laya {
namespace {
std::string dump_python(const json& value, bool ascii = false) {
    if (!value.is_structured()) return value.dump(-1, ' ', ascii);
    std::string text = value.is_object() ? "{" : "[";
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first) text += ", ";
        first = false;
        if (value.is_object()) text += json(it.key()).dump(-1, ' ', ascii) + ": ";
        text += dump_python(it.value(), ascii);
    }
    return text + (value.is_object() ? "}" : "]");
}
std::string render(const json& value) { return value.is_string() ? value.get<std::string>() : dump_python(value); }
std::string clean(std::string text, const std::string& mask) {
    size_t pos = 0;
    while ((pos = text.find(mask, pos)) != std::string::npos) { text.replace(pos, mask.size(), " "); ++pos; }
    return text;
}
bool empty(const json& x) { return x.is_null() || (x.is_string() && x.get<std::string>().empty()); }
double rounded(double x) { return std::nearbyint(x*10000.0)/10000.0; }
}
agent::agent(const std::filesystem::path& directory, bool cuda, bool bf16, bool flash, bool tensor_core, bool allow_truncation)
    : agent(directory, cuda ? backend_type::cuda : backend_type::cpu, bf16, flash, tensor_core, allow_truncation) {}
agent::agent(const std::filesystem::path& directory, backend_type backend, bool bf16, bool flash, bool tensor_core, bool allow_truncation)
    : agent(directory,backend,bf16 ? precision_type::bf16 : precision_type::fp32,flash,tensor_core,allow_truncation) {}
agent::agent(const std::filesystem::path& directory, backend_type backend, precision_type precision, bool flash, bool tensor_core, bool allow_truncation)
    : model(directory, backend, precision, flash, tensor_core), tok(directory / "tokenizer/tokenizer.json"), allow_truncation(allow_truncation) {
    std::ifstream f(directory / "tokenizer/tokenizer_config.json"); settings = json::parse(f);
}
std::string agent::backend_name() const { return model.backend_name(); }
std::string agent::device_name() const { return model.device_name(); }
batch agent::prepare(const json& requests, json& metadata) const {
    if (!requests.is_array() || requests.empty()) throw std::invalid_argument("requests must be a nonempty array");
    auto token_text = [&](const std::string& key) {
        auto value = settings.at(key+"_token");
        return value.is_string() ? value.get<std::string>() : value.at("content").get<std::string>();
    };
    const auto mask_text = token_text("mask");
    const auto mask_id = tok.token_id(mask_text), cls_id = tok.token_id(token_text("cls"));
    const auto sep_id = tok.token_id(token_text("sep")), pad_id = tok.token_id(token_text("pad"));
    auto encode = [&](const std::string& text) { return tok.encode(clean(text, mask_text)); };
    int limit = model.config().value("max_len", 512), budget = model.config().value("head_max_len", 192);
    std::vector<std::vector<int32_t>> sequences, positions;
    batch result;
    metadata = json::array();
    int request_index = 0;
    for (auto& request : requests) {
        auto state = encode(render(request.at("state")));
        auto questions = request.at("questions");
        if (!questions.is_object() || questions.empty()) throw std::invalid_argument("questions must be a nonempty object");
        for (auto& [id, definition] : questions.items()) {
            auto type = definition.at("type").get<std::string>();
            int qtype = type == "choice" ? 0 : type == "score" ? 1 : type == "noul" ? 2 : -1;
            if (qtype < 0) throw std::invalid_argument("Unsupported question type: " + type);
            auto criteria = definition.value("criteria", json());
            std::vector<std::string> options;
            if (qtype == 0) {
                if (criteria.is_array()) {
                    auto mapping = json::object();
                    for (auto& value : criteria) mapping[value.get<std::string>()] = nullptr;
                    criteria = mapping;
                }
                if (!criteria.is_object()) throw std::invalid_argument("Choice criteria must be a list or object");
                for (auto& [name, value] : criteria.items()) options.push_back(empty(value) ? name : name + ": " + render(value));
            } else if (qtype == 1) {
                if (!criteria.is_array()) throw std::invalid_argument("Score criteria must be an array");
                for (size_t i = 0; i < criteria.size(); ++i) options.push_back("level " + std::to_string(i) + ": " + render(criteria[i]));
            } else {
                if (criteria.is_null()) criteria = json::object();
                if (!criteria.is_object()) throw std::invalid_argument("Boolean criteria must be an object");
                auto no = criteria.value("false", json()), yes = criteria.value("true", json());
                options = {"false: " + (empty(no) ? std::string("no, the statement does not hold") : render(no)),
                           "true: " + (empty(yes) ? std::string("yes, the statement holds") : render(yes))};
            }
            if (options.size() < 2 || options.size() > 255) throw std::invalid_argument("Questions require 2 through 255 options");
            auto instruction = definition.at("instructions");
            auto heading = encode(type + " question: " + (instruction.is_string() ? instruction.get<std::string>() : dump_python(instruction, true)));
            std::vector<std::vector<int32_t>> encoded;
            int used = 0;
            for (auto& option : options) {
                auto ids = encode(" " + option);
                if (ids.size() > 48) {
                    if (!allow_truncation) throw std::invalid_argument("Question '" + id + "' exceeds option token limit (48)");
                    ids.resize(48);
                }
                ids.insert(ids.begin(), mask_id); used += ids.size(); encoded.push_back(std::move(ids));
            }
            int remaining = budget - used;
            if (remaining < 16) {
                int per = std::max(4, (budget-16)/int(options.size()));
                if (!allow_truncation) {
                    for (const auto& ids : encoded)
                        if (int(ids.size()) > per) throw std::invalid_argument("Question '" + id + "' exceeds option token budget (" + std::to_string(per) + " tokens per option)");
                }
                used = 0;
                for (auto& ids : encoded) { if (int(ids.size()) > per) ids.resize(per); used += ids.size(); }
                remaining = budget-used;
            }
            int heading_limit = std::max(8, remaining);
            if (int(heading.size()) > heading_limit) {
                if (!allow_truncation) throw std::invalid_argument("Question '" + id + "' exceeds heading token budget (" + std::to_string(heading_limit) + " tokens)");
                heading.resize(heading_limit);
            }
            if (!allow_truncation && used + int(heading.size()) > budget)
                throw std::invalid_argument("Question '" + id + "' exceeds question head token budget (" + std::to_string(budget) + " tokens)");
            std::vector<int32_t> ids{cls_id}, markers;
            ids.insert(ids.end(), heading.begin(), heading.end()); ids.push_back(sep_id);
            for (auto& option : encoded) {
                markers.push_back(ids.size()); ids.insert(ids.end(), option.begin(), option.end());
            }
            ids.push_back(sep_id);
            int room = std::max(0, limit-int(ids.size())-1);
            if (!allow_truncation && state.size() > static_cast<size_t>(room))
                throw std::invalid_argument("Question '" + id + "' exceeds state context limit (" + std::to_string(room) + " tokens)");
            ids.insert(ids.end(), state.begin(), state.begin()+std::min(size_t(room), state.size())); ids.push_back(sep_id);
            if (int(ids.size()) > limit) {
                if (!allow_truncation) throw std::invalid_argument("Question '" + id + "' exceeds final sequence limit (" + std::to_string(limit) + " tokens)");
                ids.resize(limit);
            }
            if (markers.back() >= limit) throw std::invalid_argument("Question '" + id + "' exceeds final sequence limit (" + std::to_string(limit) + " tokens)");
            result.length = std::max(result.length, int(ids.size()));
            result.options = std::max(result.options, int(markers.size()));
            result.lengths.push_back(ids.size()); result.counts.push_back(markers.size()); result.types.push_back(qtype);
            sequences.push_back(std::move(ids)); positions.push_back(std::move(markers));
            metadata.push_back({{"request", request_index}, {"id", id}, {"type", type}, {"criteria", criteria}});
        }
        ++request_index;
    }
    result.size = sequences.size();
    result.ids.assign(result.size*result.length, pad_id);
    result.markers.resize(result.size*result.options);
    for (int row = 0; row < result.size; ++row) {
        std::copy(sequences[row].begin(), sequences[row].end(), result.ids.begin()+row*result.length);
        for (int k = 0; k < result.options; ++k)
            result.markers[row*result.options+k] = row*result.length + (k < result.counts[row] ? positions[row][k] : 0);
    }
    return result;
}
json agent::prepare_json(const json& requests) const {
    json metadata; auto input = prepare(requests, metadata);
    return {{"batch", input.size}, {"length", input.length}, {"options", input.options}, {"ids", input.ids},
            {"lengths", input.lengths}, {"markers", input.markers}, {"counts", input.counts}, {"types", input.types}};
}
json agent::predict(const json& requests, bool raw) {
    json metadata; auto input = prepare(requests, metadata); auto values = model.forward(input);
    if (raw) return {{"inputs", {{"batch", input.size}, {"length", input.length}, {"options", input.options}, {"ids", input.ids},
                                {"lengths", input.lengths}, {"markers", input.markers}, {"counts", input.counts}, {"types", input.types}}},
                     {"logits", values.logits}, {"actions", values.actions},
                     {"action_count", values.action_count}, {"compute_ms", values.compute_ms}};
    auto output = json::array();
    for (size_t i = 0; i < requests.size(); ++i)
        output.push_back({{"model", "laya-rl-agent"}, {"answers", json::object()}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}});
    for (int row = 0; row < input.size; ++row) {
        auto& meta = metadata[row];
        const int count = input.counts[row];
        std::string type = meta["type"], bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
        double temperature = model.config().value("temperature", json::array({1,1,1}))[input.types[row]].get<double>();
        auto temperatures = model.config().value("temperature_by_options", json::object());
        if (temperatures.contains(type+":"+bucket)) temperature = temperatures[type+":"+bucket];
        std::vector<float> probs(count);
        for (int k = 0; k < count; ++k) probs[k] = values.logits[row*input.options+k]/float(std::max(1e-3, temperature));
        float maximum = *std::max_element(probs.begin(), probs.end()), total = 0;
        for (auto& v : probs) total += v = std::exp(v-maximum);
        float entropy = 0;
        double score = 0;
        for (int k = 0; k < count; ++k) { probs[k] /= total; entropy -= probs[k]*std::log(std::max(probs[k], 1e-12f)); score += k*double(probs[k]); }
        auto action = values.actions.data()+row*values.action_count;
        float max_action = *std::max_element(action, action+values.action_count), action_sum = 0;
        for (int k = 0; k < values.action_count; ++k) action_sum += std::exp(action[k]-max_action);
        json answer = {{"type", type}, {"confidence", rounded(std::clamp(1.0-double(entropy)/std::log(double(count)), 0.0, 1.0))},
                       {"action", {{"act_probability", rounded(std::exp(action[0]-max_action)/action_sum)}}}};
        auto criteria = meta["criteria"];
        if (type == "choice") {
            auto selected = std::max_element(probs.begin(), probs.end())-probs.begin();
            auto probabilities = json::object(); int k = 0;
            for (auto& [name, unused] : criteria.items()) {
                probabilities[name] = rounded(probs[k]); if (k == selected) answer["choice"] = name; ++k;
            }
            answer["probabilities"] = probabilities;
        } else if (type == "score") {
            answer["score"] = rounded(score); auto probabilities = json::object(), legend = json::object();
            for (int k = 0; k < count; ++k) { probabilities[std::to_string(k)] = rounded(probs[k]); legend[std::to_string(k)] = criteria[k]; }
            answer["probabilities"] = probabilities; answer["legend"] = legend;
        } else {
            answer["noul"] = rounded(probs[1]); answer["confidence"] = rounded(std::max(double(probs[1]), 1.0-double(probs[1])));
        }
        int request = meta["request"];
        output[request]["answers"][meta["id"].get<std::string>()] = answer;
        int tokens = output[request]["usage"]["input_tokens"];
        output[request]["usage"]["input_tokens"] = tokens+input.lengths[row];
    }
    return output;
}
}
