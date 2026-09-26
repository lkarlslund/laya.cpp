#include "laya/runtime.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using laya::json;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

json choice_question(std::string instruction, json criteria = json::array({"A", "B"})) {
    return {{"type", "choice"}, {"instructions", std::move(instruction)}, {"criteria", std::move(criteria)}};
}

json request(json state, json questions) {
    return json::array({json{{"state", std::move(state)}, {"questions", std::move(questions)}}});
}

void rejects(laya::agent& agent, const json& input, const std::string& fragment) {
    try {
        (void)agent.prepare_json(input);
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find(fragment) != std::string::npos,
                "expected error containing '" + fragment + "', got '" + error.what() + "'");
        return;
    }
    throw std::runtime_error("strict preparation accepted input expected to be rejected: " + fragment);
}

std::string repeated(const std::string& phrase, int count) {
    std::string result;
    for (int i = 0; i < count; ++i) {
        if (i) result += ' ';
        result += phrase;
    }
    return result;
}

std::string value_for_option_tokens(const laya::tokenizer& tokenizer, const std::string& label,
                                    const std::string& word, int target) {
    for (int count = 1; count <= target * 3; ++count) {
        const auto candidate = repeated(word, count);
        if (tokenizer.encode(" " + label + ": " + candidate).size() == static_cast<size_t>(target)) return candidate;
    }
    throw std::runtime_error("could not construct an option with the requested exact token count");
}

std::string phrase_with_exact_tokens(const laya::tokenizer& tokenizer, const std::string& word,
                                     int target) {
    for (int count = 1; count <= target * 3; ++count) {
        const auto candidate = repeated(word, count);
        if (tokenizer.encode(" " + candidate).size() == static_cast<size_t>(target)) return candidate;
    }
    throw std::runtime_error("could not construct text with the requested exact token count");
}

int option_span(const json& prepared, size_t index, int sep_id) {
    const auto start = prepared.at("markers").at(index).get<int>();
    int end;
    if (index + 1 < prepared.at("counts").at(0).get<int>()) {
        end = prepared.at("markers").at(index + 1).get<int>();
    } else {
        const auto& ids = prepared.at("ids");
        end = start;
        while (end < prepared.at("lengths").at(0).get<int>() && ids.at(end).get<int>() != sep_id) ++end;
    }
    return end - start;
}

int redistribution_count(int head_budget) {
    for (int count = 2; count <= 255; ++count) {
        const int per = std::max(4, (head_budget - 16) / count);
        if (per < 47 && head_budget - count * (per + 2) < 16) return count;
    }
    throw std::runtime_error("could not choose an option count that triggers redistribution");
}

int exact_fit_count(int head_budget) {
    for (int count = 2; count <= 255; ++count) {
        const int encoded_tokens = (head_budget - 16) / count - 1;
        if ((head_budget - 16) % count == 0 && encoded_tokens > 0 && encoded_tokens <= 47) return count;
    }
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: test-request-preparation MODEL_DIR");
        const std::filesystem::path model_dir(argv[1]);
        std::ifstream config_file(model_dir / "rl_agent_config.json");
        if (!config_file) throw std::runtime_error("cannot open model config");
        const auto config = json::parse(config_file);
        const int max_len = config.value("max_len", 512);
        const int head_budget = config.value("head_max_len", 192);

        laya::agent strict(model_dir, laya::backend_type::cpu);
        laya::agent compatibility(model_dir, laya::backend_type::cpu, false, false, false, true);
        laya::tokenizer tokenizer(model_dir / "tokenizer/tokenizer.json");
        std::ifstream tokenizer_config_file(model_dir / "tokenizer/tokenizer_config.json");
        const auto tokenizer_config = json::parse(tokenizer_config_file);
        const auto sep_setting = tokenizer_config.at("sep_token");
        const auto sep_text = sep_setting.is_string() ? sep_setting.get<std::string>()
                                                       : sep_setting.at("content").get<std::string>();
        const int sep_id = tokenizer.token_id(sep_text);

        // Structured state/instructions and multiple questions remain byte-for-byte
        // identical when no shortening is needed.
        const auto ordinary = request(json{{"status", "ready"}, {"count", 3}}, json{
            {"first", choice_question("Choose the best option", json::array({"one", "two"}))},
            {"second", {{"type", "score"}, {"instructions", json{{"goal", "rate"}}},
                        {"criteria", json::array({"low", "high"})}}}
        });
        require(strict.prepare_json(ordinary) == compatibility.prepare_json(ordinary),
                "strict preparation changed an input that fits all limits");

        // Exactly filling the option budget is accepted; shortening starts only
        // when the remaining budget falls below 16 tokens.
        const int exact_count = exact_fit_count(head_budget);
        if (exact_count > 0) {
            const int option_tokens = (head_budget - 16) / exact_count - 1; // one mask token per option
            json exact_options = json::object();
            for (int i = 0; i < exact_count; ++i) {
                const auto label = "exact-" + std::to_string(i);
                exact_options[label] = value_for_option_tokens(tokenizer, label, "word", option_tokens);
            }
            const auto exact_fit = request("state", json{{"q", choice_question("Choose", exact_options)}});
            require(strict.prepare_json(exact_fit) == compatibility.prepare_json(exact_fit),
                    "strict preparation rejected or changed an exact-fit option budget");
        }

        // Each case reaches a distinct legacy shortening point. Verify that strict
        // mode rejects it and that the explicit compatibility mode still prepares it.
        auto too_long_option = phrase_with_exact_tokens(tokenizer, "option", 50);
        auto long_option = request("state", json{{"q", choice_question("Choose", json::array({too_long_option, "other"}))}});
        rejects(strict, long_option, "option token limit");
        const auto legacy_option = compatibility.prepare_json(long_option);
        require(option_span(legacy_option, 0, sep_id) == 49,
                "compatibility mode did not preserve the 48-token option cap plus mask token");

        json many_options = json::object();
        const int option_count = redistribution_count(head_budget);
        const int per_option = std::max(4, (head_budget - 16) / option_count);
        for (int i = 0; i < option_count; ++i) {
            const auto label = "option-" + std::to_string(i);
            many_options[label] = value_for_option_tokens(tokenizer, label, "word", per_option + 1);
        }
        auto redistribution = request("state", json{{"q", choice_question("Choose", many_options)}});
        rejects(strict, redistribution, "option token budget");
        const auto legacy_redistribution = compatibility.prepare_json(redistribution);
        for (int i = 0; i < option_count; ++i) {
            const int actual = option_span(legacy_redistribution, i, sep_id);
            require(actual == per_option,
                    "compatibility mode option " + std::to_string(i) + " span was " +
                    std::to_string(actual) + ", expected redistributed span " + std::to_string(per_option));
        }

        auto long_heading = request("state", json{{"q", choice_question(repeated("instruction", 500))}});
        rejects(strict, long_heading, "heading token budget");
        const auto legacy_heading = compatibility.prepare_json(long_heading);
        int option_used = 0;
        for (const auto& option : {std::string("A"), std::string("B")})
            option_used += 1 + static_cast<int>(tokenizer.encode(" " + option).size());
        int remaining = head_budget - option_used;
        if (remaining < 16) {
            const int per = std::max(4, (head_budget - 16) / 2);
            option_used = 2 * per;
            remaining = head_budget - option_used;
        }
        const int expected_heading_tokens = std::max(8, remaining);
        require(legacy_heading.at("markers").at(0).get<int>() == expected_heading_tokens + 2,
                "compatibility mode did not preserve legacy heading truncation");

        auto long_state = request(repeated("context", max_len * 3), json{{"q", choice_question("Choose")}});
        rejects(strict, long_state, "state context limit");
        const auto legacy_state = compatibility.prepare_json(long_state);
        require(legacy_state.at("lengths").at(0).get<int>() == max_len,
                "compatibility mode did not preserve legacy max_len state truncation");

        // A large head layout can exceed max_len before state is appended. This
        // final guard is conditional because current published variants generally
        // keep head_max_len well below max_len.
        if (head_budget > max_len) {
            json options = json::object();
            for (int i = 0; i < 5; ++i) options["option-" + std::to_string(i)] = repeated("word", 30);
            auto final_cap = request("", json{{"q", choice_question(repeated("instruction", head_budget), options)}});
            try {
                (void)strict.prepare_json(final_cap);
                throw std::runtime_error("strict preparation accepted an input beyond max_len");
            } catch (const std::invalid_argument& error) {
                require(std::string(error.what()).find("final sequence limit") != std::string::npos ||
                        std::string(error.what()).find("heading token budget") != std::string::npos ||
                        std::string(error.what()).find("option token budget") != std::string::npos,
                        "unexpected rejection before final sequence limit");
            }
        }

        std::cout << "request preparation truncation cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
