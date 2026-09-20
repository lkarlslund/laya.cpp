#include "laya/runtime.hpp"
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace laya {
namespace {
std::string utf8(int codepoint) {
    icu::UnicodeString u;
    u.append(static_cast<UChar32>(codepoint));
    std::string out;
    u.toUTF8String(out);
    return out;
}
}
struct tokenizer::impl {
    struct added { std::string text; int32_t id; bool normalized, lstrip; };
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, int> ranks;
    std::vector<added> extra;
    std::array<std::string, 256> bytes;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache;
    mutable size_t cache_bytes = 0;
    std::unique_ptr<icu::RegexPattern> pattern;
    const icu::Normalizer2* normalizer;

    explicit impl(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open tokenizer: " + path.string());
        auto data = json::parse(file);
        auto& spec = data.at("model");
        if (spec.at("type") != "BPE" || data.at("normalizer").at("type") != "NFC" ||
            data.at("pre_tokenizer").at("type") != "ByteLevel" ||
            data.at("pre_tokenizer").at("add_prefix_space") != false ||
            data.at("pre_tokenizer").at("use_regex") != true ||
            !spec.at("dropout").is_null() || spec.value("byte_fallback", false) ||
            spec.value("ignore_merges", false))
            throw std::runtime_error("Unsupported tokenizer configuration");
        for (auto& [name, id] : spec.at("vocab").items()) vocab.emplace(name, id.get<int32_t>());
        int rank = 0;
        for (auto& pair : spec.at("merges")) {
            if (!pair.is_array() || pair.size() != 2) throw std::runtime_error("Unsupported BPE merge format");
            ranks.emplace(pair[0].get<std::string>() + '\0' + pair[1].get<std::string>(), rank++);
        }
        for (auto& t : data.at("added_tokens")) {
            if (t.at("single_word") != false || t.at("rstrip") != false)
                throw std::runtime_error("Unsupported added-token boundary flags");
            extra.push_back({t.at("content"), t.at("id"), t.at("normalized"), t.at("lstrip")});
            vocab[t.at("content").get<std::string>()] = t.at("id").get<int32_t>();
        }
        std::stable_sort(extra.begin(), extra.end(), [](auto& a, auto& b) { return a.text.size() > b.text.size(); });
        int next = 256;
        for (int b = 0; b < 256; ++b) {
            bool visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            bytes[b] = utf8(visible ? b : next++);
        }
        UErrorCode error = U_ZERO_ERROR;
        normalizer = icu::Normalizer2::getNFCInstance(error);
        pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, error));
        if (U_FAILURE(error)) throw std::runtime_error("Cannot initialize Unicode tokenizer");
    }

    void bpe(const std::string& word, std::vector<int32_t>& output) const {
        if (auto found = cache.find(word); found != cache.end()) {
            output.insert(output.end(), found->second.begin(), found->second.end());
            return;
        }
        std::vector<std::string> parts;
        for (unsigned char byte : word) parts.push_back(bytes[byte]);
        while (parts.size() > 1) {
            int best = std::numeric_limits<int>::max();
            size_t at = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto found = ranks.find(parts[i] + '\0' + parts[i + 1]);
                if (found != ranks.end() && found->second < best) { best = found->second; at = i; }
            }
            if (best == std::numeric_limits<int>::max()) break;
            parts[at] += parts[at + 1];
            parts.erase(parts.begin() + at + 1);
        }
        std::vector<int32_t> tokens;
        tokens.reserve(parts.size());
        for (auto& part : parts) tokens.push_back(vocab.at(part));
        output.insert(output.end(), tokens.begin(), tokens.end());
        if (word.size() <= 4096) {
            if (cache.size() >= 16384 || cache_bytes > 4*1024*1024) { cache.clear(); cache_bytes = 0; }
            cache_bytes += word.size() + tokens.size()*sizeof(int32_t);
            cache.emplace(word, std::move(tokens));
        }
    }

    void ordinary(const std::string& text, std::vector<int32_t>& output) const {
        UErrorCode error = U_ZERO_ERROR;
        auto unicode = icu::UnicodeString::fromUTF8(text);
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(unicode, error));
        while (matcher->find(error)) {
            std::string piece;
            matcher->group(error).toUTF8String(piece);
            bpe(piece, output);
        }
        if (U_FAILURE(error)) throw std::runtime_error("Unicode tokenization failed");
    }

    void split(const std::string& text, bool normalized, std::vector<int32_t>& output) const {
        size_t begin = 0;
        while (begin < text.size()) {
            size_t first = std::string::npos;
            const added* selected = nullptr;
            for (auto& token : extra) {
                if (token.normalized != normalized) continue;
                size_t pos = text.find(token.text, begin);
                if (pos < first) { first = pos; selected = &token; }
            }
            size_t end = selected ? first : text.size();
            std::string prefix = text.substr(begin, end - begin);
            if (selected && selected->lstrip) {
                auto value = icu::UnicodeString::fromUTF8(prefix);
                int32_t len = value.length();
                while (len > 0) {
                    auto cp = value.char32At(value.moveIndex32(len, -1));
                    if (!u_isUWhiteSpace(cp)) break;
                    len = value.moveIndex32(len, -1);
                }
                value.truncate(len);
                prefix.clear(); value.toUTF8String(prefix);
            }
            if (normalized) ordinary(prefix, output);
            else {
                UErrorCode error = U_ZERO_ERROR;
                icu::UnicodeString value;
                normalizer->normalize(icu::UnicodeString::fromUTF8(prefix), value, error);
                if (U_FAILURE(error)) throw std::runtime_error("NFC normalization failed");
                std::string nfc; value.toUTF8String(nfc);
                split(nfc, true, output);
            }
            if (!selected) break;
            output.push_back(selected->id);
            begin = first + selected->text.size();
        }
    }
};

tokenizer::tokenizer(const std::filesystem::path& path) : p(std::make_unique<impl>(path)) {}
tokenizer::~tokenizer() = default;
std::vector<int32_t> tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> result;
    p->split(text, false, result);
    return result;
}
int32_t tokenizer::token_id(const std::string& text) const { return p->vocab.at(text); }
}
