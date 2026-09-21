#pragma once
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace laya {
enum class backend_type { cuda, cpu, vulkan };
using json = nlohmann::ordered_json;
struct batch {
    int size = 0, length = 0, options = 0;
    std::vector<int32_t> ids, lengths, markers, counts, types;
};
struct raw_result {
    std::vector<float> logits, actions;
    int action_count = 0;
    double compute_ms = 0;
};
class tokenizer {
public:
    explicit tokenizer(const std::filesystem::path& path);
    ~tokenizer();
    std::vector<int32_t> encode(const std::string& text) const;
    int32_t token_id(const std::string& text) const;
private:
    struct impl;
    std::unique_ptr<impl> p;
};
class runtime {
public:
    runtime(const std::filesystem::path& directory, bool cuda = true, bool bf16 = false, bool flash = false, bool tensor_core = false);
    runtime(const std::filesystem::path& directory, backend_type backend, bool bf16 = false, bool flash = false, bool tensor_core = false);
    ~runtime();
    raw_result forward(const batch& input);
    const json& config() const;
    std::string backend_name() const;
private:
    struct impl;
    std::unique_ptr<impl> p;
};
class agent {
public:
    agent(const std::filesystem::path& directory, bool cuda = true, bool bf16 = false, bool flash = false, bool tensor_core = false);
    agent(const std::filesystem::path& directory, backend_type backend, bool bf16 = false, bool flash = false, bool tensor_core = false);
    json predict(const json& requests, bool raw = false);
    json prepare_json(const json& requests) const;
    std::string backend_name() const;
private:
    runtime model;
    tokenizer tok;
    json settings;
    batch prepare(const json& requests, json& metadata) const;
};
}
