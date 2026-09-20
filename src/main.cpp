#include "laya/runtime.hpp"
#include "laya/http.hpp"
#include "ggml.h"
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        ggml_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
        }, nullptr);
        std::string model = "models/laya", input_file, variant = "english";
        bool cuda = true, bf16 = false, flash = false, tensor_core = false, raw = false, prepare = false;
        bool server = false, http_option = false;
        laya::http_options http;
        auto number = [](const std::string& value, int maximum) {
            int n = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data()+value.size(), n);
            if (error != std::errc{} || end != value.data()+value.size() || n < 1 || n > maximum)
                throw std::invalid_argument("Invalid numeric option: " + value);
            return n;
        };
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i+1 < argc) model = argv[++i];
            else if (arg == "--variant" && i+1 < argc) variant = argv[++i];
            else if (arg == "--input" && i+1 < argc) input_file = argv[++i];
            else if (arg == "--server") server = true;
            else if (arg == "--host" && i+1 < argc) { http.host = argv[++i]; http_option = true; }
            else if (arg == "--port" && i+1 < argc) { http.port = number(argv[++i], 65535); http_option = true; }
            else if (arg == "--max-questions" && i+1 < argc) { http.max_questions = number(argv[++i], 4096); http_option = true; }
            else if (arg == "--cpu") cuda = false;
            else if (arg == "--fp32") { bf16 = false; flash = false; }
            else if (arg == "--flash-fp32") { bf16 = false; flash = true; }
            else if (arg == "--tensor-core-fp32") { bf16 = false; tensor_core = true; }
            else if (arg == "--bf16" || arg == "--experimental-bf16") { bf16 = true; flash = true; }
            else if (arg == "--no-flash") flash = false;
            else if (arg == "--raw") raw = true;
            else if (arg == "--prepare") prepare = true;
            else if (arg == "--help") {
                std::cout << "laya-cli [--model DIR] [--variant english|multilingual|typed-decisions] [--input JSON] [--raw|--prepare] [--fp32] [--cpu]\n"
                             "--tensor-core-fp32 --flash-fp32 enables the optimized CUDA path.\n"
                             "--bf16 enables matching-precision CUDA inference (requires the toolchain in docs/precision.md).\n--experimental-bf16 is a compatibility alias. --no-flash is supported only in FP32 mode.\n"
                             "Reads JSON lines from stdin when --input is absent. Each line is a request or request array.\n";
                std::cout << "--server listens on HTTP: POST /v1/systemone (JEV schema), POST /predict (batch), GET /health, GET /v1/models.\n"
                             "--host ADDRESS (127.0.0.1), --port PORT (8080), --max-questions N (8).\n"
                             "Set LAYA_API_KEY to require a bearer token; /health is unauthenticated.\n";
                return 0;
            } else throw std::invalid_argument("Unknown or incomplete option: " + arg);
        }
        if (variant!="english" && variant!="multilingual" && variant!="typed-decisions")
            throw std::invalid_argument("Unknown model variant: "+variant);
        if (server && (!input_file.empty() || raw || prepare))
            throw std::invalid_argument("--server cannot be combined with --input, --raw or --prepare");
        if (http_option && !server) throw std::invalid_argument("HTTP options require --server");
        if (variant!="english") model=(std::filesystem::path(model)/variant).string();
        laya::agent agent(model, cuda, bf16, flash, tensor_core);
        std::cerr << "Ready: " << agent.backend_name() << '\n';
        if (server) {
            // A direct --model checkpoint path must identify its actual variant too.
            std::ifstream metadata_file(std::filesystem::path(model)/"rl_agent_config.json");
            const auto metadata = laya::json::parse(metadata_file);
            http.variant = metadata.value("model_name", "") == "laya-typed-decisions" ? "typed-decisions" :
                           metadata.value("encoder", "") == "jhu-clsp/mmBERT-base" ? "multilingual" : "english";
            http.backend = agent.backend_name();
            if (const char* key = std::getenv("LAYA_API_KEY")) http.api_key = key;
            return laya::serve_http(http, [&](const laya::json& requests) { return agent.predict(requests); });
        }
        auto run = [&](const laya::json& value) {
            auto requests = value.is_array() ? value : laya::json::array({value});
            auto start = std::chrono::steady_clock::now();
            auto result = prepare ? agent.prepare_json(requests) : agent.predict(requests, raw);
            auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            return laya::json{{"results", result}, {"elapsed_ms", elapsed}, {"backend", agent.backend_name()}};
        };
        if (!input_file.empty()) {
            std::ifstream file(input_file);
            if (!file) throw std::runtime_error("Cannot open input file");
            std::cout << run(laya::json::parse(file)).dump() << '\n';
        } else {
            std::string line;
            while (std::getline(std::cin, line)) {
                try { std::cout << run(laya::json::parse(line)).dump() << std::endl; }
                catch (const std::exception& e) { std::cout << laya::json{{"error", e.what()}}.dump() << std::endl; }
            }
        }
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
