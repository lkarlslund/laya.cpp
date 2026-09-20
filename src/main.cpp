#include "laya/runtime.hpp"
#include "ggml.h"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        ggml_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
        }, nullptr);
        std::string model = "models/laya", input_file, variant = "english";
        bool cuda = true, bf16 = false, flash = false, tensor_core = false, raw = false, prepare = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i+1 < argc) model = argv[++i];
            else if (arg == "--variant" && i+1 < argc) variant = argv[++i];
            else if (arg == "--input" && i+1 < argc) input_file = argv[++i];
            else if (arg == "--cpu") cuda = false;
            else if (arg == "--fp32") { bf16 = false; flash = false; }
            else if (arg == "--flash-fp32") { bf16 = false; flash = true; }
            else if (arg == "--tensor-core-fp32") { bf16 = false; tensor_core = true; }
            else if (arg == "--experimental-bf16") { bf16 = true; flash = true; }
            else if (arg == "--no-flash") flash = false;
            else if (arg == "--raw") raw = true;
            else if (arg == "--prepare") prepare = true;
            else if (arg == "--help") {
                std::cout << "laya-cli [--model DIR] [--variant english|multilingual|typed-decisions] [--input JSON] [--raw|--prepare] [--fp32] [--cpu]\n"
                             "--tensor-core-fp32 --flash-fp32 enables the optimized CUDA path.\n"
                             "--experimental-bf16 enables an unvalidated lower-precision mode; --no-flash disables its fused attention.\n"
                             "Reads JSON lines from stdin when --input is absent. Each line is a request or request array.\n";
                return 0;
            } else throw std::invalid_argument("Unknown or incomplete option: " + arg);
        }
        if (variant!="english" && variant!="multilingual" && variant!="typed-decisions")
            throw std::invalid_argument("Unknown model variant: "+variant);
        if (variant!="english") model=(std::filesystem::path(model)/variant).string();
        laya::agent agent(model, cuda, bf16, flash, tensor_core);
        std::cerr << "Ready: " << agent.backend_name() << '\n';
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
