#include "laya/runtime.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        laya::tokenizer tokenizer(argv[1]);
        std::ifstream file(argv[2]);
        auto cases = laya::json::parse(file);
        for (const auto& test : cases) {
            auto actual = tokenizer.encode(test.at("text").get<std::string>());
            auto expected = test.at("ids").get<std::vector<int32_t>>();
            if (actual != expected) {
                std::cerr << "Token mismatch: " << test.at("text").dump() << "\nExpected: "
                          << laya::json(expected).dump() << "\nActual: " << laya::json(actual).dump() << '\n';
                return 1;
            }
        }
        std::cout << cases.size() << " tokenizer fixtures passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
