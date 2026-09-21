#include "laya/runtime.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using laya::json;
void compare(const json& expected, const json& actual, const std::string& path = "result") {
    if (expected.is_number() && actual.is_number()) {
        const double a = expected.get<double>(), b = actual.get<double>();
        if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 0.000100000001)
            throw std::runtime_error(path + ": numeric mismatch");
    } else if (expected.is_object() && actual.is_object()) {
        if (expected.size() != actual.size()) throw std::runtime_error(path + ": key count mismatch");
        for (auto it = expected.begin(); it != expected.end(); ++it)
            compare(it.value(), actual.at(it.key()), path + "." + it.key());
    } else if (expected.is_array() && actual.is_array()) {
        if (expected.size() != actual.size()) throw std::runtime_error(path + ": row count mismatch");
        for (size_t i = 0; i < expected.size(); ++i) compare(expected[i], actual[i], path + "." + std::to_string(i));
    } else if (expected != actual) throw std::runtime_error(path + ": categorical mismatch");
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected checkpoint directory");
        // Exercise the legacy constructor and the explicit backend API together.
        laya::agent reference(argv[1], true);
        laya::agent candidate(argv[1], laya::backend_type::vulkan);
        if (!reference.backend_name().starts_with("CUDA") || !candidate.backend_name().starts_with("Vulkan"))
            throw std::runtime_error("Wrong execution backend");
        std::string anchor;
        for (int i = 0; i < 128; ++i) anchor += "word ";
        json request = {{"state", anchor}, {"questions", {{"refund", {
            {"type", "noul"}, {"instructions", "Does the customer request a refund?"}}}}}};
        auto first = json::array({request, request});
        first[1]["state"] = "Please refund this payment.";
        auto second = first;
        second[1]["state"] = "The product works correctly. ";
        for (int i = 0; i < 12; ++i) second[1]["state"].get_ref<std::string&>() += "I am happy with it. ";
        const auto shape_a = candidate.prepare_json(first), shape_b = candidate.prepare_json(second);
        if (shape_a.at("length") != shape_b.at("length") || shape_a.at("lengths") == shape_b.at("lengths"))
            throw std::runtime_error("Fixture must reuse a shape with different padding");
        const auto before = candidate.predict(first);
        compare(reference.predict(first), before);
        const auto changed = candidate.predict(second);
        compare(reference.predict(second), changed);
        if (candidate.predict(first) != before || candidate.predict(second) != changed)
            throw std::runtime_error("Replayed graph changed its answer after updating padding");
        std::cout << "CUDA/Vulkan parity and changing-padding graph replays passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
