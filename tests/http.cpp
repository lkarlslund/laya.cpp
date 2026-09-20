#include "laya/http.hpp"
#include "httplib.h"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using json = laya::http_server::json;
using namespace std::chrono_literals;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct fixture {
    laya::http_server server;
    int port;
    std::thread thread;
    fixture(laya::http_options options, laya::http_server::predictor predict)
        : server(std::move(options), std::move(predict)), port(server.bind()) {
        check(port > 0, "bind failed");
        thread = std::thread([this] { server.listen(); });
        for (int i = 0; i < 1000 && !server.running(); ++i) std::this_thread::sleep_for(1ms);
        if (!server.running()) { server.stop(); thread.join(); throw std::runtime_error("listener startup timed out"); }
    }
    ~fixture() { server.stop(); thread.join(); }
};
json body(const httplib::Result& response, int status) {
    check(bool(response), "HTTP transport failed");
    if (response->status != status)
        throw std::runtime_error("Unexpected status " + std::to_string(response->status) + ": " + response->body);
    check(response->get_header_value("Content-Type").starts_with("application/json"), "missing JSON content type");
    return json::parse(response->body);
}
int main() {
    try {
        laya::http_options options;
        options.port = 0;
        options.backend = "test";
        options.max_body_bytes = 4096;
        std::atomic<int> active{0}, peak{0}, calls{0};
        auto predict = [&](const json& requests) {
            int count = ++active;
            peak.store(std::max(peak.load(), count));
            ++calls;
            std::this_thread::sleep_for(5ms);
            --active;
            if (requests[0]["state"] == "fail") throw std::runtime_error("private failure detail");
            if (requests[0]["state"] == "invalid") throw std::invalid_argument("invalid criteria");
            auto result = json::array();
            for (const auto& request : requests)
                result.push_back({{"model", "laya-rl-agent"}, {"answers", {{"echo", request["state"]}}},
                                  {"usage", {{"input_tokens", 1}, {"output_tokens", 0}}}});
            return result;
        };
        fixture service(options, predict);
        httplib::Client client("127.0.0.1", service.port);
        client.set_keep_alive(true);
        const json request = {{"model", "jev-latest"}, {"state", "hello"},
            {"questions", {{"test", {{"type", "noul"}, {"instructions", "Is this a greeting?"}}}}}};
        check(body(client.Get("/health"), 200)["model"] == "laya", "health model");
        check(body(client.Get("/v1/models"), 200)["models"][0]["name"] == "laya", "model discovery");
        auto response = client.Post("/v1/systemone", request.dump(), "application/json");
        auto result = body(response, 200);
        check(result["answers"]["echo"] == "hello" && !result.contains("results"), "JEV response envelope");
        check(response->has_header("x-typesafe-request-id"), "request ID missing");
        check(body(client.Post("/predict", json::array({request, request}).dump(), "application/json"), 200)["results"].size() == 2, "batch route");
        body(client.Post("/v1/systemone", "[", "application/json"), 400);
        body(client.Post("/v1/systemone", std::string("{\"state\":\"\xff\"}"), "application/json"), 400);
        body(client.Post("/v1/systemone", "[]", "application/json"), 422);
        body(client.Post("/predict", "[]", "application/json"), 422);
        body(client.Post("/v1/systemone", "{}", "application/json"), 422);
        auto bad = request;
        bad["model"] = "laya-multilingual";
        body(client.Post("/v1/systemone", bad.dump(), "application/json"), 422);
        bad = request; bad["state"] = nullptr;
        body(client.Post("/v1/systemone", bad.dump(), "application/json"), 422);
        bad = request; bad["questions"]["test"]["type"] = "chat";
        body(client.Post("/v1/systemone", bad.dump(), "application/json"), 422);
        auto many = json::array();
        for (int i = 0; i < 9; ++i) many.push_back(request);
        body(client.Post("/predict", many.dump(), "application/json"), 413);
        body(client.Get("/v1/systemone"), 405);
        body(client.Get("/missing"), 404);
        body(client.Post("/v1/systemone", std::string(5000, 'x'), "application/json"), 413);
        bad = request; bad["state"] = "fail";
        auto failure = body(client.Post("/v1/systemone", bad.dump(), "application/json"), 500);
        check(failure.dump().find("private") == std::string::npos, "internal failure leaked");
        bad["state"] = "invalid";
        body(client.Post("/v1/systemone", bad.dump(), "application/json"), 422);
        // Concurrent callers must get their own answer while sharing one predictor.
        std::vector<std::future<void>> pending;
        for (int i = 0; i < 8; ++i) pending.push_back(std::async(std::launch::async, [&, i] {
            httplib::Client parallel("127.0.0.1", service.port);
            auto item = request; item["state"] = std::to_string(i);
            check(body(parallel.Post("/v1/systemone", item.dump(), "application/json"), 200)["answers"]["echo"] == std::to_string(i), "cross-request contamination");
        }));
        for (auto& work : pending) work.get();
        check(peak == 1, "predictor called concurrently");
        for (const std::string variant : {"multilingual", "typed-decisions"}) {
            options.variant = variant;
            options.api_key = "test-key";
            fixture secured(options, predict);
            httplib::Client remote("127.0.0.1", secured.port);
            body(remote.Get("/health"), 200);
            const auto before = calls.load();
            body(remote.Post("/v1/systemone", request.dump(), "application/json"), 401);
            body(remote.Get("/v1/models"), 401);
            check(calls == before, "unauthenticated inference executed");
            auto reply = remote.Post("/v1/systemone", {{"Authorization", "Bearer test-key"}}, request.dump(), "application/json");
            check(body(reply, 200)["model"] == "laya-" + variant, "variant identity");
        }
        std::cout << "HTTP routes, JEV envelopes, validation, limits, auth and concurrent serialization passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
