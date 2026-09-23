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
        options.batch_wait_ms = 30;
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
        json request = {{"state", "hello"},
            {"questions", {{"test", {{"type", "noul"}, {"instructions", "Is this a greeting?"}}}}}};
        check(body(client.Get("/health"), 200)["model"] == "laya", "health model");
        const auto models = body(client.Get("/v1/models"), 200)["models"];
        check(models.size() == 1 && models[0]["name"] == "laya", "model discovery");
        check(models[0]["release_date"] == "2026-09-20", "service release date");
        request["model"] = models[0]["name"];
        auto response = client.Post("/v1/systemone", request.dump(), "application/json");
        auto result = body(response, 200);
        check(result["model"] == models[0]["name"] && result["answers"]["echo"] == "hello" && !result.contains("results"), "discovered model request");
        auto without_model = request;
        without_model.erase("model");
        check(body(client.Post("/v1/systemone", without_model.dump(), "application/json"), 200)["model"] == models[0]["name"], "omitted model uses loaded checkpoint");
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
        for (const char* alias : {"jev-latest", "laya-latest", "laya-rl-agent", "english"}) {
            bad["model"] = alias;
            body(client.Post("/v1/systemone", bad.dump(), "application/json"), 422);
        }
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
        const int before_batch = calls.load();
        std::vector<std::future<void>> pending;
        for (int i = 0; i < 8; ++i) pending.push_back(std::async(std::launch::async, [&, i] {
            httplib::Client parallel("127.0.0.1", service.port);
            auto item = request; item["state"] = std::to_string(i);
            check(body(parallel.Post("/v1/systemone", item.dump(), "application/json"), 200)["answers"]["echo"] == std::to_string(i), "cross-request contamination");
        }));
        for (auto& work : pending) work.get();
        check(calls - before_batch < 8, "concurrent requests were not batched");
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
            const auto discovered = body(remote.Get("/v1/models", {{"Authorization", "Bearer test-key"}}), 200)["models"];
            check(discovered.size() == 1 && discovered[0]["name"] == "laya-" + variant, "variant discovery");
            auto selected = request;
            selected["model"] = discovered[0]["name"];
            auto reply = remote.Post("/v1/systemone", {{"Authorization", "Bearer test-key"}}, selected.dump(), "application/json");
            check(body(reply, 200)["model"] == discovered[0]["name"], "discovered variant identity");
            body(remote.Post("/v1/systemone", {{"Authorization", "Bearer test-key"}}, request.dump(), "application/json"), 422);
        }
        // Pack whole calls by question count, preserving array offsets across routes.
        auto packed_options = options;
        packed_options.api_key.clear();
        packed_options.variant = "english";
        packed_options.max_questions = 2;
        packed_options.max_batch_questions = 4;
        packed_options.batch_wait_ms = 1000;
        std::atomic<int> packed_calls{0};
        fixture packed(packed_options, [&](const json& items) {
            ++packed_calls;
            size_t questions = 0;
            for (const auto& item : items) questions += item["questions"].size();
            check(questions <= 4, "GPU question budget exceeded");
            return predict(items);
        });
        auto pair = std::async(std::launch::async, [&] {
            httplib::Client remote("127.0.0.1", packed.port);
            return remote.Post("/predict", json::array({request, request}).dump(), "application/json");
        });
        httplib::Client packed_client("127.0.0.1", packed.port);
        auto two_questions = request;
        two_questions["questions"]["second"] = request["questions"]["test"];
        auto packed_reply = packed_client.Post("/v1/systemone", two_questions.dump(), "application/json");
        auto pair_reply = pair.get();
        check(body(pair_reply, 200)["results"].size() == 2, "array response slice");
        body(packed_reply, 200);
        check(packed_calls == 1, "mixed routes did not share a full batch");
        check(pair_reply->get_header_value("X-Laya-Batch-Id") == packed_reply->get_header_value("X-Laya-Batch-Id"),
              "batch identity differs");
        const auto pair_offset = pair_reply->get_header_value("X-Laya-Batch-Offset");
        const auto single_offset = packed_reply->get_header_value("X-Laya-Batch-Offset");
        check((pair_offset == "0" && single_offset == "2") || (pair_offset == "1" && single_offset == "0"),
              "request offsets are incorrect");
        // Hold inference so admission and shutdown behavior are deterministic.
        options.api_key.clear();
        options.variant = "english";
        options.max_pending_requests = 1;
        options.batch_wait_ms = 0;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        fixture bounded(options, [&](const json& items) {
            entered.set_value();
            gate.wait();
            return predict(items);
        });
        auto running = std::async(std::launch::async, [&] {
            httplib::Client remote("127.0.0.1", bounded.port);
            return remote.Post("/predict", request.dump(), "application/json");
        });
        check(entered.get_future().wait_for(2s) == std::future_status::ready, "worker did not start");
        httplib::Client excess("127.0.0.1", bounded.port);
        auto overloaded = excess.Post("/predict", request.dump(), "application/json");
        const bool rejected = overloaded && overloaded->status == 503 && overloaded->has_header("Retry-After");
        release.set_value();
        body(running.get(), 200);
        check(rejected, "overload was not rejected");
        options.max_pending_requests = 32;
        options.batch_wait_ms = 50;
        std::atomic<int> mixed_calls{0};
        fixture mixed(options, [&](const json& items) {
            ++mixed_calls;
            for (const auto& item : items)
                if (item["state"] == "invalid") throw std::invalid_argument("invalid criteria");
            return predict(items);
        });
        auto invalid = request; invalid["state"] = "invalid";
        auto one = std::async(std::launch::async, [&] {
            httplib::Client remote("127.0.0.1", mixed.port);
            return remote.Post("/predict", invalid.dump(), "application/json");
        });
        httplib::Client valid("127.0.0.1", mixed.port);
        body(valid.Post("/predict", json::array({request, request}).dump(), "application/json"), 200);
        body(one.get(), 422);
        check(mixed_calls >= 2, "invalid request isolation failed");
        options.max_pending_requests = 2;
        options.batch_wait_ms = 0;
        std::promise<void> shutdown_entered, shutdown_release;
        auto shutdown_gate = shutdown_release.get_future().share();
        fixture stopping(options, [&](const json& items) {
            shutdown_entered.set_value();
            shutdown_gate.wait();
            return predict(items);
        });
        auto active_request = std::async(std::launch::async, [&] {
            httplib::Client remote("127.0.0.1", stopping.port);
            return remote.Post("/predict", request.dump(), "application/json");
        });
        shutdown_entered.get_future().wait();
        auto queued_request = std::async(std::launch::async, [&] {
            httplib::Client remote("127.0.0.1", stopping.port);
            return remote.Post("/predict", request.dump(), "application/json");
        });
        httplib::Client health("127.0.0.1", stopping.port);
        bool queued = false;
        for (int i = 0; i < 1000; ++i) {
            if (body(health.Get("/health"), 200)["queued_requests"] == 1) { queued = true; break; }
            std::this_thread::sleep_for(1ms);
        }
        stopping.server.stop();
        shutdown_release.set_value();
        body(active_request.get(), 200);
        body(queued_request.get(), 503);
        check(queued, "shutdown test did not enqueue request");
        options.batching = false;
        fixture separate(options, predict);
        httplib::Client single("127.0.0.1", separate.port);
        check(body(single.Post("/predict", json::array({request, request}).dump(), "application/json"), 200)["results"].size() == 2,
              "disabling batching broke explicit arrays");
        fixture malformed(options, [](const json&) { return json::array(); });
        httplib::Client broken("127.0.0.1", malformed.port);
        body(broken.Post("/predict", request.dump(), "application/json"), 500);
        std::cout << "HTTP routes, JEV envelopes, validation, limits, auth and concurrent serialization passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
