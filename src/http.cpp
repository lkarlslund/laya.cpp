#include "laya/http.hpp"
#include "httplib.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>

namespace laya {
namespace {
using json = http_server::json;
void send(httplib::Response& response, int status, const json& body) {
    response.status = status;
    response.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}
void error(httplib::Response& response, int status, const std::string& message) {
    send(response, status, {{"error", {{"message", message}, {"status", status}}}});
}
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> interrupted{false};
void interrupt(int) { interrupted.store(true, std::memory_order_relaxed); }
}

struct http_server::impl {
    http_options options;
    predictor predict;
    httplib::Server server;
    struct unavailable : std::runtime_error { using std::runtime_error::runtime_error; };
    struct reply {
        json results;
        double elapsed;
        uint64_t batch;
        size_t offset;
    };
    struct job {
        json requests;
        size_t questions;
        std::chrono::steady_clock::time_point arrived = std::chrono::steady_clock::now();
        std::promise<reply> result;
    };
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<job>> queue;
    size_t pending = 0;
    bool stopping = false;
    uint64_t batch_id = 0;
    std::thread worker;
    std::atomic<uint64_t> request_id{0};
    std::string model;

    impl(http_options value, predictor callback) : options(std::move(value)), predict(std::move(callback)) {
        if (options.variant != "english" && options.variant != "multilingual" && options.variant != "typed-decisions")
            throw std::invalid_argument("Unknown HTTP model variant");
        if (options.port < 0 || options.port > 65535 || options.max_questions == 0 || options.max_body_bytes == 0)
            throw std::invalid_argument("Invalid HTTP port or request limit");
        if (!options.max_batch_questions) options.max_batch_questions = options.max_questions;
        if (options.max_batch_questions < options.max_questions || options.max_pending_requests == 0 ||
            options.max_pending_requests > 256 || options.batch_wait_ms > 1000)
            throw std::invalid_argument("Invalid HTTP batching limits");
        model = options.variant == "english" ? "laya" : "laya-" + options.variant;
        // Bound both active connections and queued sockets. Excess sockets close.
        server.new_task_queue = [this] { return new httplib::ThreadPool(options.max_pending_requests + 8, 32); };
        server.set_payload_max_length(options.max_body_bytes);
        server.set_tcp_nodelay(true);
        server.set_read_timeout(10);
        server.set_write_timeout(10);
        server.set_keep_alive_timeout(2);
        server.set_keep_alive_max_count(100);
        server.set_pre_routing_handler([this](const auto& request, auto& response) {
            response.set_header("x-typesafe-request-id", "laya-" + std::to_string(++request_id));
            if (request.path != "/health" && !options.api_key.empty() &&
                request.get_header_value("Authorization") != "Bearer " + options.api_key) {
                response.set_header("WWW-Authenticate", "Bearer");
                error(response, 401, "Invalid or missing bearer token");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.Get("/health", [this](const auto&, auto& response) {
            std::lock_guard lock(mutex);
            send(response, 200, {{"status", "ok"}, {"model", model}, {"variant", options.variant},
                                 {"backend", options.backend}, {"max_questions", options.max_questions},
                                 {"pending_requests", pending}, {"queued_requests", queue.size()},
                                 {"batching", options.batching}, {"max_batch_questions", options.max_batch_questions},
                                 {"max_pending_requests", options.max_pending_requests}, {"batch_wait_ms", options.batch_wait_ms}});
        });
        server.Get("/v1/models", [this](const auto&, auto& response) {
            send(response, 200, {{"models", json::array({{{"name", model},
                {"description", "Laya " + options.variant + " native typed decisions"},
                {"release_date", "2026-09-20"}}})}});
        });
        server.Post("/v1/systemone", [this](const auto& request, auto& response) { evaluate(request, response, false); });
        server.Post("/predict", [this](const auto& request, auto& response) { evaluate(request, response, true); });
        server.set_error_handler([](const auto& request, auto& response) {
            if (!response.body.empty()) return;
            if (response.status == 404 && (request.path == "/v1/systemone" || request.path == "/predict" ||
                                          request.path == "/health" || request.path == "/v1/models")) {
                response.status = 405;
                response.set_header("Allow", request.path == "/health" || request.path == "/v1/models" ? "GET, HEAD" : "POST");
            }
            error(response, response.status, httplib::status_message(response.status));
        });
        server.set_exception_handler([](const auto&, auto& response, std::exception_ptr exception) {
            try { if (exception) std::rethrow_exception(exception); }
            catch (const std::exception& e) { std::cerr << "HTTP handler error: " << e.what() << '\n'; }
            catch (...) { std::cerr << "Unknown HTTP handler error\n"; }
            error(response, 500, "Internal server error");
        });
    }

    ~impl() {
        stop();
        if (worker.joinable()) worker.join();
    }

    void stop() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            for (auto& item : queue) {
                item->result.set_exception(std::make_exception_ptr(unavailable("Server is stopping")));
                --pending;
            }
            queue.clear();
        }
        changed.notify_all();
        server.stop();
    }

    reply submit(json requests, size_t questions) {
        auto item = std::make_shared<job>();
        item->requests = std::move(requests);
        item->questions = questions;
        auto future = item->result.get_future();
        {
            std::lock_guard lock(mutex);
            if (stopping) throw unavailable("Server is stopping");
            if (pending >= options.max_pending_requests) throw unavailable("Inference queue is full");
            // Lazy start keeps constructor failure and unused listeners thread-free.
            if (!worker.joinable()) worker = std::thread([this] { dispatch(); });
            queue.push_back(item);
            ++pending;
        }
        changed.notify_one();
        return future.get();
    }

    void execute(const std::vector<std::shared_ptr<job>>& jobs) {
        const auto id = ++batch_id;
        auto requests = json::array();
        for (const auto& item : jobs)
            for (const auto& request : item->requests) requests.push_back(request);
        try {
            const auto start = std::chrono::steady_clock::now();
            auto results = predict(requests);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!results.is_array() || results.size() != requests.size())
                throw std::runtime_error("Predictor returned an invalid result count");
            size_t offset = 0;
            // Prepare every slice before fulfilling promises.
            std::vector<reply> replies;
            for (const auto& item : jobs) {
                auto slice = json::array();
                for (size_t i = 0; i < item->requests.size(); ++i) slice.push_back(results.at(offset + i));
                replies.push_back({std::move(slice), elapsed, id, offset});
                offset += item->requests.size();
            }
            for (size_t i = 0; i < jobs.size(); ++i) jobs[i]->result.set_value(std::move(replies[i]));
        } catch (...) {
            auto failure = std::current_exception();
            bool input_error = false;
            try { std::rethrow_exception(failure); }
            catch (const std::invalid_argument&) { input_error = true; }
            catch (const std::length_error&) { input_error = true; }
            catch (const json::exception&) { input_error = true; }
            catch (...) {}
            // Late preprocessing errors must not reject unrelated callers.
            if (input_error && jobs.size() > 1) {
                for (const auto& item : jobs) execute({item});
            } else {
                for (const auto& item : jobs) item->result.set_exception(failure);
            }
        }
    }

    void dispatch() {
        for (;;) {
            std::vector<std::shared_ptr<job>> jobs;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping) return;
                const auto deadline = queue.front()->arrived + std::chrono::milliseconds(options.batch_wait_ms);
                size_t questions = 0;
                for (;;) {
                    while (!queue.empty() && questions + queue.front()->questions <= options.max_batch_questions) {
                        questions += queue.front()->questions;
                        jobs.push_back(queue.front());
                        queue.pop_front();
                        if (!options.batching) break;
                    }
                    if (stopping || !options.batching || questions == options.max_batch_questions || !queue.empty() ||
                        std::chrono::steady_clock::now() >= deadline) break;
                    changed.wait_until(lock, deadline);
                }
            }
            // Already selected jobs finish during shutdown; queued jobs get 503.
            execute(jobs);
            {
                std::lock_guard lock(mutex);
                pending -= jobs.size();
            }
        }
    }

    void validate(const json& request, size_t& questions) const {
        if (!request.is_object()) throw std::invalid_argument("Request must be an object");
        const auto& state = request.at("state");
        if (!state.is_string() && !state.is_structured())
            throw std::invalid_argument("state must be a string, object or array");
        if (request.contains("model")) {
            auto name = request.at("model").get<std::string>();
            if (name != model && name != options.variant && name != "jev-latest" && name != "laya-latest" && name != "laya-rl-agent")
                throw std::invalid_argument("Requested model is not loaded; this server serves " + model);
        }
        const auto& definitions = request.at("questions");
        if (!definitions.is_object() || definitions.empty())
            throw std::invalid_argument("questions must be a nonempty object");
        if (definitions.size() > options.max_questions - questions)
            throw std::length_error("Request exceeds --max-questions limit");
        questions += definitions.size();
        for (const auto& definition : definitions) {
            const auto& instruction = definition.at("instructions");
            if (!instruction.is_string() && !instruction.is_structured())
                throw std::invalid_argument("instructions must be a string, object or array");
            const auto type = definition.at("type").get<std::string>();
            if (type != "choice" && type != "score" && type != "noul")
                throw std::invalid_argument("Unsupported question type: " + type);
        }
    }

    void evaluate(const httplib::Request& request, httplib::Response& response, bool batch_route) {
        try {
            auto value = json::parse(request.body);
            if (!batch_route && !value.is_object()) throw std::invalid_argument("/v1/systemone expects one request object");
            auto requests = value.is_array() ? value : json::array({value});
            if (requests.empty()) throw std::invalid_argument("Request array must not be empty");
            size_t questions = 0;
            for (const auto& item : requests) validate(item, questions);
            auto completed = submit(std::move(requests), questions);
            auto& results = completed.results;
            const double elapsed = completed.elapsed;
            response.set_header("X-Laya-Batch-Id", std::to_string(completed.batch));
            response.set_header("X-Laya-Batch-Offset", std::to_string(completed.offset));
            if (batch_route) {
                send(response, 200, {{"results", results}, {"elapsed_ms", elapsed}, {"backend", options.backend}});
            } else {
                auto result = results.at(0);
                result["model"] = model;
                send(response, 200, result);
            }
        } catch (const unavailable& e) {
            response.set_header("Retry-After", "1");
            error(response, 503, e.what());
        } catch (const json::parse_error& e) { error(response, 400, e.what());
        } catch (const std::length_error& e) { error(response, 413, e.what());
        } catch (const json::exception& e) { error(response, 422, e.what());
        } catch (const std::invalid_argument& e) { error(response, 422, e.what());
        } catch (const std::exception& e) {
            std::cerr << "HTTP inference error: " << e.what() << '\n';
            error(response, 500, "Inference failed");
        }
    }
};
http_server::http_server(http_options options, predictor predict) : p(std::make_unique<impl>(std::move(options), std::move(predict))) {}
http_server::~http_server() = default;
int http_server::bind() {
    if (p->options.port == 0) return p->server.bind_to_any_port(p->options.host);
    return p->server.bind_to_port(p->options.host, p->options.port) ? p->options.port : -1;
}
bool http_server::listen() { return p->server.listen_after_bind(); }
bool http_server::running() const { return p->server.is_running(); }
void http_server::stop() { p->stop(); }

int serve_http(const http_options& options, http_server::predictor predict) {
    http_server server(options, std::move(predict));
    const auto port = server.bind();
    if (port < 0) throw std::runtime_error("Cannot bind HTTP listener to " + options.host + ":" + std::to_string(options.port));
    interrupted.store(false, std::memory_order_relaxed);
    const auto old_int = std::signal(SIGINT, interrupt);
    const auto old_term = std::signal(SIGTERM, interrupt);
    struct monitor_thread {
        std::atomic<bool> stop{false};
        std::thread worker;
        ~monitor_thread() {
            stop.store(true, std::memory_order_relaxed);
            if (worker.joinable()) worker.join();
        }
    } monitor;
    monitor.worker = std::thread([&] {
        while (!monitor.stop.load(std::memory_order_relaxed)) {
            if (interrupted.load(std::memory_order_relaxed) && server.running()) { server.stop(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::cerr << "Listening: http://" << options.host << ':' << port << "/v1/systemone\n";
    const bool success = server.listen();
    monitor.stop.store(true, std::memory_order_relaxed);
    monitor.worker.join();
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    if (!success) throw std::runtime_error("HTTP listener failed");
    return 0;
}
}
