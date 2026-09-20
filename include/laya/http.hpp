#pragma once
#include <functional>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace laya {
struct http_options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string variant = "english";
    std::string backend;
    std::string api_key;
    size_t max_questions = 8;
    size_t max_body_bytes = 1024 * 1024;
};

// The callback accepts an array and returns the native prediction array.
// Calls are serialized: one runtime and tokenizer may safely back this server.
class http_server {
public:
    using json = nlohmann::ordered_json;
    using predictor = std::function<json(const json&)>;
    http_server(http_options options, predictor predict);
    ~http_server();
    int bind(); // port 0 selects an available port (useful for integration tests).
    bool listen();
    bool running() const;
    void stop();
private:
    struct impl;
    std::unique_ptr<impl> p;
};
int serve_http(const http_options& options, http_server::predictor predict);
}
