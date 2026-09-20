#include "laya/decision.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

laya::decision laya::calibrate(std::span<const float> logits, double temperature) {
    if (logits.empty() || !std::isfinite(temperature))
        throw std::invalid_argument("nonempty logits and finite temperature required");
    for (float x : logits)
        if (!std::isfinite(x)) throw std::invalid_argument("logits must be finite");
    const double scale = std::max(0.001, temperature);
    const double maximum = *std::max_element(logits.begin(), logits.end());
    decision result{{}, 0, 0, 1};
    double total = 0;
    for (float x : logits) {
        double value = std::exp((double(x) - maximum) / scale);
        result.probabilities.push_back(value);
        total += value;
    }
    double entropy = 0;
    for (unsigned i = 0; i < result.probabilities.size(); ++i) {
        double &p = result.probabilities[i];
        p /= total;
        result.expected_score += i * p;
        entropy -= p * std::log(std::max(1e-12, p));
    }
    result.choice = std::max_element(result.probabilities.begin(), result.probabilities.end()) - result.probabilities.begin();
    if (logits.size() > 1)
        result.confidence = std::clamp(1 - entropy / std::log(double(logits.size())), 0.0, 1.0);
    return result;
}
