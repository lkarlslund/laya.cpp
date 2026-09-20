#include "laya/decision.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

void check(bool ok) { if (!ok) throw std::runtime_error("decision check failed"); }
int main() {
    auto uniform = laya::calibrate(std::array{0.f, 0.f, 0.f});
    check(std::abs(uniform.expected_score - 1) < 1e-12);
    check(uniform.confidence < 1e-12);
    auto peaked = laya::calibrate(std::array{-1000.f, 1000.f});
    check(peaked.choice == 1 && peaked.probabilities[1] == 1);
    auto soft = laya::calibrate(std::array{0.f, 2.f}, 2);
    check(std::abs(soft.probabilities[1] - 0.7310585786300049) < 1e-12);
    check(laya::calibrate(std::array{5.f}).confidence == 1);
    bool threw = false;
    try { laya::calibrate(std::array{std::numeric_limits<float>::infinity()}); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw);
}
