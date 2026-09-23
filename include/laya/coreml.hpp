#pragma once

#include "laya/runtime.hpp"

namespace laya {

// Internal bridge implemented in Objective-C++. The exported model owns all
// precision and accelerator decisions; this interface stays at the existing
// batch/raw_result boundary.
class coreml_runtime {
public:
    explicit coreml_runtime(const std::filesystem::path& directory);
    ~coreml_runtime();

    coreml_runtime(const coreml_runtime&) = delete;
    coreml_runtime& operator=(const coreml_runtime&) = delete;

    raw_result forward(const batch& input);
    const json& config() const;
    std::string backend_name() const;
    std::string device_name() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

}  // namespace laya
