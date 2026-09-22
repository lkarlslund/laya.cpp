#pragma once
#include "ggml-backend.h"

// Per-backend sticky range status. Reset before a request; query after each
// graph before consuming its outputs. Both operations synchronize the backend.
extern "C" {
GGML_BACKEND_API void laya_vk_bf16_status_reset(ggml_backend_t backend);
GGML_BACKEND_API bool laya_vk_bf16_status_failed(ggml_backend_t backend);
}
