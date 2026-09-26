# Binary releases

Rolling releases provide seven raw executables, with no installer or application
library bundle:

| File | GPU backend |
|---|---|
| `laya-rNNNN-linux-amd64-cuda12` | NVIDIA CUDA 12.9, including V100 |
| `laya-rNNNN-linux-amd64-cuda13` | NVIDIA CUDA 13.0, newer GPUs |
| `laya-rNNNN-linux-amd64-vulkan` | Vulkan |
| `laya-rNNNN-windows-amd64-cuda12.exe` | NVIDIA CUDA 12.9, including V100 |
| `laya-rNNNN-windows-amd64-cuda13.exe` | NVIDIA CUDA 13.0, newer GPUs |
| `laya-rNNNN-windows-amd64-vulkan.exe` | Vulkan |
| `laya-rNNNN-macos-arm64-coreml` | Apple Core ML |

Each release includes SHA-256 checksums, dependency/build manifests, and license
notices. Models and tokenizers remain separate downloads. The Linux and macOS files
need `chmod +x` after downloading. CUDA is the default backend on CUDA builds.
Use `--vulkan` with the Vulkan executable; without it, startup requests the
unavailable CUDA backend. Use `--coreml` with the macOS executable.
Both the CLI and HTTP server are included in each executable.

## External requirements

All builds link Laya, ggml and ICU (including Unicode data) into the executable.
Linux and Windows also link the C++ runtime statically; macOS uses its system
`libc++`. Release builds disable OpenMP and host-native CPU tuning. The x64
builds target AVX2 CPUs. No Python, CUDA compiler or Vulkan SDK is needed to run.

| Build | External runtime requirements |
|---|---|
| Linux CUDA 12 | glibc 2.39 or newer (Ubuntu 24.04 baseline) and a compatible CUDA 12 NVIDIA driver providing `libcuda.so.1` |
| Linux CUDA 13 | glibc 2.39 or newer (Ubuntu 24.04 baseline) and a CUDA 13-compatible NVIDIA driver providing `libcuda.so.1` |
| Linux Vulkan | Same Linux baseline, `libvulkan.so.1` from the system package manager, and a compatible GPU driver |
| Windows CUDA 12 | Windows 10/11 x64, compatible NVIDIA driver (`nvcuda.dll`), `cublas64_12.dll` and `cublasLt64_12.dll` |
| Windows CUDA 13 | Windows 10/11 x64, compatible NVIDIA driver (`nvcuda.dll`), `cublas64_13.dll` and `cublasLt64_13.dll` |
| Windows Vulkan | Windows 10/11 x64, `vulkan-1.dll` and a compatible GPU driver |
| macOS Core ML | Apple Silicon with macOS 15 or newer and compiled `Laya.mlmodelc` buckets beside the checkpoint; Apple system frameworks only |

CUDA 12 builds use **CUDA Toolkit 12.9.1 and cuBLAS 12.9.1.4**. Choose these for
V100, or a driver that supports CUDA 12 but cannot initialize CUDA 13. CUDA 13
builds retain **CUDA Toolkit 13.0.2 / nvcc 13.0.88 and cuBLAS 13.1.0.3**, the
validated BF16 toolchain. CUDA 12 builds support FP32; `--bf16` requires the
CUDA 13 profile. Do not substitute CUDA 12 DLLs for CUDA 13 DLLs or rename them;
the executables link to their own math-library major version.
Linux links the CUDA runtime and cuBLAS/cuBLASLt statically. Windows links the
CUDA runtime statically but needs the two math DLLs. These DLLs are not included
with the NVIDIA display driver. Put both beside the `.exe`, or supply them through
`PATH` from the matching toolkit.
CUDA's Windows driver loader opens `nvcuda.dll` at runtime, so it may be absent
from the manifest's direct-import list; the NVIDIA driver is still required.

The Windows cuBLAS files are available in NVIDIA's
[cuBLAS 13.1.0.3 redistribution archive](https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.1.0.3-archive.zip).
Its SHA-256 is
`4ac4847bbe4f7709b244956fcfc32197a2954ee70b155cb67eebd9ee26f7e339`.
Extract `cublas64_13.dll` and `cublasLt64_13.dll` from its `bin` directory;
keep the accompanying NVIDIA license. No full toolkit installation is necessary.
For CUDA 12, use the
[cuBLAS 12.9.1.4 redistribution archive](https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip)
and extract `cublas64_12.dll` and `cublasLt64_12.dll`. Its SHA-256 is
`d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed`.
Keep the accompanying NVIDIA license. The release executables themselves are
individual downloads, not ZIP archives.

GPU drivers and the Vulkan loader stay system-managed. Vulkan capability and
precision-profile requirements are described in [Vulkan support](vulkan.md).
CUDA 12 binaries contain code for SM 70, 75, 80, 86, 89, 90 and 120; CUDA 13
binaries cover SM 75, 80, 86, 89, 90 and 120. CUDA chooses compatible compiled
kernels at runtime, so one download supports multiple GPU generations. This does
not split inference across multiple GPUs. There is no automatic switch between
CUDA runtime major versions; download the profile your driver supports. Compiling
for a GPU does not establish numerical equivalence on that GPU.
The macOS executable uses the Core ML backend and requires separately exported
model buckets; see [Core ML](coreml.md). Its ICU dependency is linked statically.
The macOS 15 baseline reflects the build runner and its static ICU package.

## Automation and validation

The workflow checks `main` at **00:17 and 12:17 UTC**, skipping commits already
released. A manual run offers `validate` (build only) and `release`. Numbered
`rNNNN` prereleases are published only after all seven builds, host tests and
external-dependency audits succeed. Assets are uploaded to a draft before the
release becomes visible. Incomplete matrices and mixed-commit artifacts fail
publication. A failed draft is retained for diagnosis and is not a public release.

Each target also has a standalone build workflow. For example,
`gh workflow run binary-build.yml -f target=windows-cuda12` checks only Windows
CUDA 12 and uploads a validation artifact tagged `r0000`; it does not publish a
release. Use `target=windows-cuda13` to check CUDA 13 independently.
The release workflow calls those seven builds independently, then combines their
artifacts. Successful builds cache their tested executable by source and build
configuration, so changes to packaging or release checks reuse the compiled code.
Use `-f target=macos-coreml` to validate the Apple Silicon build on its own.

Hosted runners perform compilation, host tests and dependency checks. They do
**not** run the full GPU acceptance corpus. Rolling releases are therefore marked
as prereleases, and their manifests explicitly record the absence of GPU
validation. Existing performance/accuracy measurements apply to the builds and
hardware identified in those records; Windows and newly packaged binaries need
separate matching-precision GPU validation before equivalent claims can be made.

## Building locally

Use a fresh build directory and add `-DLAYA_PORTABLE=ON` to the regular CMake
options. Static ICU libraries are required. On Windows, use the pinned vcpkg
`x64-windows-static` triplet and MSVC; on Linux, the release workflow uses Ubuntu
24.04's static ICU development libraries. GPU libraries remain dynamically
linked only where listed above. `scripts/release/automation.py package` rejects
unexpected application-library dependencies and Linux build-directory RPATHs.
