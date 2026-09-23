# Binary releases

Rolling releases provide four raw executables, with no installer or application
library bundle:

| File | GPU backend |
|---|---|
| `laya-rNNNN-linux-amd64-cuda` | NVIDIA CUDA |
| `laya-rNNNN-linux-amd64-vulkan` | Vulkan |
| `laya-rNNNN-windows-amd64-cuda.exe` | NVIDIA CUDA |
| `laya-rNNNN-windows-amd64-vulkan.exe` | Vulkan |

Each release includes SHA-256 checksums, dependency/build manifests, and license
notices. Models and tokenizers remain separate downloads. The Linux files need
`chmod +x` after downloading. Use `--vulkan` with the Vulkan executable; CUDA is
the default backend. Both the CLI and HTTP server are included in each executable.

## External requirements

All builds link Laya, ggml, ICU (including Unicode data) and the C++ runtime into
the executable. Release builds disable OpenMP and host-native CPU tuning. They
target x64 CPUs with AVX2. No Python, CUDA compiler or Vulkan SDK is needed to run.

| Build | External runtime requirements |
|---|---|
| Linux CUDA | glibc 2.39 or newer (Ubuntu 24.04 baseline) and a CUDA 13-compatible NVIDIA driver providing `libcuda.so.1` |
| Linux Vulkan | Same Linux baseline, `libvulkan.so.1` from the system package manager, and a compatible GPU driver |
| Windows CUDA | Windows 10/11 x64, compatible NVIDIA driver (`nvcuda.dll`), `cublas64_13.dll` and `cublasLt64_13.dll` |
| Windows Vulkan | Windows 10/11 x64, `vulkan-1.dll` and a compatible GPU driver |

CUDA builds use **CUDA Toolkit 13.0.2 / nvcc 13.0.88 and cuBLAS 13.1.0.3**.
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
The release executables themselves are individual downloads, not ZIP archives.

GPU drivers and the Vulkan loader stay system-managed. Vulkan capability and
precision-profile requirements are described in [Vulkan support](vulkan.md).
CUDA binaries contain code for SM 80, 86, 89 and 120. Compiling for a GPU does not
establish numerical equivalence on that GPU.

## Automation and validation

The workflow checks `main` at **00:17 and 12:17 UTC**, skipping commits already
released. A manual run offers `validate` (build only) and `release`. Numbered
`rNNNN` prereleases are published only after all four builds, host tests and
external-dependency audits succeed. Assets are uploaded to a draft before the
release becomes visible. Incomplete matrices and mixed-commit artifacts fail
publication. A failed draft is retained for diagnosis and is not a public release.

Each target also has a standalone build workflow. For example,
`gh workflow run binary-build.yml -f target=windows-cuda` checks only Windows CUDA
and uploads a validation artifact tagged `r0000`; it does not publish a release.
The release workflow calls those four builds independently, then combines their
artifacts. Successful builds cache their tested executable by source and build
configuration, so changes to packaging or release checks reuse the compiled code.

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
