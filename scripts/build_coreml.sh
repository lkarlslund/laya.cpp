#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

build_dir="${SOURCE_DIR}/build-coreml"
jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || printf '8')"
run_tests=false
fresh_config=false

usage() {
  cat <<'EOF'
Build the Apple Core ML backend on Apple Silicon.

Usage: scripts/build_coreml.sh [options]

Options:
  --build-dir DIR  Build directory (default: build-coreml)
  --jobs N         Parallel build jobs (default: logical CPU count)
  --fresh          Discard the existing CMake cache before configuring
  --test           Run CTest after a successful build
  -h, --help       Show this help

Environment:
  DEVELOPER_DIR    Xcode developer directory. Defaults to the standard
                   /Applications/Xcode.app installation.
  CMAKE_PREFIX_PATH
                   Additional CMake package search paths.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

while (($# > 0)); do
  case "$1" in
    --build-dir)
      (($# >= 2)) || fail "--build-dir requires a value"
      build_dir="$2"
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || fail "--jobs requires a value"
      jobs="$2"
      shift 2
      ;;
    --test)
      run_tests=true
      shift
      ;;
    --fresh)
      fresh_config=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || fail "Core ML builds require macOS"
[[ "$(uname -m)" == "arm64" ]] || fail "Core ML builds require Apple Silicon (arm64)"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"

require_command cmake
require_command git
require_command ninja
require_command brew
require_command xcode-select

if [[ -z "${DEVELOPER_DIR:-}" ]]; then
  if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  else
    DEVELOPER_DIR="$(xcode-select -p)"
  fi
fi
export DEVELOPER_DIR

[[ -d "$DEVELOPER_DIR" ]] || fail "DEVELOPER_DIR does not exist: $DEVELOPER_DIR"
[[ "$DEVELOPER_DIR" != *CommandLineTools* ]] || fail \
  "full Xcode is required; install Xcode or set DEVELOPER_DIR to its Developer directory"

require_command xcodebuild
require_command xcrun
xcodebuild -version
xcrun --find clang >/dev/null
xcrun --find coremlcompiler >/dev/null

readonly macos_version="$(sw_vers -productVersion)"
readonly macos_major="${macos_version%%.*}"
((macos_major >= 12)) || fail "Core ML requires macOS 12 or newer (found ${macos_version})"

icu_prefix="$(brew --prefix icu4c 2>/dev/null)" || fail \
  "Homebrew ICU is missing; run: brew install icu4c"
json_prefix="$(brew --prefix nlohmann-json 2>/dev/null)" || fail \
  "nlohmann-json is missing; run: brew install nlohmann-json"

cmake_prefix_path="${icu_prefix};${json_prefix}"
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  cmake_prefix_path="${cmake_prefix_path};${CMAKE_PREFIX_PATH}"
fi

if [[ "$build_dir" != /* ]]; then
  build_dir="${SOURCE_DIR}/${build_dir}"
fi

printf 'Configuring Core ML build\n'
printf '  source: %s\n' "$SOURCE_DIR"
printf '  build:  %s\n' "$build_dir"
printf '  Xcode:  %s\n' "$DEVELOPER_DIR"

git -C "$SOURCE_DIR" submodule update --init --recursive

cmake_args=(-S "$SOURCE_DIR" -B "$build_dir" -G Ninja)
if [[ "$fresh_config" == true ]]; then
  cmake_args=(--fresh "${cmake_args[@]}")
fi

cmake "${cmake_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
  -DLAYA_CUDA=OFF \
  -DLAYA_VULKAN=OFF \
  -DLAYA_COREML=ON

cmake --build "$build_dir" --parallel "$jobs"

if [[ "$run_tests" == true ]]; then
  ctest --test-dir "$build_dir" --output-on-failure
fi

printf 'Core ML build complete: %s/bin/laya-cli\n' "$build_dir"
