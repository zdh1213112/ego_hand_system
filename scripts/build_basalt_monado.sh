#!/usr/bin/env bash
# Build the Ubuntu/Debian Basalt CLI with system fmt/yaml-cpp, even inside Conda.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASALT_SOURCE="${BASALT_SOURCE_DIR:-${PROJECT_DIR}/third_party/basalt_monado}"
BASALT_JOBS="${BASALT_BUILD_JOBS:-2}"
[[ "$BASALT_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BASALT_BUILD_JOBS must be a positive integer' >&2; exit 2; }
[[ -f "$BASALT_SOURCE/CMakeLists.txt" ]] || { echo "Basalt checkout not found: $BASALT_SOURCE" >&2; exit 1; }
MULTIARCH="$(/usr/bin/gcc -print-multiarch)"
SYSTEM_PACKAGES="/usr/lib/${MULTIARCH}/cmake"
for package in fmt yaml-cpp; do
  [[ -d "$SYSTEM_PACKAGES/$package" ]] || {
    echo "Missing system $package CMake package; install libfmt-dev and libyaml-cpp-dev." >&2
    exit 1
  }
done

# Removing PATH alone does not clear package locations saved in CMakeCache.txt.
# Pin the conflicting cached packages explicitly as well.
IGNORE_PREFIXES="${CONDA_PREFIX:-}"
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_INSTALL="$(dirname -- "$(dirname -- "$CONDA_EXE")")"
  IGNORE_PREFIXES="${IGNORE_PREFIXES:+${IGNORE_PREFIXES};}${CONDA_INSTALL}"
fi
BUILD_DIR="${BASALT_SOURCE}/build"
CLEAN_ENV=(env -u CONDA_PREFIX -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH
  -u CPATH -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH -u PKG_CONFIG_PATH -u PKG_CONFIG_LIBDIR
  PATH=/usr/bin:/bin:/usr/sbin:/sbin "CCACHE_DIR=${BUILD_DIR}/.ccache")

"${CLEAN_ENV[@]}" /usr/bin/cmake -S "$BASALT_SOURCE" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBASALT_BUILD_SHARED_LIBRARY_ONLY=OFF \
  -DBASALT_INSTANTIATIONS_DOUBLE=OFF \
  "-Dfmt_DIR=${SYSTEM_PACKAGES}/fmt" \
  "-Dyaml-cpp_DIR=${SYSTEM_PACKAGES}/yaml-cpp" \
  "-DCMAKE_IGNORE_PREFIX_PATH=${IGNORE_PREFIXES}"
"${CLEAN_ENV[@]}" /usr/bin/cmake --build "$BUILD_DIR" --target basalt_vio -j "$BASALT_JOBS"
"${CLEAN_ENV[@]}" "$BUILD_DIR/basalt_vio" --help
