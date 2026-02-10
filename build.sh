#!/bin/bash
# Knowhere Build Script (Conan 2)
# Usage: ./build.sh [options]
#
# Options:
#   --with-diskann      Enable DiskANN support
#   --with-pageann      Enable PageANN support (requires DiskANN)
#   --with-ut           Build unit tests
#   --with-benchmark    Build benchmarks
#   --with-asan         Enable Address Sanitizer
#   --debug            Build Debug instead of Release
#   --clean            Clean build directory first
#   --help             Show this help message

set -e

# Default values
BUILD_TYPE="Release"
WITH_DISKANN="False"
WITH_PAGEANN="False"
WITH_UT="False"
WITH_BENCHMARK="False"
WITH_ASAN="False"
CLEAN_BUILD="False"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-diskann)
            WITH_DISKANN="True"
            shift
            ;;
        --with-pageann)
            WITH_PAGEANN="True"
            WITH_DISKANN="True"  # PageANN requires DiskANN
            shift
            ;;
        --with-ut)
            WITH_UT="True"
            shift
            ;;
        --with-benchmark)
            WITH_BENCHMARK="True"
            shift
            ;;
        --with-asan)
            WITH_ASAN="True"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD="True"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --with-diskann      Enable DiskANN support"
            echo "  --with-pageann      Enable PageANN support (requires DiskANN)"
            echo "  --with-ut           Build unit tests"
            echo "  --with-benchmark    Build benchmarks"
            echo "  --with-asan         Enable Address Sanitizer"
            echo "  --debug            Build Debug instead of Release"
            echo "  --clean            Clean build directory first"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --with-diskann --with-pageann --with-ut"
            echo "  $0 --with-diskann --with-ut --debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "Knowhere Build Configuration (Conan 2)"
echo "========================================"
echo "Build Type:      $BUILD_TYPE"
echo "With DiskANN:    $WITH_DISKANN"
echo "With PageANN:    $WITH_PAGEANN"
echo "With Unit Tests: $WITH_UT"
echo "With Benchmark:  $WITH_BENCHMARK"
echo "With ASAN:       $WITH_ASAN"
echo "Clean Build:     $CLEAN_BUILD"
echo "========================================"
echo ""

# Clean build directory if requested
if [ "$CLEAN_BUILD" = "True" ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Add Conan remote (Conan 2)
echo "Adding Conan remote..."
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --force 2>/dev/null || true

# Detect OS and set compiler settings
OS_TYPE=$(uname)
if [ "$OS_TYPE" = "Darwin" ]; then
    COMPILER_LIBCXX="libc++"
    echo "Detected macOS, using compiler.libcxx=libc++"
else
    COMPILER_LIBCXX="libstdc++11"
    echo "Detected Linux, using compiler.libcxx=libstdc++11"
fi

# Conan 2 install
echo ""
echo "Running conan install (Conan 2)..."
conan install .. \
  --build=missing \
  -o with_diskann=${WITH_DISKANN} \
  -o with_pageann=${WITH_PAGEANN} \
  -o with_ut=${WITH_UT} \
  -o with_benchmark=${WITH_BENCHMARK} \
  -o with_asan=${WITH_ASAN} \
  -s compiler.libcxx=${COMPILER_LIBCXX} \
  -s build_type=${BUILD_TYPE} \
  --output-folder=. \
  || { echo "❌ Conan install failed!"; exit 1; }

# Build
echo ""
echo "Building project..."
conan build .. --build-dir=. || { echo "❌ Build failed!"; exit 1; }

echo ""
echo "========================================"
echo "✓ Build successful!"
echo "========================================"
echo ""
echo "Build artifacts:"
if [ "$BUILD_TYPE" = "Release" ]; then
    echo "  Library: build/Release/lib/libknowhere.dylib (macOS) or .so (Linux)"
    if [ "$WITH_UT" = "True" ]; then
        echo "  Tests:   build/Release/tests/ut/knowhere_tests"
        echo ""
        echo "Run tests:"
        echo "  cd build"
        echo "  ./Release/tests/ut/knowhere_tests '[pageann]'"
    fi
else
    echo "  Library: build/Debug/lib/libknowhere.dylib (macOS) or .so (Linux)"
    if [ "$WITH_UT" = "True" ]; then
        echo "  Tests:   build/Debug/tests/ut/knowhere_tests"
        echo ""
        echo "Run tests:"
        echo "  cd build"
        echo "  ./Debug/tests/ut/knowhere_tests '[pageann]'"
    fi
fi
