#!/bin/bash
set -e

echo "========================================"
echo "Building Knowhere with Conan 2 (Local)"
echo "========================================"

rm -rf build
mkdir -p build && cd build

# Conan 2 install - build from source, no remote access
echo "Step 1: Installing dependencies with Conan 2..."
conan install ..   --build=missing   -o with_diskann=True   -o with_pageann=True   -o with_ut=True   -s compiler.libcxx=libstdc++11   -s build_type=Release   --output-folder=.   -pr:h=default || {
    echo "Conan install failed, trying with explicit profile..."
    conan install ..       --build=missing       -o with_diskann=True       -o with_pageann=True       -o with_ut=True       -s compiler.libcxx=libstdc++11       -s build_type=Release       --output-folder=.       -pr:h ~/.conan2/profiles/default       -pr:b ~/.conan2/profiles/default
}

echo ""
echo "Step 2: Building project..."
conan build .. --build-dir=.

echo ""
echo "✓ Build complete!"
echo ""
echo "Run tests:"
echo "  cd build"
echo "  ./Release/tests/ut/knowhere_tests '[pageann]""
