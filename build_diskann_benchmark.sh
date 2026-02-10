#!/bin/bash
set -e

echo "Building Knowhere with DiskANN and PageANN (Conan 1)..."

# Clean and create build directory
rm -rf build
mkdir -p build && cd build

# Add Conan remote (Conan 1 syntax)
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --force 2>/dev/null || true

# Install dependencies and generate build files
conan install .. \
  --build=missing \
  -o with_ut=True \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_benchmark=True \
  -s compiler.libcxx=libstdc++11 \
  -s build_type=Release \
  || { echo "Conan install failed!"; exit 1; }

# Build the project
conan build .. || { echo "Conan build failed!"; exit 1; }

echo "✓ Build complete!"
echo ""
echo "Run tests:"
echo "  cd build"
echo "  ./Release/tests/ut/knowhere_tests '[pageann]'"
