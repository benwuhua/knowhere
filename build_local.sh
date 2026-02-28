#!/bin/bash
set -e

# 使用 Conan 1（因为所有包都在其缓存中）
echo "Using Conan 1 with local cache..."

rm -rf build
mkdir -p build && cd build

# Conan 1 install - 使用本地缓存
conan install ..   --build=missing   -o with_diskann=True   -o with_pageann=True   -o with_ut=True   -s compiler.libcxx=libstdc++11   -s build_type=Release

# Build
conan build ..

echo "Build complete!"
