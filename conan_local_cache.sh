#!/bin/bash
set -e

echo "构建 Knowhere 使用本地缓存（Conan 2 兼容模式）..."

# 创建本地 Conan 2 cache 目录
mkdir -p ~/.conan2/data

# 尝试使用 --build=missing 而不是 --build=*
rm -rf build
mkdir -p build && cd build

conan install ..   --build=missing   -o with_diskann=True   -o with_pageann=True   -o with_ut=True   -s compiler.libcxx=libstdc++11   -s build_type=Release   --output-folder=.   2>&1 | tee /tmp/conan_install.log

echo "Conan install 完成，开始编译..."
conan build .. --build-dir=.

echo "构建完成！"
