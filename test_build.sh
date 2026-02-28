#!/bin/bash
rm -rf build
mkdir -p build && cd build

# 使用兼容模式
conan install ..   --build=missing   -o with_diskann=True   -o with_pageann=True   -o with_ut=True   -s compiler.libcxx=libstdc++11   -s build_type=Release   --output-folder=.

# Conan 2 build 命令
cmake --preset conan-release || cmake ..
cmake --build . --preset conan-release || make -j$(nproc)

echo "构建完成！"
