#!/bin/bash
set -e

echo "========================================"
echo "使用本地缓存的二进制文件构建 Knowhere"
echo "========================================"

# 从 Conan 1 缓存复制库文件
mkdir -p build/lib
mkdir -p build/include

# 收集所有编译好的库
find ~/.conan/data -name "*.a" -exec cp {} build/lib/ \; 2>/dev/null || true
find ~/.conan/data -name "*.so" -exec cp {} build/lib/ \; 2>/dev/null || true
find ~/.conan/data -name "*.dylib" -exec cp {} build/lib/ \; 2>/dev/null || true

# 收集头文件
echo "收集头文件..."
for pkg_dir in ~/.conan/data/*/; do
    pkg_name=$(basename $pkg_dir)
    echo "  处理 $pkg_name..."
    find $pkg_dir -name "*.h" -exec cp --parents {} build/include/ \; 2>/dev/null || true
done

# 直接使用 CMake 构建，绕过 Conan
cd build
cmake ..   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_STANDARD=17   -DWITH_DISKANN=ON   -DWITH_PAGEANN=ON   -DWITH_UT=ON   -DBUILD_SHARED_LIBS=ON   -DCMAKE_PREFIX_PATH=$(pwd)/lib   -DCMAKE_INCLUDE_PATH=$(pwd)/include

make -j$(nproc)

echo "✓ 构建完成！"
