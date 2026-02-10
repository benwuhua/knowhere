# Conan 2 快速参考 - Knowhere

## 🚀 快速开始

### 安装 Conan 2

```bash
pip install conan==2.0.0
# 或
pip install --upgrade conan
```

### 构建命令

```bash
# 方式 1: 使用新脚本（推荐）
./build.sh --with-diskann --with-pageann --with-ut

# 方式 2: 使用专用脚本
./build_diskann_benchmark.sh

# 方式 3: 手动构建
rm -rf build && mkdir build && cd build
conan install .. --build=missing \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_ut=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release \
  --output-folder=.
conan build .. --build-dir=.
```

## 📋 主要变化

### conanfile.py 变化

```diff
- from conans import tools
- required_conan_version = ">=1.55.0"
+ required_conan_version = ">=2.0.0"

- cxx_std_flag = tools.cppstd_flag(self.settings)
- cxx_std_value = cxx_std_flag.split("=")[1] if cxx_std_flag else "c++17"
+ if self.settings.compiler.get_safe("cppstd"):
+     cxx_std_value = f"c++{self.settings.compiler.cppstd}"
+ else:
+     cxx_std_value = f"c++17"
```

### 构建命令变化

```diff
# Conan 1
conan install .. \
  --build=missing \
  -o with_diskann=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release

conan build ..

# Conan 2
conan install .. \
  --build=missing \
  -o with_diskann=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release \
+  --output-folder=.

- conan build ..
+ conan build .. --build-dir=.
```

## 🛠️ 构建选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--with-diskann` | 启用 DiskANN | False |
| `--with-pageann` | 启用 PageANN（需要 DiskANN）| False |
| `--with-ut` | 构建单元测试 | False |
| `--with-benchmark` | 构建性能测试 | False |
| `--with-asan` | 启用地址 sanitizer | False |
| `--debug` | Debug 构建 | Release |
| `--clean` | 清理构建目录 | False |

## 🏗️ 编译器设置

| 操作系统 | compiler.libcxx |
|----------|-----------------|
| macOS | `libc++` |
| Linux | `libstdc++11` |

## 🧪 测试

```bash
# 运行所有测试
cd build
./Release/tests/ut/knowhere_tests

# 运行 PageANN 测试
./Release/tests/ut/knowhere_tests "[pageann]"

# 运行 DiskANN 测试
./Release/tests/ut/knowhere_tests "[diskann]"
```

## 🐛 故障排查

### 清理缓存

```bash
# 清理 Conan 缓存
conan remove "*" -c

# 清理构建目录
rm -rf build

# 重新构建
./build.sh --with-diskann --clean
```

### 查看详细日志

```bash
# Conan 详细输出
conan install .. --build=missing -v -v

# CMake 详细输出
conan build .. --build-dir=. -- -DCMAKE_VERBOSE_MAKEFILE=ON
```

## 📚 更多信息

- [完整迁移指南](CONAN2_MIGRATION_GUIDE.md)
- [Conan 2 官方文档](https://docs.conan.io/2/)
- [Knowhere GitHub](https://github.com/milvus-io/knowhere)

---

**提示:** 使用 `./build.sh --help` 查看所有选项
