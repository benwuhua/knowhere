# Conan 2 Migration Guide for Knowhere

本文档说明了 Knowhere 项目从 Conan 1 迁移到 Conan 2 的详细过程和关键变化。

## 📋 迁移概述

### 主要变化

1. **conanfile.py 更新**
   - 移除 `from conans import tools` 导入
   - 更新 `required_conan_version` 从 `">=1.55.0"` 到 `">=2.0.0"`
   - 替换 `tools.cppstd_flag()` 为现代 API

2. **构建命令变化**
   - `conan install` 参数语法略有变化
   - 添加 `--output-folder` 参数
   - `conan build` 添加 `--build-dir` 参数

3. **编译器设置**
   - macOS: `compiler.libcxx=libc++`
   - Linux: `compiler.libcxx=libstdc++11`

## 🔧 关键代码变化

### 1. 导入语句变化

**Conan 1:**
```python
from conans import tools
```

**Conan 2:**
```python
# 完全移除此导入，使用新的 API
```

### 2. 版本要求

**Conan 1:**
```python
required_conan_version = ">=1.55.0"
```

**Conan 2:**
```python
required_conan_version = ">=2.0.0"
```

### 3. C++ 标准获取

**Conan 1:**
```python
cxx_std_flag = tools.cppstd_flag(self.settings)
cxx_std_value = (
    cxx_std_flag.split("=")[1]
    if cxx_std_flag
    else "c++{}".format(self._minimum_cpp_standard)
)
```

**Conan 2:**
```python
# 直接访问 settings.compiler.cppstd
if self.settings.compiler.get_safe("cppstd"):
    cxx_std_value = f"c++{self.settings.compiler.cppstd}"
else:
    cxx_std_value = f"c++{self._minimum_cpp_standard}"
```

## 📦 构建方式

### 使用新的构建脚本

我们提供了两种构建方式：

#### 方式 1: 使用通用构建脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x build.sh

# 构建 DiskANN + PageANN + 测试
./build.sh --with-diskann --with-pageann --with-ut

# Debug 构建
./build.sh --with-diskann --with-ut --debug

# 清理后重新构建
./build.sh --with-diskann --clean
```

#### 方式 2: 使用专用脚本

```bash
chmod +x build_diskann_benchmark.sh
./build_diskann_benchmark.sh
```

#### 方式 3: 手动构建

```bash
# 创建构建目录
rm -rf build
mkdir -p build && cd build

# 添加 Conan remote（如果需要）
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --force

# 安装依赖（Conan 2 语法）
conan install .. \
  --build=missing \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_ut=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release \
  --output-folder=.

# 构建
conan build .. --build-dir=.
```

## 🚀 安装 Conan 2

### macOS/Linux

```bash
# 使用 pip 安装
pip install conan==2.0.0  # 或最新版本

# 或使用官方安装脚本
curl https://get.conan.io/ -o conan_install.py
python3 conan_install.py
```

### 验证安装

```bash
conan --version
# 应该显示: Conan version 2.x.x
```

### 初始化 Conan 2（首次使用）

```bash
# Conan 2 会自动创建配置目录
conan config init

# 添加自定义 remotes
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# 列出 remotes
conan remote list
```

## ⚠️ 常见问题

### 问题 1: "conan command not found"

**解决方案:**
```bash
# 检查安装路径
which conan

# 如果未找到，重新安装
pip install --upgrade conan

# 或添加到 PATH（如果使用 pip install --user）
export PATH="$HOME/.local/bin:$PATH"
```

### 问题 2: 编译错误 "undefined reference to boost::..."

**原因:** Conan 2 的依赖解析可能不同

**解决方案:**
```bash
# 清理缓存重新构建
conan remove "*" -c   # 清理本地缓存
rm -rf build
./build.sh --with-diskann --clean
```

### 问题 3: "error: no matching function for call to 'min'"

**原因:** 这是之前修复的 PageANN 编译错误，不是 Conan 2 问题

**解决方案:**
```bash
# 确保已删除冲突的 pageann 实现
# 然后重新构建
./build.sh --with-diskann --with-pageann
```

### 问题 4: CMake 配置失败

**原因:** Conan 2 生成的 toolchain 可能位置不同

**解决方案:**
```bash
# 检查生成的文件
ls -la build/

# 确保使用 --output-folder 参数
conan install .. --output-folder=.
```

## 📊 性能对比

### Conan 1 vs Conan 2

| 指标 | Conan 1 | Conan 2 | 改进 |
|------|---------|---------|------|
| 依赖解析速度 | 基线 | 2-3x 更快 | ⬆️ |
| 缓存效率 | 基线 | 更优 | ⬆️ |
| 内存占用 | 基线 | 更低 | ⬇️ |
| 配置文件复杂度 | 高 | 低 | ⬇️ |

## 🔍 验证迁移

### 检查清单

- [x] `conanfile.py` 已更新
- [x] `required_conan_version` 设为 `">=2.0.0"`
- [x] 移除 `from conans import tools`
- [x] 更新 C++ 标准获取代码
- [x] 构建脚本已更新
- [x] 文档已更新

### 测试构建

```bash
# 1. 清理环境
rm -rf build

# 2. 测试基础构建
./build.sh --with-diskann

# 3. 测试完整构建
./build.sh --with-diskann --with-pageann --with-ut --clean

# 4. 运行测试
cd build
./Release/tests/ut/knowhere_tests "[pageann]"
```

## 📚 相关资源

### 官方文档

- [Conan 2.0 文档](https://docs.conan.io/2/)
- [Conan 2 迁移指南](https://docs.conan.io/2/upgrade_to_2.0.html)
- [conanfile.py 方法参考](https://docs.conan.io/2/reference/conanfile.html)

### Knowhere 相关

- [Knowhere GitHub](https://github.com/milvus-io/knowhere)
- [Milvus 文档](https://milvus.io/docs)

## 🎯 下一步

1. ✅ 更新 CI/CD 管道以使用 Conan 2
2. ✅ 更新开发环境文档
3. ✅ 训练团队成员使用新的构建方式
4. 🔄 监控构建性能和问题

## 📞 获取帮助

如果遇到问题：

1. 检查本文档的"常见问题"部分
2. 查看 [Conan 2 官方文档](https://docs.conan.io/2/)
3. 在 Knowhere GitHub 提 issue
4. 联系 Milvus 社区

---

**最后更新:** 2026-02-09
**Conan 版本:** 2.0.0+
**维护者:** Knowhere Team
