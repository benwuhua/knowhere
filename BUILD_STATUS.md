# Knowhere 编译状态 - Conan 1 配置

## 📅 日期: 2026-02-10

## ✅ 当前配置

### Conan 版本
- **主版本**: Conan 1.66.0
- **文件**: `/Users/ryan/Code/knowhere/conanfile.py` (Conan 1 兼容)
- **备用版本**: `/Users/ryan/Code/knowhere/conanfile_v2.py` (Conan 2，保存供未来使用)

### PageANN 配置
- ✅ 已修复 PageANN 编译错误（删除冲突的实现）
- ✅ PageANNIndexNode 在 `src/index/diskann/diskann.cc` 中实现
- ✅ PageANNConfig 在 `src/index/diskann/pageann_config.h` 中定义
- ✅ 构建选项已启用: `-DKNOWHERE_WITH_DISKANN -DKNOWHERE_WITH_PAGEANN`

### 容器编译状态
- **容器**: knowhere-build (f92a04975eb2)
- **构建目录**: `/knowhere/build/Release/`
- **状态**: 编译进行中（9个并行进程）
- **使用**: 本地缓存（无需网络下载）

## 📁 文件结构

### 源代码文件
```
/Users/ryan/Code/knowhere/
├── conanfile.py              # Conan 1 (当前使用)
├── conanfile_v2.py           # Conan 2 (备用)
├── src/index/diskann/
│   ├── diskann.cc            # 包含 PageANNIndexNode 实现
│   └── pageann_config.h       # PageANNConfig 配置
└── tests/ut/test_pageann.cc  # PageANN 测试
```

### 已删除的冲突文件
```
include/knowhere/index/pageann/   # 整个目录已删除
src/index/pageann/                # 整个目录已删除
src/index/diskann/pageann.h     # 已删除
```

## 🔄 编译流程

### 1. 依赖安装（已完成）
```bash
conan install .. --build=missing \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_ut=True \
  -s compiler.libcxx=libstdc++11 \
  -s build_type=Release
```
所有依赖从本地缓存获取，无需网络下载。

### 2. 编译（进行中）
```bash
conan build ..
```
- 正在编译 DiskANN 组件
- 使用 4 个并行编译作业 (-j4)
- 编译路径: `/knowhere/build/Release/`

### 3. 预期结果
```
/knowhere/build/Release/
├── lib/
│   └── libknowhere.so       # 最终库文件
└── tests/ut/
    └── knowhere_tests        # 测试可执行文件
```

## 🎯 验证步骤

编译完成后，运行测试：

```bash
# 在容器中
cd /knowhere/build
./Release/tests/ut/knowhere_tests "[pageann]"

# 或只编译，不运行测试
cd /knowhere/build
./Release/lib/libknowhere.so  # 检查库文件
```

## 📝 迁移到 Conan 2 的计划

### 当前阻塞
- boost/1.83.0 - 1.86.0 的 conanfile.py 使用 Conan 1 API
- folly, glog 等其他依赖也有类似问题
- 需要等待上游包维护者更新

### 下一步
1. 关注 boost/Conan2 配方更新
2. 监控 folly、glog 等包的 Conan 2 支持
3. 使用 `conanfile_v2.py` 进行迁移测试
4. 参考 `CONAN2_MIGRATION_GUIDE.md` 进行完整迁移

## 💡 Conan 1 vs Conan 2

| 特性 | Conan 1 | Conan 2 | 状态 |
|------|----------|----------|------|
| 当前使用 | ✅ | - | 生产环境 |
| 代码准备 | ✅ | ✅ | 就绪 |
| 依赖支持 | ✅ | ⏳ | 等待上游 |
| 稳定性 | ✅ | ⏳ | 待验证 |

## 🔧 本地开发

### macOS 构建
```bash
cd /Users/ryan/Code/knowhere
rm -rf build
mkdir -p build && cd build
conan install .. \
  --build=missing \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_ut=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release
conan build ..
```

### 切换到 Conan 2（未来）
```bash
# 切换文件
mv conanfile.py conanfile_v1.py
mv conanfile_v2.py conanfile.py

# 重新构建
rm -rf build
mkdir -p build && cd build
conan install .. \
  --build=missing \
  -o with_diskann=True \
  -o with_pageann=True \
  -o with_ut=True \
  -s compiler.libcxx=libc++ \
  -s build_type=Release
conan build ..
```

---

**最后更新**: 2026-02-10 07:48 (容器编译进行中)
**状态**: ✅ PageANN 修复完成，⏳ 等待编译完成
