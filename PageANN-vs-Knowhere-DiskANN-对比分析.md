# PageANN vs Knowhere DiskANN 对比分析报告

## 概述

本报告对比了两个基于 DiskANN 的实现：
- **PageANN**: `/Users/ryan/Code/Paper/PageANN` - 论文实现，专注于页级图优化
- **Knowhere DiskANN**: `/Users/ryan/Code/knowhere/src/index/diskann/` - 生产级封装，集成到 Milvus

---

## 1. 核心架构差异

### 1.1 索引粒度

| 特性 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **索引粒度** | **页级** (多个向量/4KB页) | **向量级** (单向量/节点) |
| **典型配置** | 7-40 向量/页 | 1 向量/节点 |
| **图结构** | 页节点图 (Page Graph) | Vamana 向量图 |
| **磁盘I/O** | 顺序读取整页 | 随机读取单节点 |

**关键差异**:
```cpp
// PageANN: 页级索引
uint64_t _nnodes_per_sector = 7-40;  // 每页向量数
uint32_t page_id = node_id / _nnodes_per_sector;

// Knowhere: 向量级索引
// 每个节点独立存储，按需读取
```

### 1.2 构建流程

#### PageANN (两阶段构建)
```bash
# 阶段1: 构建标准 Vamana 索引
./build_vamana_disk_index \
  --data_path data.bin \
  -R 25 -L 150 -B 2.2

# 阶段2: 转换为页级图
./generate_page_graph \
  --vamana_index_path_prefix <prefix> \
  --R 25 --num_PQ_chunks 20 \
  --min_degree_per_node 23
```

**核心算法** (`mergeNodesIntoPage`):
1. BFS 收集拓扑临近的节点 (2-6 跳)
2. 按距离排序，选择最近的向量
3. 填充到 4KB 页面
4. 剩余空间用任意节点填充

#### Knowhere DiskANN (一次性构建)
```cpp
diskann::build_disk_index<DataType>({
    data_path,              // 原始数据
    index_prefix_,          // 输出前缀
    metric,                 // L2/IP/COSINE
    max_degree,             // R: 图度数 (默认48)
    search_list_size,       // L: 列表大小 (默认128)
    pq_code_size_gb,        // PQ 内存预算
    build_dram_gb,          // 构建内存预算
});
```

**直接生成**:
- 分治构建 Vamana 图 (如果内存不足)
- 生成 PQ 压缩代码
- 写入磁盘文件

---

## 2. 搜索算法对比

### 2.1 入口点选择

| 方法 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **默认** | LSH 哈希路由 | Medoid 质心 |
| **备选** | Medoid (可选) | 无 |
| **哈希桶** | 动态计算 | 无 |

#### PageANN LSH 路由
```cpp
// 计算 LSH 哈希
uint32_t query_hash = compute_lsh_hash(query, _projectionMatrix);

// 从哈希桶获取候选入口点
auto closeNodes = _buckets[query_hash];
for (auto node : closeNodes) {
    retset.insert(node, distance);
}
```

**优势**: 减少搜索起始点数量，加速收敛

#### Knowhere DiskANN
```cpp
// 使用预计算的 medoids
for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++) {
    auto medoid = _medoids[cur_m];
    retset.insert(medoid, distance(query, medoid));
}
```

**简单**: 固定使用质心，无额外计算

### 2.2 Beam Search 实现

#### PageANN (页级搜索)
```cpp
void page_search(const T* query, const uint64_t k_search,
                 const uint64_t l_search) {
    while (retset.has_unexpanded_node()) {
        Neighbor cand = retset.closest_unexpanded();

        // 加载整个页面 (多个向量)
        uint32_t page_id = cand.id / _nnodes_per_sector;
        auto [neighbors, coords] = load_page(page_id);

        // 评估页面内所有向量
        for (neighbor : neighbors) {
            pq_dist = compute_pq_distance(query, neighbor);
            retset.insert(neighbor, pq_dist);
        }
    }
}
```

**关键特性**:
- 一次 I/O 读取多个向量
- 页面缓存 (而非节点缓存)
- PQ 距离快速剪枝

#### Knowhere DiskANN (向量级搜索)
```cpp
void cached_beam_search(const T* query, const unsigned k,
                        const unsigned L, const unsigned beam_width) {
    while (retset.has_unexpanded_node()) {
        Neighbor cand = retset.closest_unexpanded();

        // 加载单个节点的邻居
        auto neighbors = get_node_neighbors(cand.id);

        // 评估每个邻居
        for (neighbor : neighbors) {
            pq_dist = compute_pq_distance(query, neighbor);
            retset.insert(neighbor, pq_dist);
        }
    }
}
```

**关键特性**:
- beam_width 控制并发 I/O (默认 8)
- 节点级缓存
- 支持迭代器模式

---

## 3. 缓存策略

### 3.1 缓存粒度

| 特性 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **缓存单位** | 页面 (4KB) | 节点 (向量+邻居) |
| **缓存内容** | 页内所有向量+邻居 | 节点坐标+邻居列表 |
| **内存效率** | 更高 (共享元数据) | 较低 (独立元数据) |

#### PageANN 页面缓存
```cpp
// 页面缓存映射
tsl::sparse_map<uint32_t, uint32_t> _cached_page_idx_map;
uint32_t* _nhood_cache_buf = nullptr;  // 邻居缓存
T *_coord_cache_buf = nullptr;         // 坐标缓存

// 检查页面是否缓存
if (_cached_page_idx_map.find(page_id) != _cached_page_idx_map.end()) {
    neighbors = _nhood_cache_buf + cached_offset;
    coords = _coord_cache_buf + cached_offset;
}
```

**优势**:
- 多个向量共享一次 I/O
- 减少元数据开销

#### Knowhere 节点缓存
```cpp
// BFS 缓存生成
pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list);

// 或采样查询缓存
pq_flash_index_->generate_cache_list_from_sample_queries(
    num_cache_queries, node_list);

// 加载到内存
pq_flash_index_->load_cache_list(node_list);
```

**优势**:
- 灵活的缓存策略
- 支持范围搜索的 BFS 缓存

### 3.2 内存预算适配

#### PageANN 自适应缓存
```cpp
std::pair<float, float> getCacheRatio(float memBudgetInGB) {
    if (memBudgetInGB < 1.0) {
        return {0.0, 0.0};  // 无缓存
    } else if (memBudgetInGB < 2.0) {
        return {0.2, 0.5};  // 部分缓存
    } else {
        return {0.5, 1.0};  // 全缓存
    }
}
```

**智能**: 根据可用内存自动调整缓存策略

#### Knowhere 手动配置
```cpp
// 用户指定缓存大小
diskann_config.search_cache_budget_gb.value() = 1.0;  // GB
diskann_config.search_cache_budget_gb_ratio.value() = 0.1;  // 10% 数据

// 计算: min(cache_gb, cache_gb_ratio * data_size)
```

**灵活**: 用户精确控制缓存预算

---

## 4. 性能优化技术

### 4.1 I/O 优化

| 技术 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **对齐** | 4KB 扇区对齐 | 使用 `LinuxAlignedFileReader` |
| **顺序读取** | 是 (整页) | 否 (随机节点) |
| **并发 I/O** | 隐式 (beam search) | 显式 (beamwidth 参数) |
| **预取** | 无 | 异步 I/O |

#### PageANN 磁盘布局
```
扇区 0: 元数据 (num_pages, nodes_per_sector, ...)
扇区 1: 页 0 (向量 + 邻居)
扇区 2: 页 1 (向量 + 邻居)
...
```

**优势**: 顺序读取，减少磁盘寻道

#### Knowhere 磁盘布局
```
{prefix}_pq_compressed.bin: PQ 压缩向量
{prefix}_disk.index: 图邻接表 (变长)
{prefix}_disk.max_base_norm.bin: 最大范数
```

**灵活**: 支持不同内存预算的配置

### 4.2 SIMD 优化

**两者都使用**:
- PQ 距离计算的 AVX2/SSE 优化
- `nmmintrin.h` 内置函数

```cpp
// 通用 PQ 距离计算
#include <nmmintrin.h>
inline float pq_l2_distance_avx2(const uint8_t* query_pq,
                                 const uint8_t* base_pq) {
    // AVX2 实现细节...
}
```

### 4.3 产品量化 (PQ)

| 特性 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **PQ 变体** | FixedChunkPQTable | FixedChunkPQTable |
| **分块数** | 12-20 (动态) | 可配置 (disk_pq_dims) |
| **旋转** | 是 (query_rotated) | 是 |
| **在线 PQ** | 否 | 是 (AiSAQ) |

#### PageANN PQ 配置
```cpp
// SIFT-100M 实验配置
0% 内存: 12 PQ chunks
10% 内存: 20 PQ chunks
20% 内存: 20 PQ chunks
```

#### Knowhere DiskANN PQ 配置
```cpp
// 标准模式
disk_pq_dims = 0;  // 使用默认 PQ

// AiSAQ 模式 (高级量化)
inline_pq = 256;         // 节点内联 PQ
pq_cache_size = 512MB;   // PQ 缓存
vectors_beamwidth = 2;   // 向量束宽
```

---

## 5. 高级特性

### 5.1 LSH 路由 (PageANN 独有)

**目的**: 加速入口点选择

```cpp
// 构建阶段: 生成哈希桶
void generate_hash_buckets() {
    for (uint32_t node = 0; node < num_nodes; node++) {
        uint32_t hash = compute_lsh_hash(vector[node]);
        _buckets[hash].push_back(node);
    }
}

// 搜索阶段: 使用哈希桶
void search_with_routing(const T* query) {
    uint32_t query_hash = compute_lsh_hash(query);

    // 从相关哈希桶获取候选
    auto& bucket = _buckets[query_hash];
    for (auto node : bucket) {
        retset.insert(node, distance(query, node));
    }
}
```

**优势**:
- 减少搜索起点数
- 对高维数据特别有效

**Knowhere 对应**: 仅使用 medoid 质心，无 LSH

### 5.2 EmbList 支持 (Knowhere 独有)

**目的**: 聚合多个向量为一个逻辑实体

```cpp
// 应用场景: 多模态搜索 (图像+文本)
class EmbListOffset {
    size_t get_el_id(size_t vid) const;      // 向量 -> 实体
    size_t get_el_len(size_t el_id) const;   // 实体向量数
};

// 搜索时聚合距离
float aggregate_distance(std::vector<float>& dists) {
    if (metric == MAX_SIM_L2) {
        return *std::min_element(dists.begin(), dists.end());
    }
}
```

**PageANN 对应**: 无此功能

### 5.3 联邦学习支持 (Knowhere 独有)

```cpp
// 返回中间结果用于模型训练
std::vector<FederResult> feder_result;
pq_flash_index_->cached_beam_search(
    query, k, L, result_ids, result_dists,
    beamwidth, use_reorder, &stats, &feder_result
);
```

**PageANN 对应**: 无此功能

### 5.4 AiSAQ 高级量化 (Knowhere 独有)

**目的**: 结合量化和精确搜索

```cpp
class PQFlashAisaqIndex : public PQFlashIndex {
    // 节点内联存储压缩向量
    uint8_t* _inline_pq_coords;

    // 两阶段搜索
    void aisaq_search(const T* query) {
        // 阶段1: PQ 粗筛
        coarse_search_with_pq();

        // 阶段2: 精确重排序
        refine_with_full_precision();
    }
};
```

**特性**:
- `inline_pq`: 节点内联 PQ 向量数
- `rearrange`: 向量重排序优化
- `pq_cache_size`: PQ 缓存
- `vectors_beamwidth`: 向量束宽

**PageANN 对应**: 无此优化

---

## 6. 代码组织

### 6.1 PageANN 结构

```
Paper/PageANN/
├── include/
│   ├── pq_flash_index.h         # 页索引核心
│   ├── disk_utils.h             # 页构建工具
│   ├── index.h                  # 内存 Vamana
│   ├── neighbor.h               # 邻居优先队列
│   └── parameters.h             # 参数定义
├── src/
│   ├── pq_flash_index.cpp       # 页搜索实现
│   ├── disk_utils.cpp           # 页图构建
│   └── index.cpp                # Vamana 实现
└── apps/
    ├── build_vamana_disk_index.cpp   # 构建工具
    ├── generate_page_graph.cpp       # 页转换工具
    └── search_disk_index.cpp         # 搜索工具
```

**特点**:
- 命令行工具驱动
- 两阶段构建 (Vamana → Page Graph)
- 专注算法研究

### 6.2 Knowhere DiskANN 结构

```
knowhere/
├── src/index/diskann/
│   ├── diskann.cc               # 索引节点实现
│   ├── diskann_aisaq.cc         # AiSAQ 变体
│   ├── diskann_config.h         # 配置类
│   └── aisaq_config.h           # AiSAQ 配置
├── thirdparty/DiskANN/
│   ├── include/diskann/
│   │   ├── pq_flash_index.h     # 上游 PQFlashIndex
│   │   ├── pq_flash_aisaq_index.h
│   │   ├── aux_utils.h          # 构建工具
│   │   └── parameters.h
│   └── src/                     # DiskANN 库源码
└── cmake/libs/libdiskann.cmake  # 构建集成
```

**特点**:
- 集成到 Knowhere 框架
- 统一的 IndexNode 接口
- 生产级封装

---

## 7. 性能对比 (基于论文和配置)

### 7.1 内存预算

| 内存比例 | PageANN | Knowhere DiskANN |
|----------|---------|------------------|
| **0%** | 支持 (7 向量/页) | 支持 (无缓存) |
| **10%** | 7 向量/页, 12 PQ chunks | 缓存节点 |
| **20%** | 18 向量/页, 20 PQ chunks | 更多缓存 |
| **30%+** | 接近内存索引 | 接近内存索引 |

### 7.2 I/O 效率

| 指标 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **随机 I/O** | 减少 7-40x | 基准 |
| **顺序读取** | 是 (4KB 页) | 否 |
| **缓存命中率** | 更高 (页级) | 较低 (节点级) |

**论文结果** (PageANN):
- 相比 DiskANN: QPS 提升 **2-3x**
- 延迟降低 **30-50%**

---

## 8. 适用场景

### 8.1 PageANN 更适合

✅ **SSD 存储的大规模索引** (100M+ 向量)
✅ **有限内存预算** (0-20% 数据大小)
✅ **高并发查询** (页缓存效率高)
✅ **研究环境** (算法创新)

**原因**:
- 页级组织减少随机 I/O
- LSH 路由加速搜索
- 专为 SSD 优化

### 8.2 Knowhere DiskANN 更适合

✅ **生产环境** (Milvus 集成)
✅ **多模态搜索** (EmbList)
✅ **联邦学习** (中间结果)
✅ **高级量化** (AiSAQ)
✅ **多样化硬件** (GPU+CPU 混合)

**原因**:
- 企业级封装
- 丰富的功能特性
- 完善的测试和监控

---

## 9. 潜在改进方向

### 9.1 Knowhere 可以借鉴 PageANN

1. **页级索引**:
   ```cpp
   // 添加 PageIndexNode
   class PageANNIndexNode : public DiskANNIndexNode {
       uint64_t _nnodes_per_sector;
       void merge_nodes_into_pages();
       void page_search(...);
   };
   ```

2. **LSH 路由**:
   ```cpp
   // 在 PQFlashIndex 中添加
   void generate_hash_buckets(const std::vector<uint32_t>& medoids);
   uint32_t compute_lsh_hash(const T* vector);
   ```

3. **自适应缓存**:
   ```cpp
   // 根据内存预算自动调整
   auto [cache_ratio, sample_ratio] = getCacheRatio(mem_budget_gb);
   ```

4. **两阶段构建**:
   ```cpp
   // 添加页转换工具
   int convert_vamana_to_page_graph(
       const std::string& vamana_prefix,
       const std::string& page_prefix,
       int nodes_per_sector
   );
   ```

### 9.2 PageANN 可以借鉴 Knowhere

1. **统一接口**:
   ```cpp
   // 实现 IndexNode 接口
   class PageANNIndexNode : public IndexNode {
       Status Build(...) override;
       expected<DataSetPtr> Search(...) override;
   };
   ```

2. **高级量化**:
   ```cpp
   // 添加 AiSAQ 支持
   class PQFlashAisaqPageIndex : public PQFlashPageIndex {
       uint8_t* _inline_pq_coords;
       void aisaq_page_search(...);
   };
   ```

3. **迭代器模式**:
   ```cpp
   // 支持流式搜索
   class PageANNIterator {
       std::unique_ptr<Iterator> next();
   };
   ```

---

## 10. 总结

| 维度 | PageANN | Knowhere DiskANN |
|------|---------|------------------|
| **研究价值** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **生产就绪** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **I/O 效率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **功能丰富** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**关键结论**:
- **PageANN**: 学术研究突破，页级图组织显著提升 SSD 上的 I/O 效率
- **Knowhere DiskANN**: 生产级实现，功能完善，适合集成到 Milvus

**建议**: Knowhere 可以考虑集成 PageANN 的页级索引作为新的索引类型 (INDEX_DISKANN_PAGE)，在 SSD 环境下提供更好的性能。

---

## 关键文件路径

**PageANN**:
- 核心实现: `/Users/ryan/Code/Paper/PageANN/include/pq_flash_index.h`
- 页构建: `/Users/ryan/Code/Paper/PageANN/src/disk_utils.cpp`
- 使用文档: `/Users/ryan/Code/Paper/PageANN/workflows/PageANN_usage.md`

**Knowhere DiskANN**:
- 索引节点: `/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc`
- 配置类: `/Users/ryan/Code/knowhere/src/index/diskann/diskann_config.h`
- 上游库: `/Users/ryan/Code/knowhere/thirdparty/DiskANN/`
- 测试: `/Users/ryan/Code/knowhere/tests/ut/test_diskann.cc`
