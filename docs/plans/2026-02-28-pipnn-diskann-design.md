# PiPNN-DiskANN 设计文档

**日期：** 2026-02-28
**参考论文：** PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing (arxiv 2602.21247)

---

## 目标

实现 `PIPNN_DISKANN` 索引：用 PiPNN 算法替换 DiskANN 的 Vamana 构建路径，查询路径复用现有 `PQFlashIndex`，磁盘格式严格兼容 DiskANN。

**预期收益：** 构建速度 5-13× 加速，查询性能不变。

---

## 架构

### 数据流

```
Build(dataset):
  [Stage 1 - PiPNN Graph]
  RBC 分区 (Multi-level Fanout)
     ↓
  GEMM 叶子构建 (Eigen)
     ↓
  HashPrune 流式剪枝
     ↓
  (可选) 最终 RobustPrune
     ↓ in-memory Vamana 图
  write_vamana_graph() → .graph 文件

  [Stage 2 - DiskANN Format]
  diskann::generate_quantized_data()  → _pq_pivots.bin, _pq_compressed.bin
  diskann::create_disk_layout()       → _disk.index

Search(query):
  → diskann::PQFlashIndex（零改动）
```

### 新增文件

```
src/index/diskann/
├── pipnn_diskann.cc            # PiPNNDiskANNIndexNode
├── pipnn_diskann_config.h      # Config (继承 DiskANNConfig)
└── impl/
    ├── pipnn_builder.h/.cc     # 算法入口
    ├── hash_prune.h            # HashPrune
    └── rbc_partition.h/.cc     # RBC + Multi-level Fanout

cmake/libs/
└── libpipnn_diskann.cmake      # Eigen 依赖 + 构建配置

tests/ut/
└── test_pipnn_diskann.cc       # 单元测试
```

---

## 模块设计

### 1. HashPrune

**文件：** `src/index/diskann/impl/hash_prune.h`

每个点维护固定大小的预留池（`ℓ_max` 槽位），每槽 8 bytes：

```
struct Slot {
    uint32_t id;      // 4B - 点 ID
    uint16_t hash;    // 2B - m-bit 残差哈希
    uint16_t dist_bf; // 2B - bfloat16 距离
};
```

**残差哈希：** `h_p(c) = hash(c - p)` 用 m 个随机超平面的 sketch 近似，避免存储全维向量。

**插入逻辑（history-independent）：**

```
insert(c, dist):
  h = residual_hash(sketch_p, sketch_c)
  找到碰撞槽 h' == h:
    dist < slot.dist → 替换（同方向取更近）
    dist ≥ slot.dist → 丢弃
  无碰撞，size < ℓ_max:
    直接插入
  无碰撞，池已满:
    dist < farthest.dist → 替换最远槽
    dist ≥ farthest.dist → 丢弃
```

**参数：** `m=12`（哈希位数），`ℓ_max=64`（对应 DiskANN max_degree）

### 2. RBC 分区 + Multi-level Fanout

**文件：** `src/index/diskann/impl/rbc_partition.h/.cc`

```cpp
struct Config {
    uint32_t leaf_max_size = 1000;
    uint32_t fanout_l1     = 10;   // 顶层每点分配 k1 个 leader
    uint32_t fanout_l2     = 3;    // 第二层 fanout
    uint32_t fanout_rest   = 1;    // 更深层
    uint32_t num_leaders   = 0;    // 0 = 自动: min(√n, 1000)
};
```

**递归划分（消除顶层重复，3.35× 加速）：**

```
partition_recursive(points, depth):
  if |points| ≤ leaf_max_size → 输出叶子

  fanout = fanout_l1 (depth=0) | fanout_l2 (depth=1) | 1 (depth≥2)
  选 ℓ = min(√|points|, 1000) 个随机 leader
  每点分配到 fanout 个最近 leader → ℓ 个重叠子集
  parallel_for 每个子集: partition_recursive(子集, depth+1)
```

重叠保证跨分区图连通性。

### 3. GEMM 叶子构建

**文件：** `src/index/diskann/impl/pipnn_builder.h/.cc`

```
process_leaf(leaf):
  // Step 1: 提取叶子向量 → Eigen 矩阵 X (n_leaf × d)
  // Step 2: L2 距离矩阵
  //   D = -2·X·Xᵀ + ||X||²·1ᵀ + 1·||X||²ᵀ   (Eigen GEMM)
  //   D.diagonal() = ∞
  // Step 3: 每行 partial_sort 取前 k 个
  // Step 4: 双向 k-NN 边 → 插入全局 HashPrune
  //   per-point spinlock 保护并发写入
```

**并行：** 叶子级并行（`std::async` / OpenMP），叶子间无依赖。

### 4. 最终 RobustPrune（可选）

HashPrune 输出后，对每点的邻居列表再跑一次 `diskann::RobustPrune(alpha)`，利用实数 α 参数精细调节稀疏度。候选列表已很小（HashPrune 已粗剪），开销低。

### 5. DiskANN 磁盘格式

**.graph 文件格式：**

```
Header: uint32 num_points, uint32 max_degree
Per-point: uint32 num_neighbors, uint32[] neighbor_ids
```

**Build 流程（复用 DiskANN 函数）：**

```cpp
// Stage 1: 写图文件
write_vamana_graph(graph, prefix + ".graph");
write_bin_file(data, n, d, prefix + ".fbin");

// Stage 2: PQ 训练（复用）
diskann::generate_quantized_data(prefix + ".fbin", ...);

// Stage 3: 磁盘布局（复用）
diskann::create_disk_layout(prefix + ".fbin",
                            prefix + "_pq_compressed.bin",
                            prefix + ".graph",
                            prefix + "_disk.index");
```

Search / Serialize / Deserialize 与 DiskANNIndexNode 完全相同。

---

## 配置参数

继承 DiskANNConfig 全部参数，新增：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pipnn_leaf_max_size` | int | 1000 | RBC 叶子最大点数 |
| `pipnn_fanout_l1` | int | 10 | 顶层 fanout |
| `pipnn_fanout_l2` | int | 3 | 第二层 fanout |
| `pipnn_k_nn` | int | 3 | 叶子内 k-NN 的 k |
| `pipnn_hash_bits` | int | 12 | HashPrune m 值 |
| `pipnn_final_prune` | bool | true | 是否运行最终 RobustPrune |

---

## 依赖

- **Eigen 3.4**（header-only，加入 conanfile）：GEMM 计算
- **DiskANN thirdparty**（已有）：PQ 训练 + 磁盘布局，零改动
- **OpenMP / std::async**（已有）：叶子级并行

---

## 测试策略

1. **HashPrune 单元测试**：验证 history-independent 性质（不同插入顺序结果相同）
2. **RBC 单元测试**：验证叶子覆盖率（每点至少出现一次）
3. **构建对比测试**：相同数据集，PiPNN vs DiskANN 的 10@10 recall 对比
4. **格式兼容测试**：PiPNN 建图后用 DiskANN Search 加载查询
