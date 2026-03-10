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

## 当前性能收敛（2026-03-09）

- 最新阶段级 probe 表明：当前 recall 崩塌的主损失发生在 **leaf candidate generation**，而不是 `HashPrune` 或 final `RobustPrune`。
- 观测依据：`avg_membership=4` 说明 RBC 覆盖并不稀疏，但 sampled `candidate_exact_topk_overlap` 仅约 `0.36%`；`hash_overlap` 与 `candidate_overlap` 基本一致，`final_prune` 仅带来轻微进一步损耗。
- 这意味着当前设计虽然完成了“RBC -> leaf GEMM -> HashPrune -> DiskANN postprocess` 的主链路，但与论文目标之间的关键差距已经从“后处理兼容”收敛为“候选源过局部，未能把足够多的真近邻送入后续 prune”。
- 因此下一阶段主线不再优先微调 prune，而是优先验证更强的 **candidate-source**。
- 最新收敛（2026-03-09）：`higher per-leaf candidate budget` 已在 PERF-021 中被最小实验反证——它没有提升 sampled overlap，也没有改变 edge insertion 规模，因此不能继续作为主线。
- 当前设计主线改为：
  1. 多 leaf 候选聚合 / cross-leaf candidate union
  2. 最小 leader / boundary bridge 候选来源
  3. 若 cross-leaf 原型有效，再做 candidate-source 的成本/收益小矩阵
  4. 若 cross-leaf 仍无效，再补 forced DiskANN / PiPNN 同口径候选源对照
- 2026-03-09 最新阶段切换：forced DiskANN / PiPNN 同口径对照已经完成，forced DiskANN 分支稳定恢复到 `~0.168`，而 PiPNN `graph_direct recall@10` 仍停在 `0.034~0.035`；同时 `cross_leaf_union` 只把 postprocess recall 从 `0.030` 推到 `0.039`，没有同步抬升 graph 主指标。
- 结合最新 builder probe：`candidate_exact_topk_overlap=0.0088`、`hash_exact_topk_overlap=0.0177`、`final_prune_exact_topk_overlap=0.00136`，当前设计主线不再适合继续泛化为“更强 candidate-source”，而应先补齐 **shared-leaf merge / bidirectional insertion / final retained adjacency** 的 retention 归因，确认已有候选是否在 builder 内被吞掉。
- 2026-03-09 进一步收敛：retention probe 代码已落到 `pipnn_builder`，新增 `inserted` / `pre-final retained` 指标与 insertion 结果计数；最新 Release artifact 已回收，显示 `inserted_exact_topk_overlap=0.0142276` 但 `pre_final_retained_exact_topk_overlap=0.0013587`，且 `final_prune_exact_topk_overlap` 与之基本一致。
- 这说明当前最关键的 graph-stage 损失已经从“候选不够强”进一步缩到 **retained adjacency / shared-leaf merge materialization 接缝**；`HashPrune` 与 final prune 都不是当前首要矛盾。
- 随后的 `retained_degree_limit` off/on Release A/B 已给出 stop/go：repair on 仅把 `graph_direct recall@10` 从 `0.034` 提到 `0.05`、`postprocess recall@10` 从 `0.031` 提到 `0.034`，仍远低于 `0.50` 诊断门槛。
- 因此 retention seam repair 已被降级为 **弱正信号但主线 no-go**：它说明 retained adjacency 不是完全无关，但不足以作为继续深挖的主收益方向。
- 随后 `PERF-030` 已回收最小 `boundary / leader bridge` fallback 的终态 Release artifact：`graph_direct/postprocess recall@10` 仍只到 `0.037/0.034`，而 `force_diskann_build_index=0.162`、`diskann_baseline=0.172` 继续稳定。
- 这说明 source-side fallback 与 retention seam repair 一样，都只停留在诊断失败带；当前没有证据表明再补 1-2 个局部 builder 开关就能把 PiPNN graph 拉回论文要求的可用质量。
- 因此当前设计主线已从“继续 source-side / retention-side 原型”切换为 **negative conclusion / fallback path 决策**：
  1. 正式记录当前工程约束下，PiPNN graph build path 未能恢复到 recall-gated 可用区间；
  2. 若项目仍需保留可交付路径，应优先考虑 fallback build path（例如回退到 native/forced DiskANN build path）而非继续扩 PiPNN 原型；
  3. 后续如需 benchmark，只应服务于 fallback 结论归档，而不是继续把 PiPNN graph 当作默认主线。
- 2026-03-09 PERF-032 收口后，设计口径进一步固定为：
  - PiPNN 路线当前 `graph_direct/postprocess recall@10` 处于 `0.03~0.05`，这些结果只能作为**诊断性失败证据**，不能支撑“可用性能实现”结论；
  - 同口径 `force_diskann_build_index recall@10=0.162`、`diskann_baseline recall@10=0.172` 目前只能视作 **synthetic 场景下的候选参考**，尚不能直接升级为最终 fallback 工程结论；
  - 因此默认 stop/go 决策应是：停止继续扩 source / retention 微原型，把 PiPNN graph build path 从默认主线降级为已记录的 negative conclusion；
  - 仅当最终归档明确需要 appendix 时，再执行 1 组最小 clean fallback benchmark（对应条件任务 `PERF-033`），而不是恢复 PiPNN graph 主线实验。
- 但该规则有一个前提：**参考路径自身必须足够强。**
  - 若 native/forced DiskANN 在某一测试场景下自身也低于 `recall@10 >= 0.80` 的可信区间，则该场景只能作为弱参考诊断场景；
  - 弱参考场景可以说明“PiPNN 比参考更差”，但不足以单独支撑方法级 negative conclusion；
  - 这种情况下应优先补“提升参考场景有效性/口径可信度”的任务，而不是继续沿用该场景做最终 stop/go。
- 设计约束保持不变：仍优先在 PiPNN 自有 builder/适配层内收敛，不把修改 vendored DiskANN 作为默认路径。
- 2026-03-09 PERF-033 阶段补充：当前 benchmark-validity 主线固定为“先 public dataset，再谈 fallback appendix”。
  - 已确认论文正式 benchmark 数据集至少包括 `OpenAI-ArXiv (1M, 1536d, L2)` 与 `Wikipedia-Cohere (35M, 768d, MIPS)`；
  - 当前实现侧已具备外部 `fbin/query/gt` public-dataset 入口，因此新的主 blocker 不是测试接缝，而是**缺少 paper-dataset clean artifact**；
  - 在当前远端仅 `32G RAM` 的约束下，本阶段默认优先 `Wikipedia-Cohere` 的 **1M 公共子集**：它来自论文正式数据系，数据规模可控，且可直接复用现有外部 `fbin/query/gt` 入口；全量 `Wikipedia-Cohere` 与更重的 `OpenAI-ArXiv` 下载都后置；
  - `simplewiki-openai` 之类公开数据只允许作为 smoke/接缝验证，不允许单独支撑 fallback stop/go 或 negative conclusion；
  - `Cohere 1M` 的执行入口在本阶段先固定为：`label=wikipedia-cohere-1m-ip`，远端目录 `/data/work/datasets/wikipedia-cohere-1m/{base.fbin,query.fbin,gt.ibin}`；其中 `base.fbin` 通过对论文官方 `wikipedia_base.bin` 做前缀裁切并回写 1M header 得到，`query.fbin/gt.ibin` 直接复用官方 `wikipedia_query.bin` 与 `wikipedia-1M`；后续 recall 统一通过 `KNOWHERE_PIPNN_BASE_FBIN / KNOWHERE_PIPNN_QUERY_FBIN / KNOWHERE_PIPNN_GT_IBIN / KNOWHERE_PIPNN_DATASET_LABEL` 注入，不再在测试代码里硬编码数据源；
  - Clean benchmark 命令固定为：`KNOWHERE_PIPNN_BASE_FBIN=/data/work/datasets/wikipedia-cohere-1m/base.fbin KNOWHERE_PIPNN_QUERY_FBIN=/data/work/datasets/wikipedia-cohere-1m/query.fbin KNOWHERE_PIPNN_GT_IBIN=/data/work/datasets/wikipedia-cohere-1m/gt.ibin KNOWHERE_PIPNN_DATASET_LABEL=wikipedia-cohere-1m-ip ./scripts/remote/test.sh --type Release --filter '[pipnn_diskann][e2e][recall]'`；
  - 若 `Cohere 1M` 上 native/forced baseline 自身仍 `< 0.80`，则必须把该场景继续降级为弱参考，并追加 benchmark-validity 任务，而不是直接封口。
  - 2026-03-09 最新收敛：首个 `Cohere 1M` clean run `test_20260309T102701Z_77399` 曾暴露 forced DiskANN build path 的 sample-data artifact integrity / handoff 故障；随后 `src/index/diskann/pipnn_diskann.cc` 已把 sample-query cache 与 warmup sample 的消费侧失败降级为告警继续，说明 blocker 已不再是“sample_data 一坏整轮 baseline 必挂”。
  - 当前更具体的 blocker 是 **远端 Release recall active-run hygiene / orchestration 冲突**：目标最小 rerun `test_20260309T120408Z_35719` 在测试入口即返回 `status=conflict`，因为旧的 active run `test_20260309T114120Z_21797` 仍占用同一 Release recall slot；回收日志确认它是 synthetic_diag 旧任务，而非本轮 Cohere-1M clean rerun。
  - 因此 `PERF-033.3` 的下一动作必须先处理 active run / lock hygiene，清空 Release recall slot，再重跑最小 clean Cohere-1M recall；在真正回收到 native/forced baseline 终态指标前，`Cohere 1M` 仍既不能被标成 trusted reference，也不能被标成 weak reference。

## 未来研究复活条件

当前设计文档保留一条明确边界：
- **默认工程主线** 已切到 negative conclusion + fallback build path；
- **未来研究复活** 只在额外条件满足时才重启，不应自动回流到日常性能 backlog。

如果未来需要重新挑战论文路线，优先级建议如下：
1. **oracle candidate source 对照**
   - 用 oracle/近 oracle 候选送入 HashPrune，确认问题到底在 source 还是 merge/materialization。
2. **更强的 leader / boundary bridge 原型**
   - 当前最小 fallback 已判定 no-go；若复活，必须是更强结构原型，而不是继续微调局部开关。
3. **更大 leaf / overlap / fanout 的成本-收益矩阵**
   - 只在接受更高 graph build 成本的研究模式下执行。
4. **更深的 graph import / DiskANN handoff 实验**
   - 这一步意味着不再把“不修改 vendored DiskANN”当绝对约束，而是作为新的研究分支立项。

这些方向的共通前提是：
- 结果必须先跨过 recall-gated 可讨论区间，至少进入 `recall@10 >= 0.50` 的诊断有效带；
- 在达到 `>= 0.80` 之前，所有结果都只能作为诊断证据，不能写成性能收益或可交付结论。
