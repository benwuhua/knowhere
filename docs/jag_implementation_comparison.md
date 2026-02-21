# JAG 实现对比分析：论文 vs Knowhere

## 1. 核心算法对比

### 1.1 组合距离公式

| 特性 | 论文 Weight-JAG | Knowhere 实现 |
|------|----------------|---------------|
| **公式** | `combined = vec_dist + weight * filter_dist * normalized_h[id]` | `combined = vec_dist + weight` (invalid) 或 `vec_dist` (valid) |
| **归一化** | 每点独立 `normalized_h` (通过采样计算) | 无归一化 |
| **Filter距离** | 0/1 (label) 或距离值 (range, sparse) | 0/1 (label only) |
| **权重调度** | 多层级权重 `{1, 0}`, `{2, 0}`, `{10, 0}` 等 | 单一权重 (默认 0.1) |

### 1.2 关键差异

```cpp
// 论文实现
double normalized_h[id] = std_vec_dist / std_filter_dist;  // 预计算
combined = vec_dist + weight * filter_dist * normalized_h[id];

// Knowhere 实现
combined = is_invalid ? (vec_dist + weight) : vec_dist;
```

**问题：** 我们的实现缺少 `normalized_h`，这意味着：
- 论文: 权重会根据每个点的数据分布自动调整
- 我们: 权重是全局固定值，需要手动调优

---

## 2. 过滤类型支持

| 过滤类型 | 论文支持 | Knowhere 支持 | 优先级 |
|----------|----------|---------------|--------|
| Label (分类标签) | ✅ 0/1 | ✅ 0/1 | 高 |
| Range (数值范围) | ✅ 距离到边界 | ❌ | 中 |
| Sparse (稀疏标签集) | ✅ IDF加权 | ❌ | 中 |
| Subset (位图子集) | ✅ Hamming距离 | ❌ | 低 |
| Bool (布尔表达式) | ✅ Hamming距离 | ❌ | 低 |

**结论：** 我们只支持最基础的 Label 过滤，限制了 JAG 的适用场景。

---

## 3. 搜索策略对比

### 3.1 论文 Multi-Tier 策略

```cpp
// 论文配置示例
SIFT1M:        {1}              // 单层
LAION:         {1, 0}           // 先加权搜索，再无权重搜索
arxiv+label:   {10, 0}          // 高权重探索，无权重精化
msturing10m:   {5, 1, 0}        // 三层搜索
```

**Multi-Tier 逻辑：**
1. 第一层：高权重探索，优先找有效节点
2. 后续层：逐渐降低权重，精化搜索
3. 最后一层：无权重，确保召回率

### 3.2 Knowhere 策略

```cpp
// 当前实现：单一权重搜索
enable_jag = true;
jag_filter_weight = 0.1;  // 固定权重
```

**缺失：**
- 无 Multi-Tier 搜索
- 无动态权重调整
- 无后处理精化阶段

---

## 4. 性能优化技术对比

| 技术 | 论文 | Knowhere | 状态 |
|------|------|----------|------|
| Early Pruning | ✅ 基于 filter_dist 阈值 | ✅ 基于 4x worst_dist | 实现 |
| Prefetching | ✅ 预取邻居数据 | ❌ | 缺失 |
| 批量 Filter 计算 | ✅ 预计算所有邻居 | ❌ 逐个计算 | 缺失 |
| 并行建图 | ✅ ParlayLib | ✅ OpenMP | 实现 |
| Alpha 机制 | ❌ 论文未使用 | ✅ 概率准入 | 额外实现 |

---

## 5. 性能对比

### 5.1 论文声称的性能

| 指标 | 论文结果 |
|------|----------|
| 节点访问减少 | 20-40% (高过滤率) |
| 有效访问比提升 | 50-100% |
| 召回率 | 与 Post-filtering 相当 |
| QPS | 高过滤率时提升显著 |

### 5.2 Knowhere 测试结果

| 过滤率 | QPS 变化 | 召回率变化 |
|--------|---------|-----------|
| 10% | +50-60% | 保持 |
| 30% | -10~-20% | +5% |
| 50% | -35~40% | +10% |
| 70% | -60% | +16% |

**问题：**
- 高过滤率时 QPS 下降明显
- 召回率提升需要牺牲 QPS
- 低过滤率场景收益最大

---

## 6. 生产落地评估

### 6.1 当前实现的优势 ✅

1. **简单易用** - 只需设置 `enable_jag=true`
2. **兼容性好** - 不改变索引结构，运行时开关
3. **低风险** - 可随时回退到 Alpha 模式
4. **低过滤率效果好** - 10-20% 过滤率时 QPS +50%+

### 6.2 当前实现的劣势 ❌

1. **缺少 normalized_h** - 权重需要手动调优
2. **只支持 Label 过滤** - 不支持 Range/Sparse
3. **无 Multi-Tier** - 无法兼顾高召回和高 QPS
4. **高过滤率 QPS 下降** - 50%+ 过滤率时 QPS -30~60%

### 6.3 生产建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 低过滤率 (<20%) | ✅ 启用 JAG | QPS 提升 50%+ |
| 中过滤率 (20-40%) | ⚠️ 谨慎启用 | QPS 可能下降 |
| 高过滤率 (>40%) | ❌ 不建议 | QPS 下降严重 |
| 召回率敏感 | ⚠️ 可启用 | 召回率提升但 QPS 下降 |

---

## 7. 已实现的改进 ✅

### 7.1 动态权重调整 (已实现)

在 `HnswSearcher.h` 中添加了基于观察到的过滤率的动态权重调整：

```cpp
// 基于观察到的过滤率动态调整权重
inline float get_adaptive_weight(int total_seen, int invalid_seen) const {
    if (!adaptive_weight || total_seen < 10) {
        return base_filter_weight;
    }
    float valid_ratio = static_cast<float>(total_seen - invalid_seen) / total_seen;
    valid_ratio = std::max(valid_ratio, 0.01f);
    float log_weight = std::log(1.0f / valid_ratio);
    return base_filter_weight * log_weight;
}
```

### 7.2 基于过滤率的自动权重 (已实现)

```cpp
// 根据估计的过滤率自动选择最优权重
inline float get_auto_weight_for_filter_ratio(float filter_ratio) const {
    if (filter_ratio <= 0.15f) return 0.1f;   // 低过滤率：优先 QPS
    else if (filter_ratio <= 0.30f) return 0.3f;  // 中等：平衡
    else if (filter_ratio <= 0.50f) return 0.5f;  // 高：优先召回
    else return 1.0f;  // 极高：需要激进权重
}
```

### 7.3 早剪枝优化 (已实现)

跳过明显不可能进入结果集的节点，减少不必要的向量距离计算：

```cpp
// 只有当 filter 惩罚远大于最差距离时才剪枝
inline bool should_prune_by_filter(float filter_dist, float current_weight, float worst_dist) const {
    return (filter_dist * current_weight) > (worst_dist * 4.0f);
}
```

---

## 8. 待实现的改进

### 8.1 中期优化 (1-2月)

1. ~~**实现 normalized_h**~~ ✅ 已实现在线估计版本
   - 使用在线估计而非预计算
   - 公式: `normalized_h = 0.1 * sum_vec_dist / sum_filter_dist`
   - 无需额外存储，动态适应数据分布

2. ~~**支持 Range 过滤**~~ ✅ 已实现基础版本
   - 实现 `RangeFilterSet`, `RangeFilterConstraint`, `RangeFilterDistance`
   - 距离 = 到范围边界的比例距离
   - 范围内点距离=0，范围外点距离为到最近边界的距离

3. ~~**支持 Sparse 过滤**~~ ✅ 已实现 IDF 加权版本
   - 实现 `SparseFilterSet`, `SparseFilterConstraint`, `SparseFilterDistance`
   - 每点可有多标签，支持部分匹配
   - IDF 加权：稀有标签权重更高

### 8.2 长期优化 (3-6月)

1. **完整 Filter 类型支持**
   - Sparse filter (IDF 加权)
   - Subset filter (Hamming 距离)
   - Bool filter

2. **图构建时 JAG 感知**
   - RobustPrune 使用组合距离
   - 优化邻居选择

---

## 8. 论文结论可信度评估

### 8.1 可信的结论 ✅

1. **JAG 能减少无效节点访问** - 我们的结果验证了这一点
2. **低过滤率时 QPS 提升** - 我们的测试确认 +50%+
3. **召回率可以保持或提升** - 高过滤率时召回率 +10-16%

### 8.2 需要谨慎的结论 ⚠️

1. **高过滤率 QPS 提升** - 我们的结果显示 QPS 下降
2. **通用权重设置** - 论文的权重依赖 normalized_h
3. **所有过滤类型效果** - 我们只测试了 Label

### 8.3 论文未明确说明的问题 ❓

1. **索引构建开销** - normalized_h 计算需要额外时间
2. **内存开销** - 存储每点的 normalized_h
3. **Multi-Tier 的延迟增加** - 多层搜索会增加延迟

---

## 9. 总结

### JAG 能否落地生产？

**答案：部分可以，需要场景限制**

| 落地条件 | 状态 |
|----------|------|
| 低过滤率场景 (<20%) | ✅ 可直接使用 |
| 中等过滤率 (20-40%) | ⚠️ 需要调优权重 |
| 高过滤率 (>40%) | ❌ 当前实现不适用 |
| 召回率敏感场景 | ✅ 可使用，QPS 换召回率 |

### 下一步行动

1. ~~**立即** - 添加基于过滤率的自动权重调整~~ ✅ 已完成
2. **短期** - 实现简单的 normalized_h 近似
3. **中期** - 支持 Range 过滤，增加 Multi-Tier
4. **长期** - 完整实现论文所有特性

### 已实现的改进 (2025-02)

| 改进 | 状态 | 效果 |
|------|------|------|
| 动态权重调整 | ✅ 已实现 | 根据观察到的过滤率自动调整 |
| 自动权重选择 | ✅ 已实现 | `get_auto_weight_for_filter_ratio()` |
| 早剪枝优化 | ✅ 已实现 | 减少不必要的向量距离计算 |
| 默认权重优化 | ✅ 已完成 | 0.1 (优化低过滤率场景) |
| normalized_h 在线估计 | ✅ 已实现 | 论文核心公式，无需预计算 |
| Label 过滤支持 | ✅ 已实现 | 二进制距离 (0/1) |
| Range 过滤支持 | ✅ 已实现 | 比例距离，支持边界检测 |
| Sparse 过滤支持 | ✅ 已实现 | IDF 加权多标签过滤 |
| Multi-Tier 搜索 | ⚠️ 已移除 | 复杂度不值得收益 |

---

## 附录：代码位置参考

| 功能 | 论文代码 | Knowhere 代码 | 状态 |
|------|----------|---------------|------|
| Weight-JAG 搜索 | `/Paper/JAG/parlayann/WeightJAG/index.h` | `HnswSearcher.h:search_on_a_level_jag_v2` | ✅ |
| 动态权重调整 | - | `HnswSearcher.h:get_adaptive_weight()` | ✅ |
| 自动权重选择 | - | `HnswSearcher.h:get_auto_weight_for_filter_ratio()` | ✅ |
| 早剪枝 | 论文 lines 422-427 | `HnswSearcher.h:should_prune_by_filter()` | ✅ |
| normalized_h 计算 | `WeightJAG/index.h:init()` (预计算) | `HnswSearcher.h:estimate_normalized_h()` (在线) | ✅ |
| Label 过滤距离 | `/Paper/JAG/parlayann/utils/filter_check.h` | `filter_distance.h:LabelFilterDistance` | ✅ |
| Range 过滤距离 | 论文 Section 3.2 | `filter_distance.h:RangeFilterDistance` | ✅ |
| Sparse 过滤距离 | 论文 Section 3.3 | `filter_distance.h:SparseFilterDistance` | ✅ |
| Multi-Tier 配置 | `filtered_vector_search_benchmark_main.cc` | 未实现 | ❌ |
