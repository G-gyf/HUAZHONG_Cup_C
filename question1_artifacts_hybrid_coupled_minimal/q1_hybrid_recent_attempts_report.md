# 第一问 Hybrid 近期几轮尝试报告

## 1. 报告目的

这份报告只总结最近几轮围绕 `solve_question1_hybrid.py` 做的尝试，不回溯更早的贪心、route-pool、MILP 迭代版本。重点回答 4 个问题：

1. 这几轮分别做了什么。
2. 为什么要这么做。
3. 每一轮的实施步骤是什么。
4. 每一轮最后得到什么结果，以及如何解释。

当前相关代码与结果均已推送到 GitHub：

- 最新提交：`9a7fa5f`
- 分支：`main`
- 远程：`origin/main`

---

## 2. 对照基线

在进入 hybrid 主线之前，当前主线基线结果来自 `question1_artifacts/q1_cost_summary.json`：

- `total_cost = 87616.6098`
- `route_count = 132`
- `single_stop_route_count = 97`
- `late_positive_stops = 18`
- `max_late_min = 99.0040`
- `split_customer_count = 37`
- `final_solution_source = pass2`

这一组结果是后续 hybrid 所有尝试的统一对照基线。

---

## 3. 尝试一：Hybrid Minimal

### 3.1 发现的问题

在前面的主线代码里，虽然已经做出了比较强的 cost-first 结果，但有一个明显问题：

- 当前脚本已经不是最初设想的“概率构造 + 信息素 + 柯西变异 + 粒子协同”的原定方案。
- 现有主线更像“确定性构造 + 局部修复 + route-pool + MILP 全局重优化”。
- 这样做的结果是：内层精修已经很强，但外层搜索的多样性和跨代记忆不够。

因此，第一轮 hybrid 的目标不是重写主线，而是搭建一个真正的“外层群智能 + 内层 cost-first 精修”的混合框架。

### 3.2 解决方案

新增独立脚本 `solve_question1_hybrid.py`，不覆盖原 `solve_question1.py`，并在新脚本中实现：

1. 外层概率式初始解构造  
   不再只走确定性 `_insert_unit_best`，而是对候选插入位置做带温度的概率选择。

2. 信息素记忆  
   在 `ServiceUnit` 相邻关系和路线起点上维护信息素：
   - `pheromone_edge[(unit_i, unit_j, vehicle_family)]`
   - `pheromone_start[(unit_i, vehicle_family)]`

3. 粒子协同  
   每个粒子维护：
   - `current_solution`
   - `personal_best`
   群体维护：
   - `global_best`
   - `elite_archive`

4. 柯西扰动  
   对 `destroy_ratio`、`construction_temperature`、`operator_weights` 做柯西分布扰动。

5. 内层精修复用现有主线  
   外层每次生成候选后，统一送进当前已经验证有效的：
   - `improve_solution`
   - `route-pool`
   - `set partitioning / MILP`
   - `cost-first` 精修链

### 3.3 实施步骤

1. 新建 `solve_question1_hybrid.py`。
2. 复制当前主线中与 `ServiceUnit`、可行性评估、MILP 精修相关的稳定实现。
3. 在外层增加：
   - 概率构造
   - 信息素更新
   - `pbest / gbest`
   - 柯西变异
4. 增加独立结果目录：
   - `question1_artifacts_hybrid/`
5. 跑一轮 `minimal` 验证：
   - `particles = 4`
   - `generations = 3`

### 3.4 结果

`question1_artifacts_hybrid/q1_hybrid_cost_summary.json`：

- `total_cost = 87452.8964`
- `route_count = 132`
- `single_stop_route_count = 96`
- `late_positive_stops = 16`
- `max_late_min = 72.9191`
- `split_customer_count = 37`
- `final_solution_source = hybrid_pass1`

### 3.5 结果解读

这一轮说明两件事：

1. hybrid 框架是可行的  
   相比主线基线，成本真实下降：
   - `87616.61 -> 87452.90`

2. 内外层确实接起来了  
   `q1_hybrid_outer_search_trace.csv` 中已经能看到：
   - 不同粒子初解成本不同
   - `source_mode = current / personal_best / global_best`
   - destroy 算子切换
   - 外层候选进入内层再被 `pass1/pass2` 精修

### 3.6 这一轮的结论

Hybrid Minimal 成功证明：

- “外层搜索 + 内层精修”的混合主线是成立的。
- 但这时的提升仍偏轻量，外层对内层的影响主要是“换了一个更好的起点”，还没有真正改变内层 route-pool 的候选结构。

---

## 4. 尝试二：Hybrid Standard

### 4.1 发现的问题

Hybrid Minimal 证明方案可行后，下一步最自然的问题是：

- 如果只是加搜索预算，结果还能不能继续往下压？
- 外层收益是偶然一次改进，还是可持续？

### 4.2 解决方案

不改代码，只把运行预算从 `minimal` 提高到 `standard`，验证“纯加预算”的收益上限。

配置为：

- `particles = 8`
- `generations = 6`

输出目录：

- `question1_artifacts_hybrid_standard/`

### 4.3 实施步骤

1. 保持 `solve_question1_hybrid.py` 不改。
2. 使用 `standard` 配置运行。
3. 对比三组结果：
   - baseline
   - hybrid minimal
   - hybrid standard
4. 检查 `q1_hybrid_outer_search_trace.csv` 中后几代是否仍在刷新 `global_best`。

### 4.4 结果

`question1_artifacts_hybrid_standard/q1_hybrid_cost_summary.json`：

- `total_cost = 87440.5376`
- `route_count = 132`
- `single_stop_route_count = 97`
- `late_positive_stops = 17`
- `max_late_min = 221.7827`
- `split_customer_count = 37`
- `final_solution_source = hybrid_pass2`

### 4.5 结果解读

这一轮的结果很典型：

1. 只加预算，确实还能继续降成本  
   - `87452.90 -> 87440.54`

2. 但收益已经明显变小  
   相比 Minimal，只再降了约 `12.36`。

3. 同时暴露出副作用  
   `standard` 的尾部迟到更差：
   - `late_positive_stops: 16 -> 17`
   - `max_late_min: 72.92 -> 221.78`

### 4.6 为什么会这样

原因不是代码坏了，而是当时 hybrid 仍然是“弱耦合”：

- 外层负责找到新的 incumbent
- 内层再按现有 route-pool 和 MILP 精修

但外层的 archive、pheromone 等信息还没有真正参与内层候选列生成和排序。  
所以继续加预算时，外层更多是在做“更大范围的随机探索”，而不是“更深度地影响内层结构重构”。  
这就导致：

- 成本还能继续降
- 但改善幅度越来越小
- 且可能开始牺牲尾部调度均衡性

### 4.7 这一轮的结论

Hybrid Standard 证明：

- 继续加预算是有收益的
- 但收益已经进入“边际变小”的阶段
- 下一步不应该再只靠堆粒子和代数，而应该增强“外层对内层的耦合”

---

## 5. 尝试三：Hybrid Coupled Minimal

### 5.1 发现的问题

从 Standard 可以清楚看到当前瓶颈：

1. 外层已经有效，但对内层影响不够深
2. 继续纯加预算，成本改善有限
3. 还可能把尾部迟到推坏

因此需要解决的问题不再是“能不能继续搜索”，而是：

**如何让外层学到的结构信息，真正进入内层 route-pool 和 MILP 选路。**

### 5.2 解决方案

在 `solve_question1_hybrid.py` 上做耦合增强，只改 hybrid，不碰原主线脚本。主要做了四件事：

#### 5.2.1 双结果输出：`cost_best + balanced_best`

目的：

- `cost_best` 继续严格对齐题目主目标：总成本最低
- `balanced_best` 作为调度更稳的备选解，避免只盯总成本

实现方式：

- 维护两套全局最优：
  - `global_best_cost`
  - `global_best_balanced`
- 维护两套粒子最优：
  - `personal_best`
  - `personal_balanced_best`
- 在主输出目录中写 `cost_best`
- 在 `balanced/` 子目录中额外写 `balanced_best`

#### 5.2.2 Archive 注入 route-pool

目的：

- 让外层 elite archive 中积累出来的好路线，不只影响下一轮初解
- 还直接变成内层 MILP 可选的候选列

实现方式：

- 从 elite archive 取前 `2` 个精英解
- 每个解最多注入 `20` 条路线
- 注入角色包括：
  - `flex_small`
  - `piggyback_big`
  - `promotion_like_big`
  - 其它非重复支持路线
- 统一打上 `archive_seed` 标记

#### 5.2.3 Pheromone / archive 进入列排序

目的：

- 让外层学到的“好边、好簇、好模式”直接影响内层 route-pool 的列优先级

实现方式：

- 为每条候选列增加：
  - `archive_support_count`
  - `pheromone_support_score`
  - `coupled_candidate_score`
- 排序不再只看成本与时间窗，还看：
  - archive 支持
  - pheromone 强度

#### 5.2.4 新增 `cluster_remove`

目的：

- 让外层 destroy 不只删 unit，而能删掉一个小 route-cluster
- 为内层提供更大结构重组空间

实现方式：

- 从当前解中选一条小路线作为中心
- 按空间邻近 + 时间窗相容性拉入 `1-2` 条相关路线
- 整簇移除，再交给内层重建

### 5.3 实施步骤

1. 在 hybrid 脚本中补齐双结果数据结构。
2. 改写外层采样逻辑，使其能从：
   - `current`
   - `personal_best`
   - `global_best_cost`
   - `global_best_balanced`
   四类来源中采样。
3. 在 route-pool 构造前注入 archive 路线。
4. 修改 route-pool 列打分与排序。
5. 新增 `cluster_remove` destroy。
6. 新增输出字段：
   - `archive_injected_route_count`
   - `archive_selected_column_count`
   - `pheromone_bonus_selected_column_count`
   - `cluster_remove_attempt_count`
   - `cluster_remove_accepted_count`
7. 先用 `minimal` 配置验证耦合增强是否真的生效。

输出目录：

- `question1_artifacts_hybrid_coupled_minimal/`

### 5.4 结果

`question1_artifacts_hybrid_coupled_minimal/q1_hybrid_cost_summary.json`：

#### cost_best

- `total_cost = 87384.2007`
- `route_count = 132`
- `single_stop_route_count = 97`
- `late_positive_stops = 17`
- `max_late_min = 99.0040`
- `split_customer_count = 37`
- `final_solution_source = hybrid_pass1`

#### balanced_best

- `total_cost = 87695.2931`
- `route_count = 132`
- `single_stop_route_count = 96`
- `late_positive_stops = 14`
- `max_late_min = 99.0040`
- `final_solution_source = hybrid_pass1`

#### 耦合诊断计数

- `archive_injected_route_count = 25`
- `archive_selected_column_count = 22`
- `pheromone_bonus_selected_column_count = 132`
- `cluster_remove_attempt_count = 1`
- `cluster_remove_accepted_count = 0`

### 5.5 结果解读

这一轮是最近几轮里最关键的一次，因为它证明了“耦合增强”不是概念，而是真实生效：

#### 5.5.1 成本主结果继续下降

相比前几轮：

- baseline：`87616.61`
- hybrid minimal：`87452.90`
- hybrid standard：`87440.54`
- coupled minimal：`87384.20`

所以当前最新的 `cost_best` 已经是最近几轮里最低的成本结果。

#### 5.5.2 平衡解也真实存在

`balanced_best` 虽然成本高于 `cost_best`，但晚点站点更少：

- `cost_best late_positive_stops = 17`
- `balanced_best late_positive_stops = 14`

这说明双结果链路已经打通，不再只有一个“成本单目标”解。

#### 5.5.3 Archive 和 pheromone 已经真正进入内层

这点不是靠猜，而是有直接证据：

- `archive_injected_route_count = 25`
- `archive_selected_column_count = 22`
- `pheromone_bonus_selected_column_count = 132`
- `q1_hybrid_route_pool_summary.csv` 中对应字段非零

说明 archive 注入进了 route-pool，且被 MILP 选中了。

#### 5.5.4 cluster_remove 还没真正起作用

虽然 `cluster_remove` 已经被调度到了：

- `attempt_count = 1`

但目前：

- `accepted_count = 0`

所以这轮真正起作用的是“软耦合”：

- archive 注入
- pheromone 进入列排序
- 双结果记忆

而不是更强的“结构破坏式重组”。

### 5.6 这一轮的结论

Hybrid Coupled Minimal 的结论很明确：

1. 耦合增强是必要的  
   因为只加预算已经进入边际收益下降，而耦合增强后成本继续明显下降。

2. 耦合增强是有效的  
   Archive 与 pheromone 已经真正进入内层 route-pool。

3. 当前最有效的是软耦合，不是 `cluster_remove`  
   `cluster_remove` 还需要下一轮专门加强。

---

## 6. 三轮尝试的横向对比

| 方案 | total_cost | route_count | single_stop | late+ | max_late |
|---|---:|---:|---:|---:|---:|
| baseline | 87616.61 | 132 | 97 | 18 | 99.00 |
| hybrid minimal | 87452.90 | 132 | 96 | 16 | 72.92 |
| hybrid standard | 87440.54 | 132 | 97 | 17 | 221.78 |
| hybrid coupled minimal cost_best | 87384.20 | 132 | 97 | 17 | 99.00 |
| hybrid coupled minimal balanced_best | 87695.29 | 132 | 96 | 14 | 99.00 |

可以看出：

- **成本最低**：`hybrid coupled minimal cost_best`
- **均衡性最好**：当前最近几轮里，`balanced_best` 的 `late_positive_stops = 14` 最低
- **最差尾部迟到**：`hybrid standard`

---

## 7. 当前结论与下一步建议

### 7.1 当前结论

到这一步可以明确说：

1. Hybrid 主线是成立的。
2. 仅靠加预算有收益，但收益开始变小。
3. 真正有价值的下一步，不是继续盲目堆粒子和代数，而是继续增强内外层耦合。

### 7.2 当前最佳结果

如果按题目第一问“总配送成本最低”：

- 当前推荐主结果：`question1_artifacts_hybrid_coupled_minimal/`
- 主解：`cost_best = 87384.2007`

如果同时给一个调度更稳的备选：

- 备选解：`balanced_best = 87695.2931`

### 7.3 下一步建议

下一步最值得做的是：

1. 先跑一轮 `hybrid coupled standard`
   - 验证耦合增强后加预算是否还能继续稳定降成本

2. 若 `cluster_remove` 仍然几乎不被接受
   - 优先改它的结构选择与接受逻辑
   - 不要先继续盲目扩大搜索预算

3. 暂时不回退原主线脚本
   - 原 `solve_question1.py` 保持不动
   - 后续继续沿 `solve_question1_hybrid.py` 推进

---

## 8. 本次报告对应文件

- 主结果目录：`question1_artifacts_hybrid_coupled_minimal/`
- 平衡解目录：`question1_artifacts_hybrid_coupled_minimal/balanced/`
- 外层轨迹：`question1_artifacts_hybrid_coupled_minimal/q1_hybrid_outer_search_trace.csv`
- 信息素诊断：`question1_artifacts_hybrid_coupled_minimal/q1_hybrid_pheromone_top_edges.csv`
- 基线对比：`question1_artifacts_hybrid_coupled_minimal/q1_hybrid_compare_to_baseline.json`

