# 问题一、问题二 Hybrid 总结报告

## 1. 报告范围与取材说明

本报告统一以“外层 hybrid 搜索 + 内层 cost-first 精修”作为问题一、问题二的最终建模口径，并综合以下材料整理：

- 早期问题一迭代总结：[question1_artifacts/对话方案迭代汇总报告.md](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/question1_artifacts/%E5%AF%B9%E8%AF%9D%E6%96%B9%E6%A1%88%E8%BF%AD%E4%BB%A3%E6%B1%87%E6%80%BB%E6%8A%A5%E5%91%8A.md)
- 问题一版本复盘：[question1_artifacts/q1_optimization_versions_report.md](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/question1_artifacts/q1_optimization_versions_report.md)
- 问题一 hybrid 近期尝试：[question1_artifacts_hybrid_coupled_minimal/q1_hybrid_recent_attempts_report.md](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/question1_artifacts_hybrid_coupled_minimal/q1_hybrid_recent_attempts_report.md)
- 问题一主脚本：[solve_question1.py](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:228)
- 问题一 hybrid 脚本：[solve_question1_hybrid.py](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1_hybrid.py:87)
- 问题二策略脚本：[solve_question2.py](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:33)
- 问题二 cost-first v2 脚本：[solve_question2_costfirst_v2.py](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_costfirst_v2.py:22)
- 问题二 hybrid 脚本：[solve_question2_hybrid.py](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_hybrid.py:23)

最终结果读取的是你后续加大搜索预算后的最新目录，而不是最早的 minimal 试跑目录。

## 2. 最终采用结果

### 2.1 问题一最终结果

问题一最终采用目录：

- 主结果：[question1_artifacts_hybrid_coupled_heavy_s23](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/question1_artifacts_hybrid_coupled_heavy_s23)

多 seed 对比如下：

| seed | 目录 | total_cost | route_count | single_stop | late_positive_stops | max_late_min |
|---|---|---:|---:|---:|---:|---:|
| 23 | `question1_artifacts_hybrid_coupled_heavy_s23` | 86568.3157 | 131 | 97 | 13 | 59.4258 |
| 11 | `question1_artifacts_hybrid_coupled_heavy_s11` | 86666.8103 | 131 | 98 | 9 | 24.9448 |
| 17 | `question1_artifacts_hybrid_coupled_heavy_s17` | 87008.1920 | 132 | 97 | 12 | 24.9448 |

选择 `s23` 的原因是题目主目标是总成本最小。按总成本排序，`s23` 最优；按更平滑的调度质量看，`s11` 可作为补充参考，但不是主解。

相对问题一此前主基线 `question1_artifacts/q1_cost_summary.json`：

- `87616.6098 -> 86568.3157`
- 成本下降 `1048.2941`
- 路线数 `132 -> 131`
- 迟到站点 `18 -> 13`
- 最大迟到 `99.0040 -> 59.4258`

### 2.2 问题二最终结果

问题二最终采用目录：

- 主结果：[question2_artifacts_hybrid_standard_s11](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/question2_artifacts_hybrid_standard_s11)

多 seed 对比如下：

| seed | 目录 | total_cost | route_count | single_stop | late_positive_stops | max_late_min |
|---|---|---:|---:|---:|---:|---:|
| 11 | `question2_artifacts_hybrid_standard_s11` | 89850.0134 | 132 | 97 | 15 | 30.8868 |
| 23 | `question2_artifacts_hybrid_standard_s23` | 90166.0513 | 133 | 95 | 13 | 65.3202 |
| 17 | `question2_artifacts_hybrid_standard_s17` | 90226.5185 | 133 | 93 | 17 | 99.0040 |

选择 `s11` 的原因同样是题目主目标仍是总成本最小。`s23` 的迟到站点更少，但总成本更高，因此不作为主解。

相对问题二正式基线 `question2_artifacts_costfirst_v2/q2_cost_summary.json`：

- `90142.3299 -> 89850.0134`
- 成本下降 `292.3165`
- 路线数 `133 -> 132`
- `policy_violation_count = 0`
- `mandatory_ev_served_by_non_ev_count = 0`
- `fuel_route_green_zone_pre16_visit_count = 0`

问题二还保持了政策可见性：

- `green_zone_customer_count_used = 12`
- `q1_ev_usage_baseline = 25`
- `q2_ev_usage_total = 25`
- `ordinary_customers_served_by_ev = 12`
- `must_use_ev_customers_served_by_ev_only = 8`

这说明问题二是在不增加电动车总占用的情况下，完成了政策重分配。

## 3. 两问统一后的最终模型框架

两问最终都不是“单层启发式”，而是统一为四层结构：

1. `ServiceUnit` 层：先把客户需求转化为可服务单元。
2. 时间依赖路线评估层：对给定路线和车型，精确计算发车时刻、等候、迟到、能耗、碳排和回仓时间。
3. 内层 cost-first 精修层：先做局部结构优化，再做 route-pool 生成，再用 set partitioning / MILP 从候选路线集合中重组选路。
4. 外层 hybrid 搜索层：用概率构造、信息素、`pbest/gbest`、elite archive、柯西扰动和 destroy/repair 扩大搜索空间，再把候选解送回内层精修。

统一口径的好处是：

- 问题一和问题二都可以共享同一套可解释的求解骨架。
- 问题二只在 `ServiceUnit` 定义、策略窗、候选列偏好和硬门槛上增加政策约束，不需要重新发明一套独立求解框架。
- 最终的主目标始终没有变化，仍然是 `total_cost` 最小；hybrid 只是扩展搜索，不改变目标函数本身。

## 4. 问题一：做了什么、如何实施、为什么这样做

### 4.1 问题一最终建模口径

问题一最终不是“所有客户自由拆分”的模型，而是：

- 普通客户默认一次服务。
- 只有对所有车型都无法单次承运的客户，才允许必要拆分。
- 每辆车每天只执行一条路线。
- 时间窗是纯软时间窗，早到等待、晚到罚款。
- `21:00` 不是硬截止；`21:00` 后允许继续服务和回仓。

这个口径的核心目的是把模型重新收回到题意，而不是为了追求“更好看”的路线结构。

### 4.2 问题一如何从数据落地成模型

#### 步骤 1：构造 `ServiceUnit`

实现位置：

- [solve_question1.py:689](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:689)

做法：

- 若客户总重量、总体积能被某类车一次承运，则生成一个 `normal` 型 `ServiceUnit`。
- 若所有车型都无法一次承运，则进入“必要拆分”逻辑。
- 必要拆分不是直接按固定重量切块，而是先把订单切成碎片，再对不同 `bin_count` 做重复随机贪心装箱。

为什么这么做：

- 这样可以保证“普通客户不误拆”。
- 同时保留必要拆分客户在后续路线组合中的可行性和灵活性。

#### 步骤 2：对必要拆分客户做多次随机贪心装箱

实现位置：

- [solve_question1.py:640](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:640)
- [solve_question1.py:678](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:678)

做法：

- `PACKING_ATTEMPTS = 96`
- 对每个候选访问次数 `bin_count`，重复 96 次带随机种子的贪心装箱。
- 评分优先级不是单纯“装得下”，而是综合考虑：
  - 可服务车型集合尽量大
  - 空箱尽量少
  - 剩余重量/体积松弛尽量合理

为什么这么做：

- 这是一个组合爆炸很快的子问题，不适合直接在外层求全局最优。
- 96 次重复随机贪心不是物理参数，而是工程预算，用来在可接受时间内逼近“访问次数尽量少、车型兼容性尽量强”的拆分方案。
- 这类随机贪心只作用于候选生成，不直接决定最终主解，后面还会经过路线级和 MILP 级再筛选。

#### 步骤 3：额外保留 3000kg 车辆冗余

实现位置：

- [solve_question1.py:42](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:42)
- [solve_question1.py:783](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:783)

做法：

- `BIG_VEHICLE_RESERVE = 5`
- 在必要拆分阶段，不允许所有 3000kg 车都被“重货大件”一次性吃满。
- 如果当前拆分方案让 `heavy_big_only` 单元占满了大车容量，就允许少量增加访问次数，换取更低的重型大车依赖。

为什么这么做：

- 这是从早期迭代中反复暴露出来的问题：如果拆分阶段就把大车库存耗尽，后面 route merge、promotion、piggyback 再聪明也无车可用。
- 因此这 5 台不是题目给的业务参数，而是搜索保护量，用来避免“拆分阶段过早锁死后续结构优化”。

#### 步骤 4：建立时间依赖路线评估器

实现位置：

- [solve_question1.py:1026](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:1026)

做法：

- 以 `travel_time_lookup.npz` 为主表。
- 对非整数发车时刻做线性插值。
- 成本由启动、能耗、碳排、等待、迟到五部分组成。
- `DAY_END_MIN = 780` 对应 `21:00`。
- `21:00` 后不截断，而是用 `AFTER_HOURS_SPEED_KMH = 55.3` 做确定性延拓。

为什么这么做：

- 早期迭代中反复出现过“把 21:00 当硬边界”导致的伪不可行。
- 题目本质上已经有迟到罚款，所以如果再硬截断，会人为放大某些路线的不可行性。
- `55.3 km/h` 不是题目目标系数，而是查表结束后的确定性补充口径，目的是让 `21:00` 之后的继续服务和回仓可被统一计算，而不是让模型在边界处失真。

#### 步骤 5：先做局部结构优化，再做全局 route-pool + MILP

实现位置：

- 局部优化：[solve_question1.py:3741](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:3741)
- 全局重组：[solve_question1.py:3048](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1.py:3048)

做法：

- 局部阶段只保留三类高价值动作：
  - `route_merge`
  - `route_type_change`
  - `relocate`
- 全局阶段生成 route-pool，候选角色包括：
  - `singleton`
  - `flex_small`
  - `promotion`
  - `piggyback_big`
  - `rigid_big`
  - `residual_promotion`
  - `cluster_flex_small`
- 再把候选路线送进 set partitioning / MILP 做两轮选路。

为什么这么做：

- 纯局部搜索很难“一次替换几条路线”。
- 纯 MILP 又不能直接在原始客户序列空间上求解。
- 所以采用“先用启发式造列，再用 MILP 选列”的折中方案：既保留工程可跑性，又能做全局重组。

### 4.3 问题一为什么还要加 hybrid 外层

实现位置：

- [solve_question1_hybrid.py:87](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1_hybrid.py:87)
- [solve_question1_hybrid.py:1699](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question1_hybrid.py:1699)

原因：

- 问题一在进入 route-pool + MILP 后，内层已经很强，但也更依赖初始结构。
- 只用确定性构造，容易在相似结构附近打转。
- 因此 hybrid 外层负责扩大搜索半径，内层负责把外层送来的候选结构压实成可解释的高质量解。

具体做法：

- 概率构造：不再只走单一路径插入，而是对若干较优插入位置按温度概率采样。
- 信息素：记录“哪些单元适合做起点”“哪些相邻关系值得保留”。
- `pbest / gbest`：粒子记住自己的最好结构，全局再记住群体最好结构。
- elite archive：把过去几代里有价值的整条路线注入新的 route-pool。
- 柯西扰动：对 `destroy_ratio`、温度、算子权重做重尾扰动，避免搜索太快收缩。
- `cluster_remove`：不只删若干单元，也删小 route-cluster，为重组腾出更大的空间。
- `balanced_best`：在成本容忍带内保留一个更注重迟到控制的备选解，容忍带定义为 `max(300, 0.5% * total_cost)`。

### 4.4 问题一迭代中发现了什么问题，因此采用了哪些措施

综合前述迭代报告与最新脚本，可以把问题一的“发现问题 - 采取措施”概括为：

| 发现的问题 | 采取的措施 | 为什么必须这么做 |
|---|---|---|
| 普通客户被自由拆分，出现低成本假优解 | 改为客户级 `ServiceUnit`，只允许必要拆分 | 先把模型口径纠正，否则后续优化都建立在错解上 |
| 21:00 被误当成硬边界，导致回仓和服务逻辑失真 | 用 `55.3 km/h` 对 21:00 后做确定性延拓 | 让边界之后的继续服务和回仓可计算，避免伪不可行 |
| 必要拆分阶段耗尽大车库存，后续 merge 无法发生 | 设置 `BIG_VEHICLE_RESERVE = 5` | 给后续结构重组留出库存弹性 |
| 单纯局部搜索无法一次替换多条路线 | 建 route-pool，再做 set partitioning / MILP | 把优化对象从单个单元提升为整条路线集合 |
| 继续加局部算子收益变小 | 加 hybrid 外层 | 让搜索不再完全依赖单一确定性 warm start |
| 只看总成本会牺牲调度平滑性 | 额外输出 `balanced_best` | 方便论文里讨论“最低成本”和“更稳调度”的取舍 |

### 4.5 问题一最终结果如何解释

最终选择的 `s23` 主解说明：

- 模型口径已经稳定，`split_customer_count = mandatory_split_customer_count = 37`。
- 重预算 hybrid 外层确实起作用，因为最优解已从原主基线再下降 `1048.2941`。
- 这次改进不是单纯减少路线数造成的错觉，而是同时降低了总成本、迟到站点和最大迟到。
- 车辆结构上仍然几乎不用 `fuel_1250`，这不是求解器 bug，而是脚本自己给出的“车型支配诊断”结果：在当前成本模型下，所有车型启动费相同，能耗和碳排又只区分燃油/电动，因此 `fuel_1500` 对 `fuel_1250` 构成严格支配。

## 5. 问题二：做了什么、如何实施、为什么这样做

### 5.1 问题二不是另起炉灶，而是在问题一骨架上加政策层

问题二直接继承问题一的建模骨架：

- 仍然是 `ServiceUnit`
- 仍然是一车一天一路线
- 仍然是时间依赖路线评估
- 仍然是内层局部优化 + route-pool + MILP
- 再额外叠加绿色配送区、强制电动车客户和燃油车限时服务策略

实现位置：

- [solve_question2.py:33](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:33)

### 5.2 问题二如何把政策写进 `ServiceUnit`

实现位置：

- [solve_question2.py:265](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:265)

做法：

- 在 `ServiceUnit` 上新增：
  - `in_green_zone`
  - `must_use_ev_under_policy`
  - `fuel_allowed_after_16`
  - `ev_tw_start_min / ev_tw_end_min`
  - `fuel_tw_start_min / fuel_tw_end_min`
- 若客户必须用电动车，则可服务车型集合直接过滤成 EV。
- 若客户属于“燃油车只能 16:00 后进入”的类，则燃油车时间窗单独收紧。

为什么这么做：

- 政策不是简单加一个罚款项就能表达的，它直接改变了不同动力类型车辆的可行服务窗口。
- 因此问题二必须在 `ServiceUnit` 层就把 EV 与 fuel 的服务窗口分开，否则后面无论局部搜索还是 MILP 都是在错误的可行域上搜索。

### 5.3 问题二为什么要固定绿配区口径

实现位置：

- [solve_question2.py:17](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:17)
- [solve_question2.py:954](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:954)

脚本采用的固定口径是：

- `green_zone_basis = attachment_geometry_radius_10km`

这样做的原因是：

- 题面文字口径与附件几何口径不完全一致。
- 如果不固定一条可复现规则，后续“燃油车是否违规进入”的统计就会不稳定。
- 脚本最终用的是几何口径；在全部几何客户中为 15 个，进入活动客户集后实际参与约束的是 12 个，所以结果里出现 `green_zone_customer_count_used = 12`。

这不是为了调结果而“手工改数字”，而是为了让策略约束有唯一可编程定义。

### 5.4 问题二必要拆分为什么要再做一次政策感知修正

实现位置：

- [solve_question2.py:127](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:127)
- [solve_question2.py:265](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2.py:265)

做法：

- 仍保留重复随机贪心装箱，但评分改成“政策感知型”：
  - 尽量减少 `ev3000_only` 单元数量
  - 尽量减少 `heavy_big_only` 单元数量
  - 尽量增加可服务车型集合
- 若策略感知拆分后仍超过 `ev_3000` 容量，则继续往增加访问次数的方向调，直到不超 EV 大车库存。

为什么这么做：

- 在问题二里，纯容量可行不代表策略可行。
- 某些客户如果被拆成太多 `ev_3000_only` 单元，会在还没开始路线优化时就把 EV 大车库存锁死。
- 因此问题二的拆分阶段必须先把“政策引起的车型紧约束”前置消化。

### 5.5 问题二 cost-first v2 为什么要额外强化三类候选列

实现位置：

- [solve_question2_costfirst_v2.py:171](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_costfirst_v2.py:171)
- [solve_question2_costfirst_v2.py:352](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_costfirst_v2.py:352)

新增三类 Q2 专属角色：

- `ev_flex_small_q2`
- `late_fuel_cluster_q2`
- `policy_piggyback_q2`

对应的原因是：

- `ev_flex_small_q2`：把必须用 EV 的客户优先聚合进 EV 小路线，减少对 `ev_3000` 的过度依赖。
- `late_fuel_cluster_q2`：把允许燃油车 16:00 后服务的客户聚成燃油路线，减少无谓的 EV 占用。
- `policy_piggyback_q2`：保留那些虽然在大车上混装、但从政策角度并不坏、且能降成本的搭载方案。

这里的加分项 `18 / 8 / 15 / 4 / -12 / -6 / 4` 不是业务成本，不进入最终目标函数，只是 route-pool 排序和列筛选时的启发式偏好，作用是把政策友好的列更早送进 MILP。

### 5.6 问题二为什么保留确定性贪心优先级

实现位置：

- [solve_question2_costfirst_v2.py:51](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_costfirst_v2.py:51)

`_q2_choice_rank_key` 的策略顺序大致是：

- 普通客户如果本来就能用燃油车，就尽量别先占用 EV。
- 对“燃油车 16:00 后可进”的客户，优先让燃油车服务。
- 对必须用 EV 的客户，如果 `ev_1250` 就够，就优先别直接占 `ev_3000`。

为什么需要这种确定性贪心：

- 问题二的难点不是单次插入的精度，而是避免早期错误决策把稀缺 EV 资源用光。
- 所以基线构造阶段必须有一个稳定、可解释、可重复的插入偏好；否则不同随机种子会频繁走进“先图省事、后面无解”的死路。

### 5.7 问题二为什么继续加 hybrid 外层

实现位置：

- [solve_question2_hybrid.py:23](/C:/Desktop/%E5%8D%8E%E4%B8%AD%E6%9D%AFA%E9%A2%98/solve_question2_hybrid.py:23)

问题二 hybrid 的思路是：

- 外层沿用问题一 hybrid 的概率构造、archive、pheromone、pbest/gbest、cluster remove。
- 内层不另造一套求解器，而是直接复用 `Question2CostFirstV2Solver`。
- 硬门槛额外增加：
  - `policy_violation_count = 0`
  - `mandatory_ev_served_by_non_ev_count = 0`
  - `fuel_route_green_zone_pre16_visit_count = 0`

为什么这么做：

- 问题二本质上是“在问题一结构上再加策略约束”，所以最稳的做法不是推翻重做，而是在已经收敛的 Q2 v2 内层上再加统一的 hybrid 外层。
- 这样做可以保持两问框架一致，也方便论文里统一叙述。

### 5.8 问题二迭代中发现了什么问题，因此采用了哪些措施

| 发现的问题 | 采取的措施 | 为什么必须这么做 |
|---|---|---|
| 题面与附件对绿配区口径不完全一致 | 固定 `attachment_geometry_radius_10km` 作为唯一实现口径 | 先保证约束可重复计算 |
| 只按容量拆分会把 EV 大车库存提前耗尽 | 在拆分评分里加入 `ev3000_only_count` 约束 | 避免还没开始排路线就把策略资源锁死 |
| 普通客户会无谓占用 EV | 用 `_q2_choice_rank_key` 在构造阶段就优先保留 EV 资源 | 让 EV 用在真正必须用 EV 的客户上 |
| 仅靠 Q2 v2 内层仍有局部依赖 | 在 v2 内层外包一层 hybrid | 用外层多样化结构突破确定性 warm start 依赖 |
| minimal hybrid 只实现了机制，未降成本 | 加大预算并做多 seed 比较 | 说明 Q2 hybrid 需要更大搜索预算才有机会超过 v2 |

### 5.9 问题二最终结果如何解释

最终选择的 `s11` 主解说明：

- 成本相对 v2 基线下降 `292.3165`。
- 路线数少 1 条。
- 三个核心政策违规指标全部为 0。
- 在不增加 EV 总使用量的前提下完成了政策重排。
- 虽然 `late_positive_stops` 从 14 变成 15，但 `max_late_min` 从 `88.6030` 大幅降到 `30.8868`，说明迟到尾部明显收缩，极端迟到被压下来了。

## 6. 脚本里的人为设定、启发式权重与确定性策略，应如何解释

下表只解释“为什么要这么设”，不把它们包装成业务真值。

| 项目 | 脚本位置 | 性质 | 解释 |
|---|---|---|---|
| `SERVICE_TIME_MIN = 20` | `solve_question1.py` | 业务参数 | 直接按题目/预处理配置使用 |
| `START_COST = 400`、等待/迟到单价 | `solve_question1.py` | 业务参数 | 直接进入目标函数 |
| `AFTER_HOURS_SPEED_KMH = 55.3` | `solve_question1.py` | 确定性延拓值 | 查表只覆盖到 21:00，必须补一条后续行驶口径，避免边界伪不可行 |
| `PACKING_ATTEMPTS = 96` | `solve_question1.py` | 搜索预算 | 让随机贪心装箱多试几次，近似更好的拆分方案 |
| `BIG_VEHICLE_RESERVE = 5` | `solve_question1.py` | 搜索保护量 | 防止必要拆分阶段耗尽 3000kg 车辆，给后续 merge 留弹性 |
| `FUEL_3000_SEARCH_RESERVE = 1` | `solve_question1.py` | 局部搜索保护量 | repair 阶段至少保留 1 台 `fuel_3000` 给结构合并 |
| route-pool 列上限、MILP 时间上限 | `solve_question1.py` | 工程截断参数 | 防止候选列和 MILP 尺度失控 |
| `HYBRID_BALANCED_COST_ABS_ALLOWANCE = 300` 与 `0.5%` 相对带 | `solve_question1_hybrid.py` | 备选解容忍带 | 让 `balanced_best` 只在“成本仍接近主解”的条件下比较迟到表现 |
| 信息素、archive、柯西、destroy/temperature 参数 | `solve_question1_hybrid.py` | 搜索超参数 | 只作用于外层搜索效率，不进入最终成本函数 |
| `Q2_V2_*` 源点/邻域/配对上限 | `solve_question2_costfirst_v2.py` | 候选生成预算 | 限制 Q2 专属候选列的规模，避免池子爆炸 |
| Q2 列加减分 `18/8/15/4/-12/-6/4` | `solve_question2_costfirst_v2.py` | 列排序启发式 | 不是业务成本，只是让策略友好的列更容易进入 MILP |
| Q2 hybrid `_rank_key_penalty` 中的 `220/140/180/...` | `solve_question2_hybrid.py` | 外层搜索引导权重 | 不是最终目标，只是控制概率插入时别过早浪费 EV 资源 |

## 7. 需要在报告里主动交代的实现口径说明

### 7.1 关于车型支配

脚本会自动输出一组“车型支配诊断”，其结论很重要：

- 所有车型启动费相同。
- 能耗与碳排模型只区分燃油/电动，不进一步区分同动力下的小车与中车。
- 因此在当前成本口径下，`fuel_1500` 对 `fuel_1250` 构成严格支配。

这意味着最终结果里 `fuel_1250` 经常不用，并不是求解器忽略了它，而是当前成本模型确实没有给它优势。

### 7.2 关于问题二的“最新加预算结果”字段解释

问题二最终采用的是 `question2_artifacts_hybrid_standard_s11` 目录，但脚本里 `hybrid_mode` 字段仍可能显示为 `minimal`。原因是：

- `solve_question2_hybrid.py` 目前只保留了 `minimal` 这个模式标签。
- 你后来通过显式传入更大的 `particles` 与 `generations` 提高了预算。

因此本报告对问题二的预算解释以真实字段为准：

- `particle_count = 8`
- `max_generations = 6`

而不是看 `hybrid_mode` 字段字面值。

### 7.3 关于问题一 `Applied budget signature`

问题一最终目录的 run report 中，`Applied budget signature` 仍带有内层 baseline reference 的残留签名，这个元数据不应用来判读最终预算。应以同一摘要文件中的真实字段为准：

- `particle_count = 12`
- `max_generations = 10`
- 多 seed 比较目录为 `s11/s17/s23`

这个字段残留不影响最终解本身，只影响报告时对预算的文字解释。

## 8. 可直接用于正文的结论性表述

可以将两问的最终方法统一表述为：

“本文针对两问均采用统一的 hybrid 求解框架。首先基于客户需求构造 `ServiceUnit`，对给定路线在时变路网下精确评估启动、能耗、碳排、等待与迟到成本；随后在内层采用局部结构优化、route-pool 生成与 set partitioning / MILP 全局重组选路，以 `total_cost` 最小为主目标；最后在外层引入概率构造、信息素记忆、elite archive、`pbest/gbest` 协同及柯西扰动，以扩大候选结构搜索范围并将外层学习到的优良结构反馈到内层 route-pool。问题二在此基础上进一步引入绿色配送区、强制电动车客户和燃油车限时进入等政策约束，并通过政策感知拆分、Q2 专属候选列和策略友好的确定性贪心优先级来保证策略可行性。”

若按最终结果写，可直接表述为：

- 问题一最终主解采用 `question1_artifacts_hybrid_coupled_heavy_s23`，总成本为 `86568.3157`，相对原主基线下降 `1048.2941`。
- 问题二最终主解采用 `question2_artifacts_hybrid_standard_s11`，总成本为 `89850.0134`，相对 Q2 v2 基线下降 `292.3165`，且全部政策硬约束均满足。

## 9. 结语

如果只看表面，问题一和问题二似乎只是“多跑一点搜索预算”。但从脚本迭代过程看，真正关键的不是单纯把粒子数和代数加大，而是先把口径一步步收正：

- 先纠正普通客户误拆。
- 再纠正 `21:00` 的错误边界解释。
- 再认识到大车库存会在拆分阶段被提前锁死。
- 再把优化对象从单点插入提升为整条路线集合重组。
- 最后才在这个稳定内层上叠加 hybrid 外层。

因此，两问最终结果的改进不是偶然的搜索运气，而是“模型口径逐步收正后，再用 hybrid 扩展搜索”的自然结果。
