# 第二问建模思路报告：相对第一问的增量建模、实施步骤与迭代路径

## 1. 报告目的与依据

本报告只讨论第二问，并且明确回答一个核心问题：第二问不是在第一问之外另写了一套新模型，而是在第一问已经成型的“服务单元 + 时变路径评估 + 内层 cost-first 精修 + 外层 hybrid 搜索”骨架上，逐层加入政策约束、资源稀缺性表达和对应的验模机制。

报告依据的脚本链条是：

- `solve_question1.py`：第一问基础求解器。
- `solve_question2.py`：在第一问基础上加入政策可行域。
- `solve_question2_costfirst_v2.py`：在 Q2 base 上加入更强的策略感知贪心与候选列塑形。
- `solve_question2_hybrid.py`：沿用第一问 hybrid 外层，把 Q2 v2 作为内层精修器。

当前采用的第二问最终结果读取自：

- `question2_artifacts_hybrid_standard_s11/q2_hybrid_cost_summary.json`

其核心结果为：

- `total_cost = 89850.01343848961`
- 相对 `question2_artifacts_costfirst_v2/q2_cost_summary.json` 的 `90142.32990705667`，改进 `292.31646856706357`
- `route_count = 132`
- `single_stop_route_count = 97`
- `split_customer_count = mandatory_split_customer_count = 37`
- `policy_violation_count = 0`
- `mandatory_ev_served_by_non_ev_count = 0`
- `fuel_route_green_zone_pre16_visit_count = 0`
- `q1_ev_usage_baseline = 25`
- `q2_ev_usage_total = 25`

因此，第二问最终不是靠“多开电动车”换取政策可行，而是在不增加电动车总占用的前提下，完成了政策重分配并进一步降本。

## 2. 第一问中保留下来的骨架

第二问保留了第一问里最重要的四个骨架层，不是推翻重做：

### 2.1 服务对象仍然先抽象成 `ServiceUnit`

第一问先把客户需求转成 `ServiceUnit`，对超载客户做必要拆分，再以 `unit` 为基本求解对象。第二问继续保留这一层，因为后续的路线构造、局部搜索、route-pool 和 MILP 都依赖这一统一对象。

### 2.2 路线成本评估仍然是时变路网下的精确仿真

第二问没有改掉第一问的成本定义，仍然是在时变旅行时间下评估：

- 启动成本
- 能耗成本
- 碳排成本
- 等待成本
- 迟到成本

也就是说，第二问不是改成“政策优先的多目标函数”，而是把政策先写进可行域，再继续以总成本最小作为主目标。

### 2.3 内层仍然是 cost-first 思路

第二问仍然保留了第一问的内层流程：

- 构造初始解
- 做局部改良
- 生成 route-pool
- 用 set partitioning / MILP 在列池上做全局重组选路

最终 MILP 的目标仍然是 `total_cost`。在 `solve_question1.py` 中，列进入 MILP 后实际使用的是 `metric_vectors["total_cost"]`，不是后面那些启发式打分。因此第二问新增的人为评分只影响“搜什么、留什么列”，不直接改写最终优化目标。

### 2.4 最终版本同样采用 hybrid 外层

为了和第一问保持统一，第二问最终也采用：

- 外层 hybrid 搜索
- 内层 Q2 cost-first v2 精修

所以现在两问可以统一理解为：

- 相同的外层全局结构搜索机制
- 相同的内层总成本 MILP 精修机制
- 第二问只是在“可行域、插入偏好、候选列构造、硬门槛诊断”上加入政策层

## 3. 第二问相对第一问的第一层变化：把政策写进 `ServiceUnit`

### 3.1 新增 `PolicyServiceUnit`

第二问在 `solve_question2.py` 中新增了 `PolicyServiceUnit`，相比第一问的 `ServiceUnit`，多出以下字段：

- `in_green_zone`
- `must_use_ev_under_policy`
- `fuel_allowed_after_16`
- `ev_allowed_flag`
- `fuel_allowed_flag`
- `ev_tw_start_min`
- `ev_tw_end_min`
- `fuel_tw_start_min`
- `fuel_tw_end_min`

这一步是第二问最本质的变化。第一问的 unit 只需要知道载重、体积和原始时间窗；第二问的 unit 必须再知道“对不同动力类型到底允许在什么窗口服务”。

### 3.2 政策不再是事后筛查，而是前置进入可行域

第二问会读取预处理产物：

- `preprocess_config.json`
- `policy_feasibility.csv`
- `ev_policy_summary.csv`

并据此构造每个客户的政策属性。这里有一个重要建模决定：

- 优化器阶段不再重新推导“是否部分重叠禁行”
- 而是直接消费预处理给出的 `fuel_service_window_start_min / fuel_service_window_end_min`

也就是说，第二问把复杂政策翻译成“车辆类型相关服务窗口”，然后在求解阶段只处理窗口可行性。这么做的原因是：

- 政策文字约束不适合在路线搜索阶段反复解释
- 窗口化之后，可行域检查可以直接嵌进 `evaluate_route`
- route-pool、局部搜索、MILP、hybrid 都能共享同一套可行性口径

### 3.3 绿配区口径被固定，而不是模糊引用题面数字

第二问脚本固定了：

- `POLICY_GREEN_ZONE_BASIS = "attachment_geometry_radius_10km"`

同时输出：

- `problem_statement_green_zone_count = 30`
- `green_zone_customer_count_total_geometry`
- `green_zone_customer_count_used`

这里体现的是一个很重要的建模态度：题面口径和几何口径存在差异时，求解器必须固定一种可复现实操口径，否则后续所有政策校验都不稳定。第二问最终采用的是预处理几何口径，而不是在优化阶段反复切换定义。

## 4. 第二问相对第一问的第二层变化：把政策写进拆分与服务窗口

### 4.1 车辆可选集合先经过政策过滤

第一问里，客户可用车型只由重量和体积决定。第二问里，多了一层：

- 若客户 `must_use_ev_under_policy = True`，则燃油车型直接从 `eligible_vehicle_types` 中删掉
- 若客户只是绿配区受限，则 EV 与 fuel 都可能保留，但 fuel 的可服务窗口会被缩窄

这一步解决的是“容量可行不等于政策可行”的问题。若只按载重容积建模，再到最后检查政策，很多路线会在搜索过程中白白进入错误区域。

### 4.2 路线评估从统一时间窗改为“按动力类型取时间窗”

第一问的 `evaluate_route` 默认所有车都看同一个 `tw_start_min / tw_end_min`。第二问增加了 `_service_window_for_vehicle`：

- `fuel` 车使用 `fuel_tw_start_min / fuel_tw_end_min`
- `ev` 车使用 `ev_tw_start_min / ev_tw_end_min`

随后在 `evaluate_route` 和 `_simulate_route_scalar` 中，实际等待与迟到都基于“当前路线车型对应的策略窗”来算。

这一步的意义是：

- 第二问的时间窗不再只是客户属性，而是“客户-车型联合属性”
- 同一个客户对 EV 和 fuel 可以有不同可行域
- 如果不这样改，fuel 车可能会在 16:00 之前被错误判成可行

### 4.3 必要拆分从“容量感知”升级为“政策感知”

第一问必要拆分的关注点主要是：

- 拆成几趟
- 会不会占用过多大车
- 每个拆分片段的可选车型集合有多宽

第二问在此基础上进一步加入了政策感知拆分。具体体现在：

- `_policy_pack_score` 会优先压低 `ev3000_only_count`
- `_policy_greedy_pack_items` 在每次装箱时会优先选择“不把碎片逼成只能用 `ev_3000`”的装法
- `_policy_pack_customer_fragments_for_count` 仍保留多次尝试，但每次尝试用确定性种子，既保留多样性，又保证复现

这里的逻辑非常关键。对于第二问，纯容量可行的拆分可能在政策上是灾难性的，因为它会把若干片段都压缩成“只能 `ev_3000` 承担”。而 `ev_3000` 是 Q2 中最稀缺的策略资源之一。

### 4.4 第二问新增了第二轮库存修正：不仅修大车，还修 `ev_3000`

第一问里，必要拆分之后会做“大车库存修正”，控制 `heavy_big_only` 的占用。第二问保留这一步，同时再加了一轮：

- 统计当前 mandatory split 中有多少片段是 `ev_3000_only`
- 若超出 `ev_3000` 可用库存，则继续把相关客户切换到访问次数更多但 `ev_3000_only_count` 更低的拆分方案

这说明第二问相对第一问的新增难点是：

- 第一问的稀缺资源主要是 3000kg 大车
- 第二问还额外出现了“必须 EV 且只能大 EV 承接”的政策型稀缺资源

如果不在拆分阶段就先把这件事修掉，后续路线搜索经常会整体不可行，或者被迫过度消耗 EV 资源。

### 4.5 仍然保留 `BIG_VEHICLE_RESERVE = 5`

第二问没有取消第一问里的 `BIG_VEHICLE_RESERVE = 5`。原因不是偷懒，而是：

- 政策约束会进一步挤压大车和大 EV 的可用空间
- 若把大车库存算得过满，route-pool 与 MILP 很容易失去缓冲
- 保留少量大车冗余，能避免 mandatory split 阶段就把大车资源吃死

这属于安全冗余，不是业务真值。它的用途是给后续结构重组留下回旋空间。

## 5. 第二问相对第一问的第三层变化：新增政策校验与结果可见性

### 5.1 第二问专门加入政策诊断

第二问新增 `_policy_solution_diagnostics`，会在最终解上统计：

- `mandatory_ev_served_by_non_ev_count`
- `fuel_route_green_zone_pre16_visit_count`
- `fuel_route_green_zone_post16_visit_count`
- `ev_route_green_zone_visit_count`
- `policy_violation_count`

这一步的重要性在于：第二问不是“算出一个低成本解再主观解释”，而是把政策是否真正满足做成可输出的机器诊断指标。

### 5.2 输出层不再只看成本，还暴露政策口径

第二问会在以下文件中补入政策信息：

- `q2_route_summary.csv`
- `q2_stop_schedule.csv`
- `q2_customer_aggregate.csv`
- `q2_service_units.csv`
- `q2_split_plan.csv`
- `q2_cost_summary.json`
- `q2_policy_summary.json`

目的很明确：

- 路线级看是否有 fuel 在绿配区 16:00 前服务
- 客户级看谁被 EV 服务、谁仍被 fuel 服务
- service unit 级看每个 unit 的 EV/fuel 策略窗
- 汇总级看最终是否真的满足政策硬约束

第二问因此形成了“求解器 + 验模器”一体化结构。

## 6. 第二问相对第一问的第四层变化：Q2 cost-first v2 的策略强化

如果说 `solve_question2.py` 只是把第一问改造成“政策可行”，那么 `solve_question2_costfirst_v2.py` 做的是第二次增强：把“政策资源稀缺性”写进构造、插入和列池塑形中。

### 6.1 单元优先级进一步改造

Q1 的 `_unit_priority_key` 主要围绕：

- 是否重货大车专用
- 时间窗紧张度
- 重量、体积

Q2 v2 在前面又加了几层：

- 必须 EV 的单元优先
- `ev_3000_only` 的单元更优先
- 必须 EV 且可由 `ev_1250` 承接的单元优先
- `fuel_allowed_after_16` 的单元优先
- `in_green_zone` 的单元优先

原因是第二问早期构造时，最怕的不是成本略高，而是先把稀缺 EV 资源错误分配给普通客户，导致后面真正需要 EV 的客户无车可用。

### 6.2 构造与插入不再只看 `delta_cost`，而是使用 Q2 专属字典序优先级

Q2 v2 新增 `_q2_choice_rank_key`。它依次判断：

- 普通且可用 fuel 的客户是否占用了 EV
- 是否进一步占用了 `ev_3000`
- `fuel_allowed_after_16` 客户是否被 fuel 承接
- 这类 late-fuel 客户是否形成了 fuel cluster
- 必须 EV 客户是否优先落到 `ev_1250`
- 必须 EV 客户是否被非 EV 承接
- 最后才看 `delta_cost`

这是一种确定性贪心的字典序决策。为什么要这么做：

- 第二问局部构造阶段没有办法实时看见全局资源影子价格
- 但必须把“保 EV 给必须 EV 客户”这件事优先级抬到成本之前
- 否则早期一个看似便宜的选择，可能把后面整个策略可行域锁死

需要明确的是：这不是最终目标函数，只是构造/插入阶段的决策规则。

### 6.3 新增三类 Q2 专属候选列

Q1 的 route-pool 已经有：

- `singleton`
- `flex_small`
- `promotion`
- `piggyback_big`
- `rigid_big`

Q2 v2 在此基础上再显式构造三类政策定向列：

- `ev_flex_small_q2`
- `late_fuel_cluster_q2`
- `policy_piggyback_q2`

它们分别解决三类问题：

- `ev_flex_small_q2`：主动搜索“多个必须 EV 单元由 EV 小车/大车承接”的小团簇
- `late_fuel_cluster_q2`：主动搜索“16:00 后可由 fuel 承接的客户聚到 fuel 路线里”
- `policy_piggyback_q2`：保留那些虽然看起来像 piggyback，但对政策资源释放有帮助的大车列

为什么 Q1 的通用列池不够用：

- Q1 的列池是按一般节约逻辑生列
- Q2 的关键结构恰恰是“政策友好但不一定马上显示最大节约”的列
- 如果不人为补这些列，它们在 route-pool 截断前就可能消失

### 6.4 新增候选生成上限，防止政策列把池子撑爆

Q2 v2 同时加入：

- `Q2_V2_EV_SOURCE_LIMIT = 32`
- `Q2_V2_LATE_FUEL_SOURCE_LIMIT = 32`
- `Q2_V2_NEIGHBOR_LIMIT = 10`
- `Q2_V2_PAIR_PARTNER_LIMIT = 6`
- `Q2_V2_TRIPLE_PARTNER_LIMIT = 4`

这些不是业务参数，而是候选生成预算。它们的用途是：

- 保证 Q2 专属列被生成
- 又不让 route-pool 因为政策定向扩展而爆炸

这类数字同样是经验定标，不是理论最优值。

### 6.5 列池排序增加了政策加减评分

Q2 v2 的 `_column_effective_saving` 在第一问 `current_cost_saving` 基础上，增加了人为加减分：

- `ev_flex_small_q2`：`+18 * must_ev_count`
- 若车型是 `ev_1250`：再 `+8 * must_ev_count`
- `late_fuel_cluster_q2` 且是 fuel：`+15 * late_fuel_count`
- `policy_piggyback_q2`：`+4`
- 若是 EV 车承接了本可用 fuel 的普通客户：`-12 * ordinary_with_fuel_option`
- 若还是 `ev_3000`：再额外 `-6 * ordinary_with_fuel_option`
- 若 fuel 车承接了 late-fuel 客户：再 `+4 * late_fuel_count`

这些值为什么要人为设：

- route-pool 截断阶段还没进入最终 MILP
- 此时必须用一组代理评分表达“哪些列更值得被保留”
- 这些分值本质上是政策资源的影子价格近似，不是业务成本

这些值的作用是什么：

- 改变列排序
- 提高策略友好列进入 MILP 的概率
- 降低“普通客户浪费 EV，尤其浪费 `ev_3000`”的概率

有没有严格检验这些具体值最优：

- 没有单独消融证明 `18` 一定优于 `15`
- 目前能证明的是这组值经过端到端结果检验，最终得到的 Q2 v2 与 Q2 hybrid 都满足政策硬约束，并且优于基线

再次强调：

- 这些分值不会直接替代最终 MILP 里的 `total_cost`
- 它们只在列池塑形阶段起作用

## 7. 第二问相对第一问的第五层变化：统一到 hybrid 外层

### 7.1 为什么第二问最终也采用 hybrid，而不是停留在 Q2 v2

Q2 v2 已经能给出政策可行解，但它仍然高度依赖：

- 确定性构造顺序
- 确定性局部改良
- 当前 route-pool 中已有的结构

这意味着它仍可能卡在“政策可行但结构局部最优”的状态。`90142.3299` 就是这样一个强基线。

因此第二问最终没有推翻 Q2 v2，而是采用和第一问相同的统一方案：

- 外层用 hybrid 负责做更大范围的结构扰动与学习
- 内层仍调用 Q2 v2 的改良、route-pool、MILP 和政策诊断

### 7.2 第二问 hybrid 沿用第一问外壳，但替换了引导逻辑

`solve_question2_hybrid.py` 直接继承第一问 hybrid 外层机制，包括：

- 概率构造
- `pbest / gbest`
- 信息素
- elite archive
- 柯西扰动
- cluster remove

但它把外层构造中的 `_rank_key_penalty` 换成 Q2 专属版本，重点惩罚：

- 普通客户占用 EV
- 普通客户占用 `ev_3000`
- late-fuel 客户没有落到 fuel
- late-fuel 没形成 fuel cluster
- 必须 EV 客户没有优先用 `ev_1250`
- 必须 EV 客户被非 EV 承接

这些权重比列池加减分大很多，是因为它们要进入概率权重的 `exp(-generalized_delta_cost / T)`。如果权重太小，外层搜索几乎只会看到原始 `delta_cost`，政策引导会失效。

这里同样要说明：

- 这些权重是经验定标
- 它们不是最终目标函数
- 它们的作用是控制外层搜索往哪里探索

### 7.3 第二问 hybrid 多了政策硬门槛

第一问 hybrid 的硬门槛主要保证：

- mandatory split 数量正确
- 普通客户不被重复路由
- 所有单元都被覆盖

第二问在此基础上再加：

- `policy_violation_count == 0`
- `mandatory_ev_served_by_non_ev_count == 0`
- `fuel_route_green_zone_pre16_visit_count == 0`

也就是说，第二问 hybrid 不是“先找到便宜解再看政策”，而是直接把政策违规解排除在外层优胜解之外。

### 7.4 第二问 hybrid 的比较基线也切换成 Q2 v2

第二问 hybrid 不再拿 Q1 或 Q2 v1 做直接对照，而是固定对照：

- `question2_artifacts_costfirst_v2`

这体现了迭代式建模思路：先把政策口径校正，再在同一口径上做结构优化，而不是跳过中间层直接对旧口径结果做比较。

## 8. 迭代路径：在第二问中发现了什么问题，因此采用了哪些措施

| 迭代中发现的问题 | 若不处理会怎样 | 最终采取的措施 |
| --- | --- | --- |
| 直接沿用 Q1 的 `ServiceUnit`，没有车型相关策略窗 | fuel 车可能在绿配区错误地被判为早于 16:00 也可服务 | 把政策写进 `PolicyServiceUnit`，按车型取服务窗口 |
| 只做容量拆分，不考虑政策 | mandatory split 片段可能大量变成 `ev_3000_only`，整体不可行 | 在拆分阶段加入 `ev3000_only_count`，并单独做 `ev_3000` 库存修正 |
| 只保证一般成本最小，未显式保护 EV 资源 | 普通客户会过早占用 EV，尤其是 `ev_3000` | 在 unit priority、choice rank key、column scoring 中显式惩罚“普通客户占 EV” |
| 只用 Q1 的通用 route-pool | 策略友好的 EV 小团簇和 late-fuel fuel 簇不一定进得了列池 | 定向生成 `ev_flex_small_q2`、`late_fuel_cluster_q2`、`policy_piggyback_q2` |
| 只用 Q2 v2 内层 | 容易停在确定性 warm start 的局部最优附近 | 在外层包一层 hybrid，增加多样化结构搜索 |
| 只看最终成本，不看政策可见性 | 结果可能“看起来更好”，但实际违反政策 | 单独输出 `policy_violation_count` 等诊断，并作为 hard guard |

这张表说明第二问的变化不是“多加几条规则”，而是一个持续迭代过程：

- 先发现 Q1 的可行域表达不足
- 再发现光有政策可行还不够，EV 资源会被浪费
- 再发现即使策略感知贪心已经很强，仍然会卡在内层局部最优
- 最终才形成现在这套“Q2 v2 内层 + Q2 hybrid 外层”的统一方案

## 9. 第二问最终模型应如何概括

如果要把第二问的建模思路浓缩成一段话，最准确的说法是：

第二问仍以第一问的总成本最小化框架为主线，保留 `ServiceUnit` 抽象、时变路径评估、route-pool 与 MILP 全局重组选路，并在其上加入政策可行域建模。具体而言，第二问把绿色配送区和必须使用电动车等政策转化为“客户-车型联合服务窗口”和“政策过滤后的可选车型集合”，在 mandatory split 阶段提前消化大车与 `ev_3000` 的双重库存压力；随后在内层通过 Q2 专属字典序贪心、定向候选列和策略感知列评分保护稀缺 EV 资源，并在外层沿用与第一问一致的 hybrid 搜索机制扩大结构搜索范围。整个流程始终以 `total_cost` 为最终优化目标，而把政策约束、候选列加减分和搜索引导权重都作为可行域控制与搜索代理，而不是直接替换目标函数。

## 10. 当前结论

第二问相对第一问，真正发生变化的不是成本定义本身，而是以下三件事：

- 可行域从“容量 + 原始时间窗”升级为“容量 + 车型相关政策窗 + EV 强制约束”
- 稀缺资源从“3000kg 大车”为主升级为“3000kg 大车 + `ev_3000` + 可在 16:00 后使用 fuel 的政策通道”
- 搜索机制从“通用节约逻辑”升级为“通用成本逻辑 + 政策资源保护逻辑”

因此，第二问本质上不是一个完全不同的问题，而是第一问在政策限制下的强化版本。最终采用 Q2 hybrid，是因为它同时保留了第一问框架的统一性、Q2 v2 的政策可行性，以及更大范围结构搜索带来的最终降本能力。
