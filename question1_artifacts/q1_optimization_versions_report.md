# 问题1优化迭代总报告

## 1. 报告目的

这份报告用于完整总结问题1从最初脚本到当前版本的所有主要优化迭代，统一回答三个问题：

1. 每一版到底发现了什么问题。
2. 每一版提出了什么解决方案。
3. 每一版是怎么实施、结果如何、为什么继续往下做。

需要先说明两点：

- 问题1的题目原始目标是：**在无政策限制条件下，以总配送成本最低为目标设计调度方案**。
- 因此，`route_count`、`single_stop_route_count`、`mixed-big`、`late_positive_stops` 这些都不是题目原始主目标，它们是帮助解释成本、发现结构问题、构造候选解的**诊断指标**。

本报告中：

- 版本1到版本9，依据本次迭代历史、阶段结果和讨论记录复原。
- 当前最新版以 `question1_artifacts/q1_cost_summary.json` 的结果为准。

---

## 2. 总体结论先行

从全局看，这 10 版迭代可以分成 4 个阶段：

### 阶段A：先修“模型口径是否正确”

- 重点解决普通客户被误拆分的问题。
- 结论是：如果拆分口径错了，会出现“路线更少、成本更低”的假优解，但这种解不符合题意。

### 阶段B：再修“为什么结构挖掘不出来”

- 重点解决 `fuel_3000 / ev_3000` 被过早占满导致 merge 无法发生的问题。
- 结论是：不是没有合并机会，而是大车库存被锁死后，机会无法落地。

### 阶段C：把搜索对象从 unit 提升到 route-set

- 重点引入 route-pool 和 set partitioning / MILP 全局选路。
- 结论是：方向正确，但一开始被“结构纯化”带偏了，过度追求消灭 mixed-big，反而损失了成本最优。

### 阶段D：回到题目原目标，做 cost-first 优化

- 重点把主目标重新对齐到 `total_cost`。
- 结论是：当前最新版虽然 `route_count` 没有继续下降，但总成本已经从 `92442.10` 降到 `87616.61`，而且 `split=37`、普通客户未误拆，这才是目前最符合题意的主结果。

**因此，当前最重要的结论不是“路线是否最少”，而是：当前最新版已经找到了一组总成本显著更低、口径正确、可交付的调度方案。**

---

## 3. 版本逐次总结

## 版本1：错误拆分造成的假优解

### 发现问题

- 早期 smoke 结果显示成本和路线数都很好看，但普通客户被错误拆成了多次访问。
- 这类结果虽然数值漂亮，但已经偏离题意，因为问题1并没有允许普通客户为了优化而随意拆分服务。

### 解决方案

- 把服务建模从“按订单碎片自由插入”收回到“按客户级 `ServiceUnit` 管理”。
- 普通客户必须单次服务。
- 只有必要拆分客户才允许拆。

### 实施流程

1. 检查 `split_customer_count` 和普通客户访问次数。
2. 对比错误拆分前后的路线数量和成本。
3. 将“普通客户不拆”设为硬口径。

### 结果

- 早期错误拆分结果大约为：
  - `total_cost = 82779.714`
  - `route_count = 114`
  - `split_customer_count = 52`
- 这组结果不能用于最终提交。

### 结论

- 这一版的价值不是得到可用解，而是明确证明了：**低成本不等于正确解**。
- 后续所有优化都必须建立在“拆分口径正确”之上。

---

## 版本2：最小可判别版，先把拆分口径修正到正确

### 发现问题

- 版本1证明了普通客户误拆是核心偏差来源。
- 但仅修口径后，局部搜索几乎没有真正挖出路线合并。

### 解决方案

- 固定问题1的建模口径：
  - 普通客户单次服务
  - 必要拆分客户按当前规则拆成 `ServiceUnit`
  - 一车一天一路线
  - `21:00` 后允许继续服务和回仓
  - 软时间窗
- 只保留最轻局部算子：
  - `route_merge`
  - `relocate`
  - `route_type_change`

### 实施流程

1. 先生成一组正确口径的初始解。
2. 跑很轻的局部搜索。
3. 重点验证 `split_customer_count == mandatory_split_customer_count`。

### 结果

- 这一版得到的关键结果是：
  - `split_customer_count = 37`
  - `mandatory_split_customer_count = 37`
  - `single_stop_route_count = 103`
  - `two_stop_route_count = 20`
  - `three_plus_route_count = 10`
  - 总路数可由上面三项推出为 `133`
  - `route_merge_success_count = 0`
  - `relocate_success_count = 0`
  - `route_type_change_success_count = 2`

### 结论

- 这一版解决了“模型口径正确性”问题。
- 但同时也暴露出下一个问题：**合并挖掘严重不足**。

---

## 版本3：大车库存预留与一次性 merge 诊断

### 发现问题

- 虽然单站路线很多，但合并几乎完全不发生。
- 进一步分析发现，不是没有可合并路线，而是 `fuel_3000 / ev_3000` 被前面阶段占满，导致合并时根本没有大车空槽可用。

### 解决方案

- 做一个最小诊断链，而不是直接重写求解器：
  - 在 repair 阶段预留 `fuel_3000` 空槽
  - 只做一次 `batch_small_to_fuel3000_merge`
- 目标不是直接得到最终最优，而是验证“先解锁大车，再做合并”这条逻辑是否成立。

### 实施流程

1. 在修复阶段引入 `fuel_3000` 预留。
2. 重新安排少量 flexible unit。
3. 用腾出来的一台大车做一次批量小路线合并。

### 结果

- 这条诊断链验证成功：
  - `reserve_repair_success_count = 1`
  - `batch_merge_success_count = 1`
- 主解从前一版约 `133 / 103` 改善到：
  - `route_count = 132`
  - `single_stop_route_count = 101`

### 结论

- 这一版证明了：**问题不是没有合并机会，而是大车库存先被锁死了。**
- 但这仍然只是诊断链，不是稳定主框架。

---

## 版本4：route-pool 原型与 route-set 重组

### 发现问题

- 单纯按 unit 插入和轻局部搜索，做不了“多条路线一起换”的动作。
- 需要把搜索对象从 unit 升级为 route-set。

### 解决方案

- 引入 `CandidateRouteSpec` 和 route-pool。
- 新增 route-set 级局部重组：
  - `release`
  - `promotion`
  - cluster cover

### 实施流程

1. 生成 `seed / singleton / rigid_big / flex_small / promotion` 候选路线。
2. 对局部 cluster 做 exact-cover 替换。
3. 用 route-set 级动作替代 unit 级试探。

### 结果

- 主解仍停留在：
  - `route_count = 132`
  - `single_stop_route_count = 101`
- 但诊断链首次跑通：
  - `diagnostic_unlock_success_count = 4`
  - `diagnostic_promotion_success_count = 1`
  - `route_pool_candidate_count = 2798`

### 结论

- 这一版证明：**route-set 级重组是必要方向。**
- 但局部重组还不够强，无法稳定产出更优整解。

---

## 版本5：集合划分 MILP 第一版，全局 unlock / compress

### 发现问题

- 局部 cluster 替换可以诊断问题，但无法全局同时选路。
- 需要一个全局层，一次性决定整套路由组合。

### 解决方案

- 在 route-pool 上增加 set partitioning MILP。
- 采用 staged objective：
  - 先 unlock big mixed
  - 再 compress single-stop 和 route count

### 实施流程

1. 构造候选路线覆盖矩阵。
2. 加入服务单元覆盖约束和车辆库存约束。
3. 分阶段求解 big mixed、single-stop、route_count 和 cost。

### 结果

- 全局模型第一次跑通，但得到的全局解虽然“结构纯”，却明显更差：
  - `global route_count = 157`
  - `global single_stop_route_count = 140`
  - `global total_cost = 104901.34`
- 但它把 mixed-big 清到了很低水平，并释放了大车库存。

### 结论

- 这一版证明：**全局模型方向是对的，但目标顺序错了。**
- 它完成了 `unlock`，没有完成 `compress`。

---

## 版本6：重排目标顺序 + 残差补列

### 发现问题

- 版本5把 `big_route_count` 过早锁死，导致释放出来的大车不能回用于压缩。
- 同时，残差列尤其是 `flex_small` 和 `promotion` 列太少。

### 解决方案

- 删除 `big_route_count` 硬锁。
- 改成“两段式大车上界 + 软偏好”。
- 引入残差补列：
  - 空间邻接
  - 时间窗重叠
  - 节约值排序

### 实施流程

1. 先跑 `pass1`。
2. 找 `singleton` 残差。
3. 生成 `residual_promotion / residual_flex_small`。
4. 再跑 `pass2`。

### 结果

- 全局解略有改善：
  - `pass1 = 157 / 140`
  - `pass2 = 156 / 138`
- 其中：
  - `selected_singleton_count: 140 -> 138`
  - `residual_promotion = 365`
  - `residual_flex_small = 4`

### 结论

- 这一版说明 `compress` 开始动起来了。
- 但问题没有根治，因为 `promotion` 仍然几乎不被选中。

---

## 版本7：promotion-like 和 bad mixed-big 语义分离

### 发现问题

- 之前所有 `promotion` 大车路线都被当成 mixed-big 一起惩罚。
- 等于模型自己把压缩型大车路线也封杀了。

### 解决方案

- 重写 mixed-big 语义：
  - `promotion_like_big_route`
  - `bad_big_flexible_route`
- unlock 阶段只惩罚坏 mixed-big，不惩罚压缩型大车路线。

### 实施流程

1. 重新定义路线结构标签。
2. 改写 MILP 的 unlock 指标。
3. 把输出和报表口径同步改掉。

### 结果

- `promotion` 不再被模型封杀：
  - `promotion_like_candidate_count = 974`
  - `global_pass1_selected_promotion_like_count = 5`
  - `global_pass2_selected_promotion_like_count = 5`
- 坏 mixed-big 清到 `0`
- 但全局解仍停在：
  - `route_count = 149`
  - `single_stop_route_count = 128`

### 结论

- 这一版解决了“promotion 进不来”的问题。
- 新瓶颈转移到了：**小车多站列不足**。

---

## 版本8：`flex_small` 强化 + `pass3` focused residual

### 发现问题

- 即使 `promotion` 已经能进来，剩余柔性 singleton 仍然太多。
- route-pool 的小车多站列供给不够，尤其是 `flex_small`。

### 解决方案

- 扩大 base `flex_small` 生成：
  - 空间邻接 + 时间窗重叠
  - 更多 small vehicle type 组合
- 增加 `pass3`：
  - 只盯 `pass2` 里的柔性 singleton 残差
  - 尝试 2/3/4-stop residual small routes

### 实施流程

1. 扩大 base flex-small 邻域。
2. 新增 `pass3`。
3. 生成 focused residual flex-small 候选。

### 结果

- 全局解进一步改善到：
  - `route_count = 145`
  - `single_stop_route_count = 118`
- 同时：
  - `selected_flexible_singleton_count = 53`
  - `selected_flex_small_count = 22`
  - `candidate_pool_cap_binding_flag = 0`

### 结论

- 这一版证明：**继续补 `flex_small` 是有效的**。
- 但只围绕 singleton 之间补列，不够。

---

## 版本9：`cluster_flex_small`，从 singleton-only 补列升级到 cluster 重组

### 发现问题

- 剩下的柔性 singleton 已经不是简单两两/三三配一下就能消掉。
- 它们需要和已经成型的小路线一起重组。

### 解决方案

- `pass3` 改为 cluster 模式：
  - 从选中的 singleton 出发
  - 把附近已选中的 `flex_small` 路线一起拉进局部 cluster
  - 重新生成 `cluster_flex_small`

### 实施流程

1. 围绕已选小路线 cluster 生成 2/3/4-stop small candidates。
2. 修复“默认取最后一个 pass”为全局解的 bug。
3. 重新跑 `pass3`。

### 结果

- 新增：
  - `cluster_flex_small = 134`
  - 其中 `2-stop = 88`
  - `3-stop = 36`
  - `4-stop = 10`
- `pass3` 进一步把结构压到：
  - `route_count = 145`
  - `single_stop_route_count = 116`
  - `selected_flexible_singleton_count = 51`
  - `selected_flex_small_count = 24`
- 但总路线数仍未继续下降。

### 结论

- 这一版说明：cluster 重组确实比 singleton-only 更有效。
- 但更深层的瓶颈已经暴露：**模型把所有 mixed-big 都当坏结构，方向带偏了。**

---

## 版本10：回到题目主目标，改成 `cost-first + piggyback_big`（当前最新版）

### 发现问题

- 前 9 版最大的问题，不是“技术不够多”，而是**主判优标准偏离了题目原目标**。
- 题目要求的是 `total_cost` 最低，不是 `route_count` 最低，也不是 mixed-big 必须清零。
- 当前数据下：
  - `startup_cost` 占总成本超过一半，路线碎片当然重要
  - 但它仍然只能通过“是否降低总成本”来判断价值
- 此外还发现：
  - baseline 里一部分 mixed-big 实际上是“有价值的顺路搭载”
  - 如果把它们强行清掉，会把总成本优化带偏

### 解决方案

- 主线改成 `cost-first`：
  - 直接最小化 `total_cost`
  - 结构指标降级为诊断指标
- mixed-big 再分类：
  - `promotion_like_big`
  - `piggyback_big`
  - `blocking_big_flexible`
- 允许：
  - 有成本价值的 `piggyback_big`
  - 有压缩价值的 `promotion_like_big`
- 只继续把真正阻塞结构的 `blocking_big_flexible` 当问题看

### 实施流程

1. route-pool 中显式加入 `piggyback_big` 候选。
2. 建立 `cost_first_total_cost` 全局 MILP 主链。
3. 跑 2 轮 route-pool iteration。
4. 仅当总成本改善时，才替换主解。
5. 同时保留：
  - `split_packing_sensitivity` 接口
  - 若 route-pool cost-first 无提升，再去看 packing 端

### 当前结果

- 当前最新版主解已经被全局 `pass2` 替换，结果为：
  - `total_cost = 87616.6098`
  - `route_count = 132`
  - `single_stop_route_count = 97`
  - `split_customer_count = 37`
  - `mandatory_split_customer_count = 37`
  - `final_solution_source = pass2`
- 相比原 baseline：
  - `baseline_total_cost = 92442.0983`
  - 成本下降约 `4825.49`
  - 下降幅度约 `5.22%`
- 结构变化：
  - `route_count` 没变，仍然是 `132`
  - `single_stop_route_count: 101 -> 97`
  - `final_piggyback_big_count = 20`
  - `final_promotion_like_big_count = 5`
  - `final_blocking_big_flexible_count = 4`
- 服务质量观察值：
  - `late_positive_stops = 18`
  - `max_late_min = 99.00`
  - `after_hours_return_count = 24`
- 这些仍然是软约束表现，不影响题目主目标口径。

### 结论

- 这一版是目前为止**最重要、也是最正确的一版**。
- 它证明了：
  - `route_count` 不是唯一关键指标
  - `mixed-big` 不是必须全部消灭
  - 只要能降低总成本并保持拆分口径正确，就是更优调度方案
- 当前这版已经是目前最符合题意的主结果。

---

## 4. 迭代主线复盘：问题是怎么一步步被找出来的

把 10 版串起来，真正解决问题的逻辑链如下：

### 第一步：先纠正建模口径

- 先确认普通客户不能误拆。
- 这一步解决的是“假优解”的问题。

### 第二步：再确认为什么 merge 不发生

- 不是没有合并机会。
- 是大车库存先被锁死，导致合并机会无法执行。

### 第三步：把搜索对象从 unit 提升到 route-set

- 因为真正有效的重构不是“插一个 unit”，而是“同时换掉几条路线”。
- 这一步带来了 route-pool、cluster cover 和 set partitioning。

### 第四步：修复模型语义

- 先修 `promotion_like` 被误杀的问题。
- 再修“所有 mixed-big 都被当成坏结构”的问题。

### 第五步：回到题目原始目标

- 最终发现，真正需要优化的不是“路线看起来最整齐”，而是“总配送成本最低”。
- 结构指标的价值，在于解释成本和生成候选，不在于替代成本目标。

---

## 5. 当前版本到底“够不够”

从题目目标看，当前最新版已经是一个**足够强且可交付**的答案，原因如下：

- 主目标 `total_cost` 已经显著下降。
- 拆分口径仍然正确：
  - `split_customer_count = mandatory_split_customer_count = 37`
  - 普通客户未误拆
- 路线结构没有恶化，反而略有改善：
  - `single_stop_route_count = 97`
- 当前解不是“结构漂亮但成本更高”的假改善，而是真正和题目目标一致的改善。

更准确地说：

- 现在这版**不是被数学上证明的全局最优**
- 但已经是当前建模和算法体系下，**最符合题目目标的最佳主解**

---

## 6. 如果还要继续优化，应该往哪里走

当前继续优化的优先级应该是：

### 优先级1：继续按成本目标优化

- 只接受能进一步降低 `total_cost` 的修改。
- 不再为了降低 `route_count` 而降低 `route_count`。

### 优先级2：运行 split/packing sensitivity

- 当前代码里已经预留了 `split_packing_sensitivity` 接口。
- 当前状态是：
  - `split_packing_sensitivity_executed = 0`
  - `split_packing_sensitivity_status = not_run`
- 下一步如果继续做，最有价值的不是再堆 route-pool，而是试探：
  - 是否能通过 packing 端减少 big-only 依赖，从而进一步降成本

### 优先级3：只把结构指标当诊断工具

- `route_count`
- `single_stop_route_count`
- `piggyback_big`
- `promotion_like_big`
- `blocking_big_flexible`

这些仍然应该输出、分析、解释，但不应再次替代成本目标。

---

## 7. 最终结论

问题1这 10 版迭代，真正的路线不是“不断减少车辆”，而是：

1. 先排除错误建模造成的假优。
2. 再识别合并失败是库存锁死还是结构弱搜索。
3. 再把局部搜索升级为全局 route-pool + MILP。
4. 再纠正 mixed-big 语义。
5. 最后回到题目原始主目标：**总配送成本最低**。

当前最新版的意义就在于：

- 它不再追求“结构最好看”
- 而是追求“在题目给定成本体系下的真实更优调度”

截至目前，**版本10 是当前最推荐作为问题1主结果的版本**。

