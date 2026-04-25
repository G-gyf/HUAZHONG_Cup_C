# 全部论文图汇总与图注报告

本报告汇总当前工作区中三个来源的论文图片：
- [question1_artifacts_hybrid_coupled_heavy_s11/paper_figures](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures>)
- [question2_artifacts_hybrid_standard_s11/paper_figures](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures>)
- [paper_compare_figures](<C:\Desktop\华中杯A题\paper_compare_figures>)

说明：
- 对于同一图同时存在 `PNG` 与 `SVG` 的情况，下文只按“一张图”统计，并给出主路径与备用路径。
- 本报告以“可直接写入论文的图注”为目标，图注偏正式、概括性强。

## 一、Q1 s11 图组

### 图 1
图名：01_total_cost_progress

主路径：[01_total_cost_progress.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\01_total_cost_progress.png>)

建议图注：  
图 X 展示了问题一混合求解过程中总成本随搜索阶段的变化轨迹，反映了从初始方案到最终方案的逐步改进过程。该图可用于说明混合启发式与后续精修策略在成本优化上的收敛效果。

### 图 2
图名：02_final_cost_composition

主路径：[02_final_cost_composition.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\02_final_cost_composition.png>)

建议图注：  
图 X 展示了问题一最终方案的成本构成，包括车辆启动成本、能耗成本、碳排放成本、等待成本和迟到成本。结果表明，启动成本与能耗成本构成总成本主体，而等待与迟到成本占比较小。

### 图 3
图名：03_baseline_vs_final_metrics

主路径：[03_baseline_vs_final_metrics.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\03_baseline_vs_final_metrics.png>)

建议图注：  
图 X 对比了问题一基线方案与最终方案的关键指标变化，包括总成本、车次数、单站路线占比和迟到指标等。该图用于说明最终方案相对于初始构造解的综合优化效果。

### 图 4
图名：04_vehicle_mix_and_stop_pattern

主路径：[04_vehicle_mix_and_stop_pattern.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\04_vehicle_mix_and_stop_pattern.png>)

建议图注：  
图 X 展示了问题一最终方案中不同车型的使用结构以及单站、双站、多站路线的构成比例。结果反映了车辆资源配置和停靠模式之间的对应关系。

### 图 5
图名：05_route_utilization_scatter

主路径：[05_route_utilization_scatter.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\05_route_utilization_scatter.png>)

建议图注：  
图 X 以散点形式展示了问题一路线载重利用率、容积利用率及其与路线代价之间的关系。该图用于观察高成本路线与低装载效率路线是否存在明显对应。

### 图 6
图名：06_route_timeline

主路径：[06_route_timeline.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\06_route_timeline.png>)

建议图注：  
图 X 展示了问题一所有路线在全天时间轴上的发车、服务与返仓分布情况。该图可用于说明配送任务在日内时段上的集中程度以及返仓压力。

### 图 7
图名：07_big_route_structure

主路径：[07_big_route_structure.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\07_big_route_structure.png>)

建议图注：  
图 X 展示了问题一中大车路线的类型结构，包括刚性大车路线、搭载型大车路线和其他大车家族。该图用于分析大车资源在最终方案中的组织方式。

### 图 8
图名：08_route_cost_pareto

主路径：[08_route_cost_pareto.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\08_route_cost_pareto.png>)

建议图注：  
图 X 展示了问题一路线成本与其他关键性能指标之间的帕累托关系，用于识别高成本路线与高迟到、高返仓压力路线的对应特征。

### 图 9
图名：09_customer_spatial_map

主路径：[09_customer_spatial_map.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\09_customer_spatial_map.png>)

建议图注：  
图 X 展示了问题一客户点的空间分布及其服务特征，可用于说明需求空间格局对路径组织和车型调配的影响。

### 图 10
图名：10_candidate_pool_family_mix

主路径：[10_candidate_pool_family_mix.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\10_candidate_pool_family_mix.png>)

建议图注：  
图 X 展示了问题一路径候选池中不同路线家族的构成情况。该图用于说明列生成或候选池扩展阶段主要依赖了哪些类型的候选路径。

### 图 11
图名：11_route_network_by_vehicle

主路径：[11_route_network_by_vehicle.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\11_route_network_by_vehicle.png>)

建议图注：  
图 X 按车辆类型展示了问题一最终配送网络结构。该图用于说明不同车型在空间覆盖范围和路线分布上的差异。

### 图 12
图名：12_big_route_network_by_family

主路径：[12_big_route_network_by_family.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\12_big_route_network_by_family.png>)

建议图注：  
图 X 按大车路线家族展示了问题一配送网络结构，用于说明不同大车路线模式在空间层面的组织特征。

### 图 13
图名：13_top_cost_route_map

主路径：[13_top_cost_route_map.png](<C:\Desktop\华中杯A题\question1_artifacts_hybrid_coupled_heavy_s11\paper_figures\13_top_cost_route_map.png>)

建议图注：  
图 X 展示了问题一中成本最高的若干条路线在空间上的分布。该图可用于定位高成本来源，并支撑后续对问题二、问题三的改进分析。

## 二、Q2 s11 图组

### 图 14
图名：q2_baseline_vs_final

主路径：[q2_baseline_vs_final.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_baseline_vs_final.png>)

建议图注：  
图 X 对比了问题二在政策约束引入前后的关键指标变化，包括总成本、车次数、迟到和政策相关服务结果。该图用于说明绿色政策约束对整体调度结果的影响。

### 图 15
图名：q2_cost_breakdown

主路径：[q2_cost_breakdown.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_cost_breakdown.png>)

建议图注：  
图 X 展示了问题二最终方案的成本分解结构。结果可用于说明在绿色政策与车型替代约束下，成本上升或变化主要来自哪些部分。

### 图 16
图名：q2_hybrid_search_trace

主路径：[q2_hybrid_search_trace.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_hybrid_search_trace.png>)

建议图注：  
图 X 展示了问题二混合求解过程中的搜索轨迹与成本变化。该图用于说明混合搜索在政策约束场景下的收敛过程与阶段性改进效果。

### 图 17
图名：q2_policy_service_mix

主路径：[q2_policy_service_mix.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_policy_service_mix.png>)

建议图注：  
图 X 展示了问题二不同政策客户类别的服务构成情况，例如必须由新能源车服务的客户、允许 16 点后燃油车进入的客户以及普通客户。该图用于说明政策执行后的服务分配结构。

### 图 18
图名：q2_route_cost_profile

主路径：[q2_route_cost_profile.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_route_cost_profile.png>)

建议图注：  
图 X 展示了问题二各条路线的成本分布特征。该图可用于分析在政策约束下高成本路线的形成原因及其尾部特征。

### 图 19
图名：q2_route_map

主路径：[q2_route_map.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_route_map.png>)

建议图注：  
图 X 展示了问题二最终配送方案的空间路径网络。该图用于说明引入绿色政策后路线覆盖范围和车型空间分工的总体格局。

### 图 20
图名：q2_seed_comparison

主路径：[q2_seed_comparison.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_seed_comparison.png>)

建议图注：  
图 X 展示了问题二不同随机种子下的解质量比较。该图用于验证混合求解器在政策场景下的结果稳定性。

### 图 21
图名：q2_vehicle_route_structure

主路径：[q2_vehicle_route_structure.png](<C:\Desktop\华中杯A题\question2_artifacts_hybrid_standard_s11\paper_figures\q2_vehicle_route_structure.png>)

建议图注：  
图 X 展示了问题二不同车型承担路线的数量结构与停靠模式特征。该图用于分析政策约束下车型替代和路径分工的变化。

## 三、综合对比图组 paper_compare_figures

### 图 22
图名：q1_q2_customer_policy_change

主路径：[q1_q2_customer_policy_change.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_customer_policy_change.png>)

建议图注：  
图 X 对比了问题一与问题二中客户服务分配的变化，重点展示政策约束对客户服务方式和车型分派带来的调整。

### 图 23
图名：q1_q2_green_zone_zoom

主路径：[q1_q2_green_zone_zoom.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_green_zone_zoom.png>)

建议图注：  
图 X 对问题一与问题二在绿色通行区附近的配送路径进行局部放大比较，用于展示绿色政策约束对核心区域路线组织的影响。

### 图 24
图名：q1_q2_overview

主路径：[q1_q2_overview.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_overview.png>)

建议图注：  
图 X 对比展示了问题一与问题二总体调度结果的核心差异，包括成本、路线结构和政策执行效果，是全文总览性质的综合图。

### 图 25
图名：q1_q2_route_difference_focus

主路径：[q1_q2_route_difference_focus.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_route_difference_focus.png>)

建议图注：  
图 X 聚焦展示了问题一与问题二差异最明显的若干条路线，用于说明绿色政策约束具体改变了哪些路线决策。

### 图 26
图名：q1_q2_route_overlay

主路径：[q1_q2_route_overlay.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_route_overlay.png>)

建议图注：  
图 X 将问题一与问题二的路线网络进行叠加比较，用于从整体空间层面展示两问方案之间的结构差异。

### 图 27
图名：q1_q2_schedule_profile

主路径：[q1_q2_schedule_profile.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_schedule_profile.png>)

建议图注：  
图 X 对比展示了问题一与问题二在全天调度时间轴上的运行结构，用于说明政策约束对发车、服务和返仓时段分布的影响。

### 图 28
图名：q1_q2_vehicle_structure

主路径：[q1_q2_vehicle_structure.png](<C:\Desktop\华中杯A题\paper_compare_figures\q1_q2_vehicle_structure.png>)

建议图注：  
图 X 对比了问题一与问题二的车型使用结构与路线承担比例，用于说明绿色政策实施后新能源车与燃油车的任务重分配情况。

### 图 29
图名：q2_seed_stability

主路径：[q2_seed_stability.png](<C:\Desktop\华中杯A题\paper_compare_figures\q2_seed_stability.png>)

建议图注：  
图 X 展示了问题二不同随机种子下结果波动的稳定性分析，用于补充说明问题二求解结果具有较好的种子鲁棒性。

## 四、第三问与敏感性分析图组

### 图 30
图名：q3_cost_route_progression

主路径：[q3_cost_route_progression.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_cost_route_progression.png>)

备用矢量路径：[q3_cost_route_progression.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_cost_route_progression.svg>)

建议图注：  
图 X 展示了第三问四次突发事件触发后，全天预测总成本和总调度车次的变化过程，并与问题二静态基线进行比较。结果表明，动态扰动持续累积后，为保证约束可行性，后续调度成本和车次数逐步上升。

### 图 31
图名：q3_lateness_cone

主路径：[q3_lateness_cone.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_lateness_cone.png>)

备用矢量路径：[q3_lateness_cone.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_lateness_cone.svg>)

建议图注：  
图 X 展示了第三问各事件处理后剩余迟到分钟与自适应时间影响锥边界 `T0` 的变化。结果表明，随着强扰动事件连续发生，迟到压力积累明显，影响锥边界也逐步向后扩张。

### 图 32
图名：q3_response_structure

主路径：[q3_response_structure.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_response_structure.png>)

备用矢量路径：[q3_response_structure.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_response_structure.svg>)

建议图注：  
图 X 展示了第三问局部重优化的响应结构，包括每次事件触发的 onboard 路线数、depot 路线数、被改动车辆数和被换车单元数。该图用于说明所设计动态策略具备分层局部调整能力。

### 图 33
图名：q1_speed_cost_late

主路径：[q1_speed_cost_late.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_cost_late.png>)

备用矢量路径：[q1_speed_cost_late.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_cost_late.svg>)

建议图注：  
图 X 展示了问题一中速度整体缩放对平均总成本和总迟到分钟的影响。结果表明，总成本对速度扰动并不敏感，而总迟到分钟对速度变化更敏感。

### 图 34
图名：q1_speed_delta_summary

主路径：[q1_speed_delta_summary.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_delta_summary.png>)

备用矢量路径：[q1_speed_delta_summary.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_delta_summary.svg>)

建议图注：  
图 X 展示了各速度场景相对于基准场景的成本变化率和路线数量变化。结果显示，速度扰动对总成本和总车次的影响幅度都较小，说明模型结构具有较好的基础稳定性。

### 图 35
图名：q1_speed_return_pressure

主路径：[q1_speed_return_pressure.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_return_pressure.png>)

备用矢量路径：[q1_speed_return_pressure.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_return_pressure.svg>)

建议图注：  
图 X 展示了速度整体缩放对最晚返仓时刻和超时返仓车次数的影响。结果表明，速度变化对返仓时效边界更为敏感，是鲁棒性分析中最值得关注的时效指标之一。

## 五、插图建议
- 若正文按问题分节组织：
  - 问题一优先用图 2、图 3、图 6、图 11。
  - 问题二优先用图 14、图 17、图 19、图 21。
  - 问题三优先用图 30、图 31、图 32。
  - 敏感性/鲁棒性优先用图 33、图 34、图 35。
- 若版面受限，可把图 24 作为 Q1/Q2 综合总览图，把图 30 作为 Q3 总结图，把图 33 或图 35 作为敏感性分析主图。
