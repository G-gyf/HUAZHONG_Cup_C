# 第三问与敏感性分析图注报告

本报告整理 [q3_sensitivity_figures](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures>) 目录下所有成品图，并给出每张图的建议图注、适用位置和文件路径。正文插图建议优先使用 `PNG`，排版或放大需求较高时可改用 `SVG`。

## 图 1
图名：Q3 Dynamic Response: Full-Day Cost and Route Count by Event

主路径：[q3_cost_route_progression.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_cost_route_progression.png>)

矢量路径：[q3_cost_route_progression.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_cost_route_progression.svg>)

建议图注：  
图 X 展示了第三问四次突发事件触发后，全天预测总成本与总调度车次的变化过程，并与 Q2 静态基线进行对比。结果表明，随着新增订单、取消订单、时间窗收紧和改址事件依次发生，动态调度方案为保持约束可行性而逐步增加后续调度成本与车次，其中 E3 和 E4 之后的增幅最为明显。

建议用途：  
放在第三问“结果分析”部分，用于说明动态响应的总代价变化。

## 图 2
图名：Q3 Dynamic Response: Local Re-optimization Load by Event

主路径：[q3_response_structure.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_response_structure.png>)

矢量路径：[q3_response_structure.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_response_structure.svg>)

建议图注：  
图 X 展示了四类事件触发后局部重优化的响应结构，包括直接受影响的 onboard 路线数、depot 路线数、被改动车辆数以及被换车的仓端单元数。可以看出，E3 时间窗收紧和 E4 改址事件同时触发了车载后缀调整与仓端重编，说明所设计的动态机制能够根据事件类型进行分层响应。

建议用途：  
放在第三问“策略机制说明”或“事件响应分析”部分，用于证明方法不是简单全局重算，而是按 onboard/depot 两层局部调整。

## 图 3
图名：Q3 Dynamic Response: Lateness Pressure and Adaptive Cone Boundary

主路径：[q3_lateness_cone.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_lateness_cone.png>)

矢量路径：[q3_lateness_cone.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q3_lateness_cone.svg>)

建议图注：  
图 X 展示了第三问各事件处理后剩余计划的总迟到分钟以及自适应时间影响锥边界 `T0` 的变化情况。结果表明，随着事件复杂度上升，剩余迟到压力持续累积，而影响锥边界也逐步向后扩张，反映出局部重优化需要纳入更晚时段的未发车路线以维持可行性。

建议用途：  
放在第三问“自适应时间影响锥”说明部分，用于解释局部锥扩张逻辑和迟到积累现象。

## 图 4
图名：Q1 Speed Sensitivity: Total Cost
与
Q1 Speed Sensitivity: Total Late Minutes

主路径：[q1_speed_cost_late.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_cost_late.png>)

矢量路径：[q1_speed_cost_late.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_cost_late.svg>)

建议图注：  
图 X 展示了问题一中全天速度整体缩放对平均总成本和总迟到分钟的影响。结果表明，在 `0.90` 至 `1.10` 的缩放范围内，总成本仅发生小幅波动，而总迟到分钟变化更明显，说明模型对速度扰动在成本层面较为稳定，但在服务时效层面更为敏感。

建议用途：  
放在“敏感性分析”部分，作为速度参数整体缩放的主图。

## 图 5
图名：Q1 Speed Sensitivity: Return-Time Pressure

主路径：[q1_speed_return_pressure.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_return_pressure.png>)

矢量路径：[q1_speed_return_pressure.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_return_pressure.svg>)

建议图注：  
图 X 展示了速度整体缩放对最晚返仓时刻和超时返仓车次数的影响。随着整体速度下降，最晚返仓时刻明显后移，超时返仓车次数略有增加；而速度提高时，这两项时效压力指标有所改善，说明速度扰动主要通过返仓边界传导到调度结果中。

建议用途：  
放在“基础鲁棒性检验”部分，用于说明模型对速度变化的时效响应。

## 图 6
图名：Q1 Speed Sensitivity: Cost Change vs Baseline
与
Q1 Speed Sensitivity: Route Count Change vs Baseline

主路径：[q1_speed_delta_summary.png](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_delta_summary.png>)

矢量路径：[q1_speed_delta_summary.svg](<C:\Desktop\华中杯A题\paper_compare_figures\q3_sensitivity_figures\q1_speed_delta_summary.svg>)

建议图注：  
图 X 展示了各速度场景相对于基准场景的成本变化率和车次变化。结果显示，速度整体扰动对总成本的影响始终控制在较小范围内，且总车次基本保持不变，说明该模型在中等速度扰动下具有较好的结构稳定性，成本层面的基础鲁棒性较强。

建议用途：  
放在“敏感性分析结论”或“鲁棒性总结”部分，用于概括“成本稳、车次稳、时效更敏感”的核心结论。

## 汇总建议
- 第三问正文建议使用图 1、图 2、图 3。
- 敏感性分析正文建议使用图 4、图 5、图 6。
- 若论文版面有限，第三问优先保留图 1 和图 2，敏感性分析优先保留图 4 和图 5。
