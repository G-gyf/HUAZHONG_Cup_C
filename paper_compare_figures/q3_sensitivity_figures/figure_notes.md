# Paper Figure Notes

## Q3 Dynamic Scheduling Figures
1. `q3_cost_route_progression`: 插入第三问结果分析部分，用于说明四次事件触发后，全天预测总成本和总调度车次如何逐步上升，并与 Q2 静态基线比较。
2. `q3_response_structure`: 插入第三问方法效果分析部分，用于说明各事件触发的 onboard/depot 局部重优化负载，以及实际被改动车辆和换车单元数量。
3. `q3_lateness_cone`: 插入第三问机制解释部分，用于说明事件推进后剩余迟到压力的累积，以及自适应影响锥边界 T0 的扩张位置。

## Q1 Sensitivity / Robustness Figures
4. `q1_speed_cost_late`: 插入敏感性分析部分，用于同时展示整体速度缩放对总成本和总迟到分钟的影响。
5. `q1_speed_return_pressure`: 插入鲁棒性分析部分，用于展示速度变化对最晚返仓时刻和超时返仓车次数的影响。
6. `q1_speed_delta_summary`: 插入结论补充部分，用于强调速度变化对成本和车次数的相对影响幅度，突出“成本稳、时效更敏感”的结论。

## Usage Note
- 所有图均同时导出为 PNG 和 SVG，可直接插入论文；若正文需要中文图题，建议在论文软件中直接改为中文题注，保留图内英文坐标即可。