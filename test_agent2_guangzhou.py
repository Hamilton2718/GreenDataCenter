#!/usr/bin/env python3
"""
测试Agent 2的能源规划功能，输入位置为广州
"""

import json
from nodes.energy_planner_node import energy_planner_node

# 模拟Agent 1的输出，位置设置为广州
test_state = {
    "user_requirements": {
        "location": "广州",
        "business_type": "通用",
        "planned_area": 10000,
        "planned_load": 5000,
        "computing_power_density": 8,
        "priority": "环保型",
        "green_energy_target": 90,
        "pue_target": 1.2,
        "budget_constraint": 10000
    },
    "environmental_data": {
        "annual_temperature": 22.0,
        "annual_wind_speed": 1.8,
        "annual_sunshine_hours": 1800,
        "carbon_emission_factor": 0.52,
        "latitude": 23.13,
        "longitude": 113.26,
        "province": "广东"
    },
    "electricity_price": {
        "尖峰电价": 0.98,
        "高峰电价": 0.85,
        "平段电价": 0.62,
        "低谷电价": 0.35,
        "深谷电价": 0.28,
        "最大峰谷价差": 0.70
    }
}

print("===== 测试Agent 2 (广州) =====")
print(f"输入位置: {test_state['user_requirements']['location']}")
print(f"输入温度: {test_state['environmental_data']['annual_temperature']}°C")

# 调用energy_planner_node
result = energy_planner_node(test_state)

# 提取能源规划方案
energy_plan = result.get("energy_plan", {})
llm_report = energy_plan.get("llm_report", "")

print("\n=== 生成的能源规划报告 ===")
print(llm_report)

# 检查报告中是否包含广州
if "广州" in llm_report:
    print("\n✅ 测试成功！报告中正确显示了广州")
else:
    print("\n❌ 测试失败！报告中没有显示广州")
