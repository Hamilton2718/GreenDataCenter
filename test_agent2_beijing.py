#!/usr/bin/env python3
"""
测试Agent 2的能源规划功能，输入位置为北京
"""

import json
from nodes.energy_planner_node import energy_planner_node

# 模拟Agent 1的输出，位置设置为北京
test_state = {
    "user_requirements": {
        "location": "北京",
        "business_type": "通用",
        "planned_area": 10000,
        "planned_load": 5000,
        "computing_power_density": 8,
        "priority": "环保型",
        "green_energy_target": 85,
        "pue_target": 1.25,
        "budget_constraint": 10000
    },
    "environmental_data": {
        "annual_temperature": 12.0,
        "annual_wind_speed": 3.0,
        "annual_sunshine_hours": 2600,
        "carbon_emission_factor": 0.48,
        "latitude": 39.90,
        "longitude": 116.40,
        "province": "北京"
    },
    "electricity_price": {
        "尖峰电价": 1.05,
        "高峰电价": 0.92,
        "平段电价": 0.68,
        "低谷电价": 0.38,
        "深谷电价": 0.30,
        "最大峰谷价差": 0.75
    }
}

print("===== 测试Agent 2 (北京) =====")
print(f"输入位置: {test_state['user_requirements']['location']}")
print(f"输入温度: {test_state['environmental_data']['annual_temperature']}°C")

# 调用energy_planner_node
result = energy_planner_node(test_state)

# 提取能源规划方案
energy_plan = result.get("energy_plan", {})
llm_report = energy_plan.get("llm_report", "")

print("\n=== 生成的能源规划报告 ===")
print(llm_report)

# 检查报告中是否包含北京
if "北京" in llm_report:
    print("\n✅ 测试成功！报告中正确显示了北京")
else:
    print("\n❌ 测试失败！报告中没有显示北京")
