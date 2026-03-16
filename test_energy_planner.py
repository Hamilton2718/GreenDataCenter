#!/usr/bin/env python3
"""
测试energy_planner_node.py的输出
"""

from nodes.energy_planner_node import energy_planner_node

# 模拟Agent 1的输出作为测试输入
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

print("===== 测试energy_planner_node =====")
print(f"输入位置: {test_state['user_requirements']['location']}")
print(f"输入温度: {test_state['environmental_data']['annual_temperature']}°C")
print(f"输入绿电目标: {test_state['user_requirements']['green_energy_target']}%")

# 调用energy_planner_node
result = energy_planner_node(test_state)

# 提取能源规划方案
energy_plan = result.get("energy_plan", {})

print("\n=== 输出结果 ===")
print(f"能源规划方案包含的字段: {list(energy_plan.keys())}")
print(f"\n1. LLM报告:")
print(energy_plan.get("llm_report", "无"))
print(f"\n2. 光伏装机容量: {energy_plan.get('pv_capacity', 0.0)} kW")
print(f"3. 风电装机容量: {energy_plan.get('wind_capacity', 0.0)} kW")
print(f"4. 储能容量: {energy_plan.get('storage_capacity', 0.0)} kWh")
print(f"5. 储能功率: {energy_plan.get('storage_power', 0.0)} kW")
print(f"6. 绿电长协比例: {energy_plan.get('ppa_ratio', 0.0)}%")
print(f"7. 电网调峰比例: {energy_plan.get('grid_ratio', 0.0)}%")
print(f"8. 预计自发自用率: {energy_plan.get('estimated_self_consumption', 0.0)}%")
print(f"9. 预计绿电占比: {energy_plan.get('estimated_green_ratio', 0.0)}%")

print("\n✅ 测试完成！")
