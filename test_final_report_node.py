#!/usr/bin/env python3
"""
测试最终报告生成节点 (final_report_node)
"""

from nodes.final_report_node import final_report_node

# 模拟输入状态
test_state = {
    "user_requirements": {
        "location": "乌兰察布",
        "business_type": "training",
        "planned_load": 40000,
        "computing_power_density": 30,
        "priority": "高",
        "green_energy_target": 90,
        "pue_target": 1.2
    },
    "environmental_data": {
        "annual_temperature": 3.61,
        "annual_wind_speed": 2.99,
        "annual_sunshine_hours": 3903.85,
        "carbon_emission_factor": 0.6479
    },
    "energy_plan": {
        "pv_capacity": 10000,
        "storage_capacity": 5000,
        "storage_power": 2500,
        "ppa_ratio": 40,
        "grid_ratio": 15
    },
    "cooling_plan": {
        "cooling_technology": "间接蒸发冷却+液冷机柜",
        "estimated_pue": 1.18
    },
    "simulation_result": {
        "summary": {
            "daily_it_energy_mwh": 960,
            "daily_green_supply_mwh": 880,
            "daily_green_ratio_pct": 91.67,
            "daily_storage_charge_mwh": 200,
            "daily_storage_discharge_mwh": 180,
            "method": "时序仿真"
        }
    },
    "financial_analysis": {
        "payback_years": 6.5,
        "irr": 12.5,
        "npv": 15000,
        "lcoe": 0.45
    }
}

# 测试 final_report_node 函数
print("测试 final_report_node 函数...")
result = final_report_node(test_state)

# 打印输出结果
print("\n输出状态包含的键:", list(result.keys()))
print("\n生成的最终报告:")
print(result.get("final_report", "无报告生成"))

print("\n测试完成！")
