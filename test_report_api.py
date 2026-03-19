#!/usr/bin/env python3
"""
测试最终报告生成API
"""

import requests
import json

# 测试数据
test_data = {
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

# 发送请求
url = "http://localhost:5001/api/generate-report"
headers = {"Content-Type": "application/json"}

print("测试最终报告生成API...")
try:
    response = requests.post(url, json=test_data, headers=headers)
    print(f"响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"成功: {result.get('success')}")
        if result.get('success'):
            final_report = result.get('data', {}).get('final_report', '')
            print(f"生成的报告长度: {len(final_report)} 字符")
            print("\n报告内容:")
            print(final_report)
        else:
            print(f"错误: {result.get('error')}")
    else:
        print(f"错误: {response.text}")
except Exception as e:
    print(f"请求失败: {str(e)}")
