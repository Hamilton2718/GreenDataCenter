#!/usr/bin/env python3
"""
测试 cooling_specialist_node 的输入输出功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nodes.cooling_specialist_node import cooling_specialist_node

def test_cooling_specialist_node():
    """
    测试 cooling_specialist_node 函数
    """
    print("===== 测试 cooling_specialist_node 输入输出 ======")
    
    # 准备测试输入
    test_input = {
        "user_requirements": {
            "location": "乌兰察布",
            "business_type": "training",
            "planned_area": 5000,
            "planned_load": 40000,  # 40 MW
            "computing_power_density": 30,  # 30 kW/机柜
            "priority": "green",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 30000
        },
        "environmental_data": {
            "annual_temperature": 8.2,
            "annual_wind_speed": 5.6,
            "annual_sunshine_hours": 1650,
            "carbon_emission_factor": 0.581,
            "raw_water_usage": 12000.0  # 假设的实时耗水量，单位 L/h
        },
        "electricity_price": {
            "尖峰电价": 1.2,
            "高峰电价": 1.0,
            "平段电价": 0.8,
            "低谷电价": 0.4,
            "深谷电价": 0.3,
            "最大峰谷价差": 0.9
        }
    }
    
    print("\n--- 测试输入 ---\n")
    print("用户需求:")
    print(f"  位置: {test_input['user_requirements']['location']}")
    print(f"  业务类型: {test_input['user_requirements']['business_type']}")
    print(f"  计划面积: {test_input['user_requirements']['planned_area']} m²")
    print(f"  计划负荷: {test_input['user_requirements']['planned_load']} kW")
    print(f"  算力密度: {test_input['user_requirements']['computing_power_density']} kW/机柜")
    print(f"  优先级: {test_input['user_requirements']['priority']}")
    print(f"  绿电目标: {test_input['user_requirements']['green_energy_target']}%")
    print(f"  PUE目标: {test_input['user_requirements']['pue_target']}")
    print(f"  预算约束: {test_input['user_requirements']['budget_constraint']} 万元")
    
    print("\n环境数据:")
    print(f"  年均温度: {test_input['environmental_data']['annual_temperature']} °C")
    print(f"  年均风速: {test_input['environmental_data']['annual_wind_speed']} m/s")
    print(f"  年日照时数: {test_input['environmental_data']['annual_sunshine_hours']} h")
    print(f"  碳排因子: {test_input['environmental_data']['carbon_emission_factor']} kg CO₂/kWh")
    print(f"  耗水量: {test_input['environmental_data']['raw_water_usage']} L/h")
    
    # 调用 cooling_specialist_node 函数
    print("\n--- 执行 cooling_specialist_node ---\n")
    try:
        result = cooling_specialist_node(test_input)
        
        print("\n--- 测试输出 ---\n")
        cooling_plan = result.get("cooling_plan", {})
        
        print("制冷方案:")
        print(f"  推荐技术: {cooling_plan.get('cooling_technology', 'N/A')}")
        print(f"  预计 PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
        print(f"  预计 WUE: {cooling_plan.get('predicted_wue', 'N/A')}")
        
        scheme_brief = cooling_plan.get("scheme_detail_brief", "")
        if scheme_brief:
            print("\n方案摘要:")
            print(scheme_brief)
        
        # 打印详细计算参数
        cooling_calc_params = cooling_plan.get("cooling_calc_params", {})
        if cooling_calc_params:
            print("\n详细计算参数:")
            for key, value in cooling_calc_params.items():
                print(f"  {key}: {value}")
        
        # 打印KPI数据
        cooling_kpis = cooling_plan.get("cooling_kpis", {})
        if cooling_kpis:
            print("\nKPI数据:")
            for key, value in cooling_kpis.items():
                print(f"  {key}: {value}")
        
        print("\n✅ 测试成功完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cooling_specialist_node()
