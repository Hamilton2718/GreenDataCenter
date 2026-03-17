#!/usr/bin/env python3
"""
测试 cooling_specialist_node 的核心计算功能（简化版，不依赖RAG和LLM）
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nodes.cooling_specialist_node import CoolingAgent3

def test_cooling_agent():
    """
    测试 CoolingAgent3 的核心计算功能
    """
    print("===== 测试 cooling_specialist_node 核心计算功能 ======")
    
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
    
    print("\n--- 测试输入 ---")
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
    
    # 创建制冷 Agent
    print("\n--- 初始化 CoolingAgent3 ---")
    cooling_agent = CoolingAgent3()
    
    # 构建 project_info
    user_reqs = test_input.get("user_requirements", {})
    env_data = test_input.get("environmental_data", {})
    
    project_info = {
        "location": user_reqs.get("location", "北京"),
        "business_type": user_reqs.get("business_type", "通用"),
        "planned_area": user_reqs.get("planned_area", 10000),
        "planned_load": user_reqs.get("planned_load", 5000),
        "computing_power_density": user_reqs.get("computing_power_density", 8),
        "priority": user_reqs.get("priority", "环保型"),
        "green_energy_target": user_reqs.get("green_energy_target", 90),
        "pue_target": user_reqs.get("pue_target", 1.2),
        "budget_constraint": user_reqs.get("budget_constraint", 10000)
    }
    
    region = project_info.get("location", "北京")
    province = "内蒙古"  # 乌兰察布属于内蒙古
    annual_temp = env_data.get("annual_temperature", 15.0)
    cabinet_power = project_info.get("computing_power_density", 8.0)
    
    print(f"\n--- 核心参数 ---")
    print(f"  区域: {region}")
    print(f"  省份: {province}")
    print(f"  年均温度: {annual_temp}°C")
    print(f"  算力密度: {cabinet_power} kW/机柜")
    
    # 测试 COP 修正因子计算
    cop_correction = cooling_agent.get_cop_correction_factor(annual_temp)
    print(f"\n--- COP 修正因子 ---")
    print(f"  修正因子: {cop_correction}")
    
    # 测试提取制冷参数（使用兜底参数）
    print("\n--- 提取制冷参数 ---")
    context = ""  # 空上下文，触发兜底参数
    extracted_params = cooling_agent.extract_cooling_params(context, project_info, province)
    print(f"  提取的参数: {extracted_params}")
    
    # 测试计算冷却 KPIs
    print("\n--- 计算冷却 KPIs ---")
    kpis = cooling_agent.calculate_cooling_kpis(extracted_params, project_info, env_data)
    print(f"  计算结果: {kpis}")
    
    # 构建简化的制冷方案
    cooling_plan = {
        "scheme_detail_brief": "基于算力密度和环境条件，推荐液冷技术方案",
        "cooling_technology": extracted_params.get("regional_cooling_preference", "风冷"),
        "estimated_pue": kpis.get("predicted_PUE", 1.3),
        "predicted_wue": kpis.get("predicted_WUE", 1.6),
        "cooling_project_info": {
            "location": region,
            "province": province,
            "it_load_kW": project_info.get("planned_load"),
            "cabinet_power_kW": cabinet_power,
            "target_pue": project_info.get("pue_target"),
            "green_energy_target": project_info.get("green_energy_target")
        },
        "cooling_calc_params": extracted_params,
        "cooling_kpis": kpis
    }
    
    print("\n--- 制冷方案 ---")
    print(f"  推荐技术: {cooling_plan.get('cooling_technology', 'N/A')}")
    print(f"  预计 PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
    print(f"  预计 WUE: {cooling_plan.get('predicted_wue', 'N/A')}")
    print(f"  方案摘要: {cooling_plan.get('scheme_detail_brief', 'N/A')}")
    
    print("\n✅ 测试成功完成！")

if __name__ == "__main__":
    test_cooling_agent()
