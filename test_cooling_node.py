#!/usr/bin/env python3
"""
测试 cooling_specialist_node 的输入输出功能
"""

import sys
import json
from nodes.cooling_specialist_node import cooling_specialist_node

# 测试用例 1: 乌兰察布 - 高算力密度
test_case_1 = {
    "user_requirements": {
        "location": "乌兰察布",
        "business_type": "training",
        "planned_area": 5000,
        "planned_load": 40000,
        "computing_power_density": 30,
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
        "raw_water_usage": 12000.0
    },
    "electricity_price": {
        "peak_price": 1.2,
        "high_price": 1.0,
        "flat_price": 0.8,
        "low_price": 0.4,
        "deep_low_price": 0.3,
        "max_price_diff": 0.9
    }
}

# 测试用例 2: 北京 - 中等算力密度
test_case_2 = {
    "user_requirements": {
        "location": "北京",
        "business_type": "storage",
        "planned_area": 3000,
        "planned_load": 15000,
        "computing_power_density": 15,
        "priority": "reliable",
        "green_energy_target": 70,
        "pue_target": 1.3,
        "budget_constraint": 20000
    },
    "environmental_data": {
        "annual_temperature": 12.3,
        "annual_wind_speed": 3.2,
        "annual_sunshine_hours": 2000,
        "carbon_emission_factor": 0.581,
        "raw_water_usage": 8000.0
    },
    "electricity_price": {
        "peak_price": 1.5,
        "high_price": 1.2,
        "flat_price": 0.9,
        "low_price": 0.5,
        "deep_low_price": 0.3,
        "max_price_diff": 1.2
    }
}

# 测试用例 3: 广州 - 低算力密度
test_case_3 = {
    "user_requirements": {
        "location": "广州",
        "business_type": "edge",
        "planned_area": 1000,
        "planned_load": 5000,
        "computing_power_density": 8,
        "priority": "economic",
        "green_energy_target": 50,
        "pue_target": 1.4,
        "budget_constraint": 10000
    },
    "environmental_data": {
        "annual_temperature": 22.5,
        "annual_wind_speed": 2.8,
        "annual_sunshine_hours": 1800,
        "carbon_emission_factor": 0.581,
        "raw_water_usage": 4000.0
    },
    "electricity_price": {
        "peak_price": 1.3,
        "high_price": 1.1,
        "flat_price": 0.8,
        "low_price": 0.4,
        "deep_low_price": 0.2,
        "max_price_diff": 1.1
    }
}

def run_test(test_case, test_name):
    """运行单个测试用例"""
    print(f"\n{'='*80}")
    print(f"测试用例: {test_name}")
    print(f"{'='*80}")
    
    try:
        # 调用 cooling_specialist_node
        result = cooling_specialist_node(test_case)
        
        # 提取制冷方案
        cooling_plan = result.get("cooling_plan", {})
        
        # 打印输出结果
        print("\n输出结果:")
        print("1. 方案摘要:")
        print(cooling_plan.get("scheme_detail_brief", "无"))
        print("\n2. 核心指标:")
        print(f"   - 推荐技术: {cooling_plan.get('cooling_technology', 'N/A')}")
        print(f"   - 预计 PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
        print(f"   - 预计 WUE: {cooling_plan.get('predicted_wue', 'N/A')}")
        print("\n3. 项目信息:")
        print(json.dumps(cooling_plan.get("cooling_project_info", {}), indent=4, ensure_ascii=False))
        print("\n4. 计算参数:")
        print(json.dumps(cooling_plan.get("cooling_calc_params", {}), indent=4, ensure_ascii=False))
        print("\n5. KPI数据:")
        print(json.dumps(cooling_plan.get("cooling_kpis", {}), indent=4, ensure_ascii=False))
        
        print(f"\n{'='*80}")
        print(f"测试用例 {test_name} 执行成功!")
        print(f"{'='*80}")
        
        return True
    except Exception as e:
        print(f"\n测试用例 {test_name} 执行失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试 cooling_specialist_node 输入输出功能")
    
    # 运行所有测试用例
    test_results = [
        run_test(test_case_1, "乌兰察布 - 高算力密度"),
        run_test(test_case_2, "北京 - 中等算力密度"),
        run_test(test_case_3, "广州 - 低算力密度")
    ]
    
    # 统计测试结果
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n{'='*80}")
    print(f"测试完成: {passed}/{total} 个测试用例通过")
    print(f"{'='*80}")
    
    if passed == total:
        print("所有测试用例执行成功!")
        sys.exit(0)
    else:
        print("部分测试用例执行失败!")
        sys.exit(1)
