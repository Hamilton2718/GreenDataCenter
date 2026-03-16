#!/usr/bin/env python3
"""
测试脚本：测试requirement_analysis_node和energy_planner_node的协作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入两个节点函数
from nodes.requirement_analysis_node import requirement_analysis_node
from nodes.energy_planner_node import energy_planner_node

if __name__ == "__main__":
    print("===== 测试两个Agent的协作 =====")
    print("\n1. 准备测试数据...")
    
    # 模拟用户输入
    test_state = {
        "user_requirements": {
            "location": "广州",
            "business_type": "通用",
            "planned_area": 10000,
            "planned_load": 5000,
            "Computing_power_density": 8,
            "priority": "环保型",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 10000
        }
    }
    
    print("\n2. 调用Agent 1 (需求与约束解析专家)...")
    # 调用requirement_analysis_node获取环境数据和电价数据
    agent1_result = requirement_analysis_node(test_state)
    
    print("\n3. 检查Agent 1的输出...")
    print(f"   - 位置: {agent1_result.get('user_requirements', {}).get('location', '未知')}")
    print(f"   - 环境数据: {agent1_result.get('environmental_data', {}).keys()}")
    print(f"   - 电价数据: {agent1_result.get('electricity_price', {}).keys()}")
    
    print("\n4. 调用Agent 2 (能源与绿电规划专家)...")
    # 将Agent 1的输出传递给energy_planner_node
    agent2_result = energy_planner_node(agent1_result)
    
    print("\n5. 检查Agent 2的输出...")
    energy_plan = agent2_result.get('energy_plan', {})
    print(f"   - 能源规划方案: {energy_plan.keys()}")
    print(f"   - 是否包含LLM报告: {'llm_report' in energy_plan}")
    
    if 'llm_report' in energy_plan:
        print("\n6. 查看LLM生成的Markdown报告...")
        print("=" * 80)
        print(energy_plan['llm_report'])
        print("=" * 80)
    
    print("\n✅ 测试完成！")
