"""
Agent 4: 虚拟运行仿真专家 (Simulator)

功能:
    - 这是最关键的一步。它拿到Agent 2和3的方案，模拟该设计在未来的典型24小时（或全年）的运行情况
    - 验证：在模拟中，如果发现绿电消纳不足或PUE超标，它会向之前的Agent发出"打回修改"的信号
    - 输出：虚拟运行性能报告

输入状态:
    - user_requirements: 用户需求
    - environmental_data: 环境数据
    - energy_plan: 能源规划方案
    - cooling_plan: 制冷方案

输出状态:
    - simulation_result: 仿真结果
    - iteration_count: 更新迭代计数器

作者: GreenDataCenter Team
版本: 1.0.0
"""

from typing import TYPE_CHECKING, List, Dict, Any
import math

if TYPE_CHECKING:
    from graph import GreenDataCenterState


def simulator_node(state: dict) -> dict:
    """
    Agent 4: 虚拟运行仿真专家 - LangGraph Node
    
    模拟数据中心24小时/全年运行情况，验证绿电消纳和PUE是否达标。
    这是LangGraph工作流的第四个节点，也是关键决策点。
    
    参数:
        state: GreenDataCenterState类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - energy_plan: 能源规划方案
            - cooling_plan: 制冷方案
            - iteration_count: 当前迭代次数
            
    返回:
        更新后的state，新增:
            - simulation_result: 仿真结果（含validation_passed字段）
            - iteration_count: 迭代计数器+1
    """
    print("\n" + "="*60)
    print("🔬 [Agent 4: 虚拟运行仿真专家] 开始工作")
    print("="*60)
    
    # 获取输入数据
    user_req = state.get('user_requirements', {})
    env_data = state.get('environmental_data', {})
    energy_plan = state.get('energy_plan', {})
    cooling_plan = state.get('cooling_plan', {})
    iteration_count = state.get('iteration_count', 0)
    
    # 提取关键参数
    green_target = user_req.get('green_energy_target', 90)
    pue_target = user_req.get('pue_target', 1.2)
    planned_load = user_req.get('planned_load', 0)
    
    # 提取方案参数
    pv_capacity = energy_plan.get('pv_capacity', 0)
    storage_capacity = energy_plan.get('storage_capacity', 0)
    estimated_pue = cooling_plan.get('estimated_pue', 1.4)
    
    print(f"📊 仿真输入:")
    print(f"  - 绿电目标: {green_target}%")
    print(f"  - PUE目标: {pue_target}")
    print(f"  - 光伏装机: {pv_capacity} kW")
    print(f"  - 储能容量: {storage_capacity} kWh")
    print(f"  - 预计PUE: {estimated_pue}")
    
    # ===== 执行24小时仿真 =====
    hourly_balance = _simulate_24h_operation(
        planned_load=planned_load,
        pv_capacity=pv_capacity,
        storage_capacity=storage_capacity,
        energy_plan=energy_plan,
        env_data=env_data
    )
    
    # ===== 计算年度指标 =====
    annual_metrics = _calculate_annual_metrics(
        hourly_balance=hourly_balance,
        planned_load=planned_load,
        estimated_pue=estimated_pue,
        energy_plan=energy_plan,
        env_data=env_data
    )
    
    # ===== 验证是否达标 =====
    validation_passed, validation_issues = _validate_design(
        annual_metrics=annual_metrics,
        green_target=green_target,
        pue_target=pue_target,
        estimated_pue=estimated_pue
    )
    
    # 更新迭代计数器
    new_iteration_count = iteration_count + 1
    
    simulation_result = {
        "hourly_power_balance": hourly_balance,
        "annual_green_consumption": round(annual_metrics['green_consumption'], 2),
        "actual_green_ratio": round(annual_metrics['green_ratio'], 2),
        "actual_pue": round(annual_metrics['actual_pue'], 3),
        "carbon_reduction": round(annual_metrics['carbon_reduction'], 2),
        "validation_passed": validation_passed,
        "validation_issues": validation_issues
    }
    
    print(f"\n📈 仿真结果:")
    print(f"  - 年绿电消纳量: {simulation_result['annual_green_consumption']:,} kWh")
    print(f"  - 实际绿电占比: {simulation_result['actual_green_ratio']:.1f}%")
    print(f"  - 实际PUE: {simulation_result['actual_pue']:.3f}")
    print(f"  - 年碳减排量: {simulation_result['carbon_reduction']:.1f} 吨CO₂")
    
    if validation_passed:
        print(f"\n✅ 验证通过: 设计满足所有约束条件")
    else:
        print(f"\n❌ 验证未通过:")
        for issue in validation_issues:
            print(f"    - {issue}")
        print(f"\n🔄 将返回Agent 1重新调整参数（第{new_iteration_count}次迭代）")
    
    print("\n" + "="*60)
    print("✅ [Agent 4: 虚拟运行仿真专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态
    return {
        **state,
        "simulation_result": simulation_result,
        "iteration_count": new_iteration_count
    }


def _simulate_24h_operation(
    planned_load: float,
    pv_capacity: float,
    storage_capacity: float,
    energy_plan: dict,
    env_data: dict
) -> List[Dict[str, Any]]:
    """
    模拟24小时电力平衡
    
    参数:
        planned_load: 计划负荷（kW）
        pv_capacity: 光伏装机容量（kW）
        storage_capacity: 储能容量（kWh）
        energy_plan: 能源规划方案
        env_data: 环境数据
        
    返回:
        24小时电力平衡数据列表
    """
    hourly_balance = []
    storage_soc = storage_capacity * 0.5  # 初始SOC为50%
    
    # 简化模型：假设夏季某天的负荷和光伏曲线
    for hour in range(24):
        # 负荷曲线（白天高、夜间低）
        if 8 <= hour <= 20:
            load = planned_load * (0.8 + 0.2 * math.sin((hour - 8) * math.pi / 12))
        else:
            load = planned_load * 0.6
        
        # 光伏出力曲线（6-18点有出力）
        if 6 <= hour <= 18:
            pv_output = pv_capacity * max(0, math.sin((hour - 6) * math.pi / 12))
        else:
            pv_output = 0
        
        # 风电出力（简化：夜间较高）
        wind_capacity = energy_plan.get('wind_capacity', 0)
        if 20 <= hour <= 6:
            wind_output = wind_capacity * 0.7
        else:
            wind_output = wind_capacity * 0.4
        
        # 可再生能源总出力
        renewable_output = pv_output + wind_output
        
        # 计算电力平衡
        net_power = renewable_output - load
        
        # 储能充放电
        if net_power > 0:  # 可再生能源盈余，充电
            charge_power = min(net_power, storage_capacity * 0.25)  # 最大0.25C充电
            charge_power = min(charge_power, (storage_capacity - storage_soc))
            storage_soc += charge_power
            grid_import = 0
            curtailment = net_power - charge_power
        else:  # 可再生能源不足，放电或购电
            discharge_needed = -net_power
            discharge_power = min(discharge_needed, storage_capacity * 0.25)  # 最大0.25C放电
            discharge_power = min(discharge_power, storage_soc)
            storage_soc -= discharge_power
            grid_import = discharge_needed - discharge_power
            curtailment = 0
        
        # 计算绿电使用情况
        green_used = min(renewable_output, load) + (charge_power if net_power > 0 else -discharge_power if net_power < 0 else 0)
        
        hourly_balance.append({
            "hour": hour,
            "load": round(load, 2),
            "pv_output": round(pv_output, 2),
            "wind_output": round(wind_output, 2),
            "renewable_output": round(renewable_output, 2),
            "storage_soc": round(storage_soc, 2),
            "grid_import": round(grid_import, 2),
            "curtailment": round(curtailment, 2),
            "green_used": round(green_used, 2)
        })
    
    return hourly_balance


def _calculate_annual_metrics(
    hourly_balance: List[Dict],
    planned_load: float,
    estimated_pue: float,
    energy_plan: dict,
    env_data: dict
) -> Dict[str, float]:
    """
    计算年度运行指标
    
    参数:
        hourly_balance: 24小时电力平衡数据
        planned_load: 计划负荷
        estimated_pue: 预计PUE
        energy_plan: 能源规划方案
        env_data: 环境数据
        
    返回:
        年度指标字典
    """
    # 从24小时数据推算全年（简化计算）
    daily_green = sum(h['green_used'] for h in hourly_balance)
    daily_load = sum(h['load'] for h in hourly_balance)
    daily_grid = sum(h['grid_import'] for h in hourly_balance)
    
    # 考虑PUE的能耗
    it_load_annual = planned_load * 0.75 * 8760  # IT设备年用电量
    total_load_annual = it_load_annual * estimated_pue  # 总用电量
    
    # PPA绿电（长协）
    ppa_ratio = energy_plan.get('ppa_ratio', 0)
    ppa_green = total_load_annual * (ppa_ratio / 100)
    
    # 分布式绿电（光伏+风电+储能）
    distributed_green = daily_green * 365
    
    # 总绿电消纳
    total_green = min(distributed_green + ppa_green, total_load_annual)
    green_ratio = (total_green / total_load_annual) * 100 if total_load_annual > 0 else 0
    
    # 碳减排计算
    carbon_factor = env_data.get('carbon_emission_factor', 0.5)
    carbon_reduction = total_green * carbon_factor / 1000  # 吨CO₂
    
    return {
        'green_consumption': total_green,
        'green_ratio': green_ratio,
        'actual_pue': estimated_pue,
        'carbon_reduction': carbon_reduction,
        'total_consumption': total_load_annual
    }


def _validate_design(
    annual_metrics: Dict[str, float],
    green_target: float,
    pue_target: float,
    estimated_pue: float
) -> tuple:
    """
    验证设计是否满足约束条件
    
    参数:
        annual_metrics: 年度指标
        green_target: 绿电目标（%）
        pue_target: PUE目标
        estimated_pue: 预计PUE
        
    返回:
        (验证是否通过, 问题列表)
    """
    issues = []
    
    # 验证绿电占比
    actual_green_ratio = annual_metrics['green_ratio']
    if actual_green_ratio < green_target:
        issues.append(f"绿电占比不足: 实际{actual_green_ratio:.1f}% < 目标{green_target}%")
    
    # 验证PUE
    if estimated_pue > pue_target * 1.05:  # 允许5%的误差
        issues.append(f"PUE超标: 预计{estimated_pue:.3f} > 目标{pue_target}")
    
    # 验证是否满足最低要求
    if actual_green_ratio < 50:
        issues.append("绿电占比过低，建议重新设计能源方案")
    
    return len(issues) == 0, issues


# 保持向后兼容的别名
agent4_node = simulator_node


# --- 主程序入口（用于独立测试） ---
if __name__ == "__main__":
    print("===== 测试 Agent 4: 虚拟运行仿真专家 =====")
    
    # 创建测试状态
    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "planned_load": 5000,
            "green_energy_target": 90,
            "pue_target": 1.2
        },
        "environmental_data": {
            "annual_temperature": 5.5,
            "carbon_emission_factor": 0.6479
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "wind_capacity": 500,
            "storage_capacity": 6000,
            "ppa_ratio": 20
        },
        "cooling_plan": {
            "estimated_pue": 1.18
        },
        "iteration_count": 0
    }
    
    # 执行节点
    result = simulator_node(test_state)
    
    print("\n===== 测试结果 =====")
    print(f"仿真结果: {result.get('simulation_result')}")
    print(f"迭代计数: {result.get('iteration_count')}")
