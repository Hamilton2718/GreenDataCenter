"""
Agent 5: 综合评价与投资决策专家 (Financial Consultant)

功能:
    - 计算CAPEX（建设成本）（地价等）和OPEX（运行成本）
    - 计算碳减排贡献
    - 输出：最终的《数据中心绿电消纳规划设计建议书》

输入状态:
    - user_requirements: 用户需求
    - environmental_data: 环境数据
    - electricity_price: 电价数据
    - energy_plan: 能源规划方案
    - cooling_plan: 制冷方案
    - simulation_result: 仿真结果

输出状态:
    - financial_analysis: 财务分析
    - final_report: 最终报告（Markdown格式）

作者: GreenDataCenter Team
版本: 1.0.0
"""

from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from graph import GreenDataCenterState


def financial_consultant_node(state: dict) -> dict:
    """
    Agent 5: 综合评价与投资决策专家 - LangGraph Node
    
    计算投资成本、运营成本和财务指标，生成最终规划设计建议书。
    这是LangGraph工作流的最后一个节点。
    
    参数:
        state: GreenDataCenterState类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - electricity_price: 电价数据
            - energy_plan: 能源规划方案
            - cooling_plan: 制冷方案
            - simulation_result: 仿真结果
            
    返回:
        更新后的state，新增:
            - financial_analysis: 财务分析
            - final_report: 最终报告（Markdown格式）
    """
    print("\n" + "="*60)
    print("💰 [Agent 5: 综合评价与投资决策专家] 开始工作")
    print("="*60)
    
    # 获取输入数据
    user_req = state.get('user_requirements', {})
    env_data = state.get('environmental_data', {})
    electricity_price = state.get('electricity_price', {})
    energy_plan = state.get('energy_plan', {})
    cooling_plan = state.get('cooling_plan', {})
    simulation_result = state.get('simulation_result', {})
    
    # 提取关键参数
    planned_area = user_req.get('planned_area', 0)
    planned_load = user_req.get('planned_load', 0)
    computing_density = user_req.get('computing_power_density', 
                                     user_req.get('Computing_power_density', 8))
    
    # 提取方案参数
    pv_capacity = energy_plan.get('pv_capacity', 0)
    wind_capacity = energy_plan.get('wind_capacity', 0)
    storage_capacity = energy_plan.get('storage_capacity', 0)
    
    print(f"📊 财务分析输入:")
    print(f"  - 计划面积: {planned_area} 平方米")
    print(f"  - 计划负荷: {planned_load} kW")
    print(f"  - 光伏装机: {pv_capacity} kW")
    print(f"  - 储能容量: {storage_capacity} kWh")
    
    # ===== 计算CAPEX（建设成本） =====
    capex = _calculate_capex(
        planned_area=planned_area,
        planned_load=planned_load,
        computing_density=computing_density,
        pv_capacity=pv_capacity,
        wind_capacity=wind_capacity,
        storage_capacity=storage_capacity,
        cooling_plan=cooling_plan
    )
    
    # ===== 计算OPEX（年运营成本） =====
    opex = _calculate_opex(
        planned_load=planned_load,
        simulation_result=simulation_result,
        electricity_price=electricity_price,
        energy_plan=energy_plan
    )
    
    # ===== 计算财务指标 =====
    financial_metrics = _calculate_financial_metrics(
        capex=capex,
        opex=opex,
        simulation_result=simulation_result
    )
    
    financial_analysis = {
        "capex": capex,
        "opex_annual": round(opex, 2),
        "payback_period": round(financial_metrics['payback_period'], 2),
        "npv": round(financial_metrics['npv'], 2),
        "irr": round(financial_metrics['irr'], 2),
        "lcoe": round(financial_metrics['lcoe'], 4)
    }
    
    print(f"\n💰 财务分析结果:")
    print(f"  - 建设投资(CAPEX): {sum(capex.values()):,.1f} 万元")
    for item, cost in capex.items():
        print(f"    · {item}: {cost:,.1f} 万元")
    print(f"  - 年运营成本(OPEX): {opex:,.1f} 万元/年")
    print(f"  - 投资回收期: {financial_analysis['payback_period']:.1f} 年")
    print(f"  - 净现值(NPV): {financial_analysis['npv']:,.1f} 万元")
    print(f"  - 内部收益率(IRR): {financial_analysis['irr']:.2f}%")
    print(f"  - 平准化电力成本(LCOE): {financial_analysis['lcoe']:.4f} 元/kWh")
    
    # ===== 生成最终报告 =====
    from graph import generate_final_report
    final_report = generate_final_report(state)
    
    print(f"\n📝 最终报告已生成")
    print(f"   报告长度: {len(final_report)} 字符")
    
    print("\n" + "="*60)
    print("✅ [Agent 5: 综合评价与投资决策专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态
    return {
        **state,
        "financial_analysis": financial_analysis,
        "final_report": final_report
    }


def _calculate_capex(
    planned_area: float,
    planned_load: float,
    computing_density: float,
    pv_capacity: float,
    wind_capacity: float,
    storage_capacity: float,
    cooling_plan: dict
) -> Dict[str, float]:
    """
    计算建设成本（CAPEX）
    
    参数:
        planned_area: 计划面积（平方米）
        planned_load: 计划负荷（kW）
        computing_density: 算力密度（kW/机柜）
        pv_capacity: 光伏装机容量（kW）
        wind_capacity: 风电装机容量（kW）
        storage_capacity: 储能容量（kWh）
        cooling_plan: 制冷方案
        
    返回:
        CAPEX明细字典（万元）
    """
    capex = {}
    
    # 1. 土建工程（元/平方米）
    construction_cost_per_sqm = 8000  # 数据中心建筑成本
    capex['土建工程'] = planned_area * construction_cost_per_sqm / 10000
    
    # 2. 机柜及IT设备
    if computing_density > 0:
        rack_count = planned_load / computing_density
    else:
        rack_count = planned_load / 8
    rack_cost = 2  # 万元/机柜（含配电）
    capex['机柜及配电'] = rack_count * rack_cost
    
    # 3. 制冷系统
    cooling_tech = cooling_plan.get('cooling_technology', '传统风冷')
    if '液冷' in cooling_tech:
        cooling_cost_per_kw = 3000  # 元/kW
    elif '间接蒸发冷却' in cooling_tech:
        cooling_cost_per_kw = 2000
    else:
        cooling_cost_per_kw = 1500
    capex['制冷系统'] = planned_load * (cooling_plan.get('estimated_pue', 1.4) - 1) * cooling_cost_per_kw / 10000
    
    # 4. 光伏系统（元/W）
    pv_cost_per_w = 3.5
    capex['光伏系统'] = pv_capacity * 1000 * pv_cost_per_w / 10000
    
    # 5. 风电系统（元/W）
    if wind_capacity > 0:
        wind_cost_per_w = 6.0
        capex['风电系统'] = wind_capacity * 1000 * wind_cost_per_w / 10000
    
    # 6. 储能系统（元/Wh）
    storage_cost_per_wh = 1.2
    capex['储能系统'] = storage_capacity * 1000 * storage_cost_per_wh / 10000
    
    # 7. 其他（UPS、监控、消防等）
    other_cost_rate = 0.15
    subtotal = sum(capex.values())
    capex['其他设备及安装'] = subtotal * other_cost_rate
    
    return {k: round(v, 2) for k, v in capex.items()}


def _calculate_opex(
    planned_load: float,
    simulation_result: dict,
    electricity_price: dict,
    energy_plan: dict
) -> float:
    """
    计算年运营成本（OPEX）
    
    参数:
        planned_load: 计划负荷（kW）
        simulation_result: 仿真结果
        electricity_price: 电价数据
        energy_plan: 能源规划方案
        
    返回:
        年运营成本（万元/年）
    """
    # 年用电量
    annual_consumption = simulation_result.get('annual_green_consumption', 0)
    total_consumption = simulation_result.get('total_consumption', planned_load * 0.75 * 8760)
    grid_consumption = total_consumption - annual_consumption
    
    # 平均电价（加权平均）
    avg_price = (
        electricity_price.get('peak_price', 0.5) * 0.1 +
        electricity_price.get('high_price', 0.4) * 0.3 +
        electricity_price.get('flat_price', 0.3) * 0.4 +
        electricity_price.get('low_price', 0.25) * 0.15 +
        electricity_price.get('deep_low_price', 0.2) * 0.05
    )
    
    # 电费成本
    electricity_cost = grid_consumption * avg_price / 10000  # 万元
    
    # 维护成本（按投资比例）
    maintenance_rate = 0.02
    pv_capacity = energy_plan.get('pv_capacity', 0)
    storage_capacity = energy_plan.get('storage_capacity', 0)
    maintenance_cost = (pv_capacity * 3.5 + storage_capacity * 1.2) * maintenance_rate
    
    # 人工成本（简化）
    labor_cost = 50  # 万元/年
    
    # 其他运营成本
    other_opex = 20
    
    return electricity_cost + maintenance_cost + labor_cost + other_opex


def _calculate_financial_metrics(
    capex: Dict[str, float],
    opex: float,
    simulation_result: dict
) -> Dict[str, float]:
    """
    计算财务指标
    
    参数:
        capex: 建设成本明细
        opex: 年运营成本
        simulation_result: 仿真结果
        
    返回:
        财务指标字典
    """
    total_capex = sum(capex.values())
    
    # 年收益（节省的电费 + 碳交易收益）
    annual_green = simulation_result.get('annual_green_consumption', 0)
    carbon_reduction = simulation_result.get('carbon_reduction', 0)
    
    # 假设电网电价0.4元/kWh，绿电成本0.2元/kWh
    electricity_saving = annual_green * (0.4 - 0.2) / 10000  # 万元
    
    # 碳交易收益（50元/吨CO₂）
    carbon_revenue = carbon_reduction * 50 / 10000  # 万元
    
    annual_benefit = electricity_saving + carbon_revenue
    
    # 投资回收期（简化计算）
    net_annual = annual_benefit - opex * 0.3  # 假设运营成本中有30%可被绿电收益抵消
    if net_annual > 0:
        payback_period = total_capex / net_annual
    else:
        payback_period = 99
    
    # NPV计算（10年期，折现率8%）
    discount_rate = 0.08
    npv = -total_capex
    for year in range(1, 11):
        npv += net_annual / ((1 + discount_rate) ** year)
    
    # IRR（简化估算）
    if net_annual > 0 and total_capex > 0:
        irr = (net_annual / total_capex) * 100
    else:
        irr = 0
    
    # LCOE（简化计算）
    total_generation = simulation_result.get('annual_green_consumption', 0)
    if total_generation > 0:
        lcoe = (total_capex * 10000 / 10 + opex * 10000) / total_generation
    else:
        lcoe = 0
    
    return {
        'payback_period': payback_period,
        'npv': npv,
        'irr': irr,
        'lcoe': lcoe
    }


# 保持向后兼容的别名
agent5_node = financial_consultant_node


# --- 主程序入口（用于独立测试） ---
if __name__ == "__main__":
    print("===== 测试 Agent 5: 综合评价与投资决策专家 =====")
    
    # 创建测试状态
    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "planned_area": 10000,
            "planned_load": 5000,
            "computing_power_density": 30
        },
        "environmental_data": {
            "annual_temperature": 5.5,
            "carbon_emission_factor": 0.6479
        },
        "electricity_price": {
            "peak_price": 0.8,
            "high_price": 0.6,
            "flat_price": 0.4,
            "low_price": 0.25,
            "deep_low_price": 0.15
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "wind_capacity": 500,
            "storage_capacity": 6000
        },
        "cooling_plan": {
            "cooling_technology": "浸没式液冷",
            "estimated_pue": 1.05
        },
        "simulation_result": {
            "annual_green_consumption": 15000000,
            "total_consumption": 20000000,
            "carbon_reduction": 9700
        }
    }
    
    # 执行节点
    result = financial_consultant_node(test_state)
    
    print("\n===== 测试结果 =====")
    print(f"财务分析: {result.get('financial_analysis')}")
    print(f"\n最终报告预览:\n{result.get('final_report')[:500]}...")
