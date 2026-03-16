"""
最终报告生成节点 (Final Report Node)

将 LangGraph 最终状态整理为 Markdown 报告，写入 state["final_report"]。
"""

from typing import Dict, Any


def generate_final_report(state: Dict[str, Any]) -> str:
    """
    根据当前状态生成最终规划设计建议书（Markdown）。

    参数:
        state: LangGraph 状态字典

    返回:
        Markdown 格式报告字符串
    """
    user_req = state.get("user_requirements", {}) or {}
    env_data = state.get("environmental_data", {}) or {}
    energy_plan = state.get("energy_plan", {}) or {}
    cooling_plan = state.get("cooling_plan", {}) or {}
    simulation = state.get("simulation_result", {}) or {}
    sim_summary = simulation.get("summary", {}) if simulation else {}
    financial = state.get("financial_analysis", {}) or {}

    report = f"""# 数据中心绿电消纳规划设计建议书

## 一、项目概况

| 项目 | 数值 |
|------|------|
| 地理位置 | {user_req.get('location', 'N/A')} |
| 业务类型 | {user_req.get('business_type', 'N/A')} |
| 计划负荷 | {user_req.get('planned_load', 'N/A')} kW |
| 算力密度 | {user_req.get('computing_power_density', 'N/A')} kW/机柜 |
| 优先级 | {user_req.get('priority', 'N/A')} |
| 绿电目标 | {user_req.get('green_energy_target', 'N/A')}% |
| PUE 目标 | {user_req.get('pue_target', 'N/A')} |

## 二、环境条件分析

| 环境参数 | 数值 |
|---------|------|
| 年均温度 | {env_data.get('annual_temperature', 'N/A')}°C |
| 年均风速 | {env_data.get('annual_wind_speed', 'N/A')} m/s |
| 年日照时长 | {env_data.get('annual_sunshine_hours', 'N/A')} 小时 |
| 碳排因子 | {env_data.get('carbon_emission_factor', 'N/A')} kgCO2/kWh |

## 三、能源配比方案

| 能源类型 | 配置 |
|---------|------|
| 分布式光伏 | {energy_plan.get('pv_capacity', 'N/A')} kW |
| 储能系统 | {energy_plan.get('storage_capacity', 'N/A')} kWh / {energy_plan.get('storage_power', 'N/A')} kW |
| 绿电长协 | {energy_plan.get('ppa_ratio', 'N/A')}% |
| 电网调峰 | {energy_plan.get('grid_ratio', 'N/A')}% |

## 四、制冷技术方案

| 技术参数 | 数值 |
|---------|------|
| 制冷技术 | {cooling_plan.get('cooling_technology', 'N/A')} |
| 预计年均 PUE | {cooling_plan.get('estimated_pue', 'N/A')} |

## 五、财务分析

| 财务指标 | 数值 |
|---------|------|
| 投资回收期 | {financial.get('payback_period', financial.get('payback_years', 'N/A'))} 年 |
| 内部收益率 (IRR) | {financial.get('irr', 'N/A')}% |
| 净现值 (NPV) | {financial.get('npv', 'N/A')} 万元 |
| 平准化电力成本 (LCOE) | {financial.get('lcoe', 'N/A')} 元/kWh |

## 六、24小时粗仿真摘要

| 指标 | 数值 |
|------|------|
| 日IT用电量 | {sim_summary.get('daily_it_energy_mwh', 'N/A')} MWh |
| 日绿电供给量 | {sim_summary.get('daily_green_supply_mwh', 'N/A')} MWh |
| 日绿电占比 | {sim_summary.get('daily_green_ratio_pct', 'N/A')}% |
| 储能日充电量 | {sim_summary.get('daily_storage_charge_mwh', 'N/A')} MWh |
| 储能日放电量 | {sim_summary.get('daily_storage_discharge_mwh', 'N/A')} MWh |
| 仿真方法 | {sim_summary.get('method', 'N/A')} |

---
*本报告由 GreenDataCenter 智能规划系统自动生成*
"""

    return report


def final_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    最终报告生成节点 - LangGraph Node

    参数:
        state: GreenDataCenterState 类型状态字典

    返回:
        更新后的状态，新增:
            - final_report: Markdown 格式最终报告
    """
    print("\n" + "=" * 60)
    print("📝 [最终报告生成节点] 开始工作")
    print("=" * 60)

    final_report = generate_final_report(state)
    print(f"✅ 最终报告生成完成，长度: {len(final_report)} 字符")

    print("\n" + "=" * 60)
    print("✅ [最终报告生成节点] 工作完成")
    print("=" * 60)

    return {
        **state,
        "final_report": final_report,
    }
