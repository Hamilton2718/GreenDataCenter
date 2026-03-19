"""
最终报告生成节点 (Final Report Node)

将 LangGraph 最终状态整理为 Markdown 报告，写入 state["final_report"]。
"""

from typing import Dict, Any


def _to_text(value: Any, default: str) -> str:
    """将空值统一转为可展示文本，避免报告中出现 N/A。"""
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def _to_num(value: Any, default: float = 0.0, digits: int = 2) -> str:
    """将数值统一格式化为字符串，空值回退到默认值。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default
    return f"{round(num, digits)}"


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

    # 项目概况兜底
    location = _to_text(user_req.get("location"), "未提供")
    business_type = _to_text(user_req.get("business_type"), "通用计算型")
    planned_load = _to_num(user_req.get("planned_load"), 0, 1)
    computing_power_density = _to_num(user_req.get("computing_power_density"), 0, 1)
    priority = _to_text(user_req.get("priority"), "环保型")
    green_target = _to_num(user_req.get("green_energy_target"), 0, 1)
    pue_target = _to_num(user_req.get("pue_target"), 1.2, 3)

    # 环境参数兜底
    annual_temperature = _to_num(env_data.get("annual_temperature"), 10.0, 2)
    annual_wind_speed = _to_num(env_data.get("annual_wind_speed"), 4.0, 2)
    annual_sunshine_hours = _to_num(env_data.get("annual_sunshine_hours"), 2500.0, 2)
    carbon_factor = _to_num(env_data.get("carbon_emission_factor"), 0.5, 4)

    # 能源方案兜底
    pv_capacity = _to_num(energy_plan.get("pv_capacity"), 0.0, 2)
    storage_capacity = _to_num(energy_plan.get("storage_capacity"), 0.0, 2)
    storage_power = _to_num(energy_plan.get("storage_power"), 0.0, 2)
    ppa_ratio = _to_num(energy_plan.get("ppa_ratio"), 0.0, 2)
    grid_ratio = _to_num(energy_plan.get("grid_ratio"), 0.0, 2)

    # 制冷方案兜底
    cooling_technology = _to_text(cooling_plan.get("cooling_technology"), "风冷")
    estimated_pue = _to_num(cooling_plan.get("estimated_pue"), 1.35, 3)

    # 财务指标兜底（优先读取统一字段，兼容旧字段）
    payback_period = financial.get("payback_period", financial.get("payback_years", 30.0))
    irr = financial.get("irr", 0.0)
    npv = financial.get("npv", 0.0)
    lcoe = financial.get("lcoe", 0.0)

    # 仿真指标兜底
    daily_it_energy = _to_num(sim_summary.get("daily_it_energy_mwh"), 0.0, 3)
    daily_green_supply = _to_num(sim_summary.get("daily_green_supply_mwh"), 0.0, 3)
    daily_green_ratio = _to_num(sim_summary.get("daily_green_ratio_pct"), 0.0, 2)
    daily_storage_charge = _to_num(sim_summary.get("daily_storage_charge_mwh"), 0.0, 3)
    daily_storage_discharge = _to_num(sim_summary.get("daily_storage_discharge_mwh"), 0.0, 3)
    sim_method = _to_text(sim_summary.get("method"), "粗仿真（典型日）")

    report = f"""# 数据中心绿电消纳规划设计建议书

## 一、项目概况


| 项目 | 数值 |
|------|------|
| 地理位置 | {location} |
| 业务类型 | {business_type} |
| 计划负荷 | {planned_load} kW |
| 算力密度 | {computing_power_density} kW/机柜 |
| 优先级 | {priority} |
| 绿电目标 | {green_target}% |
| PUE 目标 | {pue_target} |

## 二、环境条件分析

| 环境参数 | 数值 |
|---------|------|
| 年均温度 | {annual_temperature}°C |
| 年均风速 | {annual_wind_speed} m/s |
| 年日照时长 | {annual_sunshine_hours} 小时 |
| 碳排因子 | {carbon_factor} kgCO2/kWh |

## 三、能源配比方案

| 能源类型 | 配置 |
|---------|------|
| 分布式光伏 | {pv_capacity} kW |
| 储能系统 | {storage_capacity} kWh / {storage_power} kW |
| 绿电长协 | {ppa_ratio}% |
| 电网调峰 | {grid_ratio}% |

## 四、制冷技术方案

| 技术参数 | 数值 |
|---------|------|
| 制冷技术 | {cooling_technology} |
| 预计年均 PUE | {estimated_pue} |

## 五、财务分析

| 财务指标 | 数值 |
|---------|------|
| 投资回收期 | {_to_num(payback_period, 30.0, 2)} 年 |
| 内部收益率 (IRR) | {_to_num(irr, 0.0, 2)}% |
| 净现值 (NPV) | {_to_num(npv, 0.0, 2)} 万元 |
| 平准化电力成本 (LCOE) | {_to_num(lcoe, 0.0, 4)} 元/kWh |

## 六、24小时粗仿真摘要

| 指标 | 数值 |
|------|------|
| 日IT用电量 | {daily_it_energy} MWh |
| 日绿电供给量 | {daily_green_supply} MWh |
| 日绿电占比 | {daily_green_ratio}% |
| 储能日充电量 | {daily_storage_charge} MWh |
| 储能日放电量 | {daily_storage_discharge} MWh |
| 仿真方法 | {sim_method} |

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
