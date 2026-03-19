"""
Agent 4: 24小时粗仿真节点 (Simulation Node)

根据用户需求、能源方案与制冷方案，生成24小时粗仿真曲线：
- IT负载曲线 (MW)
- 绿电供应曲线 (MW)
- 储能充放电曲线 (MW，放电为正，充电为负)

说明：
- 该节点为“方案级粗仿真”，用于快速验证配置方向，不替代精细调度仿真。
- 输出结果写入 state["simulation_result"]，供财务与报告节点复用。
"""

from typing import Dict, Any, List
from pathlib import Path


def _build_it_load_curve(planned_load_kw: float, business_type: str) -> List[float]:
    """生成24小时 IT 负载曲线（MW）。"""
    base_mw = max(planned_load_kw, 0.0) / 1000.0

    # 仅按算力类型 / 负载特性划分四类：
    # 1) 通用计算型(普算) 2) 智能计算型(智算)
    # 3) 高性能计算型(超算) 4) 存储密集型(Storage Intensive)
    type_alias = {
        "通用计算型": "通用计算型",
        "普算": "通用计算型",
        "general purpose computing": "通用计算型",
        "智能计算型": "智能计算型",
        "智算": "智能计算型",
        "ai computing": "智能计算型",
        "高性能计算型": "高性能计算型",
        "超算": "高性能计算型",
        "high performance computing": "高性能计算型",
        "hpc": "高性能计算型",
        "存储密集型": "存储密集型",
        "storage intensive": "存储密集型",
    }

    raw_type = str(business_type or "").strip()
    dc_type = type_alias.get(raw_type, type_alias.get(raw_type.lower(), "通用计算型"))

    if dc_type == "智能计算型":
        # 智算：整体高负载、全天连续运行，午后达到高位平台
        factors = [
            0.92, 0.90, 0.89, 0.89, 0.90, 0.92,
            0.96, 1.00, 1.05, 1.10, 1.15, 1.20,
            1.24, 1.27, 1.28, 1.28, 1.26, 1.22,
            1.18, 1.12, 1.05, 1.00, 0.96, 0.92,
        ]
    elif dc_type == "高性能计算型":
        # 超算：任务调度特征明显，夜间和清晨可出现较高负载段
        factors = [
            1.10, 1.08, 1.06, 1.04, 1.02, 1.00,
            0.96, 0.92, 0.90, 0.92, 0.98, 1.05,
            1.12, 1.18, 1.22, 1.24, 1.23, 1.20,
            1.16, 1.14, 1.13, 1.12, 1.11, 1.10,
        ]
    elif dc_type == "存储密集型":
        # 存储密集型：负载更平稳，昼夜波动幅度小
        factors = [
            0.88, 0.87, 0.87, 0.86, 0.86, 0.87,
            0.89, 0.91, 0.93, 0.95, 0.96, 0.97,
            0.98, 0.98, 0.98, 0.97, 0.96, 0.95,
            0.94, 0.93, 0.92, 0.91, 0.90, 0.89,
        ]
    elif dc_type == "通用计算型":
        # 普算：更偏企业业务节律，白天提升，夜间回落
        factors = [
            0.75, 0.73, 0.72, 0.72, 0.74, 0.78,
            0.86, 0.95, 1.03, 1.10, 1.16, 1.20,
            1.22, 1.23, 1.22, 1.19, 1.14, 1.08,
            1.00, 0.93, 0.87, 0.82, 0.78, 0.76,
        ]
    else:
        # 未识别类型时按通用计算型兜底
        factors = [
            0.75, 0.73, 0.72, 0.72, 0.74, 0.78,
            0.86, 0.95, 1.03, 1.10, 1.16, 1.20,
            1.22, 1.23, 1.22, 1.19, 1.14, 1.08,
            1.00, 0.93, 0.87, 0.82, 0.78, 0.76,
        ]

    return [round(base_mw * f, 3) for f in factors]


def _build_pv_curve(pv_capacity_kw: float) -> List[float]:
    """生成24小时分布式光伏出力曲线（MW）。"""
    pv_cap_mw = max(pv_capacity_kw, 0.0) / 1000.0

    # 0:00-23:00 典型晴天出力系数
    pv_factors = [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.02,
        0.08, 0.18, 0.35, 0.50, 0.65, 0.78,
        0.88, 0.95, 0.98, 0.95, 0.82, 0.60,
        0.30, 0.08, 0.00, 0.00, 0.00, 0.00,
    ]
    return [round(pv_cap_mw * f, 3) for f in pv_factors]


def _estimate_ppa_green_supply(total_daily_mwh: float, ppa_ratio_pct: float) -> float:
    """估算长协绿电在日尺度的平均供给（MW，按全天均匀到货近似）。"""
    ratio = max(0.0, min(ppa_ratio_pct, 100.0)) / 100.0
    ppa_daily_mwh = total_daily_mwh * ratio
    return ppa_daily_mwh / 24.0


def _dispatch_storage(
    net_gap_curve_mw: List[float],
    storage_capacity_kwh: float,
    storage_power_kw: float,
    initial_soc: float = 0.5,
    roundtrip_efficiency: float = 0.9,
) -> Dict[str, List[float]]:
    """
    粗粒度储能调度：
    - 绿电富余时充电（曲线为负值）
    - 负荷缺口时放电（曲线为正值）
    """
    cap_mwh = max(storage_capacity_kwh, 0.0) / 1000.0
    p_mw = max(storage_power_kw, 0.0) / 1000.0

    if cap_mwh <= 0 or p_mw <= 0:
        zero = [0.0] * 24
        return {"storage_power_curve_mw": zero, "soc_curve": zero}

    charge_eff = discharge_eff = roundtrip_efficiency ** 0.5
    soc_mwh = max(0.0, min(cap_mwh, cap_mwh * initial_soc))

    storage_curve = []
    soc_curve = []

    for gap in net_gap_curve_mw:
        p = 0.0
        if gap > 0:
            # 缺口：放电支撑
            max_discharge_by_soc = soc_mwh * discharge_eff
            discharge = min(gap, p_mw, max_discharge_by_soc)
            soc_mwh -= discharge / discharge_eff
            p = discharge
        elif gap < 0:
            # 富余：充电吸收
            surplus = -gap
            max_charge_by_soc = (cap_mwh - soc_mwh) / charge_eff
            charge = min(surplus, p_mw, max_charge_by_soc)
            soc_mwh += charge * charge_eff
            p = -charge

        storage_curve.append(round(p, 3))
        soc_curve.append(round(soc_mwh / cap_mwh if cap_mwh > 0 else 0.0, 3))

    return {
        "storage_power_curve_mw": storage_curve,
        "soc_curve": soc_curve,
    }


def simulation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 4: 24小时粗仿真节点 - LangGraph Node

    输入：
    - user_requirements
    - energy_plan

    输出：
    - simulation_result
    """
    print("\n" + "=" * 60)
    print("📈 [Agent 4: 24小时粗仿真节点] 开始工作")
    print("=" * 60)

    user_req = state.get("user_requirements", {}) or {}
    energy_plan = state.get("energy_plan", {}) or {}

    planned_load_kw = float(user_req.get("planned_load", 0) or 0)
    business_type = user_req.get("business_type", "通用")

    pv_capacity_kw = float(energy_plan.get("pv_capacity", 0) or 0)
    storage_capacity_kwh = float(energy_plan.get("storage_capacity", 0) or 0)
    storage_power_kw = float(energy_plan.get("storage_power", 0) or 0)
    ppa_ratio_pct = float(energy_plan.get("ppa_ratio", 0) or 0)

    # 如果上游没有给出储能功率，给一个常见缺省：2小时系统
    if storage_capacity_kwh > 0 and storage_power_kw <= 0:
        storage_power_kw = storage_capacity_kwh / 2.0

    it_curve = _build_it_load_curve(planned_load_kw, business_type)
    pv_curve = _build_pv_curve(pv_capacity_kw)

    total_daily_it_mwh = sum(it_curve)
    ppa_base_mw = _estimate_ppa_green_supply(total_daily_it_mwh, ppa_ratio_pct)
    ppa_curve = [round(ppa_base_mw, 3)] * 24

    green_pre_storage = [round(pv_curve[i] + ppa_curve[i], 3) for i in range(24)]
    net_gap = [round(it_curve[i] - green_pre_storage[i], 3) for i in range(24)]

    storage_dispatch = _dispatch_storage(
        net_gap_curve_mw=net_gap,
        storage_capacity_kwh=storage_capacity_kwh,
        storage_power_kw=storage_power_kw,
        initial_soc=0.5,
        roundtrip_efficiency=0.9,
    )
    storage_curve = storage_dispatch["storage_power_curve_mw"]
    soc_curve = storage_dispatch["soc_curve"]

    # 供图表使用的绿电供给曲线：光伏 + 长协 + 储能放电 - 储能充电(储能负值代表充电，自然会减少可供电量)
    green_supply_curve = [
        round(green_pre_storage[i] + storage_curve[i], 3) for i in range(24)
    ]

    # 统计量
    it_daily_mwh = round(sum(it_curve), 3)
    green_daily_mwh = round(sum(max(0.0, min(green_supply_curve[i], it_curve[i])) for i in range(24)), 3)
    storage_charge_mwh = round(sum(-x for x in storage_curve if x < 0), 3)
    storage_discharge_mwh = round(sum(x for x in storage_curve if x > 0), 3)
    green_ratio = round((green_daily_mwh / it_daily_mwh) * 100, 2) if it_daily_mwh > 0 else 0.0

    simulation_result = {
        "time_labels": [f"{h}:00" for h in range(24)],
        "it_load_curve_mw": it_curve,
        "green_supply_curve_mw": green_supply_curve,
        "storage_power_curve_mw": storage_curve,
        "pv_curve_mw": pv_curve,
        "ppa_curve_mw": ppa_curve,
        "soc_curve": soc_curve,
        "summary": {
            "daily_it_energy_mwh": it_daily_mwh,
            "daily_green_supply_mwh": green_daily_mwh,
            "daily_green_ratio_pct": green_ratio,
            "daily_storage_charge_mwh": storage_charge_mwh,
            "daily_storage_discharge_mwh": storage_discharge_mwh,
            "storage_roundtrip_efficiency_assumed": 0.9,
            "method": "粗仿真（典型日）",
        },
    }

    print(f"📊 IT日用电量: {it_daily_mwh} MWh")
    print(f"🌿 绿电日供给: {green_daily_mwh} MWh")
    print(f"🔋 储能日充/放: {storage_charge_mwh}/{storage_discharge_mwh} MWh")
    print(f"✅ [Agent 4] 粗仿真完成")

    return {
        **state,
        "simulation_result": simulation_result,
        # 与旧字段兼容：写入 load_profile 的日曲线
        "load_profile": {
            "daily_load_curve": [round(v * 1000, 2) for v in it_curve],  # 转为kW
            "peak_load": round(max(it_curve) * 1000, 2) if it_curve else 0.0,
            "avg_load": round((sum(it_curve) / 24.0) * 1000, 2) if it_curve else 0.0,
            "load_factor": round((sum(it_curve) / 24.0) / max(it_curve), 3) if it_curve and max(it_curve) > 0 else 0.0,
        },
    }


def _demo_state() -> Dict[str, Any]:
    """构造可独立运行的示例输入。"""
    return {
        "user_requirements": {
            "planned_load": 50000,  # kW
            "business_type": "智能计算型",
        },
        "energy_plan": {
            "pv_capacity": 55000,       # kW
            "storage_capacity": 80000,  # kWh
            "storage_power": 20000,     # kW
            "ppa_ratio": 20,            # %
        },
    }


def _plot_simulation_curves(simulation_result: Dict[str, Any]) -> Path:
    """绘制24小时仿真曲线并输出图片文件。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        raise RuntimeError(
            "未检测到 matplotlib，请先安装：pip install matplotlib"
        ) from exc

    # 自动选择可用中文字体，避免标题/图例出现方框
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    available_fonts = [f for f in candidate_fonts if f in installed_fonts]
    if available_fonts:
        plt.rcParams["font.sans-serif"] = available_fonts
    plt.rcParams["axes.unicode_minus"] = False

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulation_curve_demo.png"

    hours = list(range(24))
    it_curve = simulation_result.get("it_load_curve_mw", [])
    green_curve = simulation_result.get("green_supply_curve_mw", [])
    storage_curve = simulation_result.get("storage_power_curve_mw", [])

    plt.figure(figsize=(12, 6))
    plt.plot(hours, it_curve, marker="o", linewidth=2, color="#3b82f6", label="IT负载 (MW)")
    plt.plot(hours, green_curve, marker="o", linewidth=2, color="#84cc16", label="绿电供应 (MW)")
    plt.bar(hours, storage_curve, width=0.7, color="#f59e0b", alpha=0.75, label="储能充放 (MW)")

    plt.title("24小时运行曲线", fontsize=14)
    plt.xlabel("小时")
    plt.ylabel("功率 (MW)")
    plt.xticks(hours)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return output_path


if __name__ == "__main__":
    demo_result = simulation_node(_demo_state())
    sim = demo_result.get("simulation_result", {})
    fig_path = _plot_simulation_curves(sim)

    summary = sim.get("summary", {})
    print("\n--- 仿真摘要 ---")
    print(f"IT日用电量: {summary.get('daily_it_energy_mwh', 0)} MWh")
    print(f"绿电日供给: {summary.get('daily_green_supply_mwh', 0)} MWh")
    print(f"绿电占比: {summary.get('daily_green_ratio_pct', 0)} %")
    print(f"储能日充电量: {summary.get('daily_storage_charge_mwh', 0)} MWh")
    print(f"储能日放电量: {summary.get('daily_storage_discharge_mwh', 0)} MWh")
    print(f"曲线图已生成: {fig_path}")
