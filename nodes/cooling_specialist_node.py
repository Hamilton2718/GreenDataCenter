"""
Agent 3: 暖通与制冷架构专家 (Cooling Specialist)

功能:
    - 根据地理位置和PUE目标选择冷却技术
    - 计算预计的年均PUE，减小的耗电量
    - 输出：制冷技术路线图

输入状态:
    - user_requirements: 用户需求（含算力密度、PUE目标）
    - environmental_data: 环境数据（温度等）
    - energy_plan: 能源规划方案

输出状态:
    - cooling_plan: 制冷方案

作者: GreenDataCenter Team
版本: 1.0.0
"""

from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from graph import GreenDataCenterState


def cooling_specialist_node(state: dict) -> dict:
    """
    Agent 3: 暖通与制冷架构专家 - LangGraph Node
    
    基于地理位置、算力密度和PUE目标，选择最优制冷技术方案。
    这是LangGraph工作流的第三个节点。
    
    参数:
        state: GreenDataCenterState类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - energy_plan: 能源规划方案
            
    返回:
        更新后的state，新增:
            - cooling_plan: 制冷方案
    """
    print("\n" + "="*60)
    print("❄️  [Agent 3: 暖通与制冷架构专家] 开始工作")
    print("="*60)
    
    # 获取输入数据
    user_req = state.get('user_requirements', {})
    env_data = state.get('environmental_data', {})
    energy_plan = state.get('energy_plan', {})
    
    # 提取关键参数
    computing_density = user_req.get('computing_power_density', 
                                     user_req.get('Computing_power_density', 8))  # kW/机柜
    pue_target = user_req.get('pue_target', 1.2)
    planned_load = user_req.get('planned_load', 0)  # kW
    
    # 提取环境参数
    annual_temp = env_data.get('annual_temperature', 10.0)
    
    print(f"📊 输入参数:")
    print(f"  - 算力密度: {computing_density} kW/机柜")
    print(f"  - PUE目标: {pue_target}")
    print(f"  - 计划负荷: {planned_load} kW")
    print(f"  - 年均温度: {annual_temp}°C")
    
    # ===== 选择制冷技术 =====
    cooling_tech = _select_cooling_technology(
        computing_density=computing_density,
        annual_temp=annual_temp,
        pue_target=pue_target
    )
    
    # ===== 计算PUE和功耗 =====
    estimated_pue = _calculate_pue(cooling_tech, annual_temp, pue_target)
    cooling_power = planned_load * (estimated_pue - 1)  # 制冷功耗 = IT负荷 * (PUE-1)
    
    # ===== 计算自然冷却小时数 =====
    free_cooling_hours = _estimate_free_cooling_hours(annual_temp, cooling_tech)
    
    # ===== 生成设备清单 =====
    equipment_list = _generate_equipment_list(cooling_tech, planned_load, computing_density)
    
    cooling_plan = {
        "cooling_technology": cooling_tech,
        "estimated_pue": round(estimated_pue, 3),
        "cooling_power_consumption": round(cooling_power, 2),
        "free_cooling_hours": free_cooling_hours,
        "equipment_list": equipment_list
    }
    
    print(f"\n❄️  制冷方案:")
    print(f"  - 制冷技术: {cooling_tech}")
    print(f"  - 预计年均PUE: {estimated_pue:.3f}")
    print(f"  - 制冷功耗: {cooling_power:.1f} kW")
    print(f"  - 自然冷却小时数: {free_cooling_hours} 小时/年")
    print(f"  - 主要设备数量: {len(equipment_list)} 项")
    
    print("\n" + "="*60)
    print("✅ [Agent 3: 暖通与制冷架构专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态
    return {
        **state,
        "cooling_plan": cooling_plan
    }


def _select_cooling_technology(computing_density: float, annual_temp: float, pue_target: float) -> str:
    """
    选择制冷技术
    
    决策逻辑:
    - 算力密度 > 20kW/机柜 → 液冷
    - 算力密度 > 15kW/机柜 且 PUE目标 < 1.25 → 液冷
    - 年均温度 < 10°C → 自然冷却/间接蒸发冷却
    - 其他 → 风冷或间接蒸发冷却
    
    参数:
        computing_density: 算力密度（kW/机柜）
        annual_temp: 年均温度（°C）
        pue_target: PUE目标
        
    返回:
        制冷技术名称
    """
    if computing_density >= 20:
        return "浸没式液冷"
    elif computing_density >= 15 and pue_target < 1.25:
        return "冷板式液冷"
    elif annual_temp < 5:
        return "自然冷却 + 机械制冷"
    elif annual_temp < 12:
        return "间接蒸发冷却"
    elif computing_density >= 10:
        return "行级风冷 + 热通道封闭"
    else:
        return "传统风冷"


def _calculate_pue(cooling_tech: str, annual_temp: float, pue_target: float) -> float:
    """
    计算预计PUE
    
    参数:
        cooling_tech: 制冷技术
        annual_temp: 年均温度
        pue_target: PUE目标
        
    返回:
        预计PUE值
    """
    # 基础PUE根据技术类型
    base_pue = {
        "浸没式液冷": 1.05,
        "冷板式液冷": 1.15,
        "自然冷却 + 机械制冷": 1.15,
        "间接蒸发冷却": 1.20,
        "行级风冷 + 热通道封闭": 1.30,
        "传统风冷": 1.40
    }.get(cooling_tech, 1.35)
    
    # 根据年均温度调整（温度越低，PUE越低）
    temp_adjustment = (10 - annual_temp) * 0.005
    adjusted_pue = base_pue - temp_adjustment
    
    # 考虑PUE目标（实际PUE不应低于目标的90%）
    min_pue = pue_target * 0.9
    
    return max(adjusted_pue, min_pue, 1.02)  # PUE最低不低于1.02


def _estimate_free_cooling_hours(annual_temp: float, cooling_tech: str) -> int:
    """
    估算自然冷却小时数
    
    参数:
        annual_temp: 年均温度
        cooling_tech: 制冷技术
        
    返回:
        自然冷却小时数
    """
    # 根据技术和温度估算
    if "自然冷却" in cooling_tech or "间接蒸发冷却" in cooling_tech:
        # 年均温度越低，自然冷却小时数越多
        if annual_temp < 5:
            return 6000
        elif annual_temp < 10:
            return 5000
        elif annual_temp < 15:
            return 3500
        else:
            return 2000
    elif "液冷" in cooling_tech:
        # 液冷系统可以利用更高温度的自然冷却
        if annual_temp < 10:
            return 7000
        elif annual_temp < 15:
            return 5500
        else:
            return 4000
    else:
        # 传统风冷
        if annual_temp < 10:
            return 3000
        elif annual_temp < 15:
            return 1500
        else:
            return 500


def _generate_equipment_list(cooling_tech: str, planned_load: float, computing_density: float) -> List[Dict[str, Any]]:
    """
    生成制冷设备清单
    
    参数:
        cooling_tech: 制冷技术
        planned_load: 计划负荷（kW）
        computing_density: 算力密度（kW/机柜）
        
    返回:
        设备清单列表
    """
    equipment = []
    
    # 计算机柜数量
    if computing_density > 0:
        rack_count = int(planned_load / computing_density)
    else:
        rack_count = int(planned_load / 8)  # 默认8kW/机柜
    
    if "液冷" in cooling_tech:
        equipment.append({
            "name": "液冷CDU（冷却分配单元）",
            "quantity": max(1, rack_count // 20),
            "unit": "台",
            "spec": f"{planned_load * 1.2 / max(1, rack_count // 20):.0f}kW"
        })
        equipment.append({
            "name": "冷却塔",
            "quantity": max(1, planned_load // 2000),
            "unit": "台",
            "spec": "横流式"
        })
    
    if "间接蒸发冷却" in cooling_tech:
        equipment.append({
            "name": "间接蒸发冷却机组",
            "quantity": max(1, planned_load // 500),
            "unit": "台",
            "spec": f"{planned_load / max(1, planned_load // 500):.0f}kW"
        })
    
    if "风冷" in cooling_tech or cooling_tech == "传统风冷":
        equipment.append({
            "name": "精密空调",
            "quantity": max(1, rack_count // 8),
            "unit": "台",
            "spec": "行级/房间级"
        })
        equipment.append({
            "name": "冷水机组",
            "quantity": max(1, planned_load // 1500),
            "unit": "台",
            "spec": "变频离心式"
        })
    
    # 通用设备
    equipment.append({
        "name": "冷冻水泵",
        "quantity": max(2, planned_load // 2000 * 2),
        "unit": "台",
        "spec": "变频"
    })
    
    equipment.append({
        "name": "冷却塔",
        "quantity": max(1, planned_load // 1500),
        "unit": "台",
        "spec": "低噪音型"
    })
    
    return equipment


# 保持向后兼容的别名
agent3_node = cooling_specialist_node


# --- 主程序入口（用于独立测试） ---
if __name__ == "__main__":
    print("===== 测试 Agent 3: 暖通与制冷架构专家 =====")
    
    # 创建测试状态
    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "computing_power_density": 30,
            "pue_target": 1.2,
            "planned_load": 5000
        },
        "environmental_data": {
            "annual_temperature": 5.5
        },
        "energy_plan": {
            "pv_capacity": 2000
        }
    }
    
    # 执行节点
    result = cooling_specialist_node(test_state)
    
    print("\n===== 测试结果 =====")
    print(f"制冷方案: {result.get('cooling_plan')}")
