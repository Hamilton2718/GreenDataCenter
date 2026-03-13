"""
GreenDataCenter - LangGraph 多Agent工作流定义

数据中心绿电消纳规划设计顾问系统
基于LangGraph的多Agent协作架构

作者: GreenDataCenter Team
版本: 1.0.0
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from dataclasses import dataclass, field
import operator

# LangGraph核心组件（延迟导入，避免循环依赖）
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

# ============================================================
# 1. 统一状态类型定义 (所有Agent共享的状态)
# ============================================================

class DataCenterRequirements(TypedDict, total=False):
    """用户需求数据结构"""
    location: str                          # 地理位置（城市名）
    business_type: str                     # 业务类型: "大模型训练"/"云存储"/"边缘计算"/"通用"
    planned_area: float                    # 计划面积（平方米）
    planned_load: float                    # 计划负荷（kW）
    computing_power_density: float         # 算力密度（kW/机柜）
    priority: str                          # 优先级: "可靠型"/"经济型"/"环保型"
    green_energy_target: float             # 绿电目标（%）
    pue_target: float                      # PUE目标
    budget_constraint: float               # 预算约束（万元）


class EnvironmentalData(TypedDict, total=False):
    """环境数据结构"""
    annual_temperature: float              # 年均温度（°C）
    annual_wind_speed: float               # 年均风速（m/s）
    annual_sunshine_hours: float           # 年均日照时长（小时）
    carbon_emission_factor: float          # 碳排因子（kgCO₂/kWh）
    latitude: float                        # 纬度
    longitude: float                       # 经度
    province: str                          # 所属省份


class ElectricityPriceData(TypedDict, total=False):
    """电价数据结构"""
    peak_price: float                      # 尖峰电价（元/kWh）
    high_price: float                      # 高峰电价（元/kWh）
    flat_price: float                      # 平段电价（元/kWh）
    low_price: float                       # 低谷电价（元/kWh）
    deep_low_price: float                  # 深谷电价（元/kWh）
    max_price_diff: float                  # 最大峰谷价差（元/kWh）


class LoadProfile(TypedDict, total=False):
    """负荷特性数据结构"""
    daily_load_curve: List[float]          # 24小时负荷曲线（kW）
    annual_load_curve: List[float]         # 全年负荷曲线（kW）
    peak_load: float                       # 峰值负荷（kW）
    avg_load: float                        # 平均负荷（kW）
    load_factor: float                     # 负荷率


class EnergyPlan(TypedDict, total=False):
    """能源规划方案数据结构"""
    # LLM 生成的完整报告 (Agent 2 - XSimple 版核心输出)
    llm_report: str                        # LLM 生成的 Markdown 格式能源规划报告
    
    # 数值化配置参数 (可由 LLM 解析或后续 Agent 计算填充)
    pv_capacity: float                     # 光伏装机容量（kW）
    wind_capacity: float                   # 风电装机容量（kW）
    storage_capacity: float                # 储能容量（kWh）
    storage_power: float                   # 储能功率（kW）
    ppa_ratio: float                       # 绿电长协比例（%）
    grid_ratio: float                      # 电网调峰比例（%）
    estimated_self_consumption: float      # 预计自发自用率（%）
    estimated_green_ratio: float           # 预计绿电占比（%）
    
    # 中间数据 (供后续 Agent 使用)
    price_data_cn: Dict[str, Any]          # 中文字段名的电价数据
    project_context: str                   # 项目背景上下文字符串
    api_data: str                          # API 返回的电网碳数据


class CoolingPlan(TypedDict, total=False):
    """制冷方案数据结构"""
    cooling_technology: str                # 制冷技术: "风冷"/"液冷"/"间接蒸发冷却"/"自然冷却"
    estimated_pue: float                   # 预计年均PUE
    cooling_power_consumption: float       # 制冷功耗（kW）
    free_cooling_hours: int                # 自然冷却小时数
    equipment_list: List[Dict[str, Any]]   # 设备清单


class SimulationResult(TypedDict, total=False):
    """仿真结果数据结构"""
    hourly_power_balance: List[Dict]       # 每小时电力平衡
    annual_green_consumption: float        # 年绿电消纳量（kWh）
    actual_green_ratio: float              # 实际绿电占比（%）
    actual_pue: float                      # 实际PUE
    carbon_reduction: float                # 碳减排量（吨CO₂/年）
    validation_passed: bool                # 验证是否通过
    validation_issues: List[str]           # 验证问题列表


class FinancialAnalysis(TypedDict, total=False):
    """财务分析数据结构"""
    capex: Dict[str, float]                # 建设成本明细（万元）
    opex_annual: float                     # 年运营成本（万元/年）
    payback_period: float                  # 投资回收期（年）
    npv: float                             # 净现值（万元）
    irr: float                             # 内部收益率（%）
    lcoe: float                            # 平准化电力成本（元/kWh）


class GreenDataCenterState(TypedDict, total=False):
    """
    LangGraph统一状态定义
    
    这是整个系统的核心状态类型，所有Agent节点都使用这个状态进行数据交换。
    状态在节点之间传递，每个节点可以读取和更新状态中的字段。
    
    字段说明:
        - user_requirements: 用户输入的需求（Agent 1填充）
        - environmental_data: 环境数据（Agent 1填充）
        - electricity_price: 电价数据（Agent 1填充）
        - load_profile: 负荷特性（Agent 1或Agent 2填充）
        - energy_plan: 能源规划方案（Agent 2填充）
        - cooling_plan: 制冷方案（Agent 3填充）
        - simulation_result: 仿真结果（Agent 4填充）
        - financial_analysis: 财务分析（Agent 5填充）
        - iteration_count: 迭代次数（用于控制循环）
        - error_message: 错误信息
        - final_report: 最终报告
    """
    # ===== Agent 1: 需求与约束解析专家 =====
    user_requirements: Optional[DataCenterRequirements]
    environmental_data: Optional[EnvironmentalData]
    electricity_price: Optional[ElectricityPriceData]
    
    # ===== Agent 2: 能源与绿电规划专家 =====
    load_profile: Optional[LoadProfile]
    energy_plan: Optional[EnergyPlan]
    
    # ===== Agent 3: 暖通与制冷架构专家 =====
    cooling_plan: Optional[CoolingPlan]
    
    # ===== Agent 4: 虚拟运行仿真专家 =====
    simulation_result: Optional[SimulationResult]
    
    # ===== Agent 5: 综合评价与投资决策专家 =====
    financial_analysis: Optional[FinancialAnalysis]
    
    # ===== 系统控制字段 =====
    iteration_count: int                   # 迭代计数器
    max_iterations: int                    # 最大迭代次数
    error_message: Optional[str]           # 错误信息
    final_report: Optional[str]            # 最终报告（Markdown格式）


# ============================================================
# 2. 状态初始化函数
# ============================================================

def create_initial_state(
    location: str = "",
    business_type: str = "通用",
    planned_area: float = 0,
    planned_load: float = 0,
    computing_power_density: float = 8,
    priority: str = "环保型",
    green_energy_target: float = 90,
    pue_target: float = 1.2,
    budget_constraint: float = 0
) -> GreenDataCenterState:
    """
    创建初始状态
    
    参数:
        location: 地理位置
        business_type: 业务类型
        planned_area: 计划面积（平方米）
        planned_load: 计划负荷（kW）
        computing_power_density: 算力密度（kW/机柜）
        priority: 优先级
        green_energy_target: 绿电目标（%）
        pue_target: PUE目标
        budget_constraint: 预算约束（万元）
    
    返回:
        初始化的GreenDataCenterState
    """
    return {
        "user_requirements": {
            "location": location,
            "business_type": business_type,
            "planned_area": planned_area,
            "planned_load": planned_load,
            "computing_power_density": computing_power_density,
            "priority": priority,
            "green_energy_target": green_energy_target,
            "pue_target": pue_target,
            "budget_constraint": budget_constraint
        },
        "environmental_data": None,
        "electricity_price": None,
        "load_profile": None,
        "energy_plan": None,
        "cooling_plan": None,
        "simulation_result": None,
        "financial_analysis": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "error_message": None,
        "final_report": None
    }


# ============================================================
# 3. 条件路由函数
# ============================================================

def should_continue_or_retry(state: GreenDataCenterState) -> str:
    """
    判断是否继续到下一个Agent或返回到Agent 1重新调整
    
    在Agent 4（仿真专家）执行后调用，根据验证结果决定流程走向。
    
    返回:
        - "continue": 验证通过，继续到Agent 5
        - "retry": 验证失败且未超过最大迭代次数，返回到Agent 1
        - "end": 验证失败且超过最大迭代次数，结束流程
    """
    simulation_result = state.get("simulation_result")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # 如果没有仿真结果，直接继续
    if simulation_result is None:
        return "continue"
    
    # 检查验证是否通过
    validation_passed = simulation_result.get("validation_passed", True)
    
    if validation_passed:
        print("✅ 仿真验证通过，继续到财务分析")
        return "continue"
    
    # 验证失败，检查是否超过最大迭代次数
    if iteration_count >= max_iterations:
        print(f"⚠️ 已达到最大迭代次数({max_iterations})，结束流程")
        return "end"
    
    print(f"🔄 仿真验证失败，返回重新调整参数（第{iteration_count + 1}次迭代）")
    return "retry"


def check_error(state: GreenDataCenterState) -> str:
    """
    检查是否有错误发生
    
    在每个Agent执行后调用，检查是否需要终止流程。
    
    返回:
        - "error": 有错误发生，结束流程
        - "continue": 无错误，继续执行
    """
    if state.get("error_message") is not None:
        print(f"❌ 发生错误: {state['error_message']}")
        return "error"
    return "continue"


# ============================================================
# 4. LangGraph构建函数
# ============================================================

def build_datacenter_workflow(
    agent1_node,
    agent2_node,
    agent3_node,
    agent4_node,
    agent5_node
) -> StateGraph:
    """
    构建数据中心绿电消纳规划工作流
    
    参数:
        agent1_node: 需求与约束解析专家节点函数
        agent2_node: 能源与绿电规划专家节点函数
        agent3_node: 暖通与制冷架构专家节点函数
        agent4_node: 虚拟运行仿真专家节点函数
        agent5_node: 综合评价与投资决策专家节点函数
    
    返回:
        编译后的StateGraph
    
    工作流结构:
        agent1 -> agent2 -> agent3 -> agent4 -> [条件判断]
                                        |
                    ┌-------------------┘
                    | (验证失败)
                    v
        agent1 <- agent4
                    |
                    | (验证通过)
                    v
                agent5 -> END
    """
    # 创建工作流图
    workflow = StateGraph(GreenDataCenterState)
    
    # 添加节点
    workflow.add_node("requirement_analysis", agent1_node)
    workflow.add_node("energy_planning", agent2_node)
    workflow.add_node("cooling_design", agent3_node)
    workflow.add_node("simulation", agent4_node)
    workflow.add_node("financial_analysis", agent5_node)
    
    # 设置入口点
    workflow.set_entry_point("requirement_analysis")
    
    # 添加边 - 顺序执行
    workflow.add_edge("requirement_analysis", "energy_planning")
    workflow.add_edge("energy_planning", "cooling_design")
    workflow.add_edge("cooling_design", "simulation")
    
    # 添加条件边 - 根据仿真结果决定流程走向
    workflow.add_conditional_edges(
        "simulation",
        should_continue_or_retry,
        {
            "continue": "financial_analysis",
            "retry": "requirement_analysis",
            "end": END
        }
    )
    
    # 财务分析后结束
    workflow.add_edge("financial_analysis", END)
    
    return workflow


def create_datacenter_agent_system(
    agent1_node,
    agent2_node,
    agent3_node,
    agent4_node,
    agent5_node,
    checkpoint_dir: Optional[str] = None
) -> Any:
    """
    创建完整的数据中心Agent系统
    
    参数:
        agent1_node: 需求与约束解析专家节点函数
        agent2_node: 能源与绿电规划专家节点函数
        agent3_node: 暖通与制冷架构专家节点函数
        agent4_node: 虚拟运行仿真专家节点函数
        agent5_node: 综合评价与投资决策专家节点函数
        checkpoint_dir: 检查点保存目录（可选）
    
    返回:
        编译后的可执行图
    """
    # 构建工作流
    workflow = build_datacenter_workflow(
        agent1_node=agent1_node,
        agent2_node=agent2_node,
        agent3_node=agent3_node,
        agent4_node=agent4_node,
        agent5_node=agent5_node
    )
    
    # 配置检查点（用于持久化状态）
    if checkpoint_dir:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app


# ============================================================
# 5. 工具函数
# ============================================================

def print_state_summary(state: GreenDataCenterState) -> None:
    """
    打印状态摘要（用于调试）
    """
    print("\n" + "="*60)
    print("📊 当前状态摘要")
    print("="*60)
    
    # 用户需求
    user_req = state.get("user_requirements")
    if user_req:
        print(f"\n📍 项目位置: {user_req.get('location', 'N/A')}")
        print(f"🏢 业务类型: {user_req.get('business_type', 'N/A')}")
        print(f"⚡ 计划负荷: {user_req.get('planned_load', 'N/A')} kW")
        print(f"🌿 绿电目标: {user_req.get('green_energy_target', 'N/A')}%")
    
    # 环境数据
    env_data = state.get("environmental_data")
    if env_data:
        print(f"\n🌡️ 年均温度: {env_data.get('annual_temperature', 'N/A')}°C")
        print(f"💨 年均风速: {env_data.get('annual_wind_speed', 'N/A')} m/s")
        print(f"☀️ 年日照时: {env_data.get('annual_sunshine_hours', 'N/A')} 小时")
    
    # 能源方案
    energy_plan = state.get("energy_plan")
    if energy_plan:
        print(f"\n🔋 储能容量: {energy_plan.get('storage_capacity', 'N/A')} kWh")
        print(f"☀️ 光伏装机: {energy_plan.get('pv_capacity', 'N/A')} kW")
        print(f"📄 长协比例: {energy_plan.get('ppa_ratio', 'N/A')}%")
    
    # 制冷方案
    cooling_plan = state.get("cooling_plan")
    if cooling_plan:
        print(f"\n❄️ 制冷技术: {cooling_plan.get('cooling_technology', 'N/A')}")
        print(f"📊 预计PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
    
    # 仿真结果
    sim_result = state.get("simulation_result")
    if sim_result:
        print(f"\n✅ 验证结果: {'通过' if sim_result.get('validation_passed') else '未通过'}")
        print(f"🌱 实际绿电占比: {sim_result.get('actual_green_ratio', 'N/A')}%")
    
    # 财务分析
    financial = state.get("financial_analysis")
    if financial:
        print(f"\n💰 投资回收期: {financial.get('payback_period', 'N/A')} 年")
        print(f"📈 内部收益率: {financial.get('irr', 'N/A')}%")
    
    print("\n" + "="*60)


def generate_final_report(state: GreenDataCenterState) -> str:
    """
    生成最终的规划设计建议书（Markdown格式）
    
    参数:
        state: 最终状态
    
    返回:
        Markdown格式的报告字符串
    """
    user_req = state.get("user_requirements", {})
    env_data = state.get("environmental_data", {})
    energy_plan = state.get("energy_plan", {})
    cooling_plan = state.get("cooling_plan", {})
    sim_result = state.get("simulation_result", {})
    financial = state.get("financial_analysis", {})
    
    report = f"""# 数据中心绿电消纳规划设计建议书

## 一、项目概述

| 项目参数 | 数值 |
|---------|------|
| 地理位置 | {user_req.get('location', 'N/A')} |
| 业务类型 | {user_req.get('business_type', 'N/A')} |
| 计划面积 | {user_req.get('planned_area', 'N/A')} 平方米 |
| 计划负荷 | {user_req.get('planned_load', 'N/A')} kW |
| 算力密度 | {user_req.get('computing_power_density', 'N/A')} kW/机柜 |
| 优先级 | {user_req.get('priority', 'N/A')} |
| 绿电目标 | {user_req.get('green_energy_target', 'N/A')}% |
| PUE目标 | {user_req.get('pue_target', 'N/A')} |

## 二、环境条件分析

| 环境参数 | 数值 |
|---------|------|
| 年均温度 | {env_data.get('annual_temperature', 'N/A')}°C |
| 年均风速 | {env_data.get('annual_wind_speed', 'N/A')} m/s |
| 年日照时长 | {env_data.get('annual_sunshine_hours', 'N/A')} 小时 |
| 碳排因子 | {env_data.get('carbon_emission_factor', 'N/A')} kgCO₂/kWh |

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
| 预计年均PUE | {cooling_plan.get('estimated_pue', 'N/A')} |
| 自然冷却小时数 | {cooling_plan.get('free_cooling_hours', 'N/A')} 小时/年 |

## 五、运行效果预测

| 性能指标 | 预测值 |
|---------|--------|
| 实际绿电占比 | {sim_result.get('actual_green_ratio', 'N/A')}% |
| 实际PUE | {sim_result.get('actual_pue', 'N/A')} |
| 年碳减排量 | {sim_result.get('carbon_reduction', 'N/A')} 吨CO₂ |

## 六、投资分析

| 财务指标 | 数值 |
|---------|------|
| 投资回收期 | {financial.get('payback_period', 'N/A')} 年 |
| 内部收益率(IRR) | {financial.get('irr', 'N/A')}% |
| 净现值(NPV) | {financial.get('npv', 'N/A')} 万元 |
| 平准化电力成本(LCOE) | {financial.get('lcoe', 'N/A')} 元/kWh |

---
*本报告由GreenDataCenter智能规划系统自动生成*
"""
    
    return report


# ============================================================
# 6. 导出接口
# ============================================================

__all__ = [
    # 状态类型
    "GreenDataCenterState",
    "DataCenterRequirements",
    "EnvironmentalData",
    "ElectricityPriceData",
    "LoadProfile",
    "EnergyPlan",
    "CoolingPlan",
    "SimulationResult",
    "FinancialAnalysis",
    
    # 函数
    "create_initial_state",
    "build_datacenter_workflow",
    "create_datacenter_agent_system",
    "should_continue_or_retry",
    "check_error",
    "print_state_summary",
    "generate_final_report",
]


# ============================================================
# 7. 主程序入口（完整工作流测试）
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("  GreenDataCenter LangGraph 多Agent工作流 - 完整测试")
    print("="*70)
    
    # ===== 1. 导入所有节点 =====
    print("\n📦 正在导入Agent节点...")
    try:
        from nodes.requirement_analysis_node import requirement_analysis_node
        print("  ✅ Agent 1: 需求与约束解析专家")
    except ImportError as e:
        print(f"  ❌ Agent 1 导入失败: {e}")
        requirement_analysis_node = None
    
    try:
        from nodes.energy_planner_node import energy_planner_node
        print("  ✅ Agent 2: 能源与绿电规划专家 (XSimple)")
    except ImportError as e:
        print(f"  ❌ Agent 2 导入失败: {e}")
        energy_planner_node = None
    
    try:
        from nodes.cooling_specialist_node import cooling_specialist_node
        print("  ✅ Agent 3: 暖通与制冷架构专家")
    except ImportError as e:
        print(f"  ❌ Agent 3 导入失败: {e}")
        cooling_specialist_node = None
    
    try:
        from nodes.simulator_node import simulator_node
        print("  ✅ Agent 4: 虚拟运行仿真专家")
    except ImportError as e:
        print(f"  ❌ Agent 4 导入失败: {e}")
        simulator_node = None
    
    try:
        from nodes.financial_consultant_node import financial_consultant_node
        print("  ✅ Agent 5: 综合评价与投资决策专家")
    except ImportError as e:
        print(f"  ❌ Agent 5 导入失败: {e}")
        financial_consultant_node = None
    
    # 检查所有节点是否可用
    all_nodes_available = all([
        requirement_analysis_node,
        energy_planner_node,
        cooling_specialist_node,
        simulator_node,
        financial_consultant_node
    ])
    
    if not all_nodes_available:
        print("\n❌ 部分Agent节点不可用，无法运行完整工作流")
        print("请检查 nodes/ 目录下的文件是否完整")
        exit(1)
    
    # ===== 2. 创建初始状态 =====
    print("\n" + "="*70)
    print("📋 创建初始状态 - 测试场景: 乌兰察布环保型数据中心")
    print("="*70)
    
    initial_state = create_initial_state(
        location="乌兰察布",
        business_type="大模型训练",
        planned_area=10000,           # 10000平方米
        planned_load=5000,            # 5000kW (5MW)
        computing_power_density=30,   # 30kW/机柜 (高密度，需要液冷)
        priority="环保型",
        green_energy_target=90,       # 90%绿电目标
        pue_target=1.2,               # PUE目标1.2
        budget_constraint=10000       # 1亿元预算
    )
    
    print("\n📊 初始状态参数:")
    print(f"  - 位置: {initial_state['user_requirements']['location']}")
    print(f"  - 业务类型: {initial_state['user_requirements']['business_type']}")
    print(f"  - 计划负荷: {initial_state['user_requirements']['planned_load']} kW")
    print(f"  - 算力密度: {initial_state['user_requirements']['computing_power_density']} kW/机柜")
    print(f"  - 绿电目标: {initial_state['user_requirements']['green_energy_target']}%")
    print(f"  - PUE目标: {initial_state['user_requirements']['pue_target']}")
    
    # ===== 3. 检查 LangGraph 是否可用 =====
    if not LANGGRAPH_AVAILABLE:
        print("\n⚠️ LangGraph 未安装，将手动逐节点执行")
        print("   如需使用完整工作流功能，请运行: pip install langgraph")
        
        # 手动逐节点执行
        print("\n" + "="*70)
        print("🚀 开始手动逐节点执行...")
        print("="*70)
        
        state = initial_state.copy()
        
        # Agent 1
        print("\n" + "-"*50)
        state = requirement_analysis_node(state)
        
        # Agent 2
        print("\n" + "-"*50)
        state = energy_planner_node(state)
        
        # Agent 3
        print("\n" + "-"*50)
        state = cooling_specialist_node(state)
        
        # Agent 4
        print("\n" + "-"*50)
        state = simulator_node(state)
        
        # 检查是否需要重试
        sim_result = state.get("simulation_result", {})
        if not sim_result.get("validation_passed", True):
            print("\n⚠️ 仿真验证未通过，在实际工作流中会返回重新调整")
            print("   问题: ", sim_result.get("validation_issues", []))
        
        # Agent 5
        print("\n" + "-"*50)
        state = financial_consultant_node(state)
        
    else:
        # ===== 4. 构建并执行 LangGraph 工作流 =====
        print("\n" + "="*70)
        print("🚀 构建 LangGraph 工作流...")
        print("="*70)
        
        try:
            app = create_datacenter_agent_system(
                agent1_node=requirement_analysis_node,
                agent2_node=energy_planner_node,
                agent3_node=cooling_specialist_node,
                agent4_node=simulator_node,
                agent5_node=financial_consultant_node
            )
            print("✅ 工作流构建成功")
            
            print("\n" + "="*70)
            print("▶️  执行工作流...")
            print("="*70)
            
            # 执行工作流
            state = app.invoke(initial_state)
            
        except Exception as e:
            print(f"❌ 工作流执行失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    # ===== 5. 打印最终结果 =====
    print("\n" + "="*70)
    print("📊 最终状态摘要")
    print("="*70)
    print_state_summary(state)
    
    # 打印 LLM 生成的能源方案报告
    energy_plan = state.get("energy_plan", {})
    llm_report = energy_plan.get("llm_report", "")
    if llm_report:
        print("\n" + "="*70)
        print("📝 Agent 2 (XSimple) 生成的能源规划报告:")
        print("="*70)
        print(llm_report)
    
    # 打印最终报告
    final_report = state.get("final_report", "")
    if final_report:
        print("\n" + "="*70)
        print("📋 最终规划设计建议书:")
        print("="*70)
        print(final_report)
    
    print("\n" + "="*70)
    print("✅ 工作流执行完成!")
    print("="*70)

