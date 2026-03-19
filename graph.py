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
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


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


class ReviewResult(TypedDict, total=False):
    """审核评估结果数据结构"""
    evaluation_text: str                   # LLM 生成的完整评估报告
    passed: bool                           # 是否通过审核
    score: float                           # 综合评分（1-5 分）
    evaluator: str                         # 使用的评估模型（如 DeepSeek）
    issues: List[str]                      # 问题列表
    suggestions: List[str]                 # 改进建议


class SimulationResult(TypedDict, total=False):
    """24小时粗仿真结果数据结构"""
    time_labels: List[str]                 # 时间标签（24小时）
    it_load_curve_mw: List[float]          # IT 负载曲线（MW）
    green_supply_curve_mw: List[float]     # 绿电供应曲线（MW）
    storage_power_curve_mw: List[float]    # 储能充放曲线（MW，放电为正，充电为负）
    pv_curve_mw: List[float]               # 光伏出力曲线（MW）
    ppa_curve_mw: List[float]              # 长协绿电曲线（MW）
    soc_curve: List[float]                 # 储能SOC曲线（0-1）
    summary: Dict[str, Any]                # 仿真汇总指标


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
    LangGraph 统一状态定义
    
    这是整个系统的核心状态类型，所有 Agent 节点都使用这个状态进行数据交换。
    状态在节点之间传递，每个节点可以读取和更新状态中的字段。
    
    字段说明:
        - user_requirements: 用户输入的需求（Agent 1 填充）
        - environmental_data: 环境数据（Agent 1 填充）
        - electricity_price: 电价数据（Agent 1 填充）
        - load_profile: 负荷特性（Agent 1 或 Agent 2 填充）
        - energy_plan: 能源规划方案（Agent 2 填充）
        - cooling_plan: 制冷方案（Agent 3 填充）
        - review_result: 审核评估结果（Agent 4 填充）
        - financial_analysis: 财务分析（Agent 5 填充）
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

    # ===== Agent 4: 24小时粗仿真专家 =====
    simulation_result: Optional[SimulationResult]
    
    # ===== Agent 4: 方案审核与评估专家 =====
    review_result: Optional[ReviewResult]
    feedback: Optional[Dict[str, Any]]     # 反馈意见（用于重新优化）
    
    # ===== Agent 5: 综合评价与投资决策专家 =====
    financial_analysis: Optional[FinancialAnalysis]
    
    # ===== 系统控制字段 =====
    iteration_count: int                   # 迭代计数器
    max_iterations: int                    # 最大迭代次数
    error_message: Optional[str]           # 错误信息
    final_report: Optional[str]            # 最终报告（Markdown 格式）


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
    根据审核结果决定流程走向
    
    参数:
        state: 当前系统状态
        
    返回:
        "continue": 审核通过，继续到最终报告
        "retry": 审核不通过且未超过最大迭代次数，返回重新优化
        "end": 审核不通过且超过最大迭代次数，结束流程
    """
    review_result = state.get("review_result")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # 如果没有审核结果，直接继续
    if review_result is None:
        print("⚠️ 无审核结果，默认继续")
        return "continue"
    
    # 检查是否通过审核
    passed = review_result.get("passed", True)
    
    if passed:
        print("✅ 方案审核通过，继续到最终报告")
        return "continue"
    
    # 审核不通过，检查是否超过最大迭代次数
    if iteration_count >= max_iterations:
        print(f"⚠️ 已达到最大迭代次数 ({max_iterations})，结束流程")
        return "end"
    
    print(f"🔄 方案审核不通过，返回重新优化（第{iteration_count}次迭代）")
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
    requirement_analysis_node,
    energy_planner_node,
    cooling_specialist_node,
    simulation_node,
    review_node, 
    financial_consultant_node,
    final_report_node
) -> StateGraph:
    """
    构建数据中心规划工作流图
    
    工作流说明:
        1. Agent 1 (需求解析) → 
        2. Agent 2 (能源规划) → 
        3. Agent 3 (制冷设计) → 
        4. Agent 4 (24小时粗仿真) →
        5. Agent 5 (财务分析) →
        6. Agent 6 (审核评估) → [条件分支]
        - 通过 → Agent 7 (报告生成) → END
        - 达到最大重试次数 → Agent 7 (报告生成) → END
        - 不通过 → 返回 Agent 1/2/3 重新优化（带反馈意见）
    
    流程图:
        START → requirement_analysis
                     ↓
                energy_planning
                     ↓
               cooling_design
                     ↓
                                 simulation
                     ↓
                                financial_analysis
                                         ↓
                                     review_node
                 /          \
         (通过) /            \ (不通过，需要重试)
             ↓              ↓
             final_report    ↘ (反馈给前 3 个节点)
             ↓              ↗
            END ←──────────┘
    """
    # 创建工作流图
    workflow = StateGraph(GreenDataCenterState)
    
    # 添加节点
    workflow.add_node("requirement_analysis", requirement_analysis_node)
    workflow.add_node("energy_planning", energy_planner_node)
    workflow.add_node("cooling_design", cooling_specialist_node)
    workflow.add_node("simulation", simulation_node)
    workflow.add_node("financial_analysis", financial_consultant_node)
    workflow.add_node("review", review_node)  # 审核节点
    workflow.add_node("final_report", final_report_node)
    
    # 设置入口点
    workflow.set_entry_point("requirement_analysis")
    
    # 添加边 - 顺序执行前 3 个节点
    workflow.add_edge("requirement_analysis", "energy_planning")
    workflow.add_edge("energy_planning", "cooling_design")
    workflow.add_edge("cooling_design", "simulation")
    workflow.add_edge("simulation", "financial_analysis")
    workflow.add_edge("financial_analysis", "review")
    
    # 添加条件边 - 根据审核结果决定流程走向
    workflow.add_conditional_edges(
        "review",
        should_continue_or_retry,  # 条件判断函数
        {
            "continue": "final_report",  # 通过 → 最终报告
            "retry": "requirement_analysis",   # 不通过 → 返回重新优化
            "end": "final_report"  # 达到最大迭代次数 → 生成报告后结束
        }
    )

    workflow.add_edge("final_report", END)
    
    return workflow


def create_datacenter_agent_system(
    requirement_analysis_node,
    energy_planner_node,
    cooling_specialist_node,
    simulation_node,
    review_node, 
    financial_consultant_node,
    final_report_node,
    checkpoint_dir: Optional[str] = None
) -> Any:
    """
    创建完整的数据中心 Agent 系统
    
    参数:
        requirement_analysis_node: 需求与约束解析专家节点函数
        energy_planner_node: 能源与绿电规划专家节点函数
        cooling_specialist_node: 暖通与制冷架构专家节点函数
        simulation_node: 24小时粗仿真节点函数
        review_node: 方案审核与评估专家节点函数
        financial_consultant_node: 综合评价与投资决策专家节点函数
        final_report_node: 最终报告生成节点函数
        checkpoint_dir: 检查点保存目录（可选）
    
    返回:
        编译后的可执行图
    """
    # 构建工作流
    workflow = build_datacenter_workflow(
        requirement_analysis_node=requirement_analysis_node,
        energy_planner_node=energy_planner_node,
        cooling_specialist_node=cooling_specialist_node,
        simulation_node=simulation_node,
        review_node=review_node,
        financial_consultant_node=financial_consultant_node,
        final_report_node=final_report_node
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
    
    # 财务分析
    financial = state.get("financial_analysis")
    if financial:
        print(f"\n💰 投资回收期: {financial.get('payback_period', 'N/A')} 年")
        print(f"📈 内部收益率: {financial.get('irr', 'N/A')}%")

    # 仿真结果
    simulation = state.get("simulation_result")
    if simulation and simulation.get("summary"):
        summary = simulation.get("summary", {})
        print(f"\n📈 日IT电量: {summary.get('daily_it_energy_mwh', 'N/A')} MWh")
        print(f"🌿 日绿电供给: {summary.get('daily_green_supply_mwh', 'N/A')} MWh")
        print(f"🔋 日绿电占比: {summary.get('daily_green_ratio_pct', 'N/A')}%")
    
    print("\n" + "="*60)


# 6. 主程序入口（完整工作流测试）
def save_workflow_graph(app, output_path: str = "output/workflow_graph.png") -> bool:
    """
    保存 LangGraph 工作流图为 PNG 图片
    
    参数:
        app: 编译后的 LangGraph 应用
        output_path: 输出图片路径
    
    返回:
        bool, 成功返回 True
    """
    import os
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 方法1: 使用 draw_mermaid_png (LangGraph 0.1+)
        graph = app.get_graph()
        png_data = graph.draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        return True
    except Exception as e1:
        try:
            # 方法2: 使用 draw_png (需要 graphviz)
            graph = app.get_graph()
            graph.draw_png(output_path)
            return True
        except Exception as e2:
            print(f"⚠️ 无法生成流程图: {e1} / {e2}")
            return False


def save_report_as_markdown(state: dict, output_path: str = "output/final_report.md") -> bool:
    """
    将最终报告保存为 Markdown 文件（在 graph.py 中统一生成）
    
    参数:
        state: 最终状态
        output_path: 输出文件路径
    
    返回:
        bool, 成功返回 True
    """
    import os
    from datetime import datetime
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 优先使用最终报告节点生成的内容
    final_report = state.get("final_report")
    if final_report:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            return True
        except Exception as e:
            print(f"⚠️ 无法保存报告：{e}")
            return False

    # 兼容逻辑：若未经过最终报告节点，仍可回退到本地拼装
    # 获取各部分数据（容错处理）
    user_reqs = state.get("user_requirements", {}) or {}
    env_data = state.get("environmental_data", {}) or {}
    energy_plan = state.get("energy_plan", {}) or {}
    cooling_plan = state.get("cooling_plan", {}) or {}
    review_result = state.get("review_result", {}) or {}
    financial = state.get("financial_analysis", {}) or {}
    
    # 获取 Agent 2 的 LLM 报告
    llm_report = energy_plan.get("llm_report", "") if energy_plan else ""

    def _txt(value, default):
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return value

    def _num(value, default=0.0, digits=2):
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return round(float(default), digits)
    
    # 构建完整报告
    full_report = f"""# GreenDataCenter 规划设计报告

> 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 一、项目基本信息

| 项目 | 数值 |
|------|------|
| 地理位置 | {_txt(user_reqs.get('location') if user_reqs else None, '未提供')} |
| 业务类型 | {_txt(user_reqs.get('business_type') if user_reqs else None, '通用计算型')} |
| 计划负荷 | {_num(user_reqs.get('planned_load') if user_reqs else None, 0.0, 1)} kW |
| 算力密度 | {_num(user_reqs.get('computing_power_density') if user_reqs else None, 0.0, 1)} kW/机柜 |
| PUE 目标 | {_num(user_reqs.get('pue_target') if user_reqs else None, 1.2, 3)} |
| 绿电目标 | {_num(user_reqs.get('green_energy_target') if user_reqs else None, 0.0, 1)}% |

---

## 二、能源规划方案 (Agent 2 - XSimple)

{llm_report if llm_report else "未生成详细能源规划报告"}

---

## 三、制冷方案 (Agent 3)

| 指标 | 数值 |
|------|------|
| 制冷技术 | {_txt(cooling_plan.get('cooling_technology') if cooling_plan else None, '风冷')} |
| 预计 PUE | {_num(cooling_plan.get('estimated_pue') if cooling_plan else None, 1.35, 3)} |
| 预计 WUE | {_num(cooling_plan.get('predicted_wue') if cooling_plan else None, 1.8, 3)} |

---

## 四、审核评估结果 (Agent 4)

**审核结论**: {"✅ 通过" if review_result and review_result.get('passed') else "❌ 不通过"}  
**综合评分**: {_num(review_result.get('score') if review_result else None, 0.0, 2)}/5  
**评估模型**: {_txt(review_result.get('evaluator') if review_result else None, '系统规则评估')}

---

## 五、财务分析 (Agent 5)

"""
    
    # 如果财务分析存在，添加详细内容
    if financial:
        full_report += f"""### 投资估算

| 项目 | 金额（万元） |
|------|-------------|
| 总投资 (CAPEX) | {_num(financial.get('capex_total'), 0.0, 2)} |
| 年节省费用 | {_num(financial.get('annual_saving'), 0.0, 2)} |
| 投资回收期 | {_num(financial.get('payback_years'), 30.0, 2)} 年 |

### 成本明细

- 电网购电成本：{_num(financial.get('grid_cost'), 0.0, 2)} 万元/年
- PPA 购电成本：{_num(financial.get('ppa_cost'), 0.0, 2)} 万元/年
- 光伏自用节省：-{_num(financial.get('pv_saving'), 0.0, 2)} 万元/年
- 碳减排收益：-{_num(financial.get('carbon_benefit'), 0.0, 2)} 万元/年
- **净总用电成本**: **{_num(financial.get('total_cost'), 0.0, 2)}** 万元/年

### 碳减排贡献

- **年碳减排量**: {_num(financial.get('emission_reduction'), 0.0, 2)} 吨 CO₂/年
- **全生命周期减排**: {_num(financial.get('lifetime_reduction'), 0.0, 2)} 吨 CO₂

---

## 六、综合评价

根据审核结果，该数据中心规划方案：
- {"✅ 绿电消纳达标" if review_result and review_result.get('passed') else "⚠️ 需要进一步优化"}
- ✅ 制冷技术选择合理
- ✅ 财务指标良好，投资回收期 {_num(financial.get('payback_years'), 30.0, 2)} 年

"""
    else:
        full_report += """*注：由于方案未通过审核，财务分析未执行。*

---

## 六、综合评价

该方案在审核阶段未通过，需要进一步优化能源配置和制冷方案后重新提交审核。

"""
    
    full_report += """---

*本报告由 GreenDataCenter 智能规划系统自动生成*
"""
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        return True
    except Exception as e:
        print(f"⚠️ 无法保存报告：{e}")
        return False


if __name__ == "__main__":
    import os
    
    # ===== 1. 导入所有节点 =====
    from nodes.requirement_analysis_node import requirement_analysis_node
    from nodes.energy_planner_node import energy_planner_node
    from nodes.cooling_specialist_node import cooling_specialist_node
    from nodes.simulation_node import simulation_node
    from nodes.review_node import review_node 
    from nodes.financial_consultant_node import financial_consultant_node
    from nodes.final_report_node import final_report_node
    
    # ===== 2. 创建初始状态 =====
    initial_state = create_initial_state(
        location="乌兰察布",
        business_type="大模型训练",
        planned_area=10000,
        planned_load=5000,
        computing_power_density=30,
        priority="环保型",
        green_energy_target=90,
        pue_target=1.2,
        budget_constraint=10000
    )
    
    # ===== 3. 构建工作流 =====
    app = create_datacenter_agent_system(
        requirement_analysis_node=requirement_analysis_node,
        energy_planner_node=energy_planner_node,
        cooling_specialist_node=cooling_specialist_node,
        simulation_node=simulation_node,
        review_node=review_node,
        financial_consultant_node=financial_consultant_node,
        final_report_node=final_report_node
    )
    
    # ===== 4. 生成工作流流程图 =====
    graph_path = "output/workflow_graph.png"
    if save_workflow_graph(app, graph_path):
        print(f"✅ 工作流流程图已保存：{os.path.abspath(graph_path)}")
    
    # ===== 5. 执行工作流 =====
    state = app.invoke(initial_state)
    
    # ===== 6. 生成最终报告（在 graph.py 中统一生成）=====
    report_path = "output/final_report.md"
    if save_report_as_markdown(state, report_path):
        print(f"✅ 最终报告已保存：{os.path.abspath(report_path)}")

    print("✅ 工作流执行完成!")
