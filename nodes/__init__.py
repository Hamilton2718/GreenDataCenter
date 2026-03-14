"""
GreenDataCenter Agents Package

这个包包含了所有用于绿色数据中心规划的专家 agent nodes。
每个 agent 都遵循 LangGraph node 的标准接口设计。

参考 AWS LangGraph 多 Agent 架构模式：
https://github.com/aws-samples/langgraph-multi-agent

Available Nodes:
    - requirement_analysis_node: 需求与约束解析专家 (Agent 1)
    - energy_planner_node: 能源与绿电规划专家 (Agent 2)
    - cooling_specialist_node: 暖通与制冷架构专家 (Agent 3)
    - review_node: 方案审核与评估专家 (Agent 4)  # 新增
    - financial_consultant_node: 综合评价与投资决策专家 (Agent 5)
    - final_report_node: 最终报告生成节点 (Agent 6)

Usage Example:
    from graph import create_datacenter_agent_system, create_initial_state
    from nodes import (
        requirement_analysis_node,
        energy_planner_node,
        cooling_specialist_node,
        review_node,  # 新增
        financial_consultant_node,
        final_report_node
    )
    
    # 创建 Agent 系统
    app = create_datacenter_agent_system(
        requirement_analysis_node=requirement_analysis_node,
        energy_planner_node=energy_planner_node,
        cooling_specialist_node=cooling_specialist_node,
        review_node=review_node,  # 新增
        financial_consultant_node=financial_consultant_node,
        final_report_node=final_report_node
    )
    
    # 创建初始状态并执行
    initial_state = create_initial_state(location="乌兰察布", ...)
    result = app.invoke(initial_state)
"""

# ============================================================
# 从 graph.py 导入统一状态类型
# ============================================================

# 统一状态类型在 graph.py 中定义，避免重复定义
from graph import (
    GreenDataCenterState,
    DataCenterRequirements,
    EnvironmentalData,
    ElectricityPriceData,
    LoadProfile,
    EnergyPlan,
    CoolingPlan,
    ReviewResult,  # 替换 SimulationResult
    FinancialAnalysis,
)

# ============================================================
# 导入各个 Agent 节点函数
# ============================================================

# Agent 1: 需求与约束解析专家 (已实现)
from nodes.requirement_analysis_node import requirement_analysis_node

# Agent 2-5: 其他专家节点 (占位导入，将在后续实现)
# 这些导入使用 try-except 包裹，以便在文件不存在时不会报错
try:
    from nodes.energy_planner_node import energy_planner_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def energy_planner_node(state):
        """占位函数：能源与绿电规划专家"""
        print("⚠️ [Agent 2] 能源与绿电规划专家 - 占位实现")
        # 返回原状态，保持流程继续
        return state

try:
    from nodes.cooling_specialist_node import cooling_specialist_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def cooling_specialist_node(state):
        """占位函数：暖通与制冷架构专家"""
        print("⚠️ [Agent 3] 暖通与制冷架构专家 - 占位实现")
        # 返回原状态，保持流程继续
        return state

try:
    from nodes.review_node import review_node  # 新增：审核节点
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def review_node(state):
        """占位函数：方案审核与评估专家"""
        print("⚠️ [Agent 4] 方案审核与评估专家 - 占位实现")
        # 默认审核通过
        state["review_result"] = {
            "passed": True,
            "score": 5.0,
            "evaluator": "Fallback"
        }
        return state

try:
    from nodes.financial_consultant_node import financial_consultant_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def financial_consultant_node(state):
        """占位函数：综合评价与投资决策专家"""
        print("⚠️ [Agent 5] 综合评价与投资决策专家 - 占位实现")
        # 生成简单报告
        state["final_report"] = "# 最终报告\n（占位内容）"
        return state

try:
    from nodes.final_report_node import final_report_node
except ImportError:
    def final_report_node(state):
        """占位函数：最终报告生成节点"""
        print("⚠️ [Agent 6] 最终报告生成节点 - 占位实现")
        state["final_report"] = state.get("final_report", "# 最终报告\n（占位内容）")
        return state


# ============================================================
# 导出接口
# ============================================================

__all__ = [
    # 统一状态类型 (从 graph.py 导入)
    'GreenDataCenterState',
    'DataCenterRequirements',
    'EnvironmentalData',
    'ElectricityPriceData',
    'LoadProfile',
    'EnergyPlan',
    'CoolingPlan',
    'ReviewResult',  # 替换 SimulationResult
    'FinancialAnalysis',
    
    # Agent 节点函数
    'requirement_analysis_node',    # Agent 1
    'energy_planner_node',          # Agent 2
    'cooling_specialist_node',      # Agent 3
    'review_node',                  # Agent 4 (新增)
    'financial_consultant_node',    # Agent 5
    'final_report_node',            # Agent 6
]
