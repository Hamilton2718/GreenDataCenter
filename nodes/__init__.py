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
    - simulator_node: 虚拟运行仿真专家 (Agent 4)
    - financial_consultant_node: 综合评价与投资决策专家 (Agent 5)

Usage Example:
    from graph import create_datacenter_agent_system, create_initial_state
    from nodes import (
        requirement_analysis_node,
        energy_planner_node,
        cooling_specialist_node,
        simulator_node,
        financial_consultant_node
    )
    
    # 创建Agent系统
    app = create_datacenter_agent_system(
        agent1_node=requirement_analysis_node,
        agent2_node=energy_planner_node,
        agent3_node=cooling_specialist_node,
        agent4_node=simulator_node,
        agent5_node=financial_consultant_node
    )
    
    # 创建初始状态并执行
    initial_state = create_initial_state(location="乌兰察布", ...)
    result = app.invoke(initial_state)
"""

# ============================================================
# 从graph.py导入统一状态类型
# ============================================================

# 统一状态类型在graph.py中定义，避免重复定义
from graph import (
    GreenDataCenterState,
    DataCenterRequirements,
    EnvironmentalData,
    ElectricityPriceData,
    LoadProfile,
    EnergyPlan,
    CoolingPlan,
    SimulationResult,
    FinancialAnalysis,
)

# ============================================================
# 导入各个Agent节点函数
# ============================================================

# Agent 1: 需求与约束解析专家 (已实现)
from nodes.requirement_analysis_node import requirement_analysis_node

# Agent 2-5: 其他专家节点 (占位导入，将在后续实现)
# 这些导入使用try-except包裹，以便在文件不存在时不会报错
try:
    from nodes.energy_planner_node import energy_planner_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def energy_planner_node(state):
        """占位函数: 能源与绿电规划专家"""
        print("⚠️ [Agent 2] 能源与绿电规划专家 - 占位实现")
        # 返回原状态，保持流程继续
        return state

try:
    from nodes.cooling_specialist_node import cooling_specialist_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def cooling_specialist_node(state):
        """占位函数: 暖通与制冷架构专家"""
        print("⚠️ [Agent 3] 暖通与制冷架构专家 - 占位实现")
        # 返回原状态，保持流程继续
        return state

try:
    from nodes.simulator_node import simulator_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def simulator_node(state):
        """占位函数: 虚拟运行仿真专家"""
        print("⚠️ [Agent 4] 虚拟运行仿真专家 - 占位实现")
        # 设置默认验证通过，以便流程可以继续
        state["simulation_result"] = {
            "validation_passed": True,
            "validation_issues": []
        }
        return state

try:
    from nodes.financial_consultant_node import financial_consultant_node
except ImportError:
    # 如果文件不存在，提供一个占位函数
    def financial_consultant_node(state):
        """占位函数: 综合评价与投资决策专家"""
        print("⚠️ [Agent 5] 综合评价与投资决策专家 - 占位实现")
        # 生成简单报告
        from graph import generate_final_report
        state["final_report"] = generate_final_report(state)
        return state


# ============================================================
# 导出接口
# ============================================================

__all__ = [
    # 统一状态类型 (从graph.py导入)
    'GreenDataCenterState',
    'DataCenterRequirements',
    'EnvironmentalData',
    'ElectricityPriceData',
    'LoadProfile',
    'EnergyPlan',
    'CoolingPlan',
    'SimulationResult',
    'FinancialAnalysis',
    
    # Agent节点函数
    'requirement_analysis_node',    # Agent 1
    'energy_planner_node',          # Agent 2
    'cooling_specialist_node',      # Agent 3
    'simulator_node',               # Agent 4
    'financial_consultant_node',    # Agent 5
]


# ============================================================
# 节点执行顺序说明
# ============================================================

"""
LangGraph工作流节点执行顺序:

1. requirement_analysis_node (需求与约束解析专家 - Agent 1)
   - 接收用户输入
   - 获取地理位置、气候、电价、碳排放因子等数据
   - 输出: user_requirements, environmental_data, electricity_price
   - 文件: nodes/requirement_analysis_node.py

2. energy_planner_node (能源与绿电规划专家 - Agent 2)
   - 由 XSimple 开发，使用 LLM (通义千问) 生成方案
   - 调用 Electricity Maps API 获取实时电网碳数据
   - 输出: energy_plan (含 llm_report 完整报告)
   - 文件: nodes/energy_planner_node.py (XSimple 版)

3. cooling_specialist_node (暖通与制冷架构专家 - Agent 3)
   - 基于地理位置、算力密度、PUE目标
   - 选择最优制冷技术
   - 输出: cooling_plan
   - 文件: nodes/cooling_specialist_node.py (待实现)

4. simulator_node (虚拟运行仿真专家 - Agent 4)
   - 模拟24小时/全年运行
   - 验证绿电消纳和PUE是否达标
   - 输出: simulation_result
   - 文件: nodes/simulator_node.py (待实现)

5. financial_consultant_node (综合评价与投资决策专家 - Agent 5)
   - 计算CAPEX和OPEX
   - 分析投资回报
   - 输出: financial_analysis, final_report
   - 文件: nodes/financial_consultant_node.py (待实现)

条件跳转 (由graph.py中的should_continue_or_retry函数控制):
- 如果simulator_node验证失败（PUE超标或绿电消纳不足）
  且未达到最大迭代次数，则返回到requirement_analysis_node重新调整参数
- 如果验证通过，继续到financial_consultant_node
- 如果超过最大迭代次数仍未通过验证，结束流程

状态传递说明:
- 所有节点共享同一个GreenDataCenterState状态对象
- 每个节点读取需要的数据，更新自己负责的字段
- 状态在节点间自动传递，无需手动管理
"""
