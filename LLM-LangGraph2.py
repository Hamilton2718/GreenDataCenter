import operator
import json
import os
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- 新增导入部分用于可视化 ---
from PIL import Image
import graphviz


# --- 新增导入结束 ---

# 1. 定义数据中心状态 (已根据“高校电力电子选题”的关注点修改因素)
class DataCenterState(TypedDict):
    # 输入参数
    predicted_green_energy_ratio: float  # 预测绿电占比 (0-1), 对应绿电出力预测
    current_datacenter_load_factor: float  # 当前数据中心负载率 (0-1), 对应负载预测与能效优化
    grid_carbon_intensity: float  # 电网碳排强度 (gCO2/kWh), 对应碳排放控制
    target_pue: float  # 数据中心PUE目标值, 对应能效评估
    energy_storage_soc_current_percent: float  # 当前储能系统荷电状态百分比 (0-100), 对应储能控制
    grid_stability_index: float  # 电网稳定性指数 (0-1, 1为稳定，0为不稳定), 对应绿电并网稳定性
    critical_workload_priority: float  # 关键工作负载优先级别 (0-1, 1为最高优先), 泛化了延迟要求

    # 逻辑计算得到的中间结果
    analysis_result: Dict
    migration_path: List[str]
    energy_storage_strategy: Dict
    green_energy_allocation: Dict

    # 大模型生成的见解与优化建议
    llm_insights: str

    # 最终输出
    final_plan: Dict
    messages: Annotated[List[BaseMessage], operator.add]


# 2. 初始化大模型 (使用 XSimple 提供的模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL
# 确保在运行前设置 DASHSCOPE_API_KEY 环境变量
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# --- 节点定义 ---

# 节点 1: 基础数据初步分析
def perform_initial_analysis(state: DataCenterState) -> DataCenterState:
    """对绿电占比、负载率、碳排强度、PUE目标等进行初步评估"""
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    carbon_intensity_grid = state["grid_carbon_intensity"]
    pue_target = state["target_pue"]
    soc = state["energy_storage_soc_current_percent"]
    grid_stability = state["grid_stability_index"]

    # 简化的综合评估逻辑
    green_availability_score = green_ratio * 100
    load_balance_score = (1 - abs(0.5 - load_factor)) * 100  # 负载越接近50%越均衡，分数越高
    carbon_impact_score = max(0, 100 - carbon_intensity_grid / 5)  # 碳排越低分数越高
    pue_efficiency_score = max(0, (2.0 - pue_target) * 50)  # PUE越接近1.0分数越高
    grid_resilience_score = grid_stability * 100

    analysis = {
        "green_availability_score": green_availability_score,
        "load_balance_score": load_balance_score,
        "carbon_impact_score": carbon_impact_score,
        "pue_efficiency_score": pue_efficiency_score,
        "grid_resilience_score": grid_resilience_score,
        "overall_score": (
                                     green_availability_score + load_balance_score + carbon_impact_score + pue_efficiency_score + grid_resilience_score) / 5,
        "urgent_action_needed": load_factor > 0.9 or green_ratio < 0.2 or grid_stability < 0.3  # 负载过高、绿电严重不足或电网不稳定
    }

    state["analysis_result"] = analysis
    state["messages"].append(
        AIMessage(content=f"✅ 数据中心初步分析完成: 综合评分 {analysis['overall_score']:.2f}")
    )
    return state


# 节点 2: 迁移决策 (结合硬逻辑，简化为是否需要外部资源)
def plan_migration_path(state: DataCenterState) -> DataCenterState:
    """根据负载率、绿电占比和电网稳定性规划是否需要外部资源或迁移"""
    load_factor = state["current_datacenter_load_factor"]
    green_ratio = state["predicted_green_energy_ratio"]
    grid_stability = state["grid_stability_index"]

    if load_factor > 0.85 and green_ratio < 0.3:
        path = ["Consider_External_Green_Cloud", "Migrate_Flexible_Workloads"]
        msg_content = "✅ 迁移路径规划完成: 负载高且绿电不足，建议考虑外部绿色云服务或迁移灵活工作负载。"
    elif grid_stability < 0.5:
        path = ["Prioritize_Local_Stability_Solutions_First"]
        msg_content = "✅ 迁移路径规划完成: 电网稳定性差，优先本地稳定性方案，谨慎对外迁移。"
    else:
        path = ["No_External_Migration_Needed"]
        msg_content = "✅ 迁移路径规划完成: 当前无需外部迁移，可在本地优化。"

    state["migration_path"] = path
    state["messages"].append(AIMessage(content=msg_content))
    return state


# 节点 3: 储能调度策略
def plan_energy_storage(state: DataCenterState) -> DataCenterState:
    """根据预测绿电、荷电状态和电网稳定性制定储能设备的充放电策略"""
    green_ratio = state["predicted_green_energy_ratio"]
    carbon_intensity_grid = state["grid_carbon_intensity"]
    soc = state["energy_storage_soc_current_percent"]
    grid_stability = state["grid_stability_index"]

    strategy = {
        "charge_periods": [],
        "discharge_periods": [],
        "capacity_utilization": {},
        "expected_grid_support": "None"
    }

    # 根据绿电、碳排、SOC和电网稳定性制定充放电策略 (示例逻辑)
    if green_ratio > 0.6 and soc < 80 and grid_stability > 0.7:  # 绿电充足，储能未满，电网稳定
        strategy["charge_periods"] = ["10:00-14:00 (光伏高峰)", "02:00-05:00 (风电高峰)"]
        strategy["discharge_periods"] = ["18:00-22:00 (用电高峰)"]
        strategy["capacity_utilization"] = {"green_energy_buffering": "60%", "peak_shaving": "40%"}
        strategy["expected_grid_support"] = "Primarily self-consumption, grid support during peak times."
    elif soc > 20 and grid_stability < 0.5:  # 储能有余量，电网不稳定
        strategy["charge_periods"] = []  # 优先放电稳定电网
        strategy["discharge_periods"] = ["Immediate (电网不稳定)", "18:00-22:00 (用电高峰)"]
        strategy["capacity_utilization"] = {"grid_stabilization": "70%", "emergency_backup": "30%"}
        strategy["expected_grid_support"] = "Active grid stabilization and frequency regulation."
    else:  # 绿电不足或储能告急
        strategy["charge_periods"] = ["00:00-06:00 (低谷电价/补电)"]
        strategy["discharge_periods"] = ["09:00-12:00 (高峰用电)", "18:00-21:00 (晚高峰)"]
        strategy["capacity_utilization"] = {"peak_shaving": "70%", "emergency_backup": "30%"}
        strategy["expected_grid_support"] = "Minimal, focused on local peak shaving."

    strategy[
        "estimated_carbon_avoidance_gCO2"] = f"{(100 - soc) * green_ratio * 0.5 * carbon_intensity_grid:.2f}"  # 模拟碳避免量

    state["energy_storage_strategy"] = strategy
    state["messages"].append(
        AIMessage(content="✅ 储能调度策略制定完成")
    )
    return state


# 节点 4: 绿电分配方案
def allocate_green_energy(state: DataCenterState) -> DataCenterState:
    """
    根据预测绿电占比、负载率和关键工作负载优先级制定绿电分配方案。
    """
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    critical_priority = state["critical_workload_priority"]

    allocation = {
        "priority_workloads": {},
        "flexible_workloads": {},
        "energy_storage_allocation": f"{min(0.2, green_ratio * 0.3) * 100:.1f}%",  # 固定一部分绿电给储能
        "total_green_usage_target": f"{green_ratio * 100:.1f}%",
        "recommendation": ""
    }

    # 考虑关键工作负载优先级
    # 假设关键负载的绿电分配上限与优先级成正比，且不超过绿电总量的某个比例
    critical_green_share = min(green_ratio * (0.4 + critical_priority * 0.3), 0.7)  # 关键负载最高占用70%绿电
    flexible_green_share = max(0.0, green_ratio - critical_green_share) - float(
        allocation["energy_storage_allocation"].replace('%', '')) / 100

    allocation["priority_workloads"] = {
        "AI_ML_training_critical": f"{critical_green_share * 0.6 * 100:.1f}%",
        "core_business_services": f"{critical_green_share * 0.4 * 100:.1f}%"
    }

    allocation["flexible_workloads"] = {
        "batch_processing": f"{max(0.0, flexible_green_share * 0.5) * 100:.1f}%",
        "data_analytics": f"{max(0.0, flexible_green_share * 0.5) * 100:.1f}%"
    }

    # 调度建议
    if green_ratio > 0.7 and load_factor < 0.6:
        allocation["recommendation"] = "绿电充足且负载适中，可优先分配给高价值和碳敏感型工作负载，考虑增加储能充电。"
    elif green_ratio > 0.4 and load_factor < 0.8:
        allocation["recommendation"] = "绿电适中，负载正常，维持当前分配策略，灵活调度以平衡绿电使用和成本。"
    else:
        allocation["recommendation"] = "绿电不足或负载过高，建议延迟非关键任务，或利用储能弥补绿电缺口，并积极寻找外部绿色算力支持。"

    state["green_energy_allocation"] = allocation
    state["messages"].append(
        AIMessage(content="✅ 绿电分配方案制定完成")
    )
    return state


# 节点 5: LLM 智能分析节点
def llm_reasoning_node(state: DataCenterState) -> DataCenterState:
    """
    利用大模型对上述所有技术参数进行综合评估并输出深度建议。
    修改后的提示词将融入“高校电力电子选题”的视角，更关注技术细节和优化潜力。
    """

    prompt = ChatPromptTemplate.from_template("""
    你是一位精通电力电子技术和绿色能源系统集成的资深专家，专门研究高校电力电子相关选题。
    根据以下数据中心实时运行数据和初步的调度结果，请提供一段专业的调度分析和优化建议（约250字）。
    请特别结合**电力电子技术**，深入分析和提出优化措施，覆盖以下关键领域：

    1.  **绿电高质量并网与功率质量**: 在当前预测绿电占比和电网稳定性下，如何确保绿电高效稳定接入？需要哪些电力电子变换器（如逆变器、APF）来保障数据中心内部的功率质量？
    2.  **储能系统最大化价值**: 基于当前储能SOC，如何通过电力电子DCDC/PCS控制优化储能系统的充放电策略，以实现削峰填谷、平抑绿电波动，并延长电池寿命？
    3.  **负载侧精细化能效与动态响应**: 结合当前负载率和PUE目标，电力电子在服务器电源、PDU等负载侧设备上如何进一步优化（如动态电压频率调节、多级转换架构），以降低损耗、提高能效，同时满足关键工作负载的优先级需求？
    4.  **整体系统协同优化**: 如何通过智能控制器将绿电、储能、负载和电网互动（Grid-Interactive）的电力电子子系统协同起来，实现数据中心整体PUE和碳排放的最低化？

    当前数据概览：
    - 预测绿电占比: {green_ratio}%
    - 当前数据中心负载率: {load_factor}%
    - 电网碳排强度: {carbon_intensity_grid} gCO2/kWh
    - 数据中心PUE目标: {pue_target}
    - 储能SOC: {soc_percent}%
    - 电网稳定性指数: {grid_stability_index} (1为稳定，0为不稳定)
    - 关键工作负载优先级: {critical_priority}
    - 预估迁移路径: {path}
    - 绿电分配方案: {allocation}
    - 储能策略: {storage_strategy}
    - 初步分析结果: {analysis}

    在你的建议中，请指出当前方案在哪些方面可以进一步利用电力电子技术进行优化，并提出可行的电力电子研究方向建议。
    """)

    chain = prompt | llm
    response = chain.invoke({
        "green_ratio": f"{state['predicted_green_energy_ratio'] * 100:.1f}",
        "load_factor": f"{state['current_datacenter_load_factor'] * 100:.1f}",
        "carbon_intensity_grid": f"{state['grid_carbon_intensity']}",
        "pue_target": f"{state['target_pue']:.2f}",
        "soc_percent": f"{state['energy_storage_soc_current_percent']:.1f}",
        "grid_stability_index": f"{state['grid_stability_index']:.2f}",
        "critical_priority": f"{state['critical_workload_priority']:.2f}",
        "path": state["migration_path"],
        "allocation": json.dumps(state["green_energy_allocation"], ensure_ascii=False, indent=2),
        "storage_strategy": json.dumps(state["energy_storage_strategy"], ensure_ascii=False, indent=2),
        "analysis": json.dumps(state["analysis_result"], ensure_ascii=False, indent=2)
    })

    state["llm_insights"] = response.content
    state["messages"].append(AIMessage(content=f"✅ 大模型智能分析完成"))
    state["messages"].append(AIMessage(content=f"大模型洞察 (电力电子优化视角): {response.content}"))
    return state


# 节点 6: 生成最终方案
def generate_final_plan(state: DataCenterState) -> DataCenterState:
    """整合所有策略和建议，生成最终调度方案"""

    final_plan = {
        "timestamp": "2024-XX-XX HH:MM:SS",  # 实际应用中替换为当前时间
        "datacenter_status_overview": {
            "predicted_green_energy_ratio": f"{state['predicted_green_energy_ratio'] * 100:.1f}%",
            "current_load_factor": f"{state['current_datacenter_load_factor'] * 100:.1f}%",
            "grid_carbon_intensity": f"{state['grid_carbon_intensity']} gCO2/kWh",
            "target_PUE": f"{state['target_pue']:.2f}",
            "energy_storage_SOC": f"{state['energy_storage_soc_current_percent']:.1f}%",
            "grid_stability": f"{state['grid_stability_index']:.2f}"
        },
        "migration_or_external_resource_plan": state["migration_path"],
        "energy_storage_strategy": state["energy_storage_strategy"],
        "green_energy_allocation_detail": state["green_energy_allocation"],  # 绿电分配方案
        "compliance_and_impact": {
            "pue_target_consideration": f"Current PUE target {state['target_pue']:.2f} needs continuous monitoring.",
            "carbon_reduction_focus": "Emphasis on maximizing green energy use and efficient storage."
        },
        "expert_advice_from_power_electronics_perspective": state["llm_insights"]  # 包含LLM的深度建议
    }

    state["final_plan"] = final_plan
    state["messages"].append(
        AIMessage(content="✅ 完整调度方案生成成功")
    )
    return state


# 构建LangGraph工作流
def create_scheduling_graph():
    workflow = StateGraph(DataCenterState)

    # 添加节点
    workflow.add_node("analyze", perform_initial_analysis)
    workflow.add_node("migrate", plan_migration_path)
    workflow.add_node("storage", plan_energy_storage)
    workflow.add_node("allocate_green", allocate_green_energy)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    workflow.add_node("final_plan", generate_final_plan)

    # 设置入口点
    workflow.set_entry_point("analyze")

    # 定义边和条件路由
    workflow.add_edge("analyze", "migrate")
    workflow.add_edge("migrate", "storage")
    workflow.add_edge("storage", "allocate_green")
    workflow.add_edge("allocate_green", "llm_reasoning")
    workflow.add_edge("llm_reasoning", "final_plan")
    workflow.add_edge("final_plan", END)

    # 编译图
    app = workflow.compile()
    return app


# --- 执行入口 ---
def main():
    app = create_scheduling_graph()

    # === 直接生成并显示流程图（已修正）===
    # graph_obj = app.get_graphviz()
    # graph_image_path = "datacenter_workflow.png"
    # graph_obj.render(filename="datacenter_workflow", format="png", view=False)
    #
    # # 使用 Pillow 库在 IDE 环境中显示图片
    # Image.open(graph_image_path).show()
    # print(f"\n✅ 流程图已生成并显示: {graph_image_path}")
    # === 显示结束 ===

    # >>>>>>>>>>>>>>>>> 关键输入数据: 根据高校电力电子选题视角输入相关数据 <<<<<<<<<<<<<<<<<
    initial_state = {
        "predicted_green_energy_ratio": 0.70,  # 预测绿电占比 70%
        "current_datacenter_load_factor": 0.60,  # 当前数据中心负载率 60%
        "grid_carbon_intensity": 550.0,  # 电网碳排强度 550 gCO2/kWh
        "target_pue": 1.25,  # 数据中心PUE目标 1.25
        "energy_storage_soc_current_percent": 75.0,  # 当前储能SOC 75%
        "grid_stability_index": 0.85,  # 电网稳定性指数 0.85 (较稳定)
        "critical_workload_priority": 0.9,  # 关键工作负载优先级 0.9 (高优先)
        "messages": [HumanMessage(content="启动数据中心绿色调度流程 (电力电子优化视角)")]
    }

    print("\n" + "=" * 60)
    print(" XSimple 绿色算力调度流程启动 (电力电子优化视角) ")
    print("=" * 60)
    print("输入参数:")
    for key, value in initial_state.items():
        if key != "messages":
            print(f"  - {key}: {value}")
    print("=" * 60)

    # 执行工作流
    result = app.invoke(initial_state)

    # 输出结果
    print("\n[最终绿电分配方案及专家建议]")
    if "final_plan" in result and "green_energy_allocation_detail" in result["final_plan"]:
        print("绿电分配细则:")
        print(json.dumps(result["final_plan"]["green_energy_allocation_detail"], indent=2, ensure_ascii=False))

    if "final_plan" in result and "expert_advice_from_power_electronics_perspective" in result["final_plan"]:
        print("\n专家LLM建议 (侧重电力电子优化):")
        print(result["final_plan"]["expert_advice_from_power_electronics_perspective"])

    print("\n[关键执行日志]")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and "大模型洞察 (电力电子优化视角)" not in msg.content:
            print(f" - {msg.content}")


if __name__ == "__main__":
    main()