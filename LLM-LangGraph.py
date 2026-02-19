      
import operator
import json
import os
from typing import TypedDict, Annotated, List, Dict, Optional
import pandas as pd
from pycaret.time_series import load_model, predict_model
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

    # 人工反馈循环
    human_feedback: str  # 人工反馈意见
    human_approved: bool # 是否通过人工审核

    # 新增：用于存储数据中心负载预测结果
    load_prediction_results: Optional[pd.DataFrame]
    # 修改：用于存储风光出力预测结果
    renewable_prediction_results: Optional[pd.DataFrame]


# 2. 初始化大模型 
# 请替换为实际可用的 API Key 和 Base URL
# 确保在运行前设置 DASHSCOPE_API_KEY 环境变量
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 3. 定义 LangGraph 的节点

# 节点 1: 初步分析节点
def perform_initial_analysis(state: DataCenterState) -> DataCenterState:
    """模拟对数据中心当前状态的初步分析"""
    print("--- 正在进行初步分析 ---")
    # 实际场景中，这里会调用API或查询数据库
    state["predicted_green_energy_ratio"] = 0.65  # 预测绿电占比
    state["current_datacenter_load_factor"] = 0.80  # 当前数据中心负载率
    state["grid_carbon_intensity"] = 500  # 电网碳排强度 gCO2/kWh
    state["target_pue"] = 1.25  # PUE目标
    state["energy_storage_soc_current_percent"] = 0.60  # 储能SOC
    state["grid_stability_index"] = 0.95  # 电网稳定性指数
    state["critical_workload_priority"] = 0.9  # 关键工作负载优先级

    # 将分析结果存入消息列表
    state["messages"].append(
        AIMessage(content="✅ 初步分析完成，数据中心状态已获取。")
    )
    return state


# 节点 2: 迁移路径规划节点
def plan_migration_path(state: DataCenterState) -> DataCenterState:
    """根据数据中心状态，规划工作负载迁移或资源调度路径"""
    print("--- 正在规划迁移或资源路径 ---")
    # 这是一个简化的决策逻辑
    if state["grid_carbon_intensity"] > 450 and state["predicted_green_energy_ratio"] < 0.7:
        path = "建议将部分非核心计算任务迁移至其他可用区或云端，以降低本地碳排放。"
    elif state["current_datacenter_load_factor"] > 0.85:
        path = "数据中心负载较高，建议启动备用服务器资源或优化现有资源分配。"
    else:
        path = "当前状态稳定，无需进行大规模迁移。"

    state["migration_path"] = path
    state["messages"].append(
        AIMessage(content=f"✅ 迁移/资源路径规划完成: {path}")
    )
    return state


# 节点 3: 储能策略规划节点
def plan_energy_storage(state: DataCenterState) -> DataCenterState:
    """根据电网稳定性和绿电比例，制定储能系统充放电策略"""
    print("--- 正在规划储能策略 ---")
    soc = state["energy_storage_soc_current_percent"]
    grid_stable = state["grid_stability_index"] > 0.9
    green_high = state["predicted_green_energy_ratio"] > 0.6

    strategy = {}
    if green_high and soc < 0.8:
        strategy["action"] = "charge"
        strategy["description"] = "绿电充足，建议储能系统充电，存储多余能量。"
    elif not grid_stable and soc > 0.4:
        strategy["action"] = "discharge"
        strategy["description"] = "电网不稳定，建议储能系统放电，保障关键负载。"
    elif state["current_datacenter_load_factor"] > 0.8 and soc > 0.5:
        strategy["action"] = "discharge_peak_shaving"
        strategy["description"] = "数据中心负载处于高峰，建议储能放电以实现削峰填谷，降低电网压力。"
    else:
        strategy["action"] = "standby"
        strategy["description"] = "储能系统待命，根据实时情况调整。"

    state["energy_storage_strategy"] = strategy
    state["messages"].append(
        AIMessage(content=f"✅ 储能策略规划完成: {strategy['description']}")
    )
    return state


# 节点 4: 绿电分配节点
def allocate_green_energy(state: DataCenterState) -> DataCenterState:
    """根据关键负载优先级，分配绿色电力"""
    print("--- 正在分配绿色电力 ---")
    critical_priority = state["critical_workload_priority"]
    green_ratio = state["predicted_green_energy_ratio"]

    # 简化分配逻辑
    critical_load_alloc = min(1.0, green_ratio * (1 + (critical_priority - 0.5)))
    non_critical_load_alloc = green_ratio * (1 - (critical_priority - 0.5))

    allocation = {
        "critical_workloads_green_supply_ratio": round(critical_load_alloc, 2),
        "non_critical_workloads_green_supply_ratio": round(non_critical_load_alloc, 2),
        "description": f"优先为关键负载分配 {critical_load_alloc:.0%} 的绿电，其余负载分配 {non_critical_load_alloc:.0%}。"
    }

    state["green_energy_allocation"] = allocation
    state["messages"].append(
        AIMessage(content="✅ 绿电分配方案制定完成")
    )
    return state


# 节点 5: 数据中心负载预测节点
def load_prediction_node(state: DataCenterState) -> DataCenterState:
    """加载并运行预训练的数据中心负载预测模型 (load.pkl)"""
    print("--- 正在调用数据中心负载预测模型 --- ")
    try:
        model = load_model('load', verbose=False)
        predictions = predict_model(model)
        print("--- ✅ 数据中心负载预测成功 ---")
        state["load_prediction_results"] = predictions
        state["messages"].append(AIMessage(content="✅ 负载预测模型调用成功，生成了未来24小时的预测数据。"))
    except Exception as e:
        error_message = f"--- ❌ 数据中心负载预测失败: {e} ---"
        print(error_message)
        state["load_prediction_results"] = pd.DataFrame({"error": [error_message]})
        state["messages"].append(AIMessage(content=error_message))
    return state


# 节点 6: 风光出力预测节点
def renewable_prediction_node(state: DataCenterState) -> DataCenterState:
    """加载并运行预训练的风光出力预测模型 (power.pkl)"""
    print("--- 正在调用风光出力预测模型 --- ")
    try:
        model = load_model('power', verbose=False)
        predictions = predict_model(model)
        print("--- ✅ 风光出力预测成功 ---")
        state["renewable_prediction_results"] = predictions
        state["messages"].append(AIMessage(content="✅ 风光出力预测模型调用成功，生成了未来24小时的预测数据。"))
    except Exception as e:
        error_message = f"--- ❌ 风光出力预测失败: {e} ---"
        print(error_message)
        state["renewable_prediction_results"] = pd.DataFrame({"error": [error_message]})
        state["messages"].append(AIMessage(content=error_message))
    return state


# 节点 7: LLM 智能分析节点
def llm_reasoning_node(state: DataCenterState) -> DataCenterState:
    # 根据人工反馈调整分析
    if state.get("human_feedback"):
        print("--- 正在根据人工反馈重新进行智能分析 ---")
        feedback_prompt = f"\n重要补充：请根据以下用户反馈调整你的建议：'{state['human_feedback']}'"
    else:
        print("--- 正在进行首次智能分析 (首席架构师视角) ---")
        feedback_prompt = ""

    prompt = ChatPromptTemplate.from_template("""你是一位零碳数据中心首席架构师。你的任务是根据下面提供的24小时“负载预测”和“风光出力预测”，制定一份清晰、可执行的调度方案。

**负载预测 (需求):**
{load_data}

**风光出力预测 (供应):**
{renewable_data}

**你的方案必须简洁地回答以下核心问题：**
1.  **供需关系**: 哪些时段绿电富余，哪些时段存在缺口？总缺口是多少？
2.  **采购策略**: 针对总缺口，应购买绿电(PPA)还是绿证(REC)？为什么？
3.  **调度指令**: 明确储能系统在各时段的充/放电策略，以及何时启动电网购电。

**当前状态参考:**
- 储能SOC: {soc_percent}%
- PUE目标: {pue_target}

**输出要求：**
请直接以结构化的清单形式输出方案，不要添加任何无关的开头、结尾、签名或日期。
{feedback_prompt}
""")

    chain = prompt | llm

    # 安全地获取和格式化预测数据
    load_df = state.get("load_prediction_results")
    renewable_df = state.get("renewable_prediction_results")

    load_str = load_df.to_string() if load_df is not None and not load_df.empty else "(无负载预测数据)"
    renewable_str = renewable_df.to_string() if renewable_df is not None and not renewable_df.empty else "(无风光出力预测数据)"

    response = chain.invoke({
        "load_data": load_str,
        "renewable_data": renewable_str,
        "soc_percent": f"{state['energy_storage_soc_current_percent']:.1f}",
        "pue_target": f"{state['target_pue']:.2f}",
        "feedback_prompt": feedback_prompt
    })

    state["llm_insights"] = response.content
    state["messages"].append(AIMessage(content=f"✅ 大模型智能分析完成 (首席架构师视角)"))
    state["messages"].append(AIMessage(content=f"大模型洞察: {response.content}"))
    return state


# 节点 8: 生成最终方案
def generate_final_plan(state: DataCenterState) -> DataCenterState:
    """整合所有策略和建议，生成最终的零碳全方位调度方案"""
    print("--- 正在生成最终的零碳全方位调度方案 ---")

    # 安全地处理可能不存在的预测结果
    load_df = state.get("load_prediction_results")
    renewable_df = state.get("renewable_prediction_results")
    load_records = load_df.to_dict('records') if load_df is not None and not load_df.empty else "N/A"
    renewable_records = renewable_df.to_dict('records') if renewable_df is not None and not renewable_df.empty else "N/A"

    final_plan = {
        "timestamp": "2024-XX-XX HH:MM:SS",  # 实际应用中替换为当前时间
        "plan_summary": "未来24小时零碳全方位调度方案",
        "load_forecast_24h": load_records,
        "renewable_energy_forecast_24h": renewable_records,
        "chief_architect_recommendations": state["llm_insights"],
        "technical_implementation_details": {
            "migration_or_resource_plan": state["migration_path"],
            "energy_storage_strategy": state["energy_storage_strategy"],
            "green_energy_allocation_detail": state["green_energy_allocation"],
        },
        "initial_conditions": {
            "predicted_green_energy_ratio": f"{state['predicted_green_energy_ratio'] * 100:.1f}%",
            "current_load_factor": f"{state['current_datacenter_load_factor'] * 100:.1f}%",
            "grid_carbon_intensity": f"{state['grid_carbon_intensity']} gCO2/kWh",
            "target_PUE": f"{state['target_pue']:.2f}",
            "energy_storage_SOC": f"{state['energy_storage_soc_current_percent']:.1f}%",
            "grid_stability": f"{state['grid_stability_index']:.2f}"
        }
    }

    state["final_plan"] = final_plan
    state["messages"].append(
        AIMessage(content="✅ 完整调度方案生成成功")
    )
    return state


# 节点 9: 人工审核节点
def human_review_node(state: DataCenterState) -> DataCenterState:
    """展示最终方案并等待人工确认"""
    plan = state["final_plan"]
    load_predictions = state.get("load_prediction_results")
    renewable_predictions = state.get("renewable_prediction_results")

    print("\n" + "=" * 30 + " 人工审核环节 " + "=" * 30)
    print("请审核以下由【首席架构师 Agent】生成的调度方案：")

    # 打印负载预测结果
    print("\n--- 未来24小时数据中心负载预测 --- ")
    if load_predictions is not None and not load_predictions.empty:
        print(load_predictions.to_string())
    else:
        print("(无负载预测数据或预测失败)")

    # 打印风光出力预测结果
    print("\n--- 未来24小时风光出力预测 --- ")
    if renewable_predictions is not None and not renewable_predictions.empty:
        print(renewable_predictions.to_string())
    else:
        print("(无风光出力预测数据或预测失败)")

    print("\n--- 首席架构师建议摘要 --- ")
    print(plan["chief_architect_recommendations"])
    print("=" * 70)

    # 获取用户输入
    print("\n>>> 请审核方案 <<<")
    user_input = input("输入 'y' 确认通过并结束，或输入具体修改意见(例如: '增加储能放电比例') 以重新生成: ")

    if user_input.strip().lower() in ['y', 'yes', 'ok', '通过', '']:
        state["human_approved"] = True
        state["messages"].append(HumanMessage(content="✅ 人工审核通过"))
        print(">>> 审核通过，流程结束。")
    else:
        state["human_approved"] = False
        state["human_feedback"] = user_input
        state["messages"].append(HumanMessage(content=f"❌ 人工审核未通过，反馈意见: {user_input}"))
        print(f">>> 收到反馈: {user_input}。正在重新进行智能分析...")

    return state


# 构建LangGraph工作流
def create_scheduling_graph():
    workflow = StateGraph(DataCenterState)

    # 添加节点
    workflow.add_node("analyze", perform_initial_analysis)
    workflow.add_node("migrate", plan_migration_path)
    workflow.add_node("storage", plan_energy_storage)
    workflow.add_node("allocate_green", allocate_green_energy)
    workflow.add_node("load_prediction", load_prediction_node) # 新增负载预测节点
    workflow.add_node("renewable_prediction", renewable_prediction_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    workflow.add_node("final_plan", generate_final_plan)
    workflow.add_node("human_review", human_review_node)

    # 设置入口点
    workflow.set_entry_point("analyze")

    # 定义边和条件路由
    workflow.add_edge("analyze", "migrate")
    workflow.add_edge("migrate", "storage")
    workflow.add_edge("storage", "allocate_green")
    workflow.add_edge("allocate_green", "load_prediction")
    workflow.add_edge("load_prediction", "renewable_prediction")
    workflow.add_edge("renewable_prediction", "llm_reasoning")
    workflow.add_edge("llm_reasoning", "final_plan")
    workflow.add_edge("final_plan", "human_review")

    # 条件边逻辑
    def should_continue(state: DataCenterState):
        if state.get("human_approved", False):
            return "end"
        else:
            return "retry"

    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "end": END,
            "retry": "llm_reasoning"
        }
    )

    # 编译图
    app = workflow.compile()
    return app


# --- 执行入口 ---
def main():
    app = create_scheduling_graph()

    # === 生成并显示流程图 ===
    try:
        print("正在生成流程图...")
        png_data = app.get_graph().draw_mermaid_png()
        graph_image_path = "datacenter_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(png_data)
        print(f"✅ 流程图已保存至: {graph_image_path}")

        # 使用 Pillow 库在 IDE 环境中显示图片
        try:
            Image.open(graph_image_path).show()
        except Exception as img_err:
            print(f"尝试打开图片失败: {img_err}")
    except Exception as e:
        print(f"⚠️ 流程图生成失败 (可能需要安装 graphviz 或 mermaid 依赖): {e}")
        # 如果 draw_mermaid_png 失败，尝试用 print 输出 mermaid 文本
        try:
            print("\nMermaid 流程图定义:")
            print(app.get_graph().draw_mermaid())
        except:
            pass
    # === 显示结束 ===

    # >>>>>>>>>>>>>>>>> 关键输入数据 <<<<<<<<<<<<<<<<<
    initial_state = {
        "predicted_green_energy_ratio": 0.70,  # 预测绿电占比 70%
        "current_datacenter_load_factor": 0.60,  # 当前数据中心负载率 60%
        "grid_carbon_intensity": 550.0,  # 电网碳排强度 550 gCO2/kWh
        "target_pue": 1.25,  # 数据中心PUE目标 1.25
        "energy_storage_soc_current_percent": 75.0,  # 当前储能SOC 75%
        "grid_stability_index": 0.85,  # 电网稳定性指数 0.85 (较稳定)
        "critical_workload_priority": 0.9,  # 关键工作负载优先级 0.9 (高优先)
        "messages": [HumanMessage(content="启动数据中心绿色调度流程 ")]
    }

    print("\n" + "=" * 60)
    print("绿色算力调度流程启动")
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

    # print("\n[关键执行日志]")
    # for msg in result["messages"]:
    #     if isinstance(msg, AIMessage) and "大模型洞察 (电力电子优化视角)" not in msg.content:
    #         print(f" - {msg.content}")


if __name__ == "__main__":
    main()

    