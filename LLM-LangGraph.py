from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
# 通过通用配置来接入大模型
from langchain_openai import ChatOpenAI
import operator
import json
import os
from IPython.display import Image, display
from PIL import Image
# 1. 定义状态结构
class DataCenterState(TypedDict):
    # 输入参数
    green_energy_ratio: float  # 绿电占比 (0-1)
    computing_surplus: float  # 算力富余度 (0-1)
    network_latency: float  # 网络延迟 (ms)
    carbon_intensity: float  # 碳排强度 (gCO2/kWh)
    latency_requirement: float  # 延迟要求 (ms)

    # 逻辑计算得到的中间结果
    analysis_result: Dict
    migration_path: List[str]
    energy_storage_strategy: Dict
    green_energy_allocation: Dict

    # 大模型生成的见解与优化建议
    llm_insights: str

    # 人工审核状态
    human_feedback: Optional[str]
    approved: Optional[bool]

    # 最终输出
    final_plan: Dict
    messages: Annotated[List[BaseMessage], operator.add]


# 2. 初始化大模型 (使用 XSimple 提供的模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# --- 节点定义 ---

# 节点 1: 基础数据计算 (硬逻辑)
def analyze_datacenter(state: DataCenterState) -> DataCenterState:
    green_ratio = state["green_energy_ratio"]
    carbon = state["carbon_intensity"]

    # 手动逻辑：简单的评分
    green_score = green_ratio * 100
    carbon_score = max(0, 100 - carbon / 10)

    state["analysis_result"] = {
        "green_score": green_score,
        "carbon_score": carbon_score,
        "status": "Green" if green_score > 60 else "Carbon-Heavy"
    }
    state["messages"].append(AIMessage(content="✅ 已完成基础数据初步分析"))
    return state


# 节点 2: 迁移决策 (结合硬逻辑)
def plan_migration_path(state: DataCenterState) -> DataCenterState:
    # 模拟路径规划
    if state["network_latency"] <= state["latency_requirement"]:
        path = ["Region_A_Green_DC", "Region_B_Edge"]
    else:
        path = ["Internal_Optimized_Node"]

    state["migration_path"] = path
    state["messages"].append(AIMessage(content="✅ 迁移路径规划完成"))
    return state


# 节点 3: 绿电分配 (硬逻辑)
def allocate_green_energy(state: DataCenterState) -> DataCenterState:
    green_ratio = state["green_energy_ratio"]
    allocation = {
        "AI_training": f"{min(green_ratio * 0.6, 0.5) * 100:.1f}%",
        "critical_services": f"{min(green_ratio * 0.3, 0.3) * 100:.1f}%",
        "total_usage": f"{green_ratio * 100:.1f}%"
    }
    state["green_energy_allocation"] = allocation
    state["messages"].append(AIMessage(content="✅ 绿电配额分配完成"))
    return state


# 节点 4: LLM 智能分析节点 (关键补充：调用 API)
def llm_reasoning_node(state: DataCenterState) -> DataCenterState:
    """利用大模型对上述所有技术参数进行综合评估并输出深度建议"""

    base_prompt = """
    你是一位资深的绿色数据中心调度专家。根据以下运行数据，请提供一段专业的调度建议（约150字）：
    - 绿电占比: {green_ratio}% 
    - 碳排强度: {carbon} gCO2/kWh
    - 网络延迟: {latency}ms (阈值: {latency_req}ms)
    - 预定迁移路径: {path}
    - 绿电分配方案: {allocation}

    请重点回答：当前方案在降低碳排和保障延迟之间是否达到了平衡？
    """

    # 如果有人工反馈，添加到 prompt 中
    if state.get("human_feedback"):
        base_prompt += f"\n\n注意：用户对上一版方案的反馈如下，请务必根据此反馈进行针对性调整和优化：\n{state['human_feedback']}"

    prompt = ChatPromptTemplate.from_template(base_prompt)

    chain = prompt | llm
    response = chain.invoke({
        "green_ratio": state["green_energy_ratio"] * 100,
        "carbon": state["carbon_intensity"],
        "latency": state["network_latency"],
        "latency_req": state["latency_requirement"],
        "path": state["migration_path"],
        "allocation": json.dumps(state["green_energy_allocation"])
    })

    state["llm_insights"] = response.content
    state["messages"].append(AIMessage(content="🤖 LLM 专家智能建议已生成"))
    return state


# 节点 5: 生成最终方案格式化
def generate_final_plan(state: DataCenterState) -> DataCenterState:
    final_plan = {
        "metrics": {
            "green_ratio": f"{state['green_energy_ratio'] * 100}%",
            "latency_compliant": state["network_latency"] <= state["latency_requirement"]
        },
        "path": state["migration_path"],
        "allocation": state["green_energy_allocation"],
        "expert_advice": state["llm_insights"]
    }
    state["final_plan"] = final_plan
    state["messages"].append(AIMessage(content="✅ 完整调度计划导出成功"))
    return state


# 节点 6: 人工审核节点
def human_review_node(state: DataCenterState) -> DataCenterState:
    print("\n" + "="*30)
    print("=== 人工审核环节 ===")
    print("="*30)
    print(f"当前 LLM 建议:\n{state['llm_insights']}")
    print("-" * 30)
    
    while True:
        user_input = input("\n请输入审核意见 (输入 'pass' 或 'ok' 通过，否则输入具体修改建议): ").strip()
        if user_input:
            break
    
    if user_input.lower() in ['pass', 'ok', 'yes', '通过']:
        print(">>> 审核通过，流程结束。")
        state["messages"].append(HumanMessage(content="审核通过"))
        return {"human_feedback": None, "approved": True}
    else:
        print(f">>> 审核不通过，反馈意见已记录: {user_input}")
        print(">>> 正在重新生成方案...")
        state["messages"].append(HumanMessage(content=f"审核不通过，意见: {user_input}"))
        return {"human_feedback": user_input, "approved": False}


def review_router(state: DataCenterState):
    """根据人工审核结果决定下一步"""
    if state.get("approved"):
        return END
    else:
        return "llm_reasoning"



# --- 构建 LangGraph 工作流 ---

def create_scheduling_graph():
    workflow = StateGraph(DataCenterState)

    # 添加节点
    workflow.add_node("analyze", analyze_datacenter)
    workflow.add_node("plan_migration", plan_migration_path)
    workflow.add_node("allocate_green", allocate_green_energy)
    workflow.add_node("llm_reasoning", llm_reasoning_node)  # 新增 LLM 节点
    workflow.add_node("generate_plan", generate_final_plan)
    workflow.add_node("human_review", human_review_node)    # 新增人工审核节点

    # 定义边（执行顺序）
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan_migration")
    workflow.add_edge("plan_migration", "allocate_green")
    workflow.add_edge("allocate_green", "llm_reasoning")  # 连接到 LLM
    workflow.add_edge("llm_reasoning", "generate_plan")  # 从 LLM 连接到生成
    
    # generate_plan 后连接到人工审核，而不是直接结束
    workflow.add_edge("generate_plan", "human_review")
    
    # 添加条件边：根据审核结果决定是结束还是重做
    workflow.add_conditional_edges(
        "human_review",
        review_router,
        {
            "llm_reasoning": "llm_reasoning", # 如果不通过，回退到 LLM 推理节点
            END: END                          # 如果通过，结束
        }
    )

    return workflow.compile()
#此处只给出了单向无分支图的构建方法，实际中可以继续构造分支结构和循环结构

# --- 执行入口 ---

def main():
    app = create_scheduling_graph()

    # 尝试可视化处理

    try:
            graph_image_path = "datacenter_workflow.png"
            # 使用 Mermaid 生成图片 (返回二进制数据)
            graph_png = app.get_graph().draw_mermaid_png()
            with open(graph_image_path, "wb") as f:
                f.write(graph_png)
            print(f"\n--- LangGraph 流程图已保存至: {graph_image_path} ---")

            # 使用 Pillow 打开并显示图片
            img = Image.open(graph_image_path)
            img.show()  # 这会打开一个新的窗口显示图片
            print("\n--- 流程图已在新的图片查看器窗口中显示 ---")

    except Exception as e:
            print(f"\n--- 无法生成流程图。请确保网络畅通或已配置相关环境。错误: {e} ---")

        # 可视化处理结束

    # 输入一组有待评估的数据中心参数
    initial_state = {
        "green_energy_ratio": 0.65,
        "computing_surplus": 0.25,
        "network_latency": 18.0,
        "carbon_intensity": 450.0,
        "latency_requirement": 30.0,
        "messages": [HumanMessage(content="启动数据中心调度分析流程")]
    }


    print("--- 正在启动 XSimple 绿色算力调度流程 ---")
    result = app.invoke(initial_state)

    print("\n[最终方案概览]")
    print(f"建议内容: {result['final_plan']['expert_advice']}")

    # print("\n[执行日志]")
    # for msg in result["messages"]:
    #     if isinstance(msg, AIMessage):
    #         print(f" - {msg.content}")


if __name__ == "__main__":
    main()