"""
Agent 2: 能源与绿电规划专家 (Energy Planner)

LLM 驱动能源规划方案
改造为 LangGraph 节点，与 Agent 1 输出格式兼容

功能:
    - 结合项目背景参数与实时电网数据，调用 LLM 生成能源配比方案
    - 输出初步电力配比方案（Markdown格式报告）

输入状态 (从 Agent 1 获取):
    - user_requirements: 用户需求 (location, pue_target, green_energy_target 等)
    - environmental_data: 环境数据 (annual_temperature, annual_sunshine_hours 等)
    - electricity_price: 电价数据 (peak_price, low_price, max_price_diff 等，英文字段名)

输出状态:
    - energy_plan: 能源规划方案 (包含 LLM 生成的报告)

"""

import os
import requests
from typing import TYPE_CHECKING, Dict, Any, Optional

# LangChain 组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from graph import GreenDataCenterState


# ============================================================
# 1. LLM 初始化 (由 XSimple 开发)
# ============================================================

def _get_llm():
    """获取 LLM 实例（通义千问）"""
    llm_api_key = os.getenv("LLM_API_KEY", "")
    if not llm_api_key:
        raise ValueError("环境变量 LLM_API_KEY 未设置，无法调用 LLM。")

    return ChatOpenAI(
        model="qwen-plus",
        api_key=llm_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


# ============================================================
# 2. API 调用函数 (保持原有逻辑)
# ============================================================

def get_detailed_energy_data(zone: str = "CN") -> str:
    """
    调用 Electricity Maps API 获取实时电网能源数据
    
    参数:
        zone: 区域代码，中国区域可用 "CN"
        
    返回:
        格式化的电网能源报告字符串
    """
    url = f"https://api.electricitymaps.com/v3/renewable-energy/latest?zone={zone}"
    headers = {
        "auth-token": "PmQxZjp5WQZZbgz844Vv"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            info = (
                f"--- 实时电网能源数据报告 ---\n"
                f"1. 目标区域 (zone): {data.get('zone')}\n"
                f"2. 可再生能源比例 (renewablePercentage): {data.get('value')}{data.get('unit', '%')}\n"
                f"3. 数据时间 (datetime): {data.get('datetime')}\n"
            )
            return info
        else:
            return f"API 请求失败，状态码：{response.status_code}，使用本地电网能源数据"
    except Exception as e:
        return f"API 调用异常：{str(e)}，将使用本地电网能源数据"


# ============================================================
# 3. LLM Agent 创建函数
# ============================================================

def create_energy_specialist_agent():
    """
    创建能源规划专家 Agent（LLM Chain）
    
    返回:
        LangChain Chain 对象
    """
    prompt_template = """
    你是一位顶尖"能源与绿电规划专家"。请结合【项目背景参数】与【实时电网数据】提供初步的电力配比方案。

    **【业务参考准则】**
    - 评估维度：需关注 PUE (能效)、REF (能源结构) 和实时碳抵消率。参考《评价标准.pdf》：零碳中心抵消比例应为100%。
    - 储能策略：建议配比储能以应对峰谷价差，实现削峰填谷。
    - 绿电消纳：优先利用本地消纳，不足部分通过绿证或长协覆盖。

    **【1. 项目背景与环境参数】**
    {project_context}

    **【2. 来自 API 的实时电网数据】**
    {api_data}

    **请输出一份深度的 Markdown 方案（≤600 字）：**
   内容需包含：
    1. **现状挑战分析**：结合当地气温评估 {pue_target} 目标的达成可能性。如果 {temp} > 17℃ 且 PUE > 1.15，必须提出液冷或自然冷却建议。
    2. **电力消纳策略**：针对 {green_gap}% 的实时绿电缺口，给出分布式光伏、PPA长协和绿证的具体配比建议。
    3. **经济性建议**：利用最大峰谷价差 {price_diff} 计算储能套利的价值，并给出具体的削峰填谷运行方案。
    4. **推演过程**：必须列出上述提到的 2-3 个核心公式，并带入本次任务的数值展示计算逻辑，确保数据闭环。
    
    **同时，请在报告末尾以 JSON 格式输出以下技术参数：**
    {{  
      "pv_capacity": 光伏装机容量（kW）,
      "wind_capacity": 风电装机容量（kW）,
      "storage_capacity": 储能容量（kWh）,
      "storage_power": 储能功率（kW）,
      "ppa_ratio": 绿电长协比例（%）,
      "grid_ratio": 电网调峰比例（%）,
      "estimated_self_consumption": 预计自发自用率（%）,
      "estimated_green_ratio": 预计绿电占比（%）
    }}
    
    请确保 JSON 格式正确，且所有数值为数字类型。
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain


# ============================================================
# 4. 工具函数
# ============================================================

def _build_project_context(
    user_req: Dict[str, Any],
    env_data: Dict[str, Any],
    price_data_cn: Dict[str, Any]
) -> str:
    """
    构建项目背景上下文字符串
    """
    location = user_req.get("location", "未知")
    pue_target = user_req.get("pue_target", 1.2)
    green_target = user_req.get("green_energy_target", 90)
    planned_load = user_req.get("planned_load", 0)
    business_type = user_req.get("business_type", "通用")
    
    annual_temp = env_data.get("annual_temperature", "N/A")
    annual_sunshine = env_data.get("annual_sunshine_hours", "N/A")
    annual_wind = env_data.get("annual_wind_speed", "N/A")
    carbon_factor = env_data.get("carbon_emission_factor", "N/A")
    
    price_diff = price_data_cn.get("最大峰谷价差", 0)
    peak_price = price_data_cn.get("尖峰电价", 0)
    low_price = price_data_cn.get("低谷电价", 0)
    
    context = (
        f"位置：{location}\n"
        f"业务类型：{business_type}\n"
        f"计划负荷：{planned_load} kW\n"
        f"目标 PUE：{pue_target}，绿电目标：{green_target}%\n"
        f"电价环境：尖峰 {peak_price} 元/kWh，低谷 {low_price} 元/kWh，峰谷价差 {price_diff} 元/kWh\n"
        f"气象：年均温 {annual_temp}℃，年日照 {annual_sunshine}h，年均风速 {annual_wind}m/s\n"
        f"当地碳排因子：{carbon_factor} kgCO₂/kWh"
    )
    return context


# ============================================================
# 5. LangGraph 节点函数 (核心接口)
# ============================================================

def energy_planner_node(state: dict) -> dict:
    """
    Agent 2: 能源与绿电规划专家 - LangGraph Node
    
    调用 LLM 生成能源规划方案，与 Agent 1 输出格式完全兼容。
    
    参数:
        state: GreenDataCenterState，包含:
            - user_requirements: 用户需求 (从 Agent 1)
            - environmental_data: 环境数据 (从 Agent 1)
            - electricity_price: 电价数据 (从 Agent 1，英文字段名)
            
    返回:
        更新后的 state，新增:
            - energy_plan: 能源规划方案
    """
    print("\n" + "="*60)
    print("⚡ [Agent 2: 能源与绿电规划专家] 开始工作")
    print("="*60)
    
    # ===== 1. 从 state 获取 Agent 1 的输出 =====
    user_req = state.get("user_requirements", {})
    env_data = state.get("environmental_data", {})
    electricity_price = state.get("electricity_price", {})
    
    # 提取关键参数
    location = user_req.get("location", "未知")
    pue_target = user_req.get("pue_target", 1.2)
    green_target = user_req.get("green_energy_target", 90)
    annual_temp = env_data.get("annual_temperature", 10)
    
    print(f"📊 从 Agent 1 获取的输入:")
    print(f"  - 位置：{location}")
    print(f"  - PUE 目标：{pue_target}")
    print(f"  - 绿电目标：{green_target}%")
    print(f"  - 年均温度：{annual_temp}°C")
    
    # ===== 2. 直接使用电价数据（已是中文格式）=====
    price_data_cn = state.get("electricity_price", {})
    # 确保price_diff的数据类型与requirement_analysis_node.py的electricity_price类型相同
    price_diff = price_data_cn.get("最大峰谷价差", 0.0)
    print(f"  - 峰谷价差：{price_diff} 元/kWh")
    
    # 接收更多数据
    peak_price = price_data_cn.get("尖峰电价", 0.0)
    high_price = price_data_cn.get("高峰电价", 0.0)
    flat_price = price_data_cn.get("平段电价", 0.0)
    valley_price = price_data_cn.get("低谷电价", 0.0)
    deep_valley_price = price_data_cn.get("深谷电价", 0.0)
    
    print(f"  - 尖峰电价：{peak_price} 元/kWh")
    print(f"  - 高峰电价：{high_price} 元/kWh")
    print(f"  - 平段电价：{flat_price} 元/kWh")
    print(f"  - 低谷电价：{valley_price} 元/kWh")
    print(f"  - 深谷电价：{deep_valley_price} 元/kWh")
    
    # ===== 3. 构建项目背景上下文 =====
    project_context = _build_project_context(user_req, env_data, price_data_cn)
    print(f"\n📋 项目背景已整理")
    
    # ===== 4. 获取实时电网碳数据 =====
    print(f"\n🌐 正在获取实时电网碳数据...")
    api_data = get_detailed_energy_data(zone="CN")
    
    if "失败" in api_data or "异常" in api_data:
        print(f"⚠️ {api_data}")
        # 使用本地碳排因子作为备用
        carbon_factor = env_data.get("carbon_emission_factor", 0.5)
        api_data = f"--- 使用本地碳排因子 ---\n碳强度: {carbon_factor * 1000} gCO2eq/kWh (本地数据)"
    else:
        print(f"✅ API 数据获取成功")
    
    # ===== 5. 调用 LLM 生成方案 =====
    print(f"\n🤖 正在调用 LLM 生成能源规划方案...")
    
    try:
        # 创建能源规划专家Agent
        agent = create_energy_specialist_agent()
        
        # 准备LLM输入参数
        llm_input = {
            "project_context": project_context,
            "api_data": api_data,
            "location": location,
            "temp": annual_temp,
            "pue_target": pue_target,
            "green_target": green_target,
            "price_diff": price_diff
        }
        
        # 调用LLM生成方案
        llm_output = agent.invoke(llm_input)
        print(f"✅ LLM方案生成成功，输出长度: {len(llm_output)} 字符")
        
        # 解析LLM输出，提取Markdown报告和JSON参数
        import json
        
        # 查找JSON部分的开始和结束位置
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            # 提取Markdown报告部分
            llm_report = llm_output[:json_start].strip()
            
            # 提取并解析JSON部分
            try:
                json_str = llm_output[json_start:json_end]
                tech_params = json.loads(json_str)
                print(f"✅ 成功解析技术参数: {list(tech_params.keys())}")
            except json.JSONDecodeError:
                print("⚠️ 无法解析JSON部分，使用默认值")
                tech_params = {}
        else:
            # 如果没有找到JSON部分，使用整个输出作为报告
            llm_report = llm_output
            tech_params = {}
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        llm_report = f"LLM 调用失败，请检查 DASHSCOPE_API_KEY 环境变量。错误: {e}"
        tech_params = {}
    
    # ===== 6. 构建 energy_plan 输出 =====
    # 保持与 graph.py 中 EnergyPlan 类型定义兼容
    energy_plan = {
        # LLM 生成的完整报告
        "llm_report": llm_report,
        
        # 使用LLM解析出的技术参数，或使用默认值
        "pv_capacity": tech_params.get("pv_capacity", 0.0),           # 光伏装机容量（kW）
        "wind_capacity": tech_params.get("wind_capacity", 0.0),         # 风电装机容量（kW）
        "storage_capacity": tech_params.get("storage_capacity", 0.0),      # 储能容量（kWh）
        "storage_power": tech_params.get("storage_power", 0.0),         # 储能功率（kW）
        "ppa_ratio": tech_params.get("ppa_ratio", 0.0),             # 绿电长协比例（%）
        "grid_ratio": tech_params.get("grid_ratio", 0.0),            # 电网调峰比例（%）
        "estimated_self_consumption": tech_params.get("estimated_self_consumption", 0.0),  # 预计自发自用率（%）
        "estimated_green_ratio": tech_params.get("estimated_green_ratio", green_target),  # 预计绿电占比（%）
        
        # 原始数据（供后续 Agent 使用）
        "price_data_cn": price_data_cn,
        "project_context": project_context,
        "api_data": api_data
    }
    
    # 打印报告预览
    print(f"\n📝 方案报告预览:")
    print("-" * 40)
    preview = llm_report[:300] + "..." if len(llm_report) > 300 else llm_report
    print(preview)
    print("-" * 40)
    
    print("\n" + "="*60)
    print("✅ [Agent 2: 能源与绿电规划专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态
    return {
        **state,
        "energy_plan": energy_plan
    }



# ============================================================
# 6. 主程序入口（用于独立测试）
# ============================================================

if __name__ == "__main__":
    print("===== 测试 Agent 2: 能源与绿电规划专家 (XSimple 版) =====")
    
    # 模拟 Agent 1 的输出作为测试输入
    test_state = {
        "user_requirements": {
            "location": "杭州",
            "business_type": "通用",
            "planned_area": 10000,
            "planned_load": 5000,
            "computing_power_density": 8,
            "priority": "环保型",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 10000
        },
        "environmental_data": {
            "annual_temperature": 17.59,
            "annual_wind_speed": 2.21,
            "annual_sunshine_hours": 3305.89,
            "carbon_emission_factor": 0.4974,
            "latitude": 30.27,
            "longitude": 120.15,
            "province": "浙江"
        },
        # Agent 1 输出的是英文字段名
        "electricity_price": {
            "peak_price": 0.9287,
            "high_price": 0.75,
            "flat_price": 0.55,
            "low_price": 0.2533,
            "deep_low_price": 0.20,
            "max_price_diff": 0.6754
        }
    }
    
    # 执行节点
    result = energy_planner_node(test_state)
    
    print("\n" + "="*80)
    print("完整报告:")
    print("="*80)
    print(result.get("energy_plan", {}).get("llm_report", "无报告"))
