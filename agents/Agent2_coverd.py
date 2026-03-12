import os
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 1. 初始化 LLM (由 XSimple 开发)
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 2. 原有 API 调用函数（保持不变）
def get_detailed_carbon_data(zone="DE"):
    """
    调用 Electricity Maps API 并提取所有返回字段
    """
    url = f"https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={zone}"
    headers = {
        "auth-token": "PmQxZjp5WQZZbgz844Vv"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            info = (
                f"--- 实时电网碳数据报告 ---\n"
                f"1. 目标区域 (zone): {data.get('zone')}\n"
                f"2. 碳强度值 (carbonIntensity): {data.get('carbonIntensity')} gCO2eq/kWh\n"
                f"3. 数据时间 (datetime): {data.get('datetime')}\n"
                f"4. 排放因子类型 (emissionFactorType): {data.get('emissionFactorType')}\n"
            )
            return info
        else:
            return f"API 请求失败，状态码：{response.status_code}"
    except Exception as e:
        return f"发生异常：{str(e)}"


# 3. 创建增强型 Agent，加入项目背景变量
def create_energy_specialist_agent():
    # 扩充 Prompt 模板，增加项目参数维度
    prompt_template = """
    你是一位顶尖“能源与绿电规划专家”。请结合【项目背景参数】与【实时电网数据】提供初步的电力配比方案。

    **【业务参考准则】**
    - 评估维度：需关注 PUE (能效)、REF (能源结构) 和实时碳抵消率。参考《评价标准.pdf》：零碳中心抵消比例应为100%。
    - 储能策略：建议配比储能以应对峰谷价差，实现削峰填谷。
    - 绿电消纳：优先利用本地消纳，不足部分通过绿证或长协覆盖。

    **【1. 项目背景与环境参数】**
    {project_context}

    **【2. 来自 API 的实时电网数据】**
    {api_data}

    **请输出一份深度的 Markdown 方案（≤600字）：**
    内容需包含：
    1. **现状挑战分析**：结合{location}的年均温({temp}℃)对 PUE {pue_target} 目标的达成可能性进行评估。
    2. **电力消纳策略**：针对 {green_target}% 的绿电目标，结合当前电网碳强度给出初步电力配比方案。
    3. **经济性建议**：利用最大峰谷价差 ({price_diff} 元/kWh) 给出具体的储能套利与运行建议。
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
            prompt
            | llm
            | StrOutputParser()
    )
    return chain


# --- 主程序 ---
if __name__ == '__main__':
    # 定义你提供的结构化输入数据
    project_input = {
        "project_info": {
            "location": "杭州",
            "business_type": "通用",
            "planned_area": 10000,
            "planned_load": 5000,
            "算力_density": 8,
            "priority": "环保型",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 10000
        },
        "environmental_data": {
            "annual_temperature": 17.59,
            "annual_wind_speed": 2.21,
            "annual_sunshine_hours": 3305.89,
            "carbon_emission_factor": 0.4974
        },
        "electricity_price": {
            "尖峰电价": 0.9287,
            "低谷电价": 0.2533,
            "最大峰谷价差": 0.6754
        },
        "timestamp": "2026-03-10"
    }

    # 格式化项目背景字符串
    project_context_str = (
        f"位置：{project_input['project_info']['location']}\n"
        f"目标 PUE：{project_input['project_info']['pue_target']}，绿电目标：{project_input['project_info']['green_energy_target']}%\n"
        f"电价环境：峰谷价差 {project_input['electricity_price']['最大峰谷价差']} 元/kWh\n"
        f"气象：年均温 {project_input['environmental_data']['annual_temperature']}℃，年日照 {project_input['environmental_data']['annual_sunshine_hours']}h"
    )

    print("🚀 XSimple 能源专家 Agent 启动中...")

    agent = create_energy_specialist_agent()

    # 调用原有的 API 获取数据逻辑（不修改内部逻辑）
    zone_code = "DE"  # 对应杭州区域，或按需保持 "DE"
    rich_api_data = get_detailed_carbon_data(zone_code)

    if "失败" in rich_api_data or "异常" in rich_api_data:
        print(rich_api_data)
    else:
        print("✅ API 数据与项目背景整合完成，正在生成分析报告...")

        # 将结构化数据传入 Agent
        report = agent.invoke({
            "project_context": project_context_str,
            "api_data": rich_api_data,
            "location": project_input['project_info']['location'],
            "temp": project_input['environmental_data']['annual_temperature'],
            "pue_target": project_input['project_info']['pue_target'],
            "green_target": project_input['project_info']['green_energy_target'],
            "price_diff": project_input['electricity_price']['最大峰谷价差']
        })

        print("\n" + "=" * 80)
        print(" 专家分析方案：")
        print(report)
        print("=" * 80 + "\n")