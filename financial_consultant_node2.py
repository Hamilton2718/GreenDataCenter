import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict
import pprint

# ==================== Agent4 经济碳排分析 ====================
def agent4_economic_carbon_analysis(state: dict) -> dict:
    """
    Agent 4: 经济效益与碳排放初步分析
    """
    # ================= 输入提取 =================
    project_info = state['project_info']
    energy_plan = state['energy_plan']
    cooling_plan = state['cooling_plan']
    electricity_price = state['electricity_price']
    env_data = state['environmental_data']

    planned_area = project_info['planned_area']
    planned_load = project_info['planned_load']
    computing_density = project_info.get('Computing_power_density', 8)
    budget_constraint = project_info['budget_constraint']
    green_target = project_info['green_energy_target']

    pv_capacity = energy_plan.get('pv_capacity', 0)
    wind_capacity = energy_plan.get('wind_capacity', 0)
    storage_capacity = energy_plan.get('storage_capacity', 0)
    cooling_tech = cooling_plan['cooling_technology']
    pue = cooling_plan['estimated_pue']
    cooling_power = cooling_plan.get('cooling_power_consumption', 0)

    avg_price = sum(electricity_price.values()) / len(electricity_price)
    carbon_factor = env_data.get('carbon_emission_factor', 0.5)

    # ================= 打印输入 =================
    print("\n📊 财务分析输入:")
    print(f"  - 计划面积: {planned_area} 平方米")
    print(f"  - 计划负荷: {planned_load} kW")
    print(f"  - 光伏装机: {pv_capacity} kW")
    print(f"  - 储能容量: {storage_capacity} kWh")
    print(f"  - 制冷功耗: {cooling_power} kW")
    print(f"  - 制冷技术: {cooling_tech}")
    print(f"  - 预算约束: {budget_constraint} 万元")
    print(f"  - 绿电目标: {green_target} %\n")

    # ================= CAPEX =================
    capex = {}
    capex['land_building'] = planned_area * 8000 / 10000
    rack_count = planned_load / computing_density
    capex['IT_equipment'] = rack_count * 2
    if '液冷' in cooling_tech:
        cooling_cost_per_kw = 3000
    elif '蒸发' in cooling_tech:
        cooling_cost_per_kw = 2000
    else:
        cooling_cost_per_kw = 1500
    capex['cooling_equipment'] = planned_load * (pue-1) * cooling_cost_per_kw / 10000
    capex['lighting_infra'] = planned_area * 200 / 10000
    capex['local_PV'] = pv_capacity * 1000 * 3.5 / 10000
    capex['wind'] = wind_capacity * 1000 * 6 / 10000
    capex['storage'] = storage_capacity * 1000 * 1.2 / 10000

    total_capex = sum(capex.values())

    # ================= OPEX =================
    annual_computing_energy = planned_load * 8760
    annual_cooling_energy = cooling_power * 8760
    annual_lighting_energy = planned_area * 50 * 8760 / 1000
    total_annual_energy = annual_computing_energy + annual_cooling_energy + annual_lighting_energy

    opex_electricity = total_annual_energy * avg_price / 10000
    rec_cost = total_annual_energy * (green_target/100) * 0.1
    depreciation = total_capex / 10
    total_opex = opex_electricity + rec_cost + depreciation

    # ================= 碳排 =================
    green_energy = pv_capacity * 8760
    net_grid_energy = total_annual_energy - green_energy
    total_carbon = net_grid_energy * carbon_factor
    carbon_offset = rec_cost / 50 * 1000
    carbon_offset_ratio = min(carbon_offset / total_carbon, 1.0)

    cost_within_budget = total_capex <= budget_constraint
    carbon_within_target = carbon_offset_ratio >= green_target/100

    # ================= 输出 =================
    print("\n💰 财务分析结果:")
    print(f"  - 建设投资(CAPEX): {total_capex:,.1f} 万元")
    for item, cost in capex.items():
        print(f"    · {item}: {cost:,.1f} 万元")
    print(f"  - 年运营成本(OPEX): {total_opex:,.1f} 万元/年")
    print(f"  - 预算是否满足: {cost_within_budget}")
    print(f"  - 碳排目标是否满足: {carbon_within_target}")
    print(f"  - 总碳排: {total_carbon:,.0f} kg CO2")
    print(f"  - 碳抵消量: {carbon_offset:,.0f} kg CO2")
    print(f"  - 碳抵消比例: {carbon_offset_ratio:.2%}\n")

    result = {
        "capex": capex,
        "total_capex": total_capex,
        "opex_estimate": total_opex,
        "carbon_estimate": {
            "total_carbon": total_carbon,
            "carbon_offset": carbon_offset,
            "carbon_offset_ratio": carbon_offset_ratio
        },
        "cost_within_budget": cost_within_budget,
        "carbon_within_target": carbon_within_target
    }
    state['agent4'] = result
    return state


# ==================== LLM Client ====================
class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("请在 .env 文件中设置 SILICONFLOW_API_KEY")
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_report(self, prompt):
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个绿色数据中心经济与投资分析专家。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            try:
                return result['choices'][0]['message']['content']
            except (KeyError, IndexError):
                raise ValueError("API 返回异常，无法获取 report_md")
        else:
            raise ValueError(f"请求失败: {response.status_code}, {response.text}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    load_dotenv()
    state = {
        "project_info": {
            "planned_area": 10000,
            "planned_load": 5000,
            "Computing_power_density": 8,
            "budget_constraint": 10000,
            "green_energy_target": 90
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "wind_capacity": 500,
            "storage_capacity": 6000
        },
        "cooling_plan": {
            "cooling_technology": "浸没式液冷",
            "estimated_pue": 1.08,
            "cooling_power_consumption": 400
        },
        "electricity_price": {
            "尖峰电价": 0.5, "高峰电价": 0.4, "平段电价": 0.3,
            "低谷电价": 0.25, "深谷电价": 0.2
        },
        "environmental_data": {
            "carbon_emission_factor": 0.5
        }
    }

    # 运行经济分析
    state = agent4_economic_carbon_analysis(state)

    # 调用 LLM 生成 Markdown 报告
    agent4_output = state['agent4']
    prompt = f"""
你是一个绿色数据中心经济评估专家。
请基于以下经济和碳排数据，撰写一份数据中心经济型评估报告（Markdown 格式）：
- 建设成本明细: {json.dumps(agent4_output['capex'], indent=2)}
- 总建设成本: {agent4_output['total_capex']} 万元
- 运营成本估算: {agent4_output['opex_estimate']:.2f} 万元/年
- 碳排放估算: {json.dumps(agent4_output['carbon_estimate'], indent=2)}
- 预算约束是否满足: {agent4_output['cost_within_budget']}
- 碳排目标是否满足: {agent4_output['carbon_within_target']}
请包括以下内容：
1. 报告概述
2. 建设成本分析
3. 年度运营成本分析
4. 碳排放与碳抵消分析
5. 综合经济评价与建议
请用 Markdown 格式输出。
"""

    client = LLMClient()
    report_md = client.generate_report(prompt)

    # 保存 Markdown
    with open("data_center_economic_evaluation.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("✅ Markdown 报告已生成: data_center_economic_evaluation.md")