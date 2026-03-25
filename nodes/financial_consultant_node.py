#!/usr/bin/env python3
"""
Agent 5: 综合评价与投资决策专家 (Financial Consultant)

基于前 4 个 Agent 的输出，计算经济指标，生成最终规划设计建议书。
适配 LangGraph 工作流版本。
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any

# ---------- 默认市场数据（可根据实际修改） ----------
DEFAULT_GRID_PRICE = 0.45          # 元/kWh，乌兰察布电价
DEFAULT_CARBON_PRICE = 85           # 元/tCO2
DEFAULT_GREEN_CERT_PRICE = 0.03     # 元/kWh，绿证价格（假设）
DEFAULT_EMISSION_FACTOR = 0.8       # kgCO2/kWh，内蒙古电网

# ---------- 单位造价（用于简单回收期估算） ----------
PV_UNIT_COST = 3000                  # 元/kW
STORAGE_UNIT_COST = 1500             # 元/kWh
DISCOUNT_RATE = 0.08                 # 8%
LIFETIME_YEARS = 20                  # 项目寿命期 20 年


# ==================== LLM Client ====================
class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
        if not self.api_key:
            # 使用默认 API key 作为备用
            self.api_key = "sk-77a4c286b27e4c06aff03cccc38cc9d1"
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.dashscope_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.dashscope_model = "qwen-plus"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_report(self, prompt):
        # 先尝试 SiliconFlow API
        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个绿色数据中心经济与投资分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data), timeout=30)
            if response.status_code == 200:
                result = response.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    print("SiliconFlow API 返回格式异常，尝试使用 DashScope API")
            else:
                print(f"SiliconFlow API 调用失败，状态码: {response.status_code}，尝试使用 DashScope API")
        except Exception as e:
            print(f"SiliconFlow API 调用异常: {str(e)}，尝试使用 DashScope API")

        # 尝试使用 DashScope API 作为备选
        try:
            data = {
                "model": self.dashscope_model,
                "messages": [
                    {"role": "system", "content": "你是一个绿色数据中心经济与投资分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            response = requests.post(self.dashscope_url, headers=self.headers, data=json.dumps(data), timeout=30)
            if response.status_code == 200:
                result = response.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    print("DashScope API 返回格式异常")
            else:
                print(f"DashScope API 调用失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"DashScope API 调用异常: {str(e)}")

        # 生成兜底报告，包含实际数据
        lines = []
        lines.append("# 财务分析报告")
        lines.append("")
        lines.append("## 报告概述")
        lines.append("由于 API 调用失败，无法生成详细报告。以下为关键数据摘要：")
        lines.append("")
        lines.append("## 建设成本分析")
        lines.append(f"- 总投资：{prompt.split('- 总建设成本: ')[1].split(' 万元')[0]} 万元")
        lines.append("")
        lines.append("## 年度运营成本分析")
        lines.append(f"- 年运营成本：{prompt.split('- 年运营成本估算: ')[1].split(' 万元/年')[0]} 万元/年")
        lines.append(f"- 年节省：{prompt.split('- 年节省: ')[1].split(' 万元/年')[0]} 万元/年")
        lines.append("")
        lines.append("## 碳排放与碳抵消分析")
        lines.append(f"- 年碳减排：{prompt.split('- 年碳减排: ')[1].split(' 吨 CO2')[0]} 吨 CO₂")
        lines.append("")
        lines.append("## 综合经济评价与建议")
        lines.append(f"- 投资回收期：{prompt.split('- 投资回收期: ')[1].split(' 年')[0]} 年")
        lines.append(f"- 绿电比例：{prompt.split('- 绿电比例: ')[1].split('%')[0]}%")
        lines.append("- 建议：请检查 API 配置后重试，或联系系统管理员。")
        
        return "\n".join(lines)


def calculate_metrics_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 LangGraph state 中提取数据并计算经济指标
    
    参数:
        state: LangGraph 状态字典，包含：
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - energy_plan: 能源规划方案
            - cooling_plan: 制冷方案
            - simulation_result: 24小时粗仿真结果（可选）
            
    返回:
        财务指标字典
    """
    # ===== 1. 从 state 提取各 Agent 的输出 =====
    user_reqs = state.get("user_requirements", {})
    env_data = state.get("environmental_data", {})
    energy_plan = state.get("energy_plan", {})
    cooling_plan = state.get("cooling_plan", {})
    simulation_result = state.get("simulation_result", {})
    
    # ===== 2. 基本参数（来自 Agent 1）=====
    location = user_reqs.get("location", "乌兰察布")
    planned_load = user_reqs.get("planned_load", 10000)  # kW
    pue_target = user_reqs.get("pue_target", 1.2)
    green_target = user_reqs.get("green_energy_target", 90) / 100.0  # 转换为小数
    
    # ===== 3. 能源方案（来自 Agent 2）=====
    # 注意：Agent 2(XSimple 版) 输出的是 LLM 报告，可能没有数值化数据
    # 这里使用默认值或从 LLM 报告中解析（简化版先用默认值）
    pv_capacity = energy_plan.get("pv_capacity", 0)  # kW
    storage_capacity = energy_plan.get("storage_capacity", 0)  # kWh
    ppa_ratio = energy_plan.get("ppa_ratio", 0) / 100.0
    
    # ===== 4. 制冷方案（来自 Agent 3）=====
    cooling_tech = cooling_plan.get("cooling_technology", "传统冷却")
    expected_pue = cooling_plan.get("estimated_pue", 1.5)
    cooling_cost = cooling_plan.get("incremental_cost", 0)  # 万元
    
    # ===== 5. 优先使用仿真结果；缺失时回退经验估算 =====
    actual_pue = expected_pue  # 假设实际 PUE 等于设计 PUE

    total_electricity = 0.0
    green_consumption = 0.0
    annual_grid_purchase = 0.0

    sim_summary = simulation_result.get("summary", {}) if simulation_result else {}
    it_curve = simulation_result.get("it_load_curve_mw", []) if simulation_result else []
    green_curve = simulation_result.get("green_supply_curve_mw", []) if simulation_result else []

    has_valid_simulation = bool(it_curve) and bool(green_curve) and len(it_curve) == len(green_curve) == 24

    if has_valid_simulation:
        daily_it_mwh = float(sim_summary.get("daily_it_energy_mwh", sum(it_curve)))
        # 绿电只按可消纳部分计入，避免高估
        daily_green_mwh = float(sim_summary.get("daily_green_supply_mwh", sum(min(green_curve[i], it_curve[i]) for i in range(24))))

        total_electricity = daily_it_mwh * 365.0
        green_consumption = max(0.0, daily_green_mwh) * 365.0
        annual_grid_purchase = max(0.0, total_electricity - green_consumption)
    else:
        # 经验公式估算：基于光伏装机与日照
        annual_sunshine_hours = env_data.get("annual_sunshine_hours", 3000)  # 小时
        pv_generation = pv_capacity * annual_sunshine_hours / 1000  # MWh/年（简单估算）

        total_electricity = planned_load * actual_pue * 8760 / 1000  # MWh/年
        green_consumption = pv_generation
        annual_grid_purchase = max(0.0, total_electricity - green_consumption)
    
    # 碳减排量（基于绿电消纳）
    emission_factor = 0.8 / 1000  # tCO2/kWh（默认值）
    carbon_reduction = green_consumption * 1000 * emission_factor  # 吨 CO2/年
    
    # ===== 6. 计算总用电量 =====
    # 方法 1: 理论计算（用于异常回退）
    total_electricity_theory = planned_load * actual_pue * 8760 / 1000  # MWh
    # 方法 2: 由仿真或估算已得到
    total_electricity_sim = annual_grid_purchase + green_consumption  # MWh

    # 优先使用已得到的数据
    total_electricity = total_electricity_sim if total_electricity_sim > 0 else total_electricity_theory
    
    # ===== 7. 绿电比例 =====
    green_ratio = green_consumption / total_electricity if total_electricity > 0 else 0
    
    # ===== 8. 成本计算 =====
    grid_price = DEFAULT_GRID_PRICE  # 元/kWh
    carbon_price = DEFAULT_CARBON_PRICE  # 元/tCO2
    green_cert_price = DEFAULT_GREEN_CERT_PRICE  # 元/kWh
    
    # 购电成本 (MWh * 元/kWh = 万元，需要除以 10)
    grid_cost = annual_grid_purchase * grid_price / 10  # 万元
    
    # PPA 购电成本（简化估算）
    ppa_volume = total_electricity * ppa_ratio  # MWh
    ppa_price = 0.35  # 元/kWh
    ppa_cost = ppa_volume * ppa_price / 10  # 万元
    
    # 光伏自用节省的电费
    pv_saving = green_consumption * grid_price / 10  # 万元
    
    # 碳排放计算
    emission_factor = DEFAULT_EMISSION_FACTOR / 1000  # tCO2/kWh
    total_emission_base = total_electricity * 1000 * emission_factor  # tCO2
    actual_emission = annual_grid_purchase * 1000 * emission_factor  # tCO2
    emission_reduction_calc = total_emission_base - actual_emission  # tCO2
    
    # 碳减排收益
    carbon_benefit = emission_reduction_calc * carbon_price / 10000  # 万元
    
    # 碳排消纳成本（绿证购买）
    if green_ratio < green_target:
        shortage = (green_target - green_ratio) * total_electricity  # MWh
        carbon_compensation_cost = shortage * green_cert_price / 10  # 万元
    else:
        carbon_compensation_cost = 0
    
    # 总用电成本
    total_cost = grid_cost + ppa_cost + carbon_compensation_cost - pv_saving - carbon_benefit
    
    # ===== 9. 投资估算 =====
    capex_pv = pv_capacity * PV_UNIT_COST / 10000  # 万元
    capex_storage = storage_capacity * STORAGE_UNIT_COST / 10000  # 万元
    capex_total = capex_pv + capex_storage + cooling_cost  # 万元
    
    # ===== 10. 投资回收期 =====
    annual_saving = pv_saving + carbon_benefit  # 万元
    payback_years = capex_total / annual_saving if annual_saving > 0 else 30.0

    # ===== 11. 补齐报告需要的核心财务字段（避免缺失值） =====
    # 年运营成本（万元/年），负值时按 0 处理
    opex_annual = max(0.0, total_cost)

    # NPV（万元）：按折现率与寿命期计算现值
    if DISCOUNT_RATE > 0:
        annuity_factor = (1 - (1 + DISCOUNT_RATE) ** (-LIFETIME_YEARS)) / DISCOUNT_RATE
    else:
        annuity_factor = float(LIFETIME_YEARS)
    npv = annual_saving * annuity_factor - capex_total

    # IRR（%）：简化近似，基于年节省与总投资强度，限制在合理区间
    irr_proxy = (annual_saving / capex_total * 100.0) if capex_total > 0 else 0.0
    irr = max(0.0, min(40.0, irr_proxy))

    # LCOE（元/kWh）：用年总成本除以年总电量
    if total_electricity > 0:
        lcoe = max(0.0, total_cost * 10000 / (total_electricity * 1000))
    else:
        lcoe = 0.0
    
    # ===== 12. 全生命周期碳减排 =====
    lifetime_reduction = emission_reduction_calc * LIFETIME_YEARS
    
    # ===== 汇总结果 =====
    results = {
        'location': location,
        'planned_load': planned_load,
        'total_electricity': round(total_electricity, 2),  # MWh
        'annual_grid_purchase': round(annual_grid_purchase, 2),  # MWh
        'green_consumption': round(green_consumption, 2),  # MWh
        'ppa_volume': round(ppa_volume, 2),  # MWh
        'green_ratio': round(green_ratio * 100, 2),  # %
        'green_target': round(green_target * 100, 2),  # %
        'actual_pue': actual_pue,
        'pue_target': pue_target,
        'grid_price': grid_price,
        'ppa_price': ppa_price,
        'carbon_price': carbon_price,
        'grid_cost': round(grid_cost, 2),  # 万元
        'ppa_cost': round(ppa_cost, 2),  # 万元
        'pv_saving': round(pv_saving, 2),  # 万元
        'carbon_benefit': round(carbon_benefit, 2),  # 万元
        'carbon_compensation_cost': round(carbon_compensation_cost, 2),  # 万元
        'total_cost': round(total_cost, 2),  # 万元
        'capex_total': round(capex_total, 2),  # 万元
        'annual_saving': round(annual_saving, 2),  # 万元
        'payback_years': round(payback_years, 2),
        'payback_period': round(payback_years, 2),
        'opex_annual': round(opex_annual, 2),
        'npv': round(npv, 2),
        'irr': round(irr, 2),
        'lcoe': round(lcoe, 4),
        'emission_reduction': round(emission_reduction_calc, 2),  # tCO2/年
        'lifetime_reduction': round(lifetime_reduction, 2),  # tCO2
        'cooling_tech': cooling_tech,
        'curtailment_rate': 0,
        'simulation_used': has_valid_simulation
    }
    
    return results


# ============================================================
# LangGraph 节点函数
# ============================================================

def financial_consultant_node(state: dict) -> dict:
    """
    Agent 5: 综合评价与投资决策专家 - LangGraph Node
    
    计算 CAPEX 和 OPEX，分析投资回报。
    
    参数:
        state: GreenDataCenterState 类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - electricity_price: 电价数据
            - energy_plan: 能源规划方案
            - cooling_plan: 制冷方案
            - review_result: 审核结果
            
    返回:
        更新后的 state，新增:
            - financial_analysis: 财务分析
    """
    print("\n" + "="*60)
    print("💰 [综合评价与投资决策专家] 开始工作")
    print("="*60)
    
    # 打印输入数据摘要
    user_reqs = state.get("user_requirements", {})
    print(f"📊 项目位置：{user_reqs.get('location', '未知')}")
    print(f"⚡ 计划负荷：{user_reqs.get('planned_load', 0)} kW")
    
    # ===== 1. 计算财务指标 =====
    print("\n📈 正在计算财务指标...")
    financial_analysis = calculate_metrics_from_state(state)
    
    print(f"  - 总投资：{financial_analysis['capex_total']} 万元")
    print(f"  - 年节省：{financial_analysis['annual_saving']} 万元")
    print(f"  - 投资回收期：{financial_analysis['payback_years']} 年")
    print(f"  - 年碳减排：{financial_analysis['emission_reduction']} 吨 CO₂")
    
    # 添加详细财务数据到 state
    financial_analysis['capex_breakdown'] = {
        'pv_system': financial_analysis.get('capex_total', 0) * 0.6,  # 假设光伏占 60%
        'storage_system': financial_analysis.get('capex_total', 0) * 0.3,  # 储能占 30%
        'cooling_system': financial_analysis.get('capex_total', 0) * 0.1  # 制冷占 10%
    }
    
    # 生成 Markdown 报告
    print("\n📝 正在生成财务分析报告...")
    client = LLMClient()
    prompt = f"""
你是一个绿色数据中心经济评估专家。
请基于以下经济和碳排数据，撰写一份数据中心经济型评估报告（Markdown 格式）：
- 建设成本明细: {json.dumps(financial_analysis['capex_breakdown'], indent=2)}
- 总建设成本: {financial_analysis['capex_total']} 万元
- 年运营成本估算: {financial_analysis['opex_annual']} 万元/年
- 年节省: {financial_analysis['annual_saving']} 万元/年
- 投资回收期: {financial_analysis['payback_years']} 年
- 年碳减排: {financial_analysis['emission_reduction']} 吨 CO2
- 绿电比例: {financial_analysis['green_ratio']}%
- 绿电目标: {financial_analysis['green_target']}%
请包括以下内容：
1. 报告概述
2. 建设成本分析
3. 年度运营成本分析
4. 碳排放与碳抵消分析
5. 综合经济评价与建议
请用 Markdown 格式输出。
"""
    
    report_md = client.generate_report(prompt)
    
    # 保存 Markdown 报告
    with open("data_center_economic_evaluation.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    
    print("✅ Markdown 报告已生成: data_center_economic_evaluation.md")
    
    # 将报告添加到财务分析结果中
    financial_analysis['report_md'] = report_md
    
    print("\n" + "="*60)
    print("✅ [综合评价与投资决策专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态
    return {
        **state,
        "financial_analysis": financial_analysis
    }


# ============================================================
# 主程序入口（独立测试）
# ============================================================

if __name__ == "__main__":
    print("===== 测试 Agent 5: 综合评价与投资决策专家 =====")
    
    # 模拟完整的 state
    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "planned_load": 5000,
            "pue_target": 1.2,
            "green_energy_target": 90
        },
        "environmental_data": {
            "annual_temperature": 5.5,
            "carbon_emission_factor": 0.6479
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "storage_capacity": 6000,
            "ppa_ratio": 20
        },
        "cooling_plan": {
            "cooling_technology": "液冷",
            "estimated_pue": 1.15,
            "incremental_cost": 500
        }
    }
    
    # 执行节点
    result = financial_consultant_node(test_state)
    
    # 显示财务分析结果
    print("\n" + "="*80)
    print("财务分析结果:")
    print("="*80)
    print(f"总投资：{result['financial_analysis']['capex_total']} 万元")
    print(f"投资回收期：{result['financial_analysis']['payback_years']} 年")
    print(f"年碳减排：{result['financial_analysis']['emission_reduction']} 吨 CO₂")
