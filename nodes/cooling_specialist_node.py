import os
import sys
import json
from typing import TypedDict, Any, Dict

# 将项目的根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入大模型相关库
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from langchain_community.chat_models import ChatTongyi

# 导入 RAG builder（使用标准接口）
from tools.rag_builder import build_or_load_vector_store
from tools import query_knowledge_base, query_knowledge_base_as_text


# ======================== 保留核心映射表 ========================
CITY_TO_PROVINCE = {
    "乌兰察布": "内蒙古", "北京": "北京", "上海": "上海", "广州": "广东",
    "深圳": "广东", "杭州": "浙江", "成都": "四川", "武汉": "湖北",
    "西安": "陕西", "南京": "江苏"
}

# 兜底用：省份级制冷基准参数映射
PROVINCE_COOLING_BASE_PARAMS = {
    "北京": {"PUE_Limit": 1.35, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "上海": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 15.0},
    "天津": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "重庆": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "河北": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "山西": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "内蒙古": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 30.0},
    "辽宁": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "吉林": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "黑龙江": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "江苏": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 15.0},
    "浙江": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 15.0},
    "安徽": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "福建": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "江西": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "山东": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "河南": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "湖北": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "湖南": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "广东": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 12.0},
    "广西": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "海南": {"PUE_Limit": 1.10, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "四川": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "贵州": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "云南": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "西藏": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "陕西": {"PUE_Limit": 1.20, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "甘肃": {"PUE_Limit": 1.20, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "青海": {"PUE_Limit": 1.20, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "宁夏": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "新疆": {"PUE_Limit": 1.20, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "香港": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 15.0},
    "澳门": {"PUE_Limit": 1.35, "WUE_Limit": 1.6, "cabinet_power_limit": 15.0},
    "台湾": {"PUE_Limit": 1.25, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "default": {"PUE_Limit": 1.30, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0}
}

# 温度对制冷能效系数（COP）的修正因子
TEMP_COP_CORRECTION = {
    "≤0": 1.2, "1-10": 1.1, "11-20": 1.0, "21-30": 0.9, ">30": 0.8
}

# ======================== Prompt模板 ========================
PARAM_EXTRACTION_PROMPT = PromptTemplate(
    template="""
你是数据中心暖通工程领域的资深专家，需从检索到的行业规范/技术文档中提取制冷规划关键计算参数，严格按JSON格式输出。
【检索知识库内容】
{retrieved_context}
【用户基础需求】
{user_requirements}
【地域基准参数(作参考)】
{regional_base_params}

【提取规则】
1. 必须提取以下参数（优先使用知识库内容，若无明确值则使用地域基准参数或行业默认值）：
   - PUE_Limit: 该地区数据中心PUE强制限值（无单位）
   - WUE_Limit: 该地区数据中心WUE强制限值（单位：L/kWh，国标红线≤1.6）
   - cooling_eff_coeff: 制冷系统能效系数（COP，制冷量/耗电量，无单位，风冷一般3-4，液冷一般4-6）
   - waste_heat_recovery_coeff: 余热回收系数（可回收余热占制冷量比例，无单位，通常0-0.8）
   - facility_loss_coeff: 基础设施损耗系数（无单位，通常0.05-0.15）
   - cabinet_power_limit: 该地区高密度机柜功率限值界线（kW/机柜）
   - regional_cooling_preference: 该地区推荐的制冷技术类型（如"液冷", "风冷", "间接蒸发冷却"）
2. 输出仅保留JSON结构，无任何多余文字、注释或说明
3. 数值均保留2位小数

【示例输出】
{{
    "PUE_Limit": 1.15,
    "WUE_Limit": 1.60,
    "cooling_eff_coeff": 4.20,
    "waste_heat_recovery_coeff": 0.60,
    "facility_loss_coeff": 0.08,
    "cabinet_power_limit": 20.00,
    "regional_cooling_preference": "液冷"
}}
""",
    input_variables=["retrieved_context", "user_requirements", "regional_base_params"]
)

COOLING_SCHEME_PROMPT = ChatPromptTemplate.from_template("""
你是一位顶级的数据中心制冷系统专家，精通各类制冷技术选型与落地。
请基于检索上下文、用户约束和地域特征，给出精准的制冷方案建议（仅保留核心数据，无冗余描述）。
**核心约束条件：**
1. 地域适配性（结合{region}的年均温度{annual_temp}℃、省份制冷规范）；
2. 算力密度：{cabinet_power} kW/机柜；
3. PUE目标：{target_pue}；
4. 绿电协同（消纳目标{green_energy_target}%）；
5. WUE合规（国标≤1.6 L/kWh）。

**检索到的当地规范上下文:**
{context}

**回答要求（必须按以下要点简洁输出，控制在200字内）：**
- 主选技术路线（精确到制冷末端和冷源设备）
- 备选技术路线
- 选型理由（结合气候、算力密度和绿电消纳能力）
- 绿电与冷库/储能的协同建议
""")

# ======================== 核心类 ========================
class CoolingAgent3:
    """
    Agent 3: 数据中心暖通与制冷架构专家
    根据 Agent 1、2 的输出和 RAG 知识库，计算 PUE/WUE，推荐制冷架构
    """
    def __init__(self):
        # 不再维护 RAG，直接使用标准接口查询
        pass
        
        # 初始化 LLM
        self.llm = ChatTongyi(model="qwen-plus", temperature=0.1)

    def retrieve_context(self, region: str, province: str, annual_temp: float, cabinet_power: float) -> str:
        """从知识库检索相关信息并构建上下文（使用标准接口）"""
        queries = [
            f"{province} 数据中心 PUE WUE 标准 规范",
            f"{province} 数据中心 算力密度 {cabinet_power}kW 制冷方案推荐",
            f"年均温度 {annual_temp}度 数据中心 制冷 节能"
        ]
        
        docs_content = []
        print("🔍 正在通过 RAG 检索当地制冷政策与技术规范...")
        
        # 使用标准接口 query_knowledge_base_as_text 直接获取拼接好的文本
        for q in queries:
            context_text = query_knowledge_base_as_text(q, k=3)
            if context_text:
                docs_content.append(context_text)
        
        # 如果没有检索到，使用内置规则生成一段上下文供大模型参考
        if not docs_content:
            print("⚠️ 未检索到有效知识，启用本地兜底规则构建上下文...")
            regional_params = PROVINCE_COOLING_BASE_PARAMS.get(province, PROVINCE_COOLING_BASE_PARAMS["default"])
            fallback_text = f"{province}地区数据中心强制要求PUE限值≤{regional_params['PUE_Limit']}，WUE限值≤{regional_params['WUE_Limit']} L/kWh。针对{cabinet_power}kW/机柜的算力（限值{regional_params['cabinet_power_limit']}），超限建议采用液冷技术，未超限建议风冷结合自然冷却。年均温{annual_temp}℃下，需注重节水与干冷。"
            docs_content.append(fallback_text)

        return "\n".join(docs_content)

    def get_cop_correction_factor(self, annual_temp: float) -> float:
        """根据年均温度获取COP修正系数"""
        if annual_temp <= 0: return TEMP_COP_CORRECTION["≤0"]
        elif 1 <= annual_temp <= 10: return TEMP_COP_CORRECTION["1-10"]
        elif 11 <= annual_temp <= 20: return TEMP_COP_CORRECTION["11-20"]
        elif 21 <= annual_temp <= 30: return TEMP_COP_CORRECTION["21-30"]
        else: return TEMP_COP_CORRECTION[">30"]

    def extract_cooling_params(self, context: str, user_reqs: Dict[str, Any], province: str) -> Dict[str, float]:
        """用 LLM 从知识库上下文中提取制冷计算核心指标"""
        regional_params = PROVINCE_COOLING_BASE_PARAMS.get(province, PROVINCE_COOLING_BASE_PARAMS["default"])
        
        if self.llm:
            print("🧠 正在调用 LLM 解析制冷核心指标(PUE/WUE/COP)...")
            prompt = PARAM_EXTRACTION_PROMPT.format(
                retrieved_context=context,
                user_requirements=json.dumps(user_reqs, ensure_ascii=False),
                regional_base_params=json.dumps(regional_params, ensure_ascii=False)
            )
            try:
                parser = JsonOutputParser()
                response = self.llm.invoke(prompt)
                params = parser.parse(response.content)
                print(f"✅ 从 RAG+LLM 提取制冷参数成功: {params}")
                return params
            except Exception as e:
                print(f"⚠️ LLM 解析制冷参数失败: {e}，回退至兜底参数...")

        # 如果没有LLM或解析失败，使用兜底参数
        cab_power = user_reqs.get('算力_density', 8)
        cab_limit = regional_params.get("cabinet_power_limit", 20.00)
        return {
            "PUE_Limit": regional_params.get("PUE_Limit", 1.15),
            "WUE_Limit": regional_params.get("WUE_Limit", 1.60),
            "cooling_eff_coeff": 4.5 if cab_power >= cab_limit else 3.8,
            "waste_heat_recovery_coeff": 0.60,
            "facility_loss_coeff": 0.07,
            "cabinet_power_limit": cab_limit,
            "regional_cooling_preference": "液冷" if cab_power >= cab_limit else "风冷"
        }

    def calculate_cooling_kpis(self, params: Dict[str, Any], project_info: Dict[str, Any], env_data: Dict[str, Any]) -> Dict[str, Any]:
        """严格根据公式推算 PUE 和 WUE"""
        it_load = project_info.get("planned_load", 0)
        cabinet_power = project_info.get("算力_density", 0)
        annual_temp = env_data.get("annual_temperature", 15.0)
        raw_water_usage_lh = env_data.get("raw_water_usage", 6000.0) # 如果环境数据中有小时耗水量，则使用

        # 密度校正和温度校正
        density_correction = 1.1 if cabinet_power >= params.get("cabinet_power_limit", 20.0) else 1.0
        cop_correction = self.get_cop_correction_factor(annual_temp)
        corrected_cop = params.get("cooling_eff_coeff", 4.0) * cop_correction

        # 1. 制冷系统原始负载需求
        cooling_load_kw = it_load * 1.1 * density_correction
        
        # 2. 余热回收能力（抵消一部分制冷能耗）
        waste_heat_recovery_kw = cooling_load_kw * params.get("waste_heat_recovery_coeff", 0.0)
        corrected_cooling_load = max(0.0, cooling_load_kw - waste_heat_recovery_kw)
        
        # 3. 实际制冷功率 = 需要排走的热量 / COP
        cooling_power_kw = corrected_cooling_load / corrected_cop if corrected_cop > 0 else 0.0
        
        # 4. 其他基础设施损耗
        facility_loss_kw = it_load * params.get("facility_loss_coeff", 0.1)

        # 5. PUE计算 = (IT负载 + 制冷能耗 + 其他设施能耗) / IT负载
        total_energy = it_load + cooling_power_kw + facility_loss_kw
        pue = total_energy / it_load if it_load > 0 else 1.0
        
        # 6. WUE计算 = 总耗水量(L/h) / IT负载(kW)
        wue = raw_water_usage_lh / it_load if it_load > 0 else 0.0

        return {
            "predicted_PUE": round(pue, 3),
            "predicted_WUE": round(wue, 3),
            "waste_heat_recovery_kw": round(waste_heat_recovery_kw, 2),
            "cooling_power_kw": round(cooling_power_kw, 2),
            "facility_loss_kw": round(facility_loss_kw, 2),
            "PUE_Limit": params.get("PUE_Limit"),
            "WUE_Limit": params.get("WUE_Limit"),
            "corrected_cop": round(corrected_cop, 2)
        }

    def generate_cooling_scheme(self, context: str, project_info: Dict[str, Any], env_data: Dict[str, Any], province: str) -> str:
        """根据前置提取的信息，调用 LLM 生成描述性制冷建议"""
        
        print("💡 正在调用 LLM 生成最终的制冷方案策略...")
        prompt_val = COOLING_SCHEME_PROMPT.format(
            region=project_info.get("location", "默认地区"),
            annual_temp=env_data.get("annual_temperature", 15.0),
            cabinet_power=project_info.get("算力_density", 8),
            target_pue=project_info.get("pue_target", 1.2),
            green_energy_target=project_info.get("green_energy_target", 90),
            context=context
        )
        try:
            response = self.llm.invoke(prompt_val)
            return response.content.strip()
        except Exception as e:
            print(f"❌ 生成详细制冷建议失败: {e}")
            return "制冷方案生成失败，请结合基础KPI数据进行手工评估。"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """主执行链路（适配 LangGraph 状态格式）"""
        # 兼容 Agent 1 的输出格式：user_requirements 包含 project_info 的所有字段
        user_reqs = state.get("user_requirements", {})
        env_data = state.get("environmental_data", {})
            
        # 构建 project_info（兼容旧版和新版字段名）
        project_info = {
            "location": user_reqs.get("location", user_reqs.get("project_info", {}).get("location", "北京")),
            "business_type": user_reqs.get("business_type", "通用"),
            "planned_area": user_reqs.get("planned_area", 10000),
            "planned_load": user_reqs.get("planned_load", 5000),
            "computing_power_density": user_reqs.get("computing_power_density", user_reqs.get("Computing_power_density", user_reqs.get("算力_density", 8))),
            "priority": user_reqs.get("priority", "环保型"),
            "green_energy_target": user_reqs.get("green_energy_target", 90),
            "pue_target": user_reqs.get("pue_target", 1.2),
            "budget_constraint": user_reqs.get("budget_constraint", 10000)
        }
            
        region = project_info.get("location", "北京")
        province = CITY_TO_PROVINCE.get(region, "北京")
        annual_temp = env_data.get("annual_temperature", 15.0)
        cabinet_power = project_info.get("算力_density", 8.0)
    
        # 1. RAG 知识检索
        context = self.retrieve_context(region, province, annual_temp, cabinet_power)
    
        # 2. 从上下文中提取核心计算参数
        extracted_params = self.extract_cooling_params(context, project_info, province)
    
        # 3. 计算物理 KPI（PUE/WUE）
        kpis = self.calculate_cooling_kpis(extracted_params, project_info, env_data)
    
        # 4. 生成报告文本
        scheme_text = self.generate_cooling_scheme(context, project_info, env_data, province)
    
        # 5. 组合标准化数据包输出（符合 graph.py 的 CoolingPlan 类型定义）
        cooling_plan = {
            # LLM 生成的详细方案
            "scheme_detail_brief": scheme_text,
                
            # 制冷系统核心参数
            "cooling_technology": extracted_params.get("regional_cooling_preference", "风冷"),
            "estimated_pue": kpis.get("predicted_PUE", 1.3),
            "predicted_wue": kpis.get("predicted_WUE", 1.6),
                
            # 详细计算参数（供后续 Agent 使用）
            "cooling_project_info": {
                "location": region,
                "province": province,
                "it_load_kW": project_info.get("planned_load"),
                "cabinet_power_kW": cabinet_power,
                "target_pue": project_info.get("pue_target"),
                "green_energy_target": project_info.get("green_energy_target")
            },
            "cooling_calc_params": extracted_params,
            "cooling_kpis": kpis
        }
            
        # 更新 state（LangGraph 标准格式）
        return {
            **state,
            "cooling_plan": cooling_plan
        }


# ======================== LangGraph 节点函数 ========================

def cooling_specialist_node(state: dict) -> dict:
    """
    Agent 3: 暖通与制冷架构专家 - LangGraph Node
    
    基于地理位置、算力密度和 PUE 目标，结合 RAG 知识库选择最优制冷技术方案。
    
    参数:
        state: GreenDataCenterState 类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - energy_plan: 能源规划方案（可选）
            
    返回:
        更新后的 state，新增:
            - cooling_plan: 制冷方案
    """
    print("\n" + "="*60)
    print("❄️  [Agent 3: 暖通与制冷架构专家] 开始工作")
    print("="*60)
    
    # 获取输入数据
    user_reqs = state.get("user_requirements", {})
    env_data = state.get("environmental_data", {})
    
    location = user_reqs.get("location", "未知")
    computing_density = user_reqs.get("computing_power_density", 
                                     user_reqs.get("Computing_power_density", 8))
    pue_target = user_reqs.get("pue_target", 1.2)
    planned_load = user_reqs.get("planned_load", 0)
    annual_temp = env_data.get("annual_temperature", 15)
    
    print(f"📊 输入参数:")
    print(f"  - 位置：{location}")
    print(f"  - 算力密度：{computing_density} kW/机柜")
    print(f"  - PUE 目标：{pue_target}")
    print(f"  - 计划负荷：{planned_load} kW")
    print(f"  - 年均温度：{annual_temp}°C")
    
    # 创建制冷 Agent 并执行
    cooling_agent = CoolingAgent3()
    result_state = cooling_agent.process(state)
    
    # 打印结果
    cooling_plan = result_state.get("cooling_plan", {})
    print(f"\n❄️  制冷方案:")
    print(f"  - 推荐技术：{cooling_plan.get('cooling_technology', 'N/A')}")
    print(f"  - 预计 PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
    print(f"  - 预计 WUE: {cooling_plan.get('predicted_wue', 'N/A')}")
    
    scheme_brief = cooling_plan.get("scheme_detail_brief", "")
    if scheme_brief:
        print(f"\n📝 方案摘要:")
        print(scheme_brief[:300] + ("..." if len(scheme_brief) > 300 else ""))
    
    print("\n" + "="*60)
    print("✅ [Agent 3: 暖通与制冷架构专家] 工作完成")
    print("="*60)
    
    return result_state


# ======================== 独立测试入口 ========================
if __name__ == "__main__":
    print("===== 开始测试 Agent 3: 数据中心暖通与制冷架构专家 =====")
    
    # 初始化制冷 Agent
    cooling_agent = CoolingAgent3()

    # 模拟上游 Agent (Agent 1 & Agent 2) 传入的 SystemState 数据
    upstream_input = {
        "project_info": {
            "location": "杭州",
            "business_type": "通用",
            "planned_area": 10000,
            "planned_load": 5000,
            "算力_density": 8,
            "priority": "环保型",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 10000,
            "timestamp": "2026-03-10"
        },
        "environmental_data": {
            "annual_temperature": 17.59,
            "annual_wind_speed": 2.21,
            "annual_sunshine_hours": 3305.89,
            "carbon_emission_factor": 0.4974,
            "raw_water_usage": 6000.0  # 假设的实时耗水量，单位 L/h，适配 5000kW
        },
        "electricity_price": {
            "尖峰电价": 0.9287, "高峰电价": 0.9287, "平段电价": 0.5629,
            "低谷电价": 0.2533, "深谷电价": 0.2533, "最大峰谷价差": 0.6754
        },
        "renewable_potential": {
            "renewable_ratio": 0.9,
            "renewable_surplus": True,
            "renewable_available_hours": 1800
        }
    }

    # 执行主流程
    print("\n--- 接收上游传入的核心参数 ---")
    print(f"项目位置: {upstream_input['project_info']['location']}")
    print(f"规划负载: {upstream_input['project_info']['planned_load']}kW")
    print(f"算力密度: {upstream_input['project_info']['算力_density']}kW/机柜")
    print(f"年均温度: {upstream_input['environmental_data']['annual_temperature']}℃")
    print(f"PUE目标: {upstream_input['project_info']['pue_target']}")

    print("\n--- 数据包已转交给 Agent 4 ---")
    print("虚拟运行仿真专家 Agent 4 可以直接读取 'cooling_plan' 进入验证阶段")