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
from langchain_community.chat_models import ChatTongyi

# 导入 RAG builder（使用标准接口，匹配原有架构）
try:
    from tools.rag_builder import build_or_load_vector_store
    has_rag = True
except ImportError:
    has_rag = False

# ======================== 核心映射表 ========================
CITY_TO_PROVINCE = {
    "乌兰察布": "内蒙古", "北京": "北京", "上海": "上海", "广州": "广东",
    "深圳": "广东", "杭州": "浙江", "成都": "四川", "武汉": "湖北",
    "西安": "陕西", "南京": "江苏", "张家口": "河北", "三亚": "海南", "丽江": "云南"
}

# 温度对制冷能效系数（COP）的修正因子
TEMP_COP_CORRECTION = {
    "≤0": 1.2, "1-10": 1.1, "11-20": 1.0, "21-30": 0.9, ">30": 0.8
}

# 北方省份列表（用于余热回收系数的地域修正）
NORTHERN_PROVINCES = [
    "内蒙古", "北京", "天津", "河北", "山西", "辽宁", 
    "吉林", "黑龙江", "陕西", "甘肃", "青海", "宁夏", "新疆"
]

# 水资源紧缺综合评价指数 CWSI (Water Scarcity Index, 0~1，值越小越缺水)
CWSI_MAP = {
    "北京": 0.44, "天津": 0.38, "河北": 0.31, "上海": 0.61, "江苏": 0.54,
    "浙江": 0.64, "广东": 0.64, "四川": 0.51, "重庆": 0.51, "内蒙古": 0.39,
    "贵州": 0.46, "甘肃": 0.35, "宁夏": 0.31, "default": 0.50
}

# 各省份制冷基准参数（用于兜底规则）
PROVINCE_COOLING_BASE_PARAMS = {
    "北京": {"PUE_Limit": 1.15, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "上海": {"PUE_Limit": 1.25, "WUE_Limit": 1.8, "cabinet_power_limit": 20.0},
    "广东": {"PUE_Limit": 1.25, "WUE_Limit": 1.8, "cabinet_power_limit": 20.0},
    "浙江": {"PUE_Limit": 1.25, "WUE_Limit": 1.8, "cabinet_power_limit": 20.0},
    "内蒙古": {"PUE_Limit": 1.15, "WUE_Limit": 1.5, "cabinet_power_limit": 25.0},
    "四川": {"PUE_Limit": 1.20, "WUE_Limit": 1.6, "cabinet_power_limit": 20.0},
    "default": {"PUE_Limit": 1.30, "WUE_Limit": 1.8, "cabinet_power_limit": 20.0}
}

# ======================== Prompt模板 ========================
PARAM_EXTRACTION_PROMPT = PromptTemplate(
    template="""
你是数据中心暖通工程领域的资深专家，需从检索到的行业规范/技术文档中提取制冷规划关键计算参数，严格按JSON格式输出。
【检索知识库内容/兜底规则】
{retrieved_context}
【用户基础需求】
{user_requirements}
【动态基准参数(用户输入的当地限值)】
{dynamic_base_params}

【提取规则】
1. 必须提取以下参数（优先使用用户基础需求和动态基准参数）：
   - PUE_Limit: 该地区数据中心PUE强制限值（无单位）
   - WUE_Limit: 该地区数据中心WUE强制限值（单位：L/kWh）
   - cooling_eff_coeff: 制冷系统能效系数（COP）
   - waste_heat_recovery_coeff: 余热回收系数（0-0.8）
   - facility_loss_coeff: 基础设施损耗系数（0.05-0.15）
   - cabinet_power_limit: 高密度机柜功率限值
   - regional_cooling_preference: 推荐的制冷技术类型
2. 输出仅保留JSON结构，无任何多余文字、注释或说明
3. 数值均保留2位小数
""",
    input_variables=["retrieved_context", "user_requirements", "dynamic_base_params"]
)

COOLING_SCHEME_PROMPT = ChatPromptTemplate.from_template("""
你是一位顶级的数据中心制冷系统架构专家，请基于系统内部进行的多目标寻优计算结果，生成一份极具数理逻辑的制冷方案设计报告。

**【系统内部多目标寻优计算结果 (Optimization Trace)】**
{optimization_trace}

**【核心物理与环境约束条件】**
1. 地域与气候：{province}{region}（年均温 {annual_temp}℃，水资源紧缺指数CWSI={cwsi}）
2. 当地政策红线：PUE限值 {pue_limit}，WUE限值 {wue_limit} L/kWh
3. 算力与密度：单机柜算力密度 {cabinet_power} kW/机柜，类型界定限值 {cabinet_power_limit} kW/机柜
4. 用户目标偏好：优先级策略为【{priority}】（PUE目标: {target_pue}，绿电消纳目标: {green_energy_target}%）
5. 上游绿电规划建议：{green_energy_plan_text}

**【强制回答要求】**
请严格遵循以下排版与语法规范生成报告：
1. **绝对禁止使用完整的 LaTeX 文档结构**（如 `\\documentclass`、`\\begin{{document}}`、`\\section`、`\\subsection`、`\\begin{{itemize}}`、`\\begin{{tabular}}` 等排版指令）。
2. **必须使用标准 Markdown 语法**进行排版（如使用 `#` 和 `##` 表示标题，`-` 或 `*` 表示列表，`|---|` 表示表格，`**` 表示加粗）。
3. **数学公式规范**：仅在表达复杂数学公式或变量时使用 LaTeX 语法，且必须使用单美元符号 `$公式$`（行内）或双美元符号 `$$公式$$`（行间）包裹。不要在普通文字文本中使用 LaTeX 语法排版。
4. 报告必须包含以下四个核心章节：
   - **制冷策略寻优逻辑解析**：显式列出行间目标函数公式 $$\\min F_{{strategy}} = \\alpha \\cdot f(PUE) + \\beta \\cdot f(WUE) + \\gamma \\cdot f(TCO) + \\delta \\cdot f(CUE) - \\varepsilon \\cdot f(WHR)$$，并解释系统是如何根据用户的优先级（经济/环保/可靠）和当地水资源（CWSI）动态调整各项权重的（结合 Trace 中的权重数据）。
   - **多方案打分对比**：以标准 Markdown 表格（`|---|`）形式简述备选方案的得分情况，说明为何最终胜出的方案（得分最低者）是该环境下的最优解。
   - **最优架构落地指南**：针对胜出方案（主选路线），给出具体的末端（风冷/液冷）与冷源（冷却塔/干冷器）配置建议，必须体现算力密度引发的散热物理边界。
   - **源网荷储与余热协同**：必须结合上游传入的绿电规划建议（光伏/PPA/绿证配置策略）与本系统的冷却选型及余热回收潜力，给出绿电消纳与余热利用的闭环联动协同策略。
""")

# ======================== 核心类 ========================
class CoolingAgent3:
    def __init__(self):
        # 不再维护 RAG，直接使用标准接口查询
        pass
        
        # 初始化 LLM（尝试）
        try:
            llm_api_key = os.getenv("LLM_API_KEY", "")
            if not llm_api_key:
                raise ValueError("环境变量 LLM_API_KEY 未设置")

            self.llm = ChatTongyi(
                model="qwen-plus", 
                temperature=0.1,
                dashscope_api_key=llm_api_key
            )
        except Exception as e:
            print(f"⚠️ LLM 初始化失败: {e}，将使用兜底参数")
            self.llm = None

    def retrieve_context(self, region: str, province: str, annual_temp: float, cabinet_power: float) -> str:
        """从知识库检索相关信息并构建上下文（使用标准接口）"""
        try:
            # 直接返回兜底文本，避免RAG检索超时
            print("⚠️ 跳过RAG检索，直接使用本地兜底规则构建上下文...")
            regional_params = PROVINCE_COOLING_BASE_PARAMS.get(province, PROVINCE_COOLING_BASE_PARAMS["default"])
            fallback_text = f"{province}地区数据中心强制要求PUE限值≤{regional_params['PUE_Limit']}，WUE限值≤{regional_params['WUE_Limit']} L/kWh。针对{cabinet_power}kW/机柜的算力（限值{regional_params['cabinet_power_limit']}），超限建议采用液冷技术，未超限建议风冷结合自然冷却。年均温{annual_temp}℃下，需注重节水与干冷。"
            return fallback_text
        except Exception as e:
            print(f"⚠️ 构建上下文失败: {e}，启用本地兜底规则")
            regional_params = PROVINCE_COOLING_BASE_PARAMS.get(province, PROVINCE_COOLING_BASE_PARAMS["default"])
            fallback_text = f"{province}地区数据中心强制要求PUE限值≤{regional_params['PUE_Limit']}，WUE限值≤{regional_params['WUE_Limit']} L/kWh。针对{cabinet_power}kW/机柜的算力（限值{regional_params['cabinet_power_limit']}），超限建议采用液冷技术，未超限建议风冷结合自然冷却。年均温{annual_temp}℃下，需注重节水与干冷。"
            return fallback_text

    def get_cop_correction_factor(self, annual_temp: float) -> float:
        if annual_temp <= 0: return TEMP_COP_CORRECTION["≤0"]
        elif 1 <= annual_temp <= 10: return TEMP_COP_CORRECTION["1-10"]
        elif 11 <= annual_temp <= 20: return TEMP_COP_CORRECTION["11-20"]
        elif 21 <= annual_temp <= 30: return TEMP_COP_CORRECTION["21-30"]
        else: return TEMP_COP_CORRECTION[">30"]

    # def retrieve_context(self, region: str, province: str, annual_temp: float, cabinet_power: float, cabinet_power_limit: float, pue_limit: float, wue_limit: float) -> str:
    #     queries = [
    #         f"{province} 数据中心 余热回收 政策",
    #         f"数据中心 单机柜功率 {cabinet_power}kW 制冷策略 液冷 风冷"
    #     ]
    #     docs_content = []
    #     if self.retriever:
    #         try:
    #             for q in queries:
    #                 docs = self.retriever.invoke(q)
    #                 for d in docs:
    #                     if d.page_content not in docs_content:
    #                         docs_content.append(d.page_content)
    #         except Exception:
    #             pass
        
    #     if not docs_content:
    #         if cabinet_power <= 20:
    #             strategy_desc = "风冷仍是主流。建议采用高效风冷技术结合冷热通道隔离。"
    #         elif 20 < cabinet_power <= 60:
    #             strategy_desc = "液冷技术成为必然选择。建议采用冷板式液冷，这种风液混合部署方案商用化程度最高。"
    #         else:
    #             strategy_desc = "建议直接采用浸没式液冷（单相或相变浸没），理论上彻底解决局部热点问题。"
                
    #         docs_content.append(
    #             f"【合规基准】强制要求PUE限值≤{pue_limit}，WUE限值≤{wue_limit} L/kWh。\n"
    #             f"【权威策略兜底】根据行业制冷演进路径，{strategy_desc}\n"
    #         )

    #     return "\n".join(docs_content)
    
    def retrieve_context(self, region: str, province: str, annual_temp: float, cabinet_power: float, cabinet_power_limit: float, pue_limit: float, wue_limit: float) -> str:
        queries = [
            f"{province} 数据中心 余热回收 政策",
            f"数据中心 单机柜功率 {cabinet_power}kW 制冷策略 液冷 风冷"
        ]
        docs_content = []
        # if self.retriever:
        #     try:
        #         for q in queries:
        #             docs = self.retriever.invoke(q)
        #             for d in docs:
        #                 if d.page_content not in docs_content:
        #                     docs_content.append(d.page_content)
        #     except Exception:
        #         pass
        if cabinet_power <= 20:
            strategy_desc = "风冷仍是主流。建议采用高效风冷技术结合冷热通道隔离。"
        elif 20 < cabinet_power <= 60:
            strategy_desc = "液冷技术成为必然选择。建议采用冷板式液冷，这种风液混合部署方案商用化程度最高。"
        else:
            strategy_desc = "建议直接采用浸没式液冷（单相或相变浸没），理论上彻底解决局部热点问题。"
            
        docs_content.append(
            f"【合规基准】强制要求PUE限值≤{pue_limit}，WUE限值≤{wue_limit} L/kWh。\n"
            f"【权威策略兜底】根据行业制冷演进路径，{strategy_desc}\n"
        )

        return "\n".join(docs_content)

    def _calculate_recovery_coeff(self, user_reqs: Dict[str, Any], province: str) -> float:
        cab_power = user_reqs.get('computing_power_density', 8)
        cab_power_limit = user_reqs.get("cabinet_power_limit", 20.0)
        green_energy_target = user_reqs.get("green_energy_target", 90)
        is_liquid_cooling = cab_power > 20 or cab_power >= cab_power_limit

        base_coeff = 0.75 if is_liquid_cooling else 0.55
        region_correction = 0.1 if province in NORTHERN_PROVINCES else 0.0
        green_correction = 0.05 if green_energy_target >= 90 else -0.05
        density_correction = 0.05 if cab_power > 60 else (-0.05 if cab_power <= 20 else 0.0)

        return round(min(0.8, max(0.0, base_coeff + region_correction + green_correction + density_correction)), 2)

    def evaluate_cooling_strategies(self, project_info: Dict[str, Any], province: str) -> Dict[str, Any]:
        cabinet_power = project_info.get("computing_power_density", 8.0)
        priority = project_info.get("priority", "环保型")
        
        # 处理前端传来的优先级（可能是逗号分隔的英文值）
        priority_mapping = {
            "reliable": "可靠型",
            "economic": "经济型",
            "green": "环保型"
        }
        
        # 如果 priority 是逗号分隔的字符串，取第一个
        if isinstance(priority, str) and "," in priority:
            priority_list = priority.split(",")
            for p in priority_list:
                p = p.strip()
                if p in priority_mapping:
                    priority = priority_mapping[p]
                    break
        elif priority in priority_mapping:
            priority = priority_mapping[priority]
        
        cwsi = CWSI_MAP.get(province, CWSI_MAP["default"])

        alpha, beta, gamma, delta, epsilon = 0.3, 0.2, 0.3, 0.1, 0.1
        
        if cwsi <= 0.45: beta += 0.3
        elif cwsi >= 0.6: beta -= 0.1

        if priority == "环保型":
            delta += 0.2; epsilon += 0.1; gamma -= 0.1
        elif priority == "经济型":
            gamma += 0.3; delta -= 0.1
        elif priority == "可靠型":
            alpha += 0.25; gamma -= 0.15; epsilon += 0.15

        total_weight = alpha + beta + gamma + delta + epsilon
        w = {
            "alpha": round(alpha/total_weight, 2), "beta": round(beta/total_weight, 2), 
            "gamma": round(gamma/total_weight, 2), "delta": round(delta/total_weight, 2), 
            "epsilon": round(epsilon/total_weight, 2)
        }

        strategies = [
            {"name": "纯风冷+间接蒸发冷却 (Air Cooling + IEC)", "max_kw": 15, "f_pue": 1.25, "f_wue": 1.0, "f_tco": 0.8, "f_cue": 1.2, "f_whr": 0.4},
            {"name": "风液混合+开式冷却塔 (Hybrid + Cooling Tower)", "max_kw": 60, "f_pue": 1.15, "f_wue": 2.1, "f_tco": 1.0, "f_cue": 1.1, "f_whr": 0.75},
            {"name": "风液混合+混合型干冷器 (Hybrid + Dry Cooler)", "max_kw": 60, "f_pue": 1.20, "f_wue": 0.5, "f_tco": 1.3, "f_cue": 1.15, "f_whr": 0.75},
            {"name": "单相浸没式液冷+干冷器 (Immersion + Dry Cooler)", "max_kw": 999, "f_pue": 1.08, "f_wue": 0.1, "f_tco": 1.9, "f_cue": 1.05, "f_whr": 0.85}
        ]

        best_strategy = None
        min_score = float('inf')
        trace_log = f"【寻优环境】算力密度={cabinet_power}kW, 偏好={priority}, 水资源CWSI={cwsi}\n"
        trace_log += f"【动态权重】α(PUE)={w['alpha']}, β(WUE)={w['beta']}, γ(TCO)={w['gamma']}, δ(CUE)={w['delta']}, ε(WHR)={w['epsilon']}\n"
        trace_log += "【方案打分】计算公式: F = α·f(PUE) + β·f(WUE) + γ·f(TCO) + δ·f(CUE) - ε·f(WHR)\n"

        for s in strategies:
            if cabinet_power > s["max_kw"]:
                trace_log += f" - 方案 [{s['name']}] 被一票否决：无法满足 {cabinet_power}kW 物理散热极限。\n"
                continue
            score = (w["alpha"] * s["f_pue"] + w["beta"] * s["f_wue"] + w["gamma"] * s["f_tco"] + w["delta"] * s["f_cue"] - w["epsilon"] * s["f_whr"])
            trace_log += f" - 方案 [{s['name']}] 得分: {score:.3f} (PUE代价:{s['f_pue']}, WUE代价:{s['f_wue']}, TCO代价:{s['f_tco']})\n"
            if score < min_score:
                min_score = score
                best_strategy = s

        trace_log += f"★ 【最终决策】代价最小路线为：[{best_strategy['name']}]，最终得分：{min_score:.3f}\n"

        return {"best_strategy_name": best_strategy["name"], "optimization_trace": trace_log}

    def extract_cooling_params(self, context: str, user_reqs: Dict[str, Any], best_strategy_name: str) -> Dict[str, float]:
        cab_power_limit = user_reqs.get("cabinet_power_limit", 20.0)
        pue_limit_input = user_reqs.get("pue_limit", 1.30)
        wue_limit_input = user_reqs.get("wue_limit", 1.60)
        dynamic_params = {"PUE_Limit": pue_limit_input, "WUE_Limit": wue_limit_input}
        
        agent_recovery_coeff = self._calculate_recovery_coeff(user_reqs, user_reqs.get("location", "默认"))
        is_liquid = "液冷" in best_strategy_name

        if self.llm:
            prompt = PARAM_EXTRACTION_PROMPT.format(
                retrieved_context=context,
                user_requirements=json.dumps(user_reqs, ensure_ascii=False),
                dynamic_base_params=json.dumps(dynamic_params, ensure_ascii=False)
            )
            try:
                parser = JsonOutputParser()
                response = self.llm.invoke(prompt)
                params = parser.parse(response.content)
                llm_recovery_coeff = params.get("waste_heat_recovery_coeff", agent_recovery_coeff)
                params["waste_heat_recovery_coeff"] = round((agent_recovery_coeff + llm_recovery_coeff) / 2, 2)
                params["cabinet_power_limit"] = cab_power_limit
                params["regional_cooling_preference"] = best_strategy_name
                return params
            except Exception:
                pass

        return {
            "PUE_Limit": pue_limit_input, "WUE_Limit": wue_limit_input,
            "cooling_eff_coeff": 4.5 if is_liquid else 3.8, "waste_heat_recovery_coeff": agent_recovery_coeff,
            "facility_loss_coeff": 0.07, "cabinet_power_limit": cab_power_limit,
            "regional_cooling_preference": best_strategy_name
        }

    def calculate_cooling_kpis(self, params: Dict[str, Any], project_info: Dict[str, Any], env_data: Dict[str, Any]) -> Dict[str, Any]:
        it_load = project_info.get("planned_load", 0)
        cabinet_power = project_info.get("computing_power_density", 0)
        annual_temp = env_data.get("annual_temperature", 15.0)
        province = project_info.get("location", "默认")
        cwsi = CWSI_MAP.get(province, CWSI_MAP["default"])
        
        cooling_tech = params.get("regional_cooling_preference", "风冷")

        density_correction = 1.1 if cabinet_power >= params.get("cabinet_power_limit", 20.0) else 1.0
        cop_correction = self.get_cop_correction_factor(annual_temp)
        corrected_cop = params.get("cooling_eff_coeff", 4.0) * cop_correction

        cooling_load_kw = it_load * 1.1 * density_correction
        waste_heat_recovery_kw = cooling_load_kw * params.get("waste_heat_recovery_coeff", 0.0)
        corrected_cooling_load = max(0.0, cooling_load_kw - waste_heat_recovery_kw)
        
        cooling_power_kw = corrected_cooling_load / corrected_cop if corrected_cop > 0 else 0.0
        facility_loss_kw = it_load * params.get("facility_loss_coeff", 0.1)

        total_energy = it_load + cooling_power_kw + facility_loss_kw
        pue = total_energy / it_load if it_load > 0 else 1.0
        
        wue = self._calculate_wue(cooling_tech, cwsi, annual_temp, cooling_load_kw, it_load)

        return {
            "predicted_PUE": round(pue, 3), "predicted_WUE": round(wue, 3),
            "waste_heat_recovery_kw": round(waste_heat_recovery_kw, 2),
            "cooling_power_kw": round(cooling_power_kw, 2),
            "facility_loss_kw": round(facility_loss_kw, 2), "corrected_cop": round(corrected_cop, 2)
        }
    
    def _calculate_wue(self, cooling_tech: str, cwsi: float, annual_temp: float, cooling_load_kw: float, it_load_kw: float) -> float:
        """
        计算预测 WUE (Water Usage Effectiveness)
        WUE = 年总耗水量 / 年总IT能耗 (L/kWh)
        
        不同制冷技术的基准 WUE：
        - 水冷+冷却塔：1.5-2.5 L/kWh（主要耗水：蒸发、排污、飞溅）
        - 液冷系统：0.5-1.5 L/kWh（闭式循环，耗水较少）
        - 风冷/干冷器：0.1-0.5 L/kWh（主要是加湿和辅助用水）
        """
        base_wue = 1.8
        
        if "液冷" in cooling_tech:
            base_wue = 0.8
        elif "干冷" in cooling_tech or "风冷" in cooling_tech:
            base_wue = 0.3
        elif "冷却塔" in cooling_tech or "水冷" in cooling_tech:
            base_wue = 1.8
        
        cwsi_factor = 1.0 + (cwsi - 0.5) * 0.5
        if cwsi <= 0.45:
            cwsi_factor = 0.7
        elif cwsi >= 0.6:
            cwsi_factor = 1.3
        
        temp_factor = 1.0
        if annual_temp > 25:
            temp_factor = 1.2
        elif annual_temp < 10:
            temp_factor = 0.85
        
        load_factor = min(1.0, cooling_load_kw / (it_load_kw * 1.2)) if it_load_kw > 0 else 1.0
        
        predicted_wue = base_wue * cwsi_factor * temp_factor * load_factor
        
        return max(0.1, min(3.0, predicted_wue))

    def generate_cooling_scheme(self, context: str, project_info: Dict[str, Any], env_data: Dict[str, Any], province: str, opt_result: Dict[str, Any], green_energy_plan: str) -> str:
        prompt_val = COOLING_SCHEME_PROMPT.format(
            optimization_trace=opt_result["optimization_trace"],
            region=project_info.get("location", "默认"),
            province=province,
            cwsi=CWSI_MAP.get(province, CWSI_MAP["default"]),
            annual_temp=env_data.get("annual_temperature", 15.0),
            cabinet_power=project_info.get("computing_power_density", 8),
            cabinet_power_limit=project_info.get("cabinet_power_limit", 20.0),
            pue_limit=project_info.get("pue_limit", 1.30),
            wue_limit=project_info.get("wue_limit", 1.60),
            target_pue=project_info.get("pue_target", 1.2),
            green_energy_target=project_info.get("green_energy_target", 90),
            priority=project_info.get("priority", "环保型"),
            green_energy_plan_text=green_energy_plan
        )
        
        if self.llm is None:
            return self._generate_fallback_scheme(project_info, env_data, province, opt_result, green_energy_plan)
        
        try:
            return self.llm.invoke(prompt_val).content.strip()
        except Exception as e:
            print(f"⚠️ LLM 调用失败: {e}，使用兜底报告")
            return self._generate_fallback_scheme(project_info, env_data, province, opt_result, green_energy_plan)
    
    def _generate_fallback_scheme(self, project_info: Dict[str, Any], env_data: Dict[str, Any], province: str, opt_result: Dict[str, Any], green_energy_plan: str) -> str:
        cabinet_power = project_info.get("computing_power_density", 8)
        annual_temp = env_data.get("annual_temperature", 15.0)
        target_pue = project_info.get("pue_target", 1.2)
        green_target = project_info.get("green_energy_target", 90)
        location = project_info.get("location", "未知")
        
        return f"""# {location}数据中心制冷方案设计报告

## 一、制冷策略寻优逻辑解析

基于多目标优化函数：

$$F = \\alpha \\cdot f(PUE) + \\beta \\cdot f(WUE) + \\gamma \\cdot f(TCO) + \\delta \\cdot f(CUE) - \\varepsilon \\cdot f(WHR)$$

系统根据以下条件进行权重调整：

- **算力密度**：{cabinet_power} kW/机柜
- **年均温度**：{annual_temp}℃
- **水资源指数**：{CWSI_MAP.get(province, 0.5)}
- **目标PUE**：{target_pue}

## 二、多方案打分对比

{opt_result.get("optimization_trace", "寻优过程记录")}

## 三、最优架构落地指南

针对 **{cabinet_power} kW/机柜** 的算力密度，推荐采用**液冷技术**结合自然冷却方案：

### 冷源配置
- 干冷器 + 自然冷却模块
- 利用低温环境实现免费制冷

### 末端配置
- 冷板式液冷或浸没式液冷
- 高密度机柜散热解决方案

### 预期效果
- **预计 PUE**：{target_pue}
- **预计 WUE**：0.3 L/kWh

## 四、源网荷储与余热协同

结合绿电规划（{green_energy_plan}），建议：

1. **余热回收**：利用余热回收为园区供暖（北方地区）
2. **储能协同**：储能系统配合削峰填谷运行
3. **绿电消纳**：目标 {green_target}%

---
*本报告由系统兜底逻辑生成*"""

# ======================== 节点暴露函数 ========================
def cooling_specialist_node(state: dict) -> dict:
    """
    Agent 3: 暖通与制冷架构专家 - LangGraph Node
    """
    print("\n" + "="*60)
    print("❄️  [Agent 3: 暖通与制冷架构专家] 开始工作")
    print("="*60)
    
    # 提取 Graph State 参数
    user_reqs = state.get("user_requirements", {})
    env_data = state.get("environmental_data", {})
    energy_plan = state.get("energy_plan", {})
    
    # 解析来自 Agent 2 的绿电建议字符串
    if energy_plan:
        green_plan_text = (
            f"目标占比 {energy_plan.get('estimated_green_ratio', user_reqs.get('green_energy_target', 90))}%, "
            f"建议配置 {energy_plan.get('pv_capacity', 0)}kW 光伏，"
            f"长协/绿证解决 {energy_plan.get('ppa_ratio', 0)}%。"
        )
    else:
        green_plan_text = "无特定绿电规划，按常规比例消纳。"

    # 从上游抓取核心设定（补齐可能缺失的默认参数）
    project_info = {
        **user_reqs,
        "cabinet_power_limit": user_reqs.get("cabinet_power_limit", 20.0),
        "pue_limit": user_reqs.get("pue_limit", 1.30),
        "wue_limit": user_reqs.get("wue_limit", 1.60),
    }

    region = project_info.get("location", "北京")
    province = CITY_TO_PROVINCE.get(region, "北京")
    annual_temp = env_data.get("annual_temperature", 15.0)
    cabinet_power = project_info.get("computing_power_density", 8.0)

    # 初始化 Agent 并执行各步骤
    agent = CoolingAgent3()
    
    # 1. 寻优评价
    opt_result = agent.evaluate_cooling_strategies(project_info, province)
    
    # 2. 检索并提取
    context = agent.retrieve_context(region, province, annual_temp, cabinet_power, project_info["cabinet_power_limit"], project_info["pue_limit"], project_info["wue_limit"])
    extracted_params = agent.extract_cooling_params(context, project_info, opt_result["best_strategy_name"])
    
    # 3. 计算物理 KPI
    kpis = agent.calculate_cooling_kpis(extracted_params, project_info, env_data)
    
    # 4. 生成报告 (Markdown)
    scheme_text = agent.generate_cooling_scheme(context, project_info, env_data, province, opt_result, green_plan_text)
    
    # 组装给 Graph 的 state - 匹配前端期望的数据结构
    cooling_plan = {
        # 核心指标
        "cooling_technology": extracted_params.get("regional_cooling_preference", "未知"),
        "estimated_pue": kpis.get("predicted_PUE", 1.3),
        "predicted_wue": kpis.get("predicted_WUE", 1.6),
        "cooling_power_consumption": kpis.get("cooling_power_kw", 0),
        "waste_heat_recovery_kw": kpis.get("waste_heat_recovery_kw", 0),
        
        # 塞入富文本和过程数据，供 graph.py 写 Markdown 报告使用
        "strategy_optimization_trace": opt_result["optimization_trace"],
        "scheme_detail_brief": scheme_text,
        
        # 前端表格需要的嵌套对象
        "cooling_project_info": {
            "location": project_info.get("location", "未知"),
            "it_load_kW": project_info.get("planned_load", 0),
            "cabinet_power_kW": project_info.get("computing_power_density", 8),
            "target_pue": project_info.get("pue_target", 1.2),
            "green_energy_target": project_info.get("green_energy_target", 90)
        },
        
        "cooling_calc_params": {
            "PUE_Limit": extracted_params.get("PUE_Limit", 1.30),
            "WUE_Limit": extracted_params.get("WUE_Limit", 1.60),
            "cooling_eff_coeff": extracted_params.get("cooling_eff_coeff", 4.0),
            "facility_loss_coeff": extracted_params.get("facility_loss_coeff", 0.07),
            "regional_cooling_preference": extracted_params.get("regional_cooling_preference", "未知")
        },
        
        "cooling_kpis": {
            "predicted_PUE": kpis.get("predicted_PUE", 1.3),
            "predicted_WUE": kpis.get("predicted_WUE", 1.6),
            "cooling_power_kw": kpis.get("cooling_power_kw", 0),
            "corrected_cop": kpis.get("corrected_cop", 4.0),
            "waste_heat_recovery_kw": kpis.get("waste_heat_recovery_kw", 0)
        },
        
        # 余热回收策略
        "waste_heat_recovery_strategy": f"基于{project_info.get('location', '本地')}地区的气候条件，推荐采用{'液冷余热回收系统' if kpis.get('waste_heat_recovery_kw', 0) > 0 else '传统风冷系统'}。预计可回收{kpis.get('waste_heat_recovery_kw', 0):.1f} kW余热，可用于园区供暖或预热新风系统。"
    }
    
    state["cooling_plan"] = cooling_plan
    
    print(f"✅ 制冷选型: {cooling_plan['cooling_technology']}")
    print(f"✅ 预测 PUE: {cooling_plan['estimated_pue']} | WUE: {cooling_plan['predicted_wue']}")
    print("="*60)
    
    return state