"""
Agent 4: 方案审核与评估专家

使用独立大模型（如 DeepSeek）对前 3 个 Agent 的方案进行综合评估，
判断技术可行性、经济合理性，决定是否通过或要求重新优化。
"""

import os
import sys
from typing import Dict, Any
from datetime import datetime

# 导入 LangChain 相关库
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 使用 langchain-openai 包中的 ChatOpenAI（新版本推荐）
from langchain_openai import ChatOpenAI



class SchemeReviewer:
    """
    方案审核专家
    使用 DeepSeek 或其他独立模型对能源、制冷方案进行第三方评估
    """
    
    def __init__(self):
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-4b521d7c953e4a2583b8cf747c96c399")
        self.llm = ChatOpenAI(
            model="deepseek-v3.2",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def evaluate_schemes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合评估所有方案
        
        参数:
            state: 包含前 3 个 Agent 的输出
            
        返回:
            评估结果和建议
        """
        user_reqs = state.get("user_requirements", {})
        energy_plan = state.get("energy_plan", {})
        cooling_plan = state.get("cooling_plan", {})
        rag_knowledge = state.get("rag_knowledge", [])
        
        # 提取关键信息
        location = user_reqs.get("location", "未知")
        pue_target = user_reqs.get("pue_target", 1.2)
        green_target = user_reqs.get("green_energy_target", 90)
        computing_density = user_reqs.get("computing_power_density", 8)
        
        # 能源方案数据
        pv_capacity = energy_plan.get("pv_capacity", 0)
        wind_capacity = energy_plan.get("wind_capacity", 0)
        storage_capacity = energy_plan.get("storage_capacity", 0)
        
        # 制冷方案数据
        cooling_tech = cooling_plan.get("cooling_technology", "风冷")
        estimated_pue = cooling_plan.get("estimated_pue", 1.3)
        predicted_wue = cooling_plan.get("predicted_wue", 1.6)
        
        # 构建评估 Prompt
        eval_prompt = f"""你是一位数据中心技术评审专家。请对以下规划方案进行严格的第三方评估：

【项目基本信息】
- 地点：{location}
- 算力密度：{computing_density} kW/机柜
- PUE 目标：{pue_target}
- 绿电目标：{green_target}%

【能源规划方案】
- 光伏装机容量：{pv_capacity} kW
- 风电装机容量：{wind_capacity} kW
- 储能容量：{storage_capacity} kWh

【制冷方案】
- 制冷技术：{cooling_tech}
- 预计 PUE: {estimated_pue}
- 预计 WUE: {predicted_wue}

【相关知识库参考】
{chr(10).join(rag_knowledge[:3]) if rag_knowledge else "无"}

请按以下维度进行评估（每项 1-5 分，5 分最优）：

1. **技术可行性**（制冷技术与算力密度的匹配度）
   评分：[ ]/5
   理由：

2. **能效达标性**（预计 PUE vs 目标 PUE）
   评分：[ ]/5
   理由：

3. **绿电消纳合理性**（可再生能源配置是否满足目标）
   评分：[ ]/5
   理由：

4. **经济性**（投资规模与收益的平衡）
   评分：[ ]/5
   理由：

5. **政策合规性**（是否符合当地 PUE/WUE 标准）
   评分：[ ]/5
   理由：

【总体评价】
□ 通过（所有维度≥4 分，可进入财务评估）
□ 有条件通过（存在小问题，建议优化后进入财务评估）
□ 不通过（存在严重缺陷，需要重新规划）

【具体问题与改进建议】
（如果不通过或有条件通过，请详细列出问题和修改建议）

【必须修改的硬伤】
（如有）

【建议优化的软伤】
（如有）

【运行策略建议】
（如有）"""

        if self.llm:
            try:
                print("🔍 正在调用 DeepSeek 进行方案评估...")
                prompt = ChatPromptTemplate.from_template(eval_prompt)
                chain = prompt | self.llm | StrOutputParser()
                evaluation_result = chain.invoke({})
                
                print("✅ 评估完成")
                return {
                    "evaluation_text": evaluation_result,
                    "llm_used": "DeepSeek"
                }
            except Exception as e:
                print(f"❌ DeepSeek 评估失败：{e}")
                return self._fallback_evaluation(state)
        else:
            # 使用备用评估逻辑
            return self._fallback_evaluation(state)
    
    def _fallback_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        备用评估逻辑（当 LLM 不可用时）
        基于规则进行简单评估
        """
        user_reqs = state.get("user_requirements", {})
        cooling_plan = state.get("cooling_plan", {})
        
        pue_target = user_reqs.get("pue_target", 1.2)
        estimated_pue = cooling_plan.get("estimated_pue", 1.3)
        
        # 简单的规则评估
        issues = []
        score = 5
        
        if estimated_pue > pue_target:
            issues.append(f"PUE 不达标：预计{estimated_pue} > 目标{pue_target}")
            score -= 2
        
        if estimated_pue > 1.4:
            issues.append("PUE 过高，建议优化制冷方案")
            score -= 1
        
        recommendation = "通过" if score >= 4 else "不通过"
        
        fallback_text = f"""【自动评估结果】
综合评分：{score}/5
结论：{recommendation}

问题列表：
{chr(10).join('  - ' + issue for issue in issues) if issues else '  无明显问题'}

建议：
{chr(10).join('  - ' + issue for issue in issues) if issues else '  方案合理，可以继续'}"""
        
        return {
            "evaluation_text": fallback_text,
            "llm_used": "Rule-based Fallback",
            "passed": score >= 4,
            "issues": issues
        }
    
    def parse_evaluation_result(self, evaluation_text: str) -> Dict[str, Any]:
        """
        解析评估结果，提取关键信息
        
        返回:
            {
                "passed": bool,  # 是否通过
                "score": float,  # 综合评分
                "issues": list,  # 问题列表
                "suggestions": list  # 改进建议
            }
        """
        # 简单的文本解析逻辑
        passed = "通过" in evaluation_text and "不通过" not in evaluation_text
        issues = []
        suggestions = []
        
        # 提取问题列表
        if "【具体问题与改进建议】" in evaluation_text:
            issues_section = evaluation_text.split("【具体问题与改进建议】")[1]
            if "【必须修改的硬伤】" in issues_section:
                issues_part = issues_section.split("【必须修改的硬伤】")[0]
                for line in issues_part.split("\n"):
                    if line.strip().startswith("-"):
                        issues.append(line.strip()[1:])
        
        # 提取建议
        if "建议优化的软伤" in evaluation_text:
            suggestions_section = evaluation_text.split("建议优化的软伤")[1]
            for line in suggestions_section.split("\n")[:5]:
                if line.strip().startswith("-"):
                    suggestions.append(line.strip()[1:])
        
        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions
        }


# ============================================================
# LangGraph 节点函数
# ============================================================

def review_node(state: dict) -> dict:
    """
    Agent 4: 方案审核与评估专家 - LangGraph Node
    
    使用独立大模型对前 3 个 Agent 的方案进行综合评估。
    
    参数:
        state: GreenDataCenterState 类型，包含:
            - user_requirements: 用户需求
            - environmental_data: 环境数据
            - electricity_price: 电价数据
            - energy_plan: 能源规划方案
            - cooling_plan: 制冷方案
            - rag_knowledge: RAG 检索知识
            
    返回:
        更新后的 state，新增:
            - review_result: 评估结果
            - iteration_count: 迭代计数器（如果未通过则 +1）
            - feedback: 反馈意见（用于重新优化）
    """
    print("\n" + "="*60)
    print("🔍 [Agent 4: 方案审核与评估专家] 开始工作")
    print("="*60)
    
    # 创建审核器
    reviewer = SchemeReviewer()
    
    # 执行评估
    evaluation = reviewer.evaluate_schemes(state)
    
    # 解析评估结果
    parsed_result = reviewer.parse_evaluation_result(evaluation["evaluation_text"])
    
    # 打印评估摘要
    print(f"\n📊 评估工具：{evaluation.get('llm_used', 'Unknown')}")
    print(f"{'✅' if parsed_result['passed'] else '❌'} 评估结论：{'通过' if parsed_result['passed'] else '不通过'}")
    
    if parsed_result.get("issues"):
        print(f"\n⚠️ 发现的问题:")
        for issue in parsed_result["issues"][:3]:
            print(f"  - {issue}")
    
    # 准备反馈信息
    feedback = {
        "passed": parsed_result["passed"],
        "issues": parsed_result.get("issues", []),
        "suggestions": parsed_result.get("suggestions", []),
        "full_evaluation": evaluation.get("evaluation_text", "")
    }
    
    # 如果未通过，增加迭代计数
    current_iteration = state.get("iteration_count", 0)
    new_iteration = current_iteration + (0 if parsed_result["passed"] else 1)
    
    if not parsed_result["passed"]:
        print(f"\n🔄 方案未通过，需要重新优化（第 {new_iteration} 次迭代）")
        feedback["retry_required"] = True
        feedback["max_iterations_reached"] = new_iteration >= 3  # 最多重试 3 次
    else:
        print("\n✅ 方案通过，可以进入财务评估阶段")
        feedback["retry_required"] = False
    
    # 返回评估结果
    result = {
        **state,
        "review_result": {
            "evaluation_text": evaluation.get("evaluation_text", ""),
            "passed": parsed_result["passed"],
            "score": parsed_result.get("score", 0),
            "evaluator": evaluation.get("llm_used", "Unknown")
        },
        "feedback": feedback,
        "iteration_count": new_iteration
    }
    
    print("\n" + "="*60)
    print("✅ [Agent 4: 方案审核与评估专家] 工作完成")
    print("="*60)
    
    return result


# ============================================================
# 独立测试入口
# ============================================================

if __name__ == "__main__":
    print("===== 测试 Agent 4: 方案审核与评估专家 =====")
    
    # 模拟测试数据
    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "planned_load": 5000,
            "pue_target": 1.2,
            "green_energy_target": 90,
            "computing_power_density": 20
        },
        "environmental_data": {
            "annual_temperature": 5.5,
            "carbon_emission_factor": 0.6479
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "wind_capacity": 1000,
            "storage_capacity": 6000
        },
        "cooling_plan": {
            "cooling_technology": "液冷",
            "estimated_pue": 1.15,
            "predicted_wue": 1.5
        },
        "rag_knowledge": [
            "内蒙古地区数据中心 PUE 限值≤1.15",
            "高密度机柜推荐采用液冷技术"
        ]
    }
    
    # 执行评估
    result = review_node(test_state)
    
    # 显示结果
    print("\n" + "="*80)
    print("评估报告预览:")
    print("="*80)
    print(result["review_result"]["evaluation_text"][:800] + "...")
    
    print("\n" + "="*80)
    print("决策信息:")
    print("="*80)
    print(f"通过：{result['review_result']['passed']}")
    print(f"需要重试：{result['feedback'].get('retry_required', False)}")
    print(f"当前迭代次数：{result['iteration_count']}")
