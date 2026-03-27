"""
Agent 4: 方案审核与评估专家

工程化评审框架：
1) 硬约束闸门（Fail Fast）
2) 量化评分（可复现）
3) LLM 解释层（解释与建议，不主导裁决）
"""

import json
import os
from typing import Dict, Any, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _load_local_dotenv() -> None:
    """轻量加载项目根目录 .env，避免直接运行脚本时取不到环境变量。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, ".env")
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# 工程阈值常量（普通变量）
min_pass_score = 3.6
green_ratio_tolerance = 2.0
max_payback_years = 10.0
min_irr = 8.0
min_npv = 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SchemeReviewer:
    """工程化方案审核器。"""

    def __init__(self):
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-4b521d7c953e4a2583b8cf747c96c399")
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=LLM_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _collect_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_req = state.get("user_requirements", {}) or {}
        energy_plan = state.get("energy_plan", {}) or {}
        cooling_plan = state.get("cooling_plan", {}) or {}
        simulation = state.get("simulation_result", {}) or {}
        sim_summary = simulation.get("summary", {}) if isinstance(simulation, dict) else {}
        financial = state.get("financial_analysis", {}) or {}

        pue_target = _to_float(user_req.get("pue_target"), 1.2)
        estimated_pue = _to_float(cooling_plan.get("estimated_pue"), 1.35)
        green_target = _to_float(user_req.get("green_energy_target"), 90.0)
        estimated_green_ratio = _to_float(energy_plan.get("estimated_green_ratio"), -1.0)
        simulated_green_ratio = _to_float(sim_summary.get("daily_green_ratio_pct"), -1.0)

        # 优先用仿真结果做评审口径
        if simulated_green_ratio >= 0:
            effective_green_ratio = simulated_green_ratio
            green_ratio_source = "simulation"
        elif estimated_green_ratio >= 0:
            effective_green_ratio = estimated_green_ratio
            green_ratio_source = "energy_plan"
        else:
            effective_green_ratio = 0.0
            green_ratio_source = "missing"

        metrics = {
            "location": user_req.get("location", "未知"),
            "business_type": user_req.get("business_type", "通用"),
            "planned_load_kw": _to_float(user_req.get("planned_load"), 0.0),
            "computing_power_density": _to_float(user_req.get("computing_power_density"), 8.0),
            "pue_target": pue_target,
            "estimated_pue": estimated_pue,
            "green_target": green_target,
            "effective_green_ratio": effective_green_ratio,
            "green_ratio_source": green_ratio_source,
            "payback_period": _to_float(financial.get("payback_period", financial.get("payback_years")), -1.0),
            "irr": _to_float(financial.get("irr"), -1.0),
            "npv": _to_float(financial.get("npv"), -1.0),
            "lcoe": _to_float(financial.get("lcoe"), -1.0),
            "storage_capacity_kwh": _to_float(energy_plan.get("storage_capacity"), 0.0),
            "storage_power_kw": _to_float(energy_plan.get("storage_power"), 0.0),
            "pv_capacity_kw": _to_float(energy_plan.get("pv_capacity"), 0.0),
            "cooling_technology": cooling_plan.get("cooling_technology", "未知"),
            "sim_method": sim_summary.get("method", "missing"),
        }
        return metrics

    def _hard_constraints(self, metrics: Dict[str, Any]) -> List[str]:
        fails: List[str] = []

        if metrics["estimated_pue"] > metrics["pue_target"]:
            fails.append(
                f"PUE 不达标：estimated_pue={metrics['estimated_pue']} > target={metrics['pue_target']}"
            )

        required_green = metrics["green_target"] - green_ratio_tolerance
        if metrics["effective_green_ratio"] < required_green:
            fails.append(
                "绿电占比未达标："
                f"effective_green_ratio={metrics['effective_green_ratio']}% < "
                f"target({metrics['green_target']}%) - tolerance({green_ratio_tolerance}%)"
            )

        payback = metrics["payback_period"]
        if payback >= 0 and payback > max_payback_years:
            fails.append(
                f"投资回收期过长：payback={payback} 年 > {max_payback_years} 年"
            )

        irr = metrics["irr"]
        if irr >= 0 and irr < min_irr:
            fails.append(f"IRR 过低：irr={irr}% < {min_irr}%")

        npv = metrics["npv"]
        if npv >= 0 and npv < min_npv:
            fails.append(f"NPV 过低：npv={npv} 万元 < {min_npv} 万元")

        return fails

    def _score(self, metrics: Dict[str, Any], hard_fails: List[str]) -> Dict[str, float]:
        # S_energy: 由绿电占比偏差计算，目标及以上给高分
        green_gap = metrics["green_target"] - metrics["effective_green_ratio"]
        s_energy = _clamp(5.0 - max(0.0, green_gap) / 8.0, 1.0, 5.0)

        # S_cooling: 由 PUE 偏差计算
        pue_gap = metrics["estimated_pue"] - metrics["pue_target"]
        s_cooling = _clamp(5.0 - max(0.0, pue_gap) * 20.0, 1.0, 5.0)

        # S_finance: 回收期 + IRR + NPV 的综合
        payback = metrics["payback_period"]
        irr = metrics["irr"]
        npv = metrics["npv"]

        s_payback = 3.0 if payback < 0 else _clamp(5.0 - max(0.0, payback - 4.0) / 2.0, 1.0, 5.0)
        s_irr = 3.0 if irr < 0 else _clamp(2.0 + irr / 6.0, 1.0, 5.0)
        s_npv = 3.0 if npv < 0 else _clamp(3.0 + npv / 3000.0, 1.0, 5.0)
        s_finance = _clamp(0.4 * s_payback + 0.3 * s_irr + 0.3 * s_npv, 1.0, 5.0)

        # S_risk: 根据硬失败数量与数据完整性
        missing_penalty = 0.0
        if metrics["green_ratio_source"] == "missing":
            missing_penalty += 0.8
        if metrics["sim_method"] == "missing":
            missing_penalty += 0.5
        s_risk = _clamp(5.0 - 0.8 * len(hard_fails) - missing_penalty, 1.0, 5.0)

        total = _clamp(
            0.30 * s_energy + 0.25 * s_cooling + 0.25 * s_finance + 0.20 * s_risk,
            1.0,
            5.0,
        )

        return {
            "s_energy": round(s_energy, 3),
            "s_cooling": round(s_cooling, 3),
            "s_finance": round(s_finance, 3),
            "s_risk": round(s_risk, 3),
            "score_total": round(total, 3),
        }

    def _build_action_items(self, hard_fails: List[str]) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = []
        for issue in hard_fails:
            if "PUE" in issue:
                actions.append({
                    "owner_node": "cooling_design",
                    "priority": "high",
                    "action": "优化制冷架构与运行策略，降低 estimated_pue 至目标以内。",
                })
            elif "绿电" in issue:
                actions.append({
                    "owner_node": "energy_planning",
                    "priority": "high",
                    "action": "提高绿电供给比例（PV/PPA/储能联动），优先提升仿真口径下绿电占比。",
                })
            elif "回收期" in issue or "IRR" in issue or "NPV" in issue:
                actions.append({
                    "owner_node": "financial_analysis",
                    "priority": "high",
                    "action": "优化 CAPEX 与收益结构，调整配置后复算财务可行性。",
                })
            else:
                actions.append({
                    "owner_node": "review",
                    "priority": "medium",
                    "action": "补充关键输入数据并重新评审。",
                })
        return actions

    def _llm_explain(
        self,
        metrics: Dict[str, Any],
        hard_fails: List[str],
        sub_scores: Dict[str, float],
        passed: bool,
    ) -> str:
        def _rule_text() -> str:
            lines = [
                "## 评审结论（规则引擎）",
                f"- 结论：{'通过' if passed else '不通过'}",
                f"- 总分：{sub_scores['score_total']}/5",
                "",
                "## 硬约束检查",
            ]
            if hard_fails:
                lines.extend([f"- {x}" for x in hard_fails])
            else:
                lines.append("- 全部通过")
            lines.extend([
                "",
                "## 分项评分",
                f"- 能源结构评分：{sub_scores['s_energy']}",
                f"- 制冷能效评分：{sub_scores['s_cooling']}",
                f"- 财务可行性评分：{sub_scores['s_finance']}",
                f"- 风险鲁棒性评分：{sub_scores['s_risk']}",
            ])
            return "\n".join(lines)

        if not self.llm:
            # LLM 不可用时，返回规则引擎说明
            return _rule_text()

        prompt = ChatPromptTemplate.from_template(
            """
你是数据中心可研评审专家，请基于以下结构化结果生成简明专业评审说明。

要求：
1) 使用 Markdown。
2) 先给“评审结论（通过/不通过）”，再给“硬约束检查”、"分项评分解释"、"整改建议"。
3) 只基于输入数据，不编造新数字。

【结论】
passed={passed}

【硬约束失败项】
{hard_fails_json}

【指标快照】
{metrics_json}

【分项分数】
{scores_json}
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke(
                {
                    "passed": passed,
                    "hard_fails_json": json.dumps(hard_fails, ensure_ascii=False, indent=2),
                    "metrics_json": json.dumps(metrics, ensure_ascii=False, indent=2),
                    "scores_json": json.dumps(sub_scores, ensure_ascii=False, indent=2),
                }
            )
        except Exception as exc:
            # 解释层失败不应阻断评估主流程
            return _rule_text() + f"\n\n> 说明：LLM解释层调用失败，已回退规则文本。错误：{exc}"

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self._collect_metrics(state)
        hard_fails = self._hard_constraints(metrics)
        sub_scores = self._score(metrics, hard_fails)

        passed = (len(hard_fails) == 0) and (sub_scores["score_total"] >= min_pass_score)
        issues = hard_fails.copy()
        suggestions = [item["action"] for item in self._build_action_items(hard_fails)]

        evaluation_text = self._llm_explain(
            metrics=metrics,
            hard_fails=hard_fails,
            sub_scores=sub_scores,
            passed=passed,
        )

        return {
            "passed": passed,
            "score": sub_scores["score_total"],
            "issues": issues,
            "suggestions": suggestions,
            "sub_scores": sub_scores,
            "metrics": metrics,
            "hard_fails": hard_fails,
            "evaluation_text": evaluation_text,
            "action_items": self._build_action_items(hard_fails),
        }


def review_node(state: dict) -> dict:
    """
    Agent 4: 方案审核与评估专家 - LangGraph Node

    工程化评估流程：硬约束 -> 量化评分 -> 评审说明。
    """
    print("\n" + "=" * 60)
    print("🔍 [Agent 4: 方案审核与评估专家] 开始工作")
    print("=" * 60)

    reviewer = SchemeReviewer()
    evaluation = reviewer.evaluate(state)

    passed = bool(evaluation["passed"])
    current_iteration = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", 3))
    new_iteration = current_iteration + (0 if passed else 1)

    print(f"\n📊 评估工具：{evaluation.get('llm_used', 'Unknown')}")
    print(f"📈 综合评分：{evaluation.get('score', 0):.2f}/5")
    print(f"{'✅' if passed else '❌'} 评估结论：{'通过' if passed else '不通过'}")

    if evaluation.get("hard_fails"):
        print("\n⚠️ 硬约束失败项:")
        for issue in evaluation["hard_fails"][:5]:
            print(f"  - {issue}")

    feedback = {
        "passed": passed,
        "issues": evaluation.get("issues", []),
        "suggestions": evaluation.get("suggestions", []),
        "action_items": evaluation.get("action_items", []),
        "hard_fails": evaluation.get("hard_fails", []),
        "score_breakdown": evaluation.get("sub_scores", {}),
        "metrics_snapshot": evaluation.get("metrics", {}),
        "full_evaluation": evaluation.get("evaluation_text", ""),
        "retry_required": not passed,
        "max_iterations_reached": (new_iteration >= max_iterations) if not passed else False,
    }

    if not passed:
        print(f"\n🔄 方案未通过，需要重新优化（第 {new_iteration} 次迭代）")
    else:
        print("\n✅ 方案通过，可以进入最终报告阶段")

    result = {
        **state,
        "review_result": {
            "evaluation_text": evaluation.get("evaluation_text", ""),
            "passed": passed,
            "score": evaluation.get("score", 0.0),
            "evaluator": evaluation.get("llm_used", "Unknown"),
            "issues": evaluation.get("issues", []),
            "suggestions": evaluation.get("suggestions", []),
        },
        "feedback": feedback,
        "iteration_count": new_iteration,
    }

    print("\n" + "=" * 60)
    print("✅ [Agent 4: 方案审核与评估专家] 工作完成")
    print("=" * 60)
    return result


if __name__ == "__main__":
    print("===== 测试 Agent 4: 方案审核与评估专家（工程化版本） =====")

    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "planned_load": 5000,
            "pue_target": 1.2,
            "green_energy_target": 90,
            "computing_power_density": 20,
        },
        "energy_plan": {
            "pv_capacity": 2000,
            "wind_capacity": 1000,
            "storage_capacity": 6000,
            "storage_power": 3000,
            "estimated_green_ratio": 85,
        },
        "cooling_plan": {
            "cooling_technology": "液冷",
            "estimated_pue": 1.15,
            "predicted_wue": 1.5,
        },
        "simulation_result": {
            "summary": {
                "daily_green_ratio_pct": 83.2,
                "method": "粗仿真（典型日）",
            }
        },
        "financial_analysis": {
            "payback_period": 7.2,
            "irr": 12.5,
            "npv": 1200,
            "lcoe": 0.45,
        },
        "iteration_count": 0,
        "max_iterations": 3,
    }

    result = review_node(test_state)

    print("\n" + "=" * 80)
    print("结构化结果:")
    print("=" * 80)
    print(f"通过：{result['review_result']['passed']}")
    print(f"评分：{result['review_result']['score']}/5")
    print(f"需要重试：{result['feedback'].get('retry_required', False)}")
    print(f"当前迭代次数：{result['iteration_count']}")
