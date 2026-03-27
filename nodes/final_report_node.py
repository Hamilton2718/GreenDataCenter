"""
最终报告生成节点 (Final Report Node)

改造目标：
1. 让 LLM 直接阅读 state 全量信息并生成可行性报告（>=1000 字）。
2. 节点内自动写入 output/final_report.md。
3. 保持 LangGraph 节点接口不变。

可选：支持工具型 Agent 创建方式（tools=[...]），便于后续接入 Tavily 等外部工具。
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


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


def _project_root() -> str:
    """返回项目根目录（nodes 的上一级目录）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _output_report_path() -> str:
    """统一报告输出路径。"""
    return os.path.join(_project_root(), "output", "final_report.md")


def _ensure_output_dir(path: str) -> None:
    """确保输出目录存在。"""
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def _safe_json_dump(data: Dict[str, Any]) -> str:
    """将 state 序列化为中文可读 JSON，便于 LLM 全量分析。"""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        # 极端情况下存在不可序列化对象，降级为字符串
        return str(data)


def _build_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """构建最终报告所需的全量快照，显式包含主要字段并保留原始 state。"""
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_requirements": state.get("user_requirements", {}),
        "environmental_data": state.get("environmental_data", {}),
        "electricity_price": state.get("electricity_price", {}),
        "load_profile": state.get("load_profile", {}),
        "energy_plan": state.get("energy_plan", {}),
        "cooling_plan": state.get("cooling_plan", {}),
        "simulation_result": state.get("simulation_result", {}),
        "review_result": state.get("review_result", {}),
        "financial_analysis": state.get("financial_analysis", {}),
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "error_message": state.get("error_message"),
        # 保留原始 state，确保“浏览所有信息”
        "raw_state": state,
    }


def _build_system_prompt() -> str:
    """面向 LLM 的系统提示词。"""
    return """
你是“绿色数据中心规划可行性总顾问”。

工作方式（必须遵守）：
1. 先阅读用户提供的 state_json，识别已给出的项目参数与缺失字段。
2. 基于 state 数据直接完整分析，并生成最终报告。
3. 输出最终报告时，必须是 Markdown 且正文不少于 1000 字。

报告硬性要求：
- 必须包含结论：可行 / 有条件可行 / 暂不可行。
- 若数据缺失，明确写出“数据缺失/待补充”及对结论影响。

建议结构：
- 标题与摘要
- 1. 项目背景与目标约束
- 2. 场址与环境可行性
- 3. 能源系统与绿电消纳策略
- 4. 制冷系统与能效路径
- 5. 仿真结果解读与运行策略
- 6. 财务可行性与投资回收
- 7. 风险清单与缓解措施
- 8. 实施路线图（近期/中期/远期）
- 9. 综合结论与建议
- 10. 关键指标汇总表
""".strip()


def generate_final_report(state: Dict[str, Any]) -> str:
    """基于全量 state 生成最终报告。"""
    snapshot = _build_state_snapshot(state)
    state_json = _safe_json_dump(snapshot)

    llm = ChatOpenAI(
        model="deepseek-v3.2",
        api_key=LLM_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=180
    )

    try:
        system_prompt = _build_system_prompt()
        user_prompt = (
            "【任务立即执行指令】\n"
            "以下就是本次项目的完整 state_json 数据，请直接阅读并基于此数据生成可行性报告\n\n"
            "```json\n"
            f"{state_json}\n"
            "```\n"
            "注意报告硬性要求：正文字数必须>=1000字，并且结论需要确切给出可行性。"
        )
        result = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        report = result.content
    except Exception as exc:
        raise RuntimeError(f"最终报告生成失败：LLM 不可用或调用异常。{exc}") from exc

    if not isinstance(report, str):
        report = str(report)

    # if len(report.strip()) < 1000:
    #     raise ValueError(
    #         f"最终报告生成失败：报告长度不足 1000 字，当前为 {len(report.strip())} 字。"
    #     )

    return report


def final_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    最终报告生成节点 - LangGraph Node

    参数:
        state: GreenDataCenterState 类型状态字典

    返回:
        更新后的状态，新增:
            - final_report: Markdown 格式最终报告
            - final_report_path: 报告落盘路径
    """
    print("\n" + "=" * 60)
    print("📝 [最终报告生成节点] 开始工作")
    print("=" * 60)

    final_report = generate_final_report(state)
    output_path = _output_report_path()
    _ensure_output_dir(output_path)

    print("\n" + "-" * 60)
    print("[finalreport]")
    print("-" * 60)
    print(final_report)
    print("-" * 60)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"✅ 最终报告生成完成，长度: {len(final_report)} 字符")
    print(f"✅ 报告已写入: {output_path}")

    print("\n" + "=" * 60)
    print("✅ [最终报告生成节点] 工作完成")
    print("=" * 60)

    return {
        **state,
        "final_report": final_report,
        "final_report_path": output_path,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 final_report_node 本地测试启动")
    print("=" * 80)

    if not LLM_API_KEY:
        print("❌ 环境变量 LLM_API_KEY 未设置，无法调用 LLM。")
        raise SystemExit(1)

    test_state = {
        "user_requirements": {
            "location": "乌兰察布",
            "business_type": "大模型训练",
            "planned_area": 12000,
            "planned_load": 8000,
            "computing_power_density": 30,
            "priority": "环保型",
            "green_energy_target": 90,
            "pue_target": 1.2,
            "budget_constraint": 15000
        },
        "environmental_data": {
            "annual_temperature": 5.5,
            "annual_wind_speed": 4.8,
            "annual_sunshine_hours": 3100,
            "carbon_emission_factor": 0.62,
            "province": "内蒙古"
        },
        "electricity_price": {
            "peak_price": 0.93,
            "high_price": 0.75,
            "flat_price": 0.55,
            "low_price": 0.26,
            "deep_low_price": 0.20,
            "max_price_diff": 0.73
        },
        "energy_plan": {
            "pv_capacity": 5000,
            "storage_capacity": 12000,
            "storage_power": 3000,
            "ppa_ratio": 45,
            "grid_ratio": 20,
            "estimated_green_ratio": 88
        },
        "cooling_plan": {
            "cooling_technology": "间接蒸发冷却+液冷",
            "estimated_pue": 1.18,
            "predicted_wue": 1.45
        },
        "simulation_result": {
            "summary": {
                "daily_it_energy_mwh": 192,
                "daily_green_supply_mwh": 156,
                "daily_green_ratio_pct": 81.25,
                "method": "典型日24小时粗仿真"
            }
        },
        "review_result": {
            "passed": True,
            "score": 4.4,
            "evaluator": "DeepSeek Reviewer"
        },
        "financial_analysis": {
            "payback_period": 6.8,
            "irr": 14.2,
            "npv": 3260,
            "lcoe": 0.43
        },
        "iteration_count": 1,
        "max_iterations": 3,
        "error_message": None,
    }

    try:
        result_state = final_report_node(test_state)
        report_path = result_state.get("final_report_path", _output_report_path())
        print("✅ 测试通过：final_report_node 执行成功")
        print(f"📄 报告路径：{report_path}")
        print(f"📝 报告长度：{len(result_state.get('final_report', ''))} 字符")
    except Exception as exc:
        print(f"❌ 测试失败：{exc}")
        raise
