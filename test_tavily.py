"""
最小化连通性测试：Tavily + ReAct Agent

用途：
1. 验证 Tavily API Key 是否可用
2. 验证 ReAct Agent 是否能调用 Tavily 工具进行联网搜索
3. 输出是否可行以及简要结果
"""

import importlib
import os
import sys
from typing import Any, List

from langchain_openai import ChatOpenAI


def _load_local_dotenv() -> None:
    """从项目根目录加载 .env（若存在）。"""
    project_root = os.path.dirname(os.path.abspath(__file__))
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


def _build_tavily_tool() -> List[Any]:
    """优先使用 langchain-tavily，兼容 langchain-community。"""
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        raise RuntimeError("缺少环境变量 TAVILY_API_KEY")

    os.environ["TAVILY_API_KEY"] = tavily_api_key

    try:
        TavilySearch = importlib.import_module("langchain_tavily").TavilySearch
        return [TavilySearch(max_results=3)]
    except Exception:
        TavilySearchResults = importlib.import_module(
            "langchain_community.tools.tavily_search"
        ).TavilySearchResults
        return [TavilySearchResults(max_results=3)]


def _create_react_agent(llm: ChatOpenAI, tools: List[Any]):
    """兼容新版/旧版 Agent 创建接口。"""
    try:
        create_agent = importlib.import_module("langchain.agents").create_agent
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是一个联网信息检索助手。请优先使用搜索工具回答问题。",
        )
    except Exception:
        create_react_agent = importlib.import_module("langgraph.prebuilt").create_react_agent
        return create_react_agent(
            llm,
            tools,
            prompt="你是一个联网信息检索助手。请优先使用搜索工具回答问题。",
        )


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item.get("text", "")))
                        else:
                            parts.append(str(item))
                    return "\n".join(parts).strip()
                return str(content)
            return str(last_msg)

    return str(result)


def _count_tool_messages(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    messages = result.get("messages")
    if not isinstance(messages, list):
        return 0

    count = 0
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if str(msg_type).lower() in {"tool", "toolmessage"}:
            count += 1
    return count


def main() -> int:
    _load_local_dotenv()

    llm_api_key = os.getenv("LLM_API_KEY", "")
    if not llm_api_key:
        print("❌ 缺少环境变量 LLM_API_KEY")
        return 1

    model = os.getenv("LLM_MODEL", "qwen-max")
    base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    timeout = int(os.getenv("TEST_LLM_TIMEOUT", "120"))

    print("=" * 72)
    print("Tavily + ReAct Agent 联网测试启动")
    print("=" * 72)
    print(f"模型: {model}")
    print(f"Base URL: {base_url}")
    print(f"Timeout: {timeout}s")

    try:
        tools = _build_tavily_tool()
        llm = ChatOpenAI(
            model=model,
            api_key=llm_api_key,
            base_url=base_url,
            timeout=timeout,
        )
        agent = _create_react_agent(llm, tools)

        query = (
            "请联网搜索并简要总结：2025年以来中国数据中心绿色低碳与PUE相关政策或标准动态，"
            "给出3条要点，并标注每条信息的来源类型。"
        )
        print("\n[查询]", query)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 8},
        )

        answer = _extract_text(result)
        tool_count = _count_tool_messages(result)

        print("\n" + "-" * 72)
        print("测试结论")
        print("-" * 72)
        print("[PASS] 可行：Agent 调用成功并返回结果")
        print(f"工具消息数量（粗略）: {tool_count}")
        print("\n[回答预览]\n")
        print(answer[:2000])
        print("\n" + "=" * 72)
        return 0
    except Exception as exc:
        print("\n" + "-" * 72)
        print("测试结论")
        print("-" * 72)
        print("[FAIL] 不可行：调用失败")
        print(f"错误: {exc.__class__.__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
