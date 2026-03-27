import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


def main() -> None:
    load_dotenv()

    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    llm_api_key = os.getenv("LLM_API_KEY", "")

    if not tavily_api_key:
        raise RuntimeError("缺少环境变量 TAVILY_API_KEY")
    if not llm_api_key:
        raise RuntimeError("缺少环境变量 LLM_API_KEY")

    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatOpenAI(
        model="deepseek-v3.2",
        api_key=llm_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60,
    )

    tools = [TavilySearch(max_results=3)]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个会使用 Tavily 搜索的助理。"
            "遵循 ReAct 模式：先思考，再调用工具，再根据结果给出最终答案。"
        ),
    )

    query = "请联网搜索：2025年中国数据中心绿电相关政策的3个要点，并给出简短来源说明。"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    print("[query]")
    print(query)
    print("\n[final answer]")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
