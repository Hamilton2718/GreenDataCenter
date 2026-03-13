# This file makes the 'tools' directory a Python package

from tools.rag_builder import (
    # 对外接口（各节点直接调用这三个函数）
    query_knowledge_base,          # 检索，返回 List[Dict]
    query_knowledge_base_as_text,  # 检索，返回拼接好的纯文本（适合直接填入Prompt）
    rebuild_knowledge_base,        # 强制重建知识库
)

__all__ = [
    "query_knowledge_base",
    "query_knowledge_base_as_text",
    "rebuild_knowledge_base",
]
