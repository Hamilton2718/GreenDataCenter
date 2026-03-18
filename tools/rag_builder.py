
import os
from typing import List, Dict, Optional
from tqdm import tqdm

# --- LangChain Core Components ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

# --- Configuration ---
# 使用绝对路径，确保从任意工作目录调用都能正确找到文件
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(_BASE_DIR, "knowledge_base")
VECTOR_STORE_PATH = os.path.join(_BASE_DIR, "vector_store", "faiss_index")

# 使用一个开源的、效果优秀的支持中文的嵌入模型
# 第一次运行时，它会自动从HuggingFace下载模型文件（约400-500MB）
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Document Loaders Mapping ---
# 将文件扩展名映射到对应的 LangChain 加载器
DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    # 您可以在这里添加更多格式, 例如 .txt, .html 等
    # ".txt": TextLoader,
}

def load_documents(path):
    """
    扫描指定路径下的所有支持的文档，并使用对应的加载器加载它们。
    """
    print(f"📚 开始从 '{path}' 文件夹加载文档...")
    documents = []
    if not os.path.exists(path):
        print(f"⚠️ 警告: 知识库文件夹 '{path}' 不存在。将创建一个空知识库。")
        return documents

    # 使用 tqdm 创建一个进度条来显示加载过程
    files_to_load = [f for f in os.listdir(path) if any(f.endswith(ext) for ext in DOCUMENT_LOADERS)]
    if not files_to_load:
        print(f"ℹ️ 信息: 在 '{path}' 中未找到支持的文档文件（.pdf, .docx）。")
        return documents

    with tqdm(total=len(files_to_load), desc="加载文档") as pbar:
        for file in files_to_load:
            file_path = os.path.join(path, file)
            ext = "." + file.rsplit(".", 1)[-1]
            if ext in DOCUMENT_LOADERS:
                try:
                    loader = DOCUMENT_LOADERS[ext](file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"❌ 加载文件 '{file_path}' 时出错: {e}")
            pbar.update(1)
            
    print(f"✅ 文档加载完成，共加载了 {len(documents)} 页/部分内容。")
    return documents

def build_vector_store(documents):
    """
    将加载的文档进行分割、嵌入，并构建一个FAISS向量存储。
    """
    if not documents:
        print("ℹ️ 信息: 没有文档可供处理，跳过向量库构建。")
        return None

    # 1. 分割 (Split)
    print("🔪 开始将文档分割成小块 (Chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 分割完成，共得到 {len(chunks)} 个文本块。")

    # 2. 嵌入 (Embed)
    print(f"🧠 正在初始化嵌入模型: '{EMBEDDING_MODEL_NAME}'...")
    print("(首次运行时会自动下载模型，请耐心等待...)")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. 存储 (Store)
    print("💾 正在构建并持久化向量数据库 (FAISS)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 向量数据库已成功构建并保存到: '{VECTOR_STORE_PATH}'")
    return vector_store

def build_or_load_vector_store(rebuild: bool = False):
    """
    主函数：构建或加载向量数据库。
    - 如果 rebuild=True，或向量存储文件不存在，则强制重新构建。
    - 否则，直接加载已有的向量存储。
    
    返回: 一个 FAISS 向量存储对象，如果没有任何文档则返回 None。
    """
    print("\n--- RAG 知识库模块 --- ")
    
    # 检查SentenceTransformerEmbeddings是否可用
    if SentenceTransformerEmbeddings is None:
        print("⚠️ SentenceTransformerEmbeddings不可用，返回 None")
        return None
    
    try:
        if rebuild or not os.path.exists(VECTOR_STORE_PATH):
            if rebuild:
                print("🔧 检测到 'rebuild=True'，将强制重建知识库。")
            else:
                print("⚠️ 未发现已缓存的向量数据库，将开始构建新的知识库。")
            
            # 执行完整的构建流程
            documents = load_documents(KNOWLEDGE_BASE_PATH)
            vector_store = build_vector_store(documents)
        else:
            # 直接加载已有的向量存储
            print(f"✅ 发现已缓存的向量数据库，直接从 '{VECTOR_STORE_PATH}' 加载。")
            print(f"🧠 正在初始化嵌入模型: '{EMBEDDING_MODEL_NAME}'...")
            try:
                # 检查SentenceTransformerEmbeddings是否可用
                if SentenceTransformerEmbeddings is None:
                    raise Exception("SentenceTransformerEmbeddings不可用")
                
                # 尝试初始化嵌入模型
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                
                # 尝试加载向量存储
                vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                print("✅ 向量数据库加载成功。")
            except Exception as e:
                print(f"⚠️ 向量数据库加载失败: {e}，返回 None")
                return None
        
        return vector_store
    except Exception as e:
        print(f"⚠️ RAG 知识库初始化失败: {e}，返回 None")
        return None


# ============================================================
# RAG 单例管理器 & 对外接口函数
# ============================================================

class _RAGManager:
    """
    RAG知识库单例管理器。
    
    保证向量库和嵌入模型在整个进程中只加载一次，
    后续所有节点调用均复用同一实例，避免重复IO和模型加载开销。
    """
    _instance: Optional["_RAGManager"] = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_vector_store(self, rebuild: bool = False):
        """
        获取向量库实例（懒加载，首次调用时才初始化）。
        
        参数:
            rebuild: 是否强制重建向量库
        返回:
            FAISS向量库实例，若知识库为空则返回None
        """
        if self._vector_store is None or rebuild:
            self._vector_store = build_or_load_vector_store(rebuild=rebuild)
        return self._vector_store

    def reset(self):
        """重置单例，强制下次调用时重新加载（用于测试或热更新知识库）。"""
        self._vector_store = None


# 模块级单例实例
_rag_manager = _RAGManager()


def query_knowledge_base(
    query: str,
    k: int = 4,
    rebuild: bool = False
) -> List[Dict]:
    """
    【对外接口】在知识库中检索与query最相关的文档片段。
    
    这是供其他节点直接调用的核心接口，内部自动管理向量库的加载与缓存。
    
    使用示例（在任意节点中）:
        from tools import query_knowledge_base
        
        results = query_knowledge_base("数据中心液冷技术要求")
        for r in results:
            print(r['content'])   # 相关文本片段
            print(r['source'])    # 来源文件名
            print(r['page'])      # 页码
    
    参数:
        query:   检索问题/关键词，支持中英文自然语言
        k:       返回最相关的片段数量，默认4条
        rebuild: 是否强制重建向量库（通常不需要），默认False
        
    返回:
        List[Dict]，每个dict包含:
            - content (str):  文档片段的文本内容
            - source  (str):  来源文件名（不含路径）
            - page    (int):  页码，Word文档则为-1
        若知识库为空或检索失败，返回空列表 []
    """
    vector_store = _rag_manager.get_vector_store(rebuild=rebuild)
    if vector_store is None:
        print("⚠️ [RAG] 知识库为空，无法检索，返回空结果")
        return []

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        results = [
            {
                "content": doc.page_content,
                "source":  os.path.basename(doc.metadata.get("source", "未知来源")),
                "page":    doc.metadata.get("page", -1),
            }
            for doc in docs
        ]
        print(f"🔍 [RAG] 查询 '{query[:30]}...' 命中 {len(results)} 条片段")
        return results
    except Exception as e:
        print(f"❌ [RAG] 检索失败: {e}")
        return []


def query_knowledge_base_as_text(
    query: str,
    k: int = 4,
    rebuild: bool = False
) -> str:
    """
    【对外接口】检索知识库并将结果拼接为纯文本，方便直接嵌入LLM Prompt。
    
    使用示例（在节点中构造Prompt时）:
        from tools import query_knowledge_base_as_text
        
        context = query_knowledge_base_as_text("PUE标准要求")
        prompt = f"根据以下知识库内容回答问题：\\n{context}\\n问题：..."
    
    参数:
        query:   检索问题/关键词
        k:       返回片段数量，默认4条
        rebuild: 是否强制重建向量库，默认False
        
    返回:
        str，格式化后的多段文本；若无结果则返回空字符串
    """
    results = query_knowledge_base(query, k=k, rebuild=rebuild)
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[片段{i}] 来源: {r['source']}"
            + (f" 第{r['page']}页" if r['page'] != -1 else "")
            + f"\n{r['content']}"
        )
    return "\n\n".join(parts)


def rebuild_knowledge_base() -> bool:
    """
    【对外接口】强制重建知识库（当knowledge_base目录有新文件时调用）。
    
    使用示例:
        from tools import rebuild_knowledge_base
        rebuild_knowledge_base()
    
    返回:
        bool，重建成功返回True，失败返回False
    """
    print("🔧 [RAG] 开始重建知识库...")
    try:
        _rag_manager.reset()
        vector_store = _rag_manager.get_vector_store(rebuild=True)
        success = vector_store is not None
        if success:
            print("✅ [RAG] 知识库重建成功")
        else:
            print("⚠️ [RAG] 知识库重建完成，但文档为空")
        return success
    except Exception as e:
        print(f"❌ [RAG] 知识库重建失败: {e}")
        return False


import argparse

# --- 主程序入口 (用于独立测试) ---
if __name__ == '__main__':
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description="构建或加载RAG知识库的向量存储。")
    parser.add_argument(
        "--rebuild",
        action="store_true", # 当出现 --rebuild 参数时，其值为 True
        help="如果指定，则强制从头开始重建知识库，而不是加载现有缓存。"
    )
    args = parser.parse_args()

    print("===== 开始独立测试 RAG 知识库构建模块 =====")
    
    # 根据命令行参数决定是否重建
    vector_store = build_or_load_vector_store(rebuild=args.rebuild)

    if vector_store:
        print("\n===== 知识库查询测试 =====")
        # 将向量数据库转换为一个“检索器”，它可以找出与问题最相关的文本块
        retriever = vector_store.as_retriever()
        
        query = "数据中心的制冷策略有哪些？"
        print(f"❓ 测试查询: {query}")
        
        # relevant_docs 是一个包含与问题最相关的 Document 对象的列表
        relevant_docs = retriever.invoke(query)
        
        print("\n🔍 检索到的最相关内容:")
        for i, doc in enumerate(relevant_docs):
            print(f"\n--- [相关片段 {i+1}] ---")
            print(doc.page_content)
            print(f"(来源: {os.path.basename(doc.metadata.get('source', 'N/A'))}, 页码: {doc.metadata.get('page', 'N/A')})")
    else:
        print("\n⚠️ 知识库为空，无法进行查询测试。")

