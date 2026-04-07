from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from config import PG_CONNECTION, OLLAMA_BASE_URL, EMBEDDING_MODEL

# 初始化嵌入模型
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

def get_vector_store():
    """
    获取向量数据库实例 (适配 langchain-postgres 最新版)
    """
    return PGVector(
        connection=PG_CONNECTION,  # 关键点：参数名从 connection_string 改为 connection
        embeddings=embeddings,      # 关键点：参数名从 embedding_function 改为 embeddings
        collection_name="medical_knowledge",
    )