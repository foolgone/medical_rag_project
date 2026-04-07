from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from config import PG_CONNECTION, OLLAMA_BASE_URL, EMBEDDING_MODEL
from sqlalchemy import create_pool_from_url,create_engine,text
from config import PG_CONNECTION

engine = create_engine(PG_CONNECTION)
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

def is_file_processed(file_hash):
    """查询数据库，判断改哈希值是否已经存在"""
    query = text("SELECT 1 FROM processed_files WHERE file_hash = :hash")
    with engine.connect() as conn:
        result = conn.execute(query, {"hash": file_hash}).fetchone()
        return result is not None

def record_file_hash(file_name, file_hash):
    """将处理完的文件信息存入数据库"""
    query = text("INSERT INTO processed_files (file_name, file_hash) VALUES (:name, :hash)")
    with engine.connect() as conn:
        conn.execute(query, {"name": file_name, "hash": file_hash})
        conn.commit()