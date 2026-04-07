from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

def process_medical_pdf(file_path):
    """
    解析医疗PDF并进行语义切分
    """
    # 1. 使用 PyMuPDF 快速读取
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # 2. 医疗文档切分策略
    # chunk_size 建议 500-800，确保包含完整的药名、症状描述
    # chunk_overlap 设为 50-100，防止语义在切分点断裂
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"📄 文档解析完成：{file_path}")
    print(f"✂️ 已切分为 {len(split_docs)} 个知识块")
    return split_docs

def get_file_hash(file_path):
    """计算文件的 MD5 值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()