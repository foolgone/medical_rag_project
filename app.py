import os
from core.database import get_vector_store
from core.parser import process_medical_pdf
from core.chain import get_medical_rag_chain


def main():
    # 1. 获取向量库连接
    vector_db = get_vector_store()

    # 2. 如果数据库是空的，则解析并导入文档
    # (企业级做法会先检查数据库记录，这里我们先手动触发一次)
    pdf_path = "data/knowledge.pdf"
    if os.path.exists(pdf_path):
        print("正在将医疗知识导入数据库...")
        chunks = process_medical_pdf(pdf_path)
        vector_db.add_documents(chunks)
        print("✅ 知识库更新完成！")

    # 3. 初始化 RAG 问答链
    rag_chain = get_medical_rag_chain(vector_db)

    # 4. 提问测试
    query = "根据文档，感冒药都是有哪些成分组成？"
    print(f"\n用户提问: {query}")

    response = rag_chain.invoke(query)
    print(f"\n🤖 助手：{response}")


if __name__ == "__main__":
    main()