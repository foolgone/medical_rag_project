import os
from core.database import get_vector_store, is_file_processed, record_file_hash, engine
from core.parser import process_medical_pdf, get_file_hash
from core.chain import get_medical_rag_chain
from sqlalchemy import text


def init_db():
    """初始化数据库表（如果不存在）"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_hash VARCHAR(32) UNIQUE NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()


def main():
    # 0. 确保哈希表存在
    init_db()

    # 1. 获取向量库连接
    vector_db = get_vector_store()

    # 2. 智能导入：基于 MD5 哈希校验
    pdf_path = "data/knowledge.pdf"
    if os.path.exists(pdf_path):
        file_name = os.path.basename(pdf_path)
        current_hash = get_file_hash(pdf_path)

        if is_file_processed(current_hash):
            print(f"⏭️  文件 [{file_name}] 已处理过，跳过导入。")
        else:
            print(f"🆕 正在解析新文档: {file_name}...")
            chunks = process_medical_pdf(pdf_path)
            vector_db.add_documents(chunks)
            record_file_hash(file_name, current_hash)
            print("✅ 知识库更新完成！")

    # 3. 初始化支持“对话记忆”的 RAG 链
    rag_chain = get_medical_rag_chain(vector_db)

    # 4. 进入交互式对话循环
    chat_history = []  # 存放对话历史 [(human_msg, ai_msg), ...]
    print("\n🏥 医疗助手已就绪（输入 'exit' 退出）")

    while True:
        query = input("\n👤 用户：").strip()
        if query.lower() in ['exit', 'quit', '退出']:
            break
        if not query:
            continue

        print("🔍 检索中...")
        # 传入 query 和历史记录
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})

        # 5. 打印答案与来源回溯
        print(f"\n🤖 助手：{response['answer']}")

        # 提取 metadata 中的页码
        pages = {str(doc.metadata.get('page', 0) + 1) for doc in response['context']}
        print(f"📍 知识来源：PDF 第 {', '.join(sorted(pages))} 页")

        # 更新对话历史
        chat_history.append((query, response['answer']))
        # 保持记忆不过长（可选，例如只保留最近5轮）
        if len(chat_history) > 5:
            chat_history.pop(0)


if __name__ == "__main__":
    main()