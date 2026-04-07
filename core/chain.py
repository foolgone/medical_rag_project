from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import OLLAMA_BASE_URL


def get_medical_rag_chain(vector_store):
    # 1. 加载模型
    llm = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)

    # 2. 设定检索器 (RAG 原理：检索相关文档)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. 设定 Prompt (RAG 原理：喂给AI)
    template = """你是一个专业的医疗助手。请根据提供的资料回答问题。
    资料中如果没有，请说不知道。不要编造。

    背景资料:
    {context}

    问题:
    {question}

    回答:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 4. 构建链 (RAG 原理：生成回答)
    # 这就是所谓的 LCEL 写法，用 | 符号连接各组件
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain