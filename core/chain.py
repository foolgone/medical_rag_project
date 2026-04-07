from typing import Any, List, Tuple

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaLLM

from config import OLLAMA_BASE_URL


def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """
    将对话历史格式化为 LangChain 消息格式
    """
    formatted_history = []
    for human_msg, ai_msg in chat_history:
        formatted_history.append(HumanMessage(content=human_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history


def get_medical_rag_chain(vector_store: Any) -> Any:
    """
    构建具备历史对话能力的医疗 RAG 链
    """
    llm = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_q_system_prompt = (
        "给定聊天记录和一个最新的用户问题，该问题可能引用了聊天记录中的上下文。"
        "请根据这些信息，将其改写为一个独立的、即使没有历史记录也能被理解的问题。"
        "不要回答问题，只需重写它，如果不需要重写则原样返回。"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    system_prompt = (
        "你是一个专业的医疗助手。请根据下方提供的资料回答用户的问题。"
        "如果资料中没有相关内容，请礼貌地告知你不知道，不要尝试编造答案。"
        "请优先使用背景资料中的术语进行回答。"
        "\n\n背景资料:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def process_rag(input_data: dict) -> dict:
        """
        RAG 处理逻辑：历史感知检索 + 文档问答
        """
        chat_history = input_data.get("chat_history", [])
        input_question = input_data["input"]

        if chat_history:
            formatted_history = format_chat_history(chat_history)
            rewritten_question = contextualize_chain.invoke({
                "input": input_question,
                "chat_history": formatted_history
            })
        else:
            rewritten_question = input_question
            formatted_history = []

        docs = retriever.invoke(rewritten_question)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        answer = qa_prompt.invoke({
            "context": context_text,
            "chat_history": formatted_history,
            "input": input_question
        })
        answer = llm.invoke(answer)
        answer = StrOutputParser().invoke(answer)

        return {
            "answer": answer,
            "context": docs,
            "input": input_question
        }

    rag_chain = RunnableLambda(process_rag)

    return rag_chain
