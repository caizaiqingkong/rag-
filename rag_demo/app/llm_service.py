"""
LLM 服务模块

封装大语言模型调用逻辑：
- ask_llm: 基础问答函数，单次对话
- ask_llm_with_memory: 带长期记忆的问答函数，支持多轮对话
  - 自动加载历史消息作为上下文
  - 自动保存用户问题和 AI 回复到记忆数据库
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .exceptions import GenerationError
from .logger_config import logger


def ask_llm(envs: dict, context: str, query: str) -> str:
    try:
        llm = ChatOpenAI(
            api_key=envs["OPENAI_API_KEY"],
            base_url=envs["OPENAI_BASE_URL"],
            model=envs["MODEL_NAME"],
            temperature=0
        )

        prompt = f"""你是一个严谨的知识库问答助手。
请严格根据参考内容回答问题。
如果参考内容不足，就直接回答：我不知道。
回答时尽量说明信息来自哪些文件。
不要编造参考内容中没有的信息。

参考内容：
{context}

用户问题：
{query}

请按下面格式回答：
1. 先给出答案
2. 再给出“来源文件：...”
"""

        answer = llm.invoke(prompt)
        return answer.content

    except Exception as e:
        logger.exception("大模型调用失败")
        raise GenerationError("大模型调用失败，请检查 API_KEY / BASE_URL / MODEL_NAME / 网络连接") from e


def ask_llm_with_memory(
    envs: dict,
    context: str,
    query: str,
    session_id: str = "default",
    max_history_turns: int = 10
) -> str:
    """带长期记忆的 LLM 调用"""
    from .memory import get_session_history, format_history_for_llm

    try:
        llm = ChatOpenAI(
            api_key=envs["OPENAI_API_KEY"],
            base_url=envs["OPENAI_BASE_URL"],
            model=envs["MODEL_NAME"],
            temperature=0
        )

        # 构建系统提示
        system_prompt = f"""你是一个严谨的知识库问答助手。
请严格根据参考内容回答问题。
如果参考内容不足，就直接回答：我不知道。
回答时尽量说明信息来自哪些文件。
不要编造参考内容中没有的信息。

参考内容：
{context}"""

        # 获取历史消息
        history = get_session_history(session_id)
        history_messages = format_history_for_llm(session_id, max_history_turns)

        # 构建完整消息列表
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history_messages)
        messages.append(HumanMessage(content=query))

        # 调用 LLM
        response = llm.invoke(messages)
        answer = response.content

        # 保存到记忆
        history.add_user_message(query)
        history.add_ai_message(answer)

        return answer

    except Exception as e:
        logger.exception("大模型调用失败")
        raise GenerationError("大模型调用失败，请检查 API_KEY / BASE_URL / MODEL_NAME / 网络连接") from e