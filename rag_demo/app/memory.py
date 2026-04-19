"""
记忆管理模块

管理对话历史的持久化存储（SQLite 数据库）：
- get_session_history: 获取或创建指定会话的历史记录
- clear_session_history: 清空指定会话的历史
- get_all_session_ids: 获取所有已存在的会话 ID 列表
- get_session_preview: 获取会话预览内容（用于界面显示）
- delete_session: 删除指定会话的所有记录
- format_history_for_llm: 格式化历史消息供 LLM 使用
"""
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from .configs import MEMORY_DB_PATH
import sqlite3



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建指定会话的消息历史"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{MEMORY_DB_PATH}"
    )


def clear_session_history(session_id: str) -> None:
    """清空指定会话的历史"""
    history = get_session_history(session_id)
    history.clear()


def get_history_messages(session_id: str):
    """获取历史消息列表（用于展示）"""
    history = get_session_history(session_id)
    return history.messages


def format_history_for_llm(session_id: str, max_turns: int = 10):
    """格式化历史消息给 LLM 使用，限制最大轮数"""
    history = get_session_history(session_id)
    messages = history.messages[-(max_turns * 2):]  # 每轮包含 user + assistant
    return messages


def get_all_session_ids() -> list:
    """获取所有已存在的会话ID列表"""
    if not MEMORY_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT DISTINCT session_id FROM message_store ORDER BY session_id")
        session_ids = [row[0] for row in cursor.fetchall()]
        return session_ids
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def get_session_preview(session_id: str, max_length: int = 50) -> str:
    """获取会话的预览内容（第一条用户消息）"""
    history = get_session_history(session_id)
    messages = history.messages

    for msg in messages:
        if msg.type == "human":
            content = msg.content
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content

    return "（空会话）"


def delete_session(session_id: str) -> None:
    """删除指定会话的所有记录"""
    if not MEMORY_DB_PATH.exists():
        return

    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def clear_session_history(session_id: str) -> None:
    """清空指定会话的历史"""
    history = get_session_history(session_id)
    history.clear()


def get_history_messages(session_id: str):
    """获取历史消息列表（用于展示）"""
    history = get_session_history(session_id)
    return history.messages


def format_history_for_llm(session_id: str, max_turns: int = 10):
    """格式化历史消息给 LLM 使用，限制最大轮数"""
    history = get_session_history(session_id)
    messages = history.messages[-(max_turns * 2):]  # 每轮包含 user + assistant
    return messages
