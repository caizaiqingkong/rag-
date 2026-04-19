import streamlit as st
from app.configs import DATA_DIR, validate_env, VECTOR_SEARCH_MODE, USE_RERANK
from app.loaders import (
    scan_supported_files,
    load_documents_from_files,
    split_documents,
    group_docs_by_source,
    build_filter_dict,
    format_docs,
    collect_sources,
)
from app.index_manager import init_or_update_vector_store
from app.retrieval import (
    build_bm25_retriever,
    retrieve_vector_docs,
    retrieve_bm25_docs,
    reciprocal_rank_fusion,
    rerank_docs,
)
from app.llm_service import ask_llm_with_memory
from app.memory import (
    get_session_history,
    clear_session_history,
    get_all_session_ids,
    get_session_preview,
    delete_session,
)


# ========== 页面配置 ==========
st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide"
)


# ========== 初始化 Session State ==========
def init_session_state():
    """初始化 Streamlit session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "bm25_retriever" not in st.session_state:
        st.session_state.bm25_retriever = None

    if "envs" not in st.session_state:
        st.session_state.envs = None


def load_messages_from_memory(session_id: str) -> list:
    """从数据库加载会话历史消息到界面"""
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        messages.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
            "sources": None  # 历史消息不保存来源信息
        })
    return messages


# ========== 加载知识库索引 ==========
@st.cache_resource
def load_knowledge_base():
    """加载知识库（只执行一次）"""
    try:
        envs = validate_env()

        files = scan_supported_files(DATA_DIR)
        if not files:
            return None, None, envs, "document 目录中没有找到文件"

        raw_docs = load_documents_from_files(files)
        chunk_docs = split_documents(raw_docs)
        grouped_chunk_docs = group_docs_by_source(chunk_docs)

        vector_store = init_or_update_vector_store(files, chunk_docs, grouped_chunk_docs)
        bm25_retriever = build_bm25_retriever(chunk_docs)

        return vector_store, bm25_retriever, envs, f"已加载 {len(files)} 个文件，{len(chunk_docs)} 个文档块"
    except Exception as e:
        return None, None, None, f"加载失败: {str(e)}"


# ========== RAG 检索函数 ==========
def retrieve_context(query: str, vector_store, bm25_retriever, filter_dict=None):
    """执行 RAG 检索，返回上下文和来源"""
    vector_docs = retrieve_vector_docs(vector_store, query, filter_dict=filter_dict)
    bm25_docs = retrieve_bm25_docs(bm25_retriever, query, filter_dict=filter_dict)

    hybrid_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
    final_docs = rerank_docs(hybrid_docs, query)

    if not final_docs:
        return None, None

    context = format_docs(final_docs)
    sources = collect_sources(final_docs)
    return context, sources


# ========== 主界面 ==========
def main():
    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.title(" RAG 知识库问答")
        st.markdown("---")

        # 会话管理
        st.subheader("会话管理")

        # 获取所有会话ID
        all_sessions = get_all_session_ids()

        # 显示会话列表
        if all_sessions:
            st.write("**历史会话：**")
            for sid in all_sessions:
                col1, col2 = st.columns([4, 3])
                with col1:
                    preview = get_session_preview(sid)
                    is_current = sid == st.session_state.session_id
                    label = f"{'[now] ' if is_current else ''}{sid}"
                    if st.button(
                        f"{label}",
                        key=f"select_{sid}",
                        help=preview,
                        use_container_width=True
                    ):
                        if sid != st.session_state.session_id:
                            st.session_state.session_id = sid
                            st.session_state.messages = load_messages_from_memory(sid)
                            st.rerun()
                with col2:
                    if st.button("删除", key=f"del_{sid}", help="删除此会话"):
                        delete_session(sid)
                        if sid == st.session_state.session_id:
                            st.session_state.session_id = "default"
                            st.session_state.messages = load_messages_from_memory("default")
                        st.rerun()

        st.markdown("---")

        # 新建会话
        new_session_id = st.text_input("新建会话ID", key="new_session_input", placeholder="输入新会话名称")
        if st.button("创建新会话", use_container_width=True):
            if new_session_id and new_session_id.strip():
                new_id = new_session_id.strip()
                st.session_state.session_id = new_id
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # 当前会话信息
        st.write(f"**当前会话：** `{st.session_state.session_id}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空会话", use_container_width=True):
                clear_session_history(st.session_state.session_id)
                st.session_state.messages = []
                st.success("已清空")
                st.rerun()
        with col2:
            if st.button("刷新历史", use_container_width=True):
                st.session_state.messages = load_messages_from_memory(st.session_state.session_id)
                st.rerun()

        st.markdown("---")

        # 检索设置
        st.subheader("检索设置")
        source_filter = st.text_input("来源文件过滤（可选）", key="source_filter")
        file_type_filter = st.selectbox(
            "文件类型过滤",
            ["全部", "txt", "md", "pdf"],
            key="file_type_filter"
        )

        st.markdown("---")
        st.caption(f"向量检索模式: {VECTOR_SEARCH_MODE}")
        st.caption(f"Rerank: {'开启' if USE_RERANK else '关闭'}")

    # 加载知识库
    vector_store, bm25_retriever, envs, status_msg = load_knowledge_base()

    if envs is None:
        st.error(status_msg)
        return

    st.sidebar.success(status_msg)

    # 如果 messages 为空，尝试从数据库加载
    if not st.session_state.messages:
        st.session_state.messages = load_messages_from_memory(st.session_state.session_id)

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.caption(f"📄 来源: {message['sources']}")

    # 用户输入
    if prompt := st.chat_input("请输入问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 构建过滤条件
        filter_dict = build_filter_dict(
            source_filter if source_filter else None,
            file_type_filter if file_type_filter != "全部" else None
        )

        # 检索 + 生成
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # RAG 检索
                context, sources = retrieve_context(
                    prompt, vector_store, bm25_retriever, filter_dict
                )

                if context is None:
                    response = "没有检索到足够相关的内容，我不知道。"
                    sources = None
                else:
                    # 调用带记忆的 LLM
                    response = ask_llm_with_memory(
                        envs, context, prompt,
                        session_id=st.session_state.session_id
                    )

            st.markdown(response)
            if sources:
                st.caption(f"📄 来源: {sources}")

        # 保存助手回复（界面显示用）
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
