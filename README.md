# RAG 知识库问答系统

基于 LangChain + FAISS 的检索增强生成（RAG）系统，支持长期记忆和多轮对话。

## 功能特性

- **混合检索**：向量检索 + BM25 关键词检索 + RRF 融合排序
- **重排序优化**：使用 Flashrank 对检索结果进行重排序
- **长期记忆**：SQLite 持久化存储对话历史，支持多会话管理
- **增量更新**：基于文件哈希值检测变更，智能增量更新向量索引
- **Web 界面**：Streamlit 实现的友好对话界面

## 项目结构

```
rag_demo/
├── app/
│   ├── __init__.py          # 包初始化
│   ├── configs.py           # 配置管理（路径、检索参数、环境变量）
│   ├── exceptions.py        # 自定义异常类
│   ├── logger_config.py     # 日志配置
│   ├── loaders.py           # 文档加载与处理
│   ├── index_manager.py     # 向量索引管理（FAISS）
│   ├── retrieval.py         # 检索模块（向量+BM25+RRF+重排序）
│   ├── llm_service.py       # LLM 服务（支持长期记忆）
│   └── memory.py            # 记忆管理（SQLite 持久化）
├── document/                # 知识库文档目录（txt/md/pdf）
├── faiss_index/             # FAISS 向量索引存储
├── logs/                    # 日志文件目录
├── main.py                  # 命令行入口
├── streamlit_app.py         # Streamlit Web 界面
├── chat_memory.db           # 对话记忆数据库（自动生成）
├── .env                     # 环境变量配置
└── requirements.txt         # Python 依赖
```

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag_demo
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### 5. 准备知识库文档

将你的文档（支持 `.txt`、`.md`、`.pdf` 格式）放入 `document/` 目录。

### 6. 启动应用

**Web 界面（推荐）：**

```bash
streamlit run streamlit_app.py
```

访问 http://localhost:8501

**命令行模式：**

```bash
python main.py
```

## 核心模块说明

### configs.py - 配置管理

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `VECTOR_SEARCH_MODE` | `mmr` | 向量检索模式：similarity/mmr/threshold |
| `VECTOR_K` | `8` | 向量检索返回文档数 |
| `BM25_K` | `8` | BM25 检索返回文档数 |
| `FINAL_TOP_K` | `6` | 最终返回文档数 |
| `VECTOR_WEIGHT` | `0.6` | 向量检索权重（RRF 融合） |
| `BM25_WEIGHT` | `0.4` | BM25 检索权重（RRF 融合） |
| `USE_RERANK` | `True` | 是否启用重排序 |

### memory.py - 记忆管理

- `get_session_history(session_id)` - 获取会话历史
- `clear_session_history(session_id)` - 清空会话记忆
- `get_all_session_ids()` - 获取所有会话 ID
- `delete_session(session_id)` - 删除会话

### retrieval.py - 检索策略

1. **向量检索**：使用 FAISS + BGE-small-zh 向量模型
2. **BM25 检索**：关键词匹配，弥补向量检索的不足
3. **RRF 融合**：倒数排名融合，合并两种检索结果
4. **重排序**：Flashrank 对最终结果进行精细化排序

## Web 界面功能

- **会话管理**：创建、切换、删除会话
- **对话历史**：自动保存和加载历史对话
- **检索过滤**：按来源文件或文件类型过滤
- **来源追溯**：显示回答的来源文件

## 依赖说明

| 依赖包 | 用途 |
|--------|------|
| `langchain` | RAG 框架 |
| `langchain-community` | 社区集成（BM25、向量存储等） |
| `langchain-huggingface` | HuggingFace 嵌入模型 |
| `langchain-openai` | OpenAI 兼容 API 调用 |
| `faiss-cpu` | 向量索引 |
| `streamlit` | Web 界面 |
| `python-dotenv` | 环境变量管理 |
| `pypdf` | PDF 文档解析 |

## 常见问题

### Q: 首次启动很慢？

A: 首次运行需要下载嵌入模型 `BAAI/bge-small-zh-v1.5`，请耐心等待。

### Q: 如何更换 LLM？

A: 修改 `.env` 文件中的 `OPENAI_BASE_URL` 和 `MODEL_NAME`，支持任何 OpenAI 兼容的 API。

### Q: 如何调整检索效果？

A: 修改 `app/configs.py` 中的参数：
- 提高 `VECTOR_WEIGHT` 偏重语义相似
- 提高 `BM25_WEIGHT` 偏重关键词匹配
- 调整 `FINAL_TOP_K` 控制返回文档数量

## License

MIT
