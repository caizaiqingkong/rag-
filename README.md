# RAG 知识库问答系统

基于 LangChain + FAISS 的检索增强生成（RAG）系统，支持长期记忆和多轮对话。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 混合检索 | 向量检索 + BM25 关键词检索 + RRF 融合排序 |
| 重排序优化 | Flashrank 对检索结果进行精细化重排序 |
| 长期记忆 | SQLite 持久化存储对话历史，支持多会话管理 |
| 增量更新 | 基于文件哈希值检测变更，智能增量更新向量索引 |
| Web 界面 | Streamlit 实现的友好对话界面 |

---

## 项目结构

```
rag_demo/
├── app/                        # 核心应用模块
│   ├── __init__.py             # 包初始化
│   ├── configs.py              # 配置管理（路径、检索参数、环境变量）
│   ├── exceptions.py           # 自定义异常类
│   ├── logger_config.py        # 日志配置
│   ├── loaders.py              # 文档加载与处理
│   ├── index_manager.py        # 向量索引管理（FAISS）
│   ├── retrieval.py            # 检索模块（向量+BM25+RRF+重排序）
│   ├── llm_service.py          # LLM 服务（支持长期记忆）
│   └── memory.py               # 记忆管理（SQLite 持久化）
├── document/                   # 知识库文档目录（txt/md/pdf）
├── faiss_index/                # FAISS 向量索引存储（自动生成）
├── logs/                       # 日志文件目录
├── main.py                     # 命令行入口
├── streamlit_app.py            # Streamlit Web 界面
├── chat_memory.db              # 对话记忆数据库（自动生成）
├── .env                        # 环境变量配置
├── requirements.txt            # Python 依赖
└── README.md                   # 项目说明
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag_demo
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

**激活虚拟环境：**

```bash
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

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

> 支持 any OpenAI 兼容的 API（DeepSeek、智谱、通义千问等）

### 5. 准备知识库文档

将文档放入 `document/` 目录，支持以下格式：
- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.pdf` - PDF 文档

### 6. 启动应用

**方式一：Web 界面（推荐）**

```bash
streamlit run streamlit_app.py
```

浏览器访问：http://localhost:8501

**方式二：命令行模式**

```bash
python main.py
```

---

## 核心模块说明

### configs.py - 配置管理

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `VECTOR_SEARCH_MODE` | `mmr` | 向量检索模式：similarity / mmr / threshold |
| `VECTOR_K` | `8` | 向量检索返回文档数 |
| `BM25_K` | `8` | BM25 检索返回文档数 |
| `FINAL_TOP_K` | `6` | 最终返回文档数 |
| `VECTOR_WEIGHT` | `0.6` | 向量检索权重（RRF 融合） |
| `BM25_WEIGHT` | `0.4` | BM25 检索权重（RRF 融合） |
| `USE_RERANK` | `True` | 是否启用重排序 |

### memory.py - 记忆管理

| 函数 | 说明 |
|------|------|
| `get_session_history(session_id)` | 获取或创建会话历史 |
| `clear_session_history(session_id)` | 清空指定会话记忆 |
| `get_all_session_ids()` | 获取所有会话 ID 列表 |
| `get_session_preview(session_id)` | 获取会话预览内容 |
| `delete_session(session_id)` | 删除指定会话 |

### retrieval.py - 检索流程

```
用户查询
    ↓
┌─────────────────────────────────────┐
│  向量检索 (FAISS + BGE-small-zh)     │
│  BM25 关键词检索                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  RRF 倒数排名融合                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Flashrank 重排序优化                │
└─────────────────────────────────────┘
    ↓
返回 Top-K 相关文档
```

---

## Web 界面功能

### 会话管理

- **历史会话列表**：侧边栏显示所有已保存的会话
- **切换会话**：点击会话名即可切换，自动加载历史对话
- **新建会话**：输入新会话 ID 创建独立对话
- **删除会话**：删除不需要的会话记录

### 检索设置

- **来源文件过滤**：只检索指定文件
- **文件类型过滤**：按 txt/md/pdf 过滤

### 对话功能

- **多轮对话**：AI 能记住之前的对话内容
- **来源追溯**：显示回答引用的来源文件

---

## 依赖说明

| 依赖包 | 用途 |
|--------|------|
| `langchain` | RAG 框架核心 |
| `langchain-community` | 社区集成（BM25、向量存储等） |
| `langchain-huggingface` | HuggingFace 嵌入模型 |
| `langchain-openai` | OpenAI 兼容 API 调用 |
| `faiss-cpu` | Facebook 向量索引库 |
| `sentence-transformers` | 句向量嵌入模型 |
| `streamlit` | Web 界面 |
| `python-dotenv` | 环境变量管理 |
| `pypdf` | PDF 文档解析 |
| `flashrank` | 重排序模型 |

---

## 常见问题

### Q: 首次启动很慢？

A: 首次运行需要下载嵌入模型 `BAAI/bge-small-zh-v1.5`（约 100MB），请耐心等待。模型会缓存到本地，后续启动会很快。

### Q: 如何更换 LLM？

A: 修改 `.env` 文件：

```env
# 使用 DeepSeek
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat

# 使用智谱 GLM
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MODEL_NAME=glm-4

# 使用通义千问
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
```

### Q: 如何调整检索效果？

A: 修改 `app/configs.py`：

```python
# 偏重语义相似
VECTOR_WEIGHT = 0.8
BM25_WEIGHT = 0.2

# 偏重关键词匹配
VECTOR_WEIGHT = 0.3
BM25_WEIGHT = 0.7

# 返回更多文档
FINAL_TOP_K = 10
```

### Q: 如何更换嵌入模型？

A: 修改 `app/configs.py`：

```python
# 使用其他中文嵌入模型
EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# 使用多语言模型
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

## 开发说明

### 运行测试

```bash
# 测试记忆模块
python -c "from app.memory import get_session_history; print('OK')"

# 测试 LLM 服务
python -c "from app.llm_service import ask_llm_with_memory; print('OK')"
```

### 查看日志

```bash
# 实时查看日志
tail -f logs/rag.log
```

---

## License

MIT License
