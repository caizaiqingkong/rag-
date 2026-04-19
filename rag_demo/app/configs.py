"""
配置管理模块

负责管理项目的所有配置项，包括：
- 路径配置（数据目录、索引目录、日志目录等）
- 检索参数配置（向量检索、BM25、混合检索权重等）
- 环境变量校验（API Key、Base URL、模型名称等）
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from .exceptions import ConfigError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "document"
INDEX_DIR = BASE_DIR / "faiss_index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "rag.log"

MEMORY_DB_PATH = Path(__file__).resolve().parent.parent / "chat_memory.db"

REBUILD_INDEX = False
USE_RERANK = True

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_SEARCH_MODE = "mmr"   # similarity / mmr / threshold

VECTOR_K = 8
BM25_K = 8
FINAL_TOP_K = 6

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RRF_K = 60


def validate_env() -> dict:
    required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME"]
    envs = {}

    missing = []
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            missing.append(key)
        else:
            envs[key] = value

    if missing:
        raise ConfigError(f".env 缺少必要配置项: {missing}")

    return envs