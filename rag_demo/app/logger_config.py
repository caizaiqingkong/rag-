"""
日志配置模块

配置项目的日志系统：
- 日志输出到文件（logs/rag.log）和控制台
- 统一的日志格式：时间 | 日志级别 | 消息
"""
import logging

from .configs import LOG_DIR, LOG_FILE

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)