"""
异常定义模块

定义项目中所有自定义异常类，用于区分不同类型的错误：
- ConfigError: 配置错误（如 .env 缺少变量）
- DataLoadError: 文档加载失败
- IndexBuildError: 索引构建或加载失败
- RetrievalError: 检索阶段失败
- GenerationError: 大模型生成失败
"""
class ConfigError(Exception):
    """配置错误，比如 .env 缺少变量"""
    pass


class DataLoadError(Exception):
    """文档加载失败"""
    pass


class IndexBuildError(Exception):
    """索引构建或加载失败"""
    pass


class RetrievalError(Exception):
    """检索阶段失败"""
    pass


class GenerationError(Exception):
    """大模型生成失败"""
    pass