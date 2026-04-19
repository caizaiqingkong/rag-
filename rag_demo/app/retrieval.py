"""
检索模块

实现混合检索策略，包括：
- 向量检索（FAISS，支持 similarity/mmr/threshold 三种模式）
- BM25 关键词检索
- 倒数排名融合（RRF）合并向量与 BM25 结果
- Flashrank 重排序优化
"""
import hashlib

from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank

from .configs import (
    BM25_K,
    VECTOR_K,
    FINAL_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RRF_K,
    VECTOR_SEARCH_MODE,
    USE_RERANK,
)
from .exceptions import RetrievalError
from .logger_config import logger
from .loaders import doc_matches_filter, build_doc_key


def build_vector_retriever(vector_store):
    if VECTOR_SEARCH_MODE == "similarity":
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_K}
        )
    elif VECTOR_SEARCH_MODE == "threshold":
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": VECTOR_K}
        )
    else:
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": VECTOR_K}
        )


def retrieve_vector_docs(vector_store, query: str, filter_dict=None):
    try:
        retriever = build_vector_retriever(vector_store)
        return retriever.invoke(query, filter=filter_dict)
    except Exception:
        logger.exception("向量检索失败，返回空结果")
        return []


def build_bm25_retriever(chunk_docs):
    try:
        bm25 = BM25Retriever.from_documents(chunk_docs)
        bm25.k = BM25_K
        return bm25
    except Exception as e:
        logger.exception("初始化 BM25 失败")
        raise RetrievalError("初始化 BM25 失败") from e


def retrieve_bm25_docs(bm25_retriever, query: str, filter_dict=None):
    try:
        docs = bm25_retriever.invoke(query)
        docs = [doc for doc in docs if doc_matches_filter(doc, filter_dict)]
        return docs[:BM25_K]
    except Exception:
        logger.exception("BM25 检索失败，返回空结果")
        return []


def reciprocal_rank_fusion(
    vector_docs,
    bm25_docs,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
):
    score_map = {}
    doc_map = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = build_doc_key(doc)
        score_map[key] = score_map.get(key, 0.0) + vector_weight / (rrf_k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = build_doc_key(doc)
        score_map[key] = score_map.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        doc_map[key] = doc

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:top_k]]


def rerank_docs(docs, query: str):
    if not USE_RERANK or not docs:
        return docs

    try:
        compressor = FlashrankRerank()
        return list(compressor.compress_documents(docs, query))
    except Exception:
        logger.exception("rerank 失败，自动降级为不 rerank")
        return docs