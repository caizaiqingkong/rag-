"""
向量索引管理模块

管理 FAISS 向量索引的初始化和更新：
- 全量重建：首次运行或文件修改/删除时
- 增量更新：仅新增文件时追加到现有索引
- 基于文件哈希值检测变更，避免不必要的重建
"""
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .configs import (
    EMBED_MODEL_NAME,
    INDEX_DIR,
    MANIFEST_FILE,
    REBUILD_INDEX,
)
from .exceptions import IndexBuildError
from .logger_config import logger
from .loaders import (
    load_manifest,
    save_manifest,
    build_current_manifest,
    detect_changes,
)


def rebuild_all_vector_store(all_chunk_docs, embeddings):
    try:
        logger.info("执行全量重建...")
        vector_store = FAISS.from_documents(all_chunk_docs, embeddings)
        vector_store.save_local(str(INDEX_DIR))
        return vector_store
    except Exception as e:
        logger.exception("全量重建向量库失败")
        raise IndexBuildError("全量重建向量库失败") from e


def incremental_add_new_files(vector_store: FAISS, new_chunk_docs):
    if not new_chunk_docs:
        return vector_store

    try:
        ids = [str(uuid4()) for _ in range(len(new_chunk_docs))]
        vector_store.add_documents(documents=new_chunk_docs, ids=ids)
        vector_store.save_local(str(INDEX_DIR))
        logger.info(f"增量新增 chunk 数量: {len(new_chunk_docs)}")
        return vector_store
    except Exception as e:
        logger.exception("增量写入向量库失败")
        raise IndexBuildError("增量写入向量库失败") from e


def init_or_update_vector_store(files, all_chunk_docs, grouped_chunk_docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    index_file = INDEX_DIR / "index.faiss"

    current_manifest = build_current_manifest(files)
    old_manifest = load_manifest(MANIFEST_FILE)

    try:
        if REBUILD_INDEX or not index_file.exists() or not old_manifest:
            vector_store = rebuild_all_vector_store(all_chunk_docs, embeddings)
            save_manifest(MANIFEST_FILE, INDEX_DIR, current_manifest)
            return vector_store

        added, modified, deleted = detect_changes(old_manifest, current_manifest)

        logger.info(f"新增文件: {added}")
        logger.info(f"修改文件: {modified}")
        logger.info(f"删除文件: {deleted}")

        if modified or deleted:
            vector_store = rebuild_all_vector_store(all_chunk_docs, embeddings)
            save_manifest(MANIFEST_FILE, INDEX_DIR, current_manifest)
            return vector_store

        vector_store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

        if added:
            new_docs = []
            for source_name in added:
                new_docs.extend(grouped_chunk_docs.get(source_name, []))
            vector_store = incremental_add_new_files(vector_store, new_docs)
            save_manifest(MANIFEST_FILE, INDEX_DIR, current_manifest)
        else:
            logger.info("没有文件变化，直接使用已有索引。")

        return vector_store

    except Exception as e:
        logger.exception("初始化或更新向量库失败")
        raise IndexBuildError("初始化或更新向量库失败") from e