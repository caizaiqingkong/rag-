from app.configs import DATA_DIR, validate_env, VECTOR_SEARCH_MODE, USE_RERANK
from app.exceptions import (
    ConfigError,
    DataLoadError,
    IndexBuildError,
    RetrievalError,
    GenerationError,
)
from app.logger_config import logger
from app.loaders import (
    scan_supported_files,
    load_documents_from_files,
    split_documents,
    group_docs_by_source,
    build_filter_dict,
    print_docs,
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
from app.llm_service import ask_llm


def main():
    try:
        envs = validate_env()
        logger.info("环境变量校验通过")

        files = scan_supported_files(DATA_DIR)
        if not files:
            raise DataLoadError("document 目录中没有找到 txt / md / pdf 文件")
        logger.info(f"当前可用文件数: {len(files)}")

        raw_docs = load_documents_from_files(files)
        logger.info(f"原始 Document 数量: {len(raw_docs)}")

        chunk_docs = split_documents(raw_docs)
        logger.info(f"切分后 chunk 数量: {len(chunk_docs)}")

        grouped_chunk_docs = group_docs_by_source(chunk_docs)

        vector_store = init_or_update_vector_store(files, chunk_docs, grouped_chunk_docs)
        bm25_retriever = build_bm25_retriever(chunk_docs)

        query = input("请输入问题：").strip()
        if not query:
            raise RetrievalError("问题不能为空")

        source_filter = input("可选：只检索某个 source 文件名（直接回车跳过）：").strip()
        file_type_filter = input("可选：只检索某种文件类型 txt/md/pdf（直接回车跳过）：").strip()

        filter_dict = build_filter_dict(source_filter, file_type_filter)

        print(f"当前过滤条件: {filter_dict}")
        print(f"向量检索模式: {VECTOR_SEARCH_MODE}")
        print(f"是否启用 rerank: {USE_RERANK}")

        vector_docs = retrieve_vector_docs(vector_store, query, filter_dict=filter_dict)
        bm25_docs = retrieve_bm25_docs(bm25_retriever, query, filter_dict=filter_dict)

        print_docs("向量召回结果", vector_docs)
        print_docs("BM25 召回结果", bm25_docs)

        hybrid_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
        print_docs("混合检索融合结果（RRF）", hybrid_docs)

        final_docs = rerank_docs(hybrid_docs, query)
        print_docs("最终结果（rerank 后）", final_docs)

        if not final_docs:
            print("\n===== 最终回答 =====")
            print("没有检索到足够相关的内容，回答：我不知道。")
            return

        context = format_docs(final_docs)
        source_summary = collect_sources(final_docs)

        print("\n===== 最终送给大模型的上下文 =====")
        print(context[:2500])

        final_answer = ask_llm(envs, context, query)

        print("\n===== 最终回答 =====")
        print(final_answer)
        print(f"\n（程序识别到的来源文件：{source_summary}）")

    except ConfigError as e:
        print(f"\n[配置错误] {e}")
    except DataLoadError as e:
        print(f"\n[数据加载错误] {e}")
    except IndexBuildError as e:
        print(f"\n[索引错误] {e}")
    except RetrievalError as e:
        print(f"\n[检索错误] {e}")
    except GenerationError as e:
        print(f"\n[生成错误] {e}")
    except Exception as e:
        logger.exception("程序发生未预期错误")
        print(f"\n[未知错误] {e}")


if __name__ == "__main__":
    main()