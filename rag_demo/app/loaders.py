"""
文档加载与处理模块

负责文档的加载、切分和管理，包括：
- 扫描支持的文件类型（txt、md、pdf）
- 加载文档内容并转换为 LangChain Document 对象
- 文档切分（RecursiveCharacterTextSplitter）
- 文件变更检测（基于哈希值的增量更新）
- 文档过滤和格式化工具函数
"""
import json
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .exceptions import DataLoadError, IndexBuildError
from .logger_config import logger


def calc_file_hash(file_path: Path) -> str:
    try:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.exception(f"计算文件哈希失败: {file_path}")
        raise DataLoadError(f"计算文件哈希失败: {file_path}") from e


def scan_supported_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise DataLoadError(f"文档目录不存在: {data_dir}")

    files = []
    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
            files.append(file_path)

    return files


def load_one_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    try:
        if suffix in [".txt", ".md"]:
            text = file_path.read_text(encoding="utf-8")
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "file_type": suffix.replace(".", "")
                    }
                )
            ]

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            pdf_docs = loader.load()
            for page_doc in pdf_docs:
                page_doc.metadata["source"] = file_path.name
                page_doc.metadata["file_path"] = str(file_path)
                page_doc.metadata["file_type"] = "pdf"
            return pdf_docs

        logger.warning(f"跳过不支持的文件类型: {file_path}")
        return []

    except UnicodeDecodeError:
        logger.exception(f"文本编码错误，跳过文件: {file_path}")
        return []
    except FileNotFoundError:
        logger.exception(f"文件不存在，跳过文件: {file_path}")
        return []
    except Exception:
        logger.exception(f"加载文件失败，跳过文件: {file_path}")
        return []


def load_documents_from_files(files: list[Path]) -> list[Document]:
    docs = []
    for file_path in files:
        one_docs = load_one_file(file_path)
        docs.extend(one_docs)

    if not docs:
        raise DataLoadError("没有成功加载任何文档，请检查 document 目录和文件内容")

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=80,
            add_start_index=True
        )

        chunk_docs = splitter.split_documents(documents)

        for i, doc in enumerate(chunk_docs):
            doc.metadata["chunk_id"] = i

        if not chunk_docs:
            raise DataLoadError("文档切分后为空，无法建立知识库")

        return chunk_docs

    except Exception as e:
        logger.exception("文档切分失败")
        raise DataLoadError("文档切分失败") from e


def group_docs_by_source(docs: list[Document]) -> dict[str, list[Document]]:
    grouped = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)
    return grouped


def load_manifest(manifest_file: Path) -> dict:
    if not manifest_file.exists():
        return {}

    try:
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("manifest.json 损坏，将视为不存在并触发重建")
        return {}
    except Exception:
        logger.exception("读取 manifest 失败，将视为不存在并触发重建")
        return {}


def save_manifest(manifest_file: Path, index_dir: Path, data: dict) -> None:
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        logger.exception("保存 manifest 失败")
        raise IndexBuildError("保存 manifest 失败") from e


def build_current_manifest(files: list[Path]) -> dict:
    return {
        file_path.name: {
            "file_path": str(file_path),
            "hash": calc_file_hash(file_path),
        }
        for file_path in files
    }


def detect_changes(old_manifest: dict, new_manifest: dict):
    old_names = set(old_manifest.keys())
    new_names = set(new_manifest.keys())

    added = sorted(new_names - old_names)
    deleted = sorted(old_names - new_names)
    modified = sorted(
        name for name in (old_names & new_names)
        if old_manifest[name]["hash"] != new_manifest[name]["hash"]
    )
    return added, modified, deleted


def build_filter_dict(source_filter: str, file_type_filter: str):
    filter_dict = {}

    if source_filter:
        filter_dict["source"] = source_filter

    if file_type_filter:
        filter_dict["file_type"] = file_type_filter

    return filter_dict if filter_dict else None


def doc_matches_filter(doc: Document, filter_dict: dict | None) -> bool:
    if not filter_dict:
        return True

    for key, value in filter_dict.items():
        if doc.metadata.get(key) != value:
            return False
    return True


def build_doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page", ""))
    chunk_id = str(doc.metadata.get("chunk_id", ""))
    start_index = str(doc.metadata.get("start_index", ""))

    if source or page or chunk_id or start_index:
        return f"{source}|{page}|{chunk_id}|{start_index}"

    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return content_hash


def print_docs(title: str, docs: list[Document], max_chars: int = 250):
    print(f"\n===== {title} =====")
    if not docs:
        print("没有结果")
        return

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Top {i} ---")
        print(
            f"source={doc.metadata.get('source')}, "
            f"file_type={doc.metadata.get('file_type')}, "
            f"page={doc.metadata.get('page', 'N/A')}, "
            f"chunk_id={doc.metadata.get('chunk_id')}"
        )
        print(doc.page_content[:max_chars])


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"[片段{i}]\n"
            f"来源文件: {doc.metadata.get('source', '未知来源')}\n"
            f"文件类型: {doc.metadata.get('file_type', '未知类型')}\n"
            f"页码: {doc.metadata.get('page', 'N/A')}\n"
            f"chunk_id: {doc.metadata.get('chunk_id', '未知chunk')}\n"
            f"内容:\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def collect_sources(docs: list[Document]) -> str:
    seen = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        if source not in seen:
            seen.append(source)
    return "、".join(seen) if seen else "无"