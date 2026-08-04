"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from functools import lru_cache
import os
from pathlib import Path
import re

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
# 800 ký tự giữ được ngữ cảnh của một mục tuyển sinh; overlap 100 ký tự hạn chế
# mất ngữ cảnh ở ranh giới giữa hai chunk.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Gọi embedding online để không phải tải/nạp model local. Task 4 và Task 5 bắt
# buộc dùng đúng model này; nếu đổi model thì phải chạy lại index.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)
EMBEDDING_DIM = 1536

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "hust_admissions_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    # documents = []
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     doc_type = "legal" if "legal" in str(md_file) else "news"
    #     documents.append({
    #         "content": content,
    #         "metadata": {"source": md_file.name, "type": doc_type}
    #     })
    # return documents
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw_content = md_file.read_text(encoding="utf-8")
        content, frontmatter = _strip_frontmatter(raw_content)
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        title = _first_heading(content) or md_file.stem.replace("-", " ")
        metadata = {
            "doc_id": frontmatter.get("doc_id", md_file.stem),
            "title": frontmatter.get("title", title),
            "source": md_file.name,
            "type": frontmatter.get("type", doc_type),
        }
        for key in ("source_url", "retrieved_at", "document_version"):
            if frontmatter.get(key) is not None:
                metadata[key] = str(frontmatter[key])
        documents.append({
            "content": content,
            "metadata": metadata,
        })
    return documents


def _strip_frontmatter(content: str) -> tuple[str, dict]:
    """Tách YAML frontmatter để không index metadata như nội dung tài liệu."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        return content, {}

    metadata = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return match.group(2), metadata


def _first_heading(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separators=["\n\n", "\n", ". ", " ", ""]
    # )
    # chunks = []
    # for doc in documents:
    #     splits = splitter.split_text(doc["content"])
    #     for i, chunk_text in enumerate(splits):
    #         chunks.append({
    #             "content": chunk_text,
    #             "metadata": {**doc["metadata"], "chunk_index": i}
    #         })
    # return chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        # Markdown "## Heading" (vd "Điểm chuẩn theo phương thức Điểm thi THPT
        # năm 2025") thường tách thành 1 chunk riêng ở boundary \n\n, mất liên
        # kết với các chunk bảng dữ liệu theo sau — retrieval trả về đúng dòng
        # "IT1 | 29.19" nhưng không biết đó là phương thức nào. Prepend heading
        # gần nhất vào mỗi chunk không tự mang theo, để cả dense và BM25 giữ
        # được ngữ cảnh "phương thức nào" trong từng chunk độc lập.
        current_heading = ""
        for i, chunk_text in enumerate(splits):
            stripped = chunk_text.strip()
            if stripped.startswith("#"):
                current_heading = stripped.splitlines()[0]
                content = chunk_text
            elif current_heading and current_heading not in chunk_text:
                content = f"{current_heading}\n\n{chunk_text}"
            else:
                content = chunk_text
            chunks.append({
                "content": content,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Tạo OpenAI client một lần; key được đọc từ OPENAI_API_KEY trong .env."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY trong file .env")
    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Sinh embedding đã chuẩn hoá L2 để cosine similarity ổn định."""
    if not texts:
        return []
    client = get_embedding_model()
    vectors = []
    for start in range(0, len(texts), 100):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + 100],
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    
    #
    # Ví dụ với sentence-transformers (local, mặc định):
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # texts = [c["content"] for c in chunks]
    # embeddings = model.encode(texts, show_progress_bar=True)
    # for chunk, emb in zip(chunks, embeddings):
    #     chunk["embedding"] = emb.tolist()
    # return chunks
    #
    # Nâng cao (optional): nếu muốn cho cả nhóm chọn được provider qua .env, viết
    # 1 hàm embed_texts(texts) dispatch theo os.getenv("EMBEDDING_PROVIDER") sang
    # sentence-transformers | Google (genai.embed_content) | OpenAI (client.embeddings.create)
    # rồi gọi lại hàm đó ở đây và ở Task 5 — tránh viết logic embed lặp lại 2 nơi.
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def get_collection():
    """Trả về persistent Chroma collection dùng cosine distance."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def recreate_collection():
    """Tạo collection sạch mỗi lần re-index, tránh lẫn chunk của corpus cũ."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        # Collection chưa tồn tại ở lần index đầu tiên.
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_ids(chunks: list[dict]) -> list[str]:
    return [
        f"{chunk['metadata']['doc_id']}_chunk_{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]


def index_to_vectorstore(chunks: list[dict], reset: bool = False):
    """
    Lưu chunks vào vector store đã chọn.
    """
    #
    # Ví dụ với ChromaDB:
    # import chromadb
    #
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # collection = client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    #
    # ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    # collection.upsert(
    #     ids=ids,
    #     documents=[c["content"] for c in chunks],
    #     embeddings=[c["embedding"] for c in chunks],
    #     metadatas=[c["metadata"] for c in chunks],
    # )
    if not chunks:
        return

    collection = recreate_collection() if reset else get_collection()
    
    ids = _chunk_ids(chunks)
    # Chroma giới hạn một upsert request; batch nhỏ cũng tránh tốn RAM với corpus lớn.
    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=ids[start : start + batch_size],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Ghi từng batch ngay sau khi tạo embedding. Nếu tiến trình bị ngắt, chạy lại
    # lệnh sẽ bỏ qua chunk đã có trong ChromaDB thay vì phải gọi API lại từ đầu.
    collection = get_collection()
    existing_ids = set(collection.get(include=[])["ids"])
    pending_chunks = [chunk for chunk in chunks if _chunk_ids([chunk])[0] not in existing_ids]
    print(f"✓ {len(existing_ids)} chunks đã có; còn {len(pending_chunks)} chunks cần index")

    batch_size = 100
    for start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[start : start + batch_size]
        index_to_vectorstore(embed_chunks(batch))
        print(f"✓ Indexed {min(start + len(batch), len(pending_chunks))}/{len(pending_chunks)} chunks mới")

    print(f"✓ ChromaDB có {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
