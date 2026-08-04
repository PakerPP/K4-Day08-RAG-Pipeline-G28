"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25 chromadb underthesea

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

-------------------------------------------------------------------------------
NGUỒN DỮ LIỆU: đọc lại chunk đã được Task 4 chunk + index vào ChromaDB, KHÔNG
tự đọc lại data/standardized/*.md và chunk lần 2. Lý do: nếu BM25 tự chunk theo
tham số khác (dù chỉ lệch chunk_size/overlap), boundary của chunk sẽ khác với
chunk trong vector store → khi Task 7 fuse RRF giữa kết quả BM25 và vector
search, 2 bên không có id nào trùng nhau, mất hết lợi ích của việc fuse.

CONTRACT MÀ TASK 4 (CHROMADB) CẦN ĐÁP ỨNG cho module này chạy đúng:
    - collection.get(include=["documents", "metadatas"]) trả về:
        ids:       list[str]   — chunk id duy nhất (vd "diem_chuan_2025_chunk_3")
        documents: list[str]   — text nội dung của chunk (KHÔNG bao gồm khối
                                  frontmatter --- ... --- ở đầu file .md gốc)
        metadatas: list[dict]  — mỗi dict nên có ít nhất:
                                    "doc_id": id của document gốc (từ frontmatter)
                                    "title": tiêu đề document gốc
                                    "chunk_index": số thứ tự chunk trong document
                                  các field khác (source_url, type, document_version...)
                                  giữ nguyên từ frontmatter nếu có, không bắt buộc.
    Nếu Task 4 đổi tên collection hoặc đường dẫn chroma_db/, sửa lại
    COLLECTION_NAME / CHROMA_DIR bên dưới cho khớp.
-------------------------------------------------------------------------------
"""

import re
from pathlib import Path

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

# =============================================================================
# CONFIGURATION — phải khớp với Task 4
# =============================================================================

BM25_K1 = 1.5   # term saturation — giá trị mặc định phổ biến, không cần tune thêm
BM25_B = 0.75   # length normalization — giá trị mặc định phổ biến

# Cache module-level để không phải build lại BM25 index mỗi lần gọi lexical_search
_CORPUS_CACHE: list[dict] | None = None
_BM25_CACHE = None


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize text tiếng Việt cho BM25.

    Dùng underthesea.word_tokenize để tách đúng từ ghép (vd "thanh toán" là
    1 token thay vì 2 token "thanh" + "toán") — quan trọng vì BM25 tính
    TF/IDF theo token, tách sai từ ghép sẽ làm lệch điểm liên quan.

    Fallback về split() đơn giản nếu underthesea lỗi (vd chưa cài), để lexical
    search vẫn chạy được (chỉ kém chính xác hơn) thay vì crash toàn bộ.
    """
    text = text.lower().strip()
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        # Fallback: bỏ dấu câu cơ bản rồi split theo khoảng trắng
        text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
        tokens = text.split()
    return [t for t in tokens if t.strip()]


# =============================================================================
# LOAD CORPUS TỪ CHROMADB (do Task 4 tạo)
# =============================================================================

def load_corpus_from_chroma(
    chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME
) -> list[dict]:
    """
    Đọc lại toàn bộ chunk (documents + metadatas) đã được Task 4 index vào
    ChromaDB. Không lấy "embeddings" vì BM25 không cần — đỡ tải dữ liệu thừa.

    Returns:
        List of {'content': str, 'metadata': dict}
        metadata luôn có thêm field 'chunk_id' (lấy từ id trong ChromaDB) để
        Task 7 (rerank_rrf) dùng làm key dedup khi fuse với vector search.

    Raises:
        RuntimeError nếu chưa chạy Task 4 (chưa có chroma_db/ hoặc collection
        rỗng) — báo lỗi rõ ràng thay vì để BM25 build trên corpus rỗng.
    """
    import chromadb

    if not chroma_dir.exists():
        raise RuntimeError(
            f"Không tìm thấy {chroma_dir}. Cần chạy Task 4 (chunking + indexing "
            f"vào ChromaDB) trước khi build BM25 index."
        )

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)

    result = collection.get(include=["documents", "metadatas"])
    ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]

    if not ids:
        raise RuntimeError(
            f"Collection '{collection_name}' rỗng. Cần chạy Task 4 để index "
            f"chunk trước."
        )

    corpus = []
    for chunk_id, content, metadata in zip(ids, documents, metadatas):
        corpus.append({
            "content": content,
            "metadata": {**(metadata or {}), "chunk_id": chunk_id},
        })
    return corpus


# =============================================================================
# BUILD BM25 INDEX
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance đã fit trên corpus (đã tokenize).
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("corpus rỗng, không thể build BM25 index")

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    return bm25


def _get_index():
    """Lazy-load + cache corpus và BM25 index (module-level singleton)."""
    global _CORPUS_CACHE, _BM25_CACHE
    if _CORPUS_CACHE is None or _BM25_CACHE is None:
        _CORPUS_CACHE = load_corpus_from_chroma()
        _BM25_CACHE = build_bm25_index(_CORPUS_CACHE)
    return _CORPUS_CACHE, _BM25_CACHE


def refresh_index():
    """
    Bắt buộc build lại corpus + BM25 index từ ChromaDB (bỏ cache).
    Gọi hàm này sau khi Task 4 reindex lại chroma_db/ (đổi corpus, thêm/bớt
    tài liệu) để lexical_search dùng dữ liệu mới thay vì cache cũ.
    """
    global _CORPUS_CACHE, _BM25_CACHE
    _CORPUS_CACHE = load_corpus_from_chroma()
    _BM25_CACHE = build_bm25_index(_CORPUS_CACHE)
    return _CORPUS_CACHE, _BM25_CACHE


# =============================================================================
# SEARCH
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending. Chỉ trả về kết quả có score > 0
        (score = 0 nghĩa là không có token nào trong query khớp document).
    """
    import numpy as np

    corpus, bm25 = _get_index()

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    if not results:
        print("Không có kết quả (hoặc chưa có chroma_db/ — chạy Task 4 trước).")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
