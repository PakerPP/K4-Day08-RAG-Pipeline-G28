"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from fpdf import FPDF
from pageindex.client import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

UPLOADED_DOC_IDS = []


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    global UPLOADED_DOC_IDS
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Thư mục không tồn tại: {STANDARDIZED_DIR}")
        return

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        pdf_path = md_file.with_suffix('.pdf')
        
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        
        with open(md_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 7, txt=text)
                
        pdf.output(str(pdf_path))

        # Upload tài liệu lên PageIndex
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                UPLOADED_DOC_IDS.append(doc_id)
                print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  ✗ Lỗi khi upload {md_file.name}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    global UPLOADED_DOC_IDS
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results = []

    for doc_id in UPLOADED_DOC_IDS:
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            
            if not retrieval_id:
                continue
                
            # Poll cho đến khi status == "completed"
            retrieval = {}
            while True:
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "").lower()
                if status == "completed" or status in ["failed", "error"]:
                    break
                time.sleep(2)
                
            if retrieval.get("status", "").lower() != "completed":
                continue

            # Parse retrieval["retrieved_nodes"] — mỗi node có "relevant_contents"
            rank = 1
            for node in retrieval.get("retrieved_nodes", [])[:2]:
                for group in node.get("relevant_contents", []):
                    for item in group:
                        content = item.get("relevant_content", "")
                        if content:
                            results.append({
                                "content": content,
                                "score": max(1.0 - (rank * 0.05), 0.1),  # Tự gán theo rank
                                "metadata": {"section": item.get("section_title")},
                                "source": "pageindex",
                            })
                            rank += 1
        except Exception as e:
            print(f"Lỗi khi truy vấn doc_id {doc_id}: {e}")

    # Sắp xếp và trả về top_k
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")