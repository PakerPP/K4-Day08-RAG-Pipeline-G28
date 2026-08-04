"""
RAG Chatbot — E-commerce Support (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption("Trợ lý hỏi đáp về chính sách thương mại điện tử và hỗ trợ khách hàng (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    show_retrieval_debug = st.checkbox(
        "Hiển thị kết quả retrieval",
        value=False,
        help="Dùng ở CP2 để đối chiếu Semantic Search và BM25 trước khi ghép pipeline.",
    )

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def render_retrieval_results(title: str, results: list[dict]) -> None:
    """Hiển thị thống nhất kết quả từ Semantic Search hoặc BM25.

    Hàm chỉ phụ thuộc vào interface chung ``content``, ``score`` và ``metadata``
    để TV2/TV3 có thể bàn giao module retrieval độc lập với UI.
    """
    st.markdown(f"#### {title}")
    if not results:
        st.caption("Không có kết quả phù hợp.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        source = metadata.get("source", "Unknown source")
        score = float(result.get("score", 0.0))
        st.markdown(f"**[{index}] {source}** — score: `{score:.4f}`")
        st.caption(result.get("content", "")[:350])


def get_retrieval_mode(sources: list[dict], declared_mode: str | None = None) -> str:
    """Xác định retrieval path để UI hiển thị đúng trạng thái CP3.

    Task 9 đánh dấu kết quả hybrid/pageindex ở field ``source`` cấp kết quả.
    ``declared_mode`` được ưu tiên để tương thích với response của Task 10 sau này.
    """
    if declared_mode in {"hybrid", "pageindex"}:
        return declared_mode
    if any(source.get("source") == "pageindex" for source in sources):
        return "pageindex"
    return "hybrid" if sources else "unknown"


def render_retrieval_mode(mode: str) -> None:
    """Thông báo retrieval path cho người dùng, nhất là khi fallback được kích hoạt."""
    if mode == "pageindex":
        st.warning(
            "⚠️ **PageIndex fallback đang được dùng.** "
            "Điểm semantic chưa đủ tin cậy nên hệ thống chuyển sang truy vấn theo cấu trúc tài liệu."
        )
    elif mode == "hybrid":
        st.caption("✅ Retrieval: Hybrid (Semantic + BM25 + RRF)")


def render_sources(sources: list[dict]) -> None:
    """Hiển thị các tài liệu đã dùng, dùng được cho cả hybrid và PageIndex."""
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            source_name = meta.get("source") or meta.get("section") or "Unknown"
            doc_type = meta.get("type", "unknown")
            retrieval_source = src.get("source", "hybrid")
            score = float(src.get("score", 0))
            st.markdown(
                f"**[{i}] {source_name}** `{doc_type}` · `{retrieval_source}` "
                f"| score: `{score:.4f}`"
            )
            st.text(src.get("content", "")[:300] + "...")
            st.divider()

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hệ thống hỏi đáp chính sách e-commerce và trợ giúp khách hàng")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            render_retrieval_mode(msg.get("retrieval_mode", "hybrid"))
            render_sources(msg["sources"])

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        # CP2: cho phép đối chiếu hai ranker độc lập trước khi Task 9 ghép
        # Hybrid Retrieval. Lỗi từ module chưa hoàn thiện được giữ trong expander
        # để luồng chat không bị dừng.
        if show_retrieval_debug:
            with st.expander("🔎 Kết quả retrieval (CP2)", expanded=True):
                dense_col, lexical_col = st.columns(2)
                try:
                    from src.task5_semantic_search import semantic_search

                    dense_results = semantic_search(query, top_k=top_k)
                    with dense_col:
                        render_retrieval_results("Semantic Search", dense_results)
                except Exception as exc:
                    with dense_col:
                        st.warning(f"Semantic Search chưa sẵn sàng: {exc}")

                try:
                    from src.task6_lexical_search import lexical_search

                    lexical_results = lexical_search(query, top_k=top_k)
                    with lexical_col:
                        render_retrieval_results("BM25 Lexical Search", lexical_results)
                except Exception as exc:
                    with lexical_col:
                        st.warning(f"BM25 chưa sẵn sàng: {exc}")

        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                # TODO (Học viên): Tích hợp hàm sinh câu trả lời từ Task 10
                # Ví dụ:
                # from src.task10_generation import generate_with_citation
                # response = generate_with_citation(query, top_k=top_k)
                # answer = response["answer"]
                # sources = response.get("sources", [])

                from src.task10_generation import generate_with_citation
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_mode = get_retrieval_mode(
                    sources, response.get("retrieval_mode")
                )

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
                retrieval_mode = "unknown"
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []
                retrieval_mode = "unknown"

            st.markdown(answer)

            if sources:
                render_retrieval_mode(retrieval_mode)
                render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_mode": retrieval_mode,
    })
