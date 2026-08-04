"""
RAG Chatbot — Tra cứu Tuyển sinh Đại học Bách khoa Hà Nội.
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
    page_title="Tra cứu Tuyển sinh BKHN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bảng màu nhận diện Bách khoa Hà Nội: đỏ chủ đạo, vàng điểm nhấn, nền trắng.
st.markdown(
    """
    <style>
      :root {
        --hust-red: #d71920;
        --hust-red-dark: #a90f1b;
        --hust-yellow: #ffc400;
        --hust-white: #ffffff;
      }

      .stApp {
        background: var(--hust-white);
      }

      h1, h2, h3 {
        color: var(--hust-red) !important;
      }

      .stApp, .stApp p, .stApp li, .stApp label, .stApp span {
        color: var(--hust-red-dark);
      }

      [data-testid="stSidebar"] {
        background: var(--hust-red);
      }

      [data-testid="stSidebar"] * {
        color: var(--hust-yellow);
      }

      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {
        color: var(--hust-yellow) !important;
      }

      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: var(--hust-white) !important;
      }

      [data-testid="stSidebar"] .stButton > button {
        background: var(--hust-white);
        border: 1px solid var(--hust-yellow);
        border-radius: 8px;
        color: var(--hust-red-dark) !important;
        font-weight: 650;
      }

      [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--hust-yellow);
        border-color: var(--hust-white);
        color: var(--hust-red-dark) !important;
      }

      [data-testid="stSidebar"] [data-testid="stRadio"] label,
      [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        color: var(--hust-yellow) !important;
      }

      /* Nhãn nhỏ và giá trị trong phần Thiết lập cần tương phản với nền đỏ. */
      [data-testid="stSidebar"] [data-testid="stSlider"] *,
      [data-testid="stSidebar"] [data-testid="stRadio"] *,
      [data-testid="stSidebar"] [data-testid="stCheckbox"] * {
        color: var(--hust-white) !important;
      }

      [data-testid="stChatInput"] {
        border: 2px solid var(--hust-red);
        border-radius: 12px;
        background: var(--hust-white);
      }

      [data-testid="stChatInput"] textarea {
        color: var(--hust-red-dark);
      }

      [data-testid="stExpander"] {
        border: 2px solid var(--hust-yellow);
        border-radius: 8px;
        background: var(--hust-white);
      }

      [data-testid="stAlert"] {
        border-left: 5px solid var(--hust-yellow);
      }

      [data-testid="stChatMessage"] {
        border-left: 4px solid var(--hust-yellow);
        border-radius: 4px;
      }

      div[data-baseweb="slider"] div[role="slider"] {
        background: var(--hust-yellow);
        border-color: var(--hust-white);
      }

      div[data-baseweb="slider"] div[data-testid="stTickBar"] {
        color: var(--hust-white);
      }

      hr {
        border-color: var(--hust-yellow);
      }

      .hust-hero {
        display: flex;
        min-height: 118px;
        margin: 0 0 0.6rem 0;
        border: 2px solid var(--hust-yellow);
        border-radius: 0;
        overflow: hidden;
        background: var(--hust-white);
      }

      .hust-hero-mark {
        display: flex;
        align-items: flex-start;
        justify-content: center;
        min-width: 86px;
        padding-top: 19px;
        background: var(--hust-red);
        color: var(--hust-yellow) !important;
        font-size: 42px;
        line-height: 1;
      }

      .hust-hero-copy {
        display: flex;
        flex: 1;
        flex-direction: column;
        justify-content: center;
        padding: 16px 24px;
        border-bottom: 18px solid var(--hust-yellow);
      }

      .hust-hero-kicker {
        color: var(--hust-red) !important;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.13em;
      }

      .hust-hero-title {
        margin-top: 0.25rem;
        color: var(--hust-red) !important;
        font-size: clamp(1.35rem, 2.5vw, 2.15rem);
        font-weight: 900;
        line-height: 1.18;
      }

      .hust-hero-subtitle {
        margin-top: 0.35rem;
        color: var(--hust-red-dark) !important;
        font-size: 0.95rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 Tra cứu Tuyển sinh BKHN")
    st.caption("Trợ lý tra cứu điểm chuẩn, phương thức xét tuyển và đề án tuyển sinh Đại học Bách khoa Hà Nội.")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Điểm chuẩn ngành CNTT: Khoa học Máy tính năm 2025 là bao nhiêu?",
        "BKHN năm 2026 có những phương thức tuyển sinh nào?",
        "Xét tuyển tài năng gồm những diện nào?",
        "Điều kiện xét tuyển bằng chứng chỉ quốc tế là gì?",
        "Hạn đăng ký xét tuyển tài năng năm 2026 là khi nào?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    response_mode = st.radio(
        "Nguồn phản hồi",
        options=("Mock demo (CP5)", "Pipeline thật (Task 10)"),
        help="Mock demo giúp TV4 hoàn thiện và trình diễn UI trước khi các module retrieval bàn giao.",
    )
    show_retrieval_debug = st.checkbox(
        "Hiển thị kết quả retrieval",
        value=False,
        help="Dùng ở CP2 để đối chiếu Semantic Search và BM25 trước khi ghép pipeline.",
    )
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → Vectorless Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# Dữ liệu demo CP5 dùng cùng schema với Task 9/10 để có thể thay bằng pipeline
# thật mà không cần sửa UI. Nội dung chỉ dùng để trình diễn giao diện.
MOCK_SOURCES = {
    "score": {
        "content": "Theo điểm chuẩn BKHN năm 2025 bằng điểm thi THPT, ngành IT1 — CNTT: Khoa học Máy tính có điểm chuẩn 29,19 cho các tổ hợp A00, A01; môn chính là Toán.",
        "score": 0.95,
        "metadata": {"source": "diem_chuan_2025.md", "type": "cutoff-score"},
        "source": "hybrid",
    },
    "methods": {
        "content": "Thông tin tuyển sinh đại học chính quy BKHN năm 2026 dự kiến 9.880 chỉ tiêu và duy trì ba phương thức: Xét tuyển tài năng, xét tuyển theo điểm Đánh giá tư duy (TSA), và xét tuyển theo điểm thi tốt nghiệp THPT.",
        "score": 0.93,
        "metadata": {"source": "thong-tin-tuyen-sinh-2026.md", "type": "admissions-plan"},
        "source": "hybrid",
    },
    "talent": {
        "content": "Xét tuyển tài năng (XTTN) năm 2026 gồm: diện 1.1 xét tuyển thẳng theo quy định của Bộ GD&ĐT; diện 1.2 dựa trên chứng chỉ quốc tế; và diện 1.3 dựa theo hồ sơ năng lực kết hợp phỏng vấn.",
        "score": 0.91,
        "metadata": {"source": "quy-dinh-xttn-2026.md", "type": "admissions-regulation"},
        "source": "hybrid",
    },
    "international": {
        "content": "Diện XTTN 1.2 dành cho thí sinh có điểm trung bình các môn văn hóa lớp 10, 11, 12 từ 8,00 trở lên và có ít nhất một chứng chỉ quốc tế còn hiệu lực như SAT, ACT, A-Level, AP hoặc IB.",
        "score": 0.88,
        "metadata": {"source": "quy-dinh-xttn-2026.md", "type": "admissions-regulation"},
        "source": "hybrid",
    },
    "timeline": {
        "content": "Theo hướng dẫn XTTN 2026, hệ thống đăng ký trực tuyến mở từ 18/5 đến hết 31/5/2026 cho diện 1.2 và 1.3; diện 1.1 mở đến hết 20/6/2026 theo quy định của Bộ GD&ĐT.",
        "score": 0.76,
        "metadata": {"section": "Mốc thời gian XTTN 2026", "type": "admissions-guide"},
        "source": "pageindex",
    },
}


def build_mock_response(query: str, top_k: int) -> dict:
    """Trả response demo có citation theo đúng interface của Task 10."""
    normalized = query.lower()
    if any(word in normalized for word in ("điểm chuẩn", "it1", "khoa học máy tính")):
        selected = [MOCK_SOURCES["score"], MOCK_SOURCES["methods"]]
        answer = (
            "Điểm chuẩn năm 2025 theo phương thức điểm thi THPT của ngành IT1 — CNTT: "
            "Khoa học Máy tính là **29,19** cho các tổ hợp A00, A01; môn chính là Toán. "
            "[diem_chuan_2025.md]"
        )
    elif any(word in normalized for word in ("phương thức", "tsa", "đánh giá tư duy", "thpt")):
        selected = [MOCK_SOURCES["methods"], MOCK_SOURCES["talent"]]
        answer = (
            "BKHN năm 2026 duy trì ba phương thức: Xét tuyển tài năng, xét tuyển theo "
            "điểm Đánh giá tư duy (TSA), và xét tuyển theo điểm thi tốt nghiệp THPT. "
            "[thong-tin-tuyen-sinh-2026.md]"
        )
    elif any(word in normalized for word in ("tài năng", "xttn", "phỏng vấn")):
        selected = [MOCK_SOURCES["talent"], MOCK_SOURCES["international"]]
        answer = (
            "Xét tuyển tài năng gồm ba diện: xét tuyển thẳng (1.1), xét tuyển dựa trên "
            "chứng chỉ quốc tế (1.2), và hồ sơ năng lực kết hợp phỏng vấn (1.3). "
            "[quy-dinh-xttn-2026.md]"
        )
    elif any(word in normalized for word in ("chứng chỉ", "sat", "act", "a-level", "ib")):
        selected = [MOCK_SOURCES["international"], MOCK_SOURCES["talent"]]
        answer = (
            "Để xét tuyển tài năng diện 1.2, thí sinh cần có điểm trung bình các môn văn "
            "hóa lớp 10–12 từ 8,00 trở lên và ít nhất một chứng chỉ quốc tế hợp lệ. "
            "[quy-dinh-xttn-2026.md]"
        )
    else:
        selected = [MOCK_SOURCES["timeline"]]
        answer = (
            "Câu hỏi này đang được mô phỏng bằng vectorless fallback. Với XTTN 2026, diện "
            "1.2 và 1.3 đăng ký từ 18/5 đến hết 31/5; diện 1.1 đến hết 20/6. "
            "[Mốc thời gian XTTN 2026]"
        )

    sources = [dict(source, metadata=dict(source["metadata"])) for source in selected[:top_k]]
    mode = get_retrieval_mode(sources)
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": mode,
        "retrieval_mode": mode,
    }


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

st.markdown(
    """
    <section class="hust-hero">
      <div class="hust-hero-mark">★</div>
      <div class="hust-hero-copy">
        <div class="hust-hero-kicker">ĐẠI HỌC BÁCH KHOA HÀ NỘI</div>
        <div class="hust-hero-title">Trợ Lý Tra Cứu Điểm Chuẩn &amp; Đề Án Tuyển Sinh</div>
        <div class="hust-hero-subtitle">Tra cứu điểm chuẩn, phương thức xét tuyển, mốc thời gian và quy định tuyển sinh.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

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

user_input = st.chat_input("Nhập câu hỏi về điểm chuẩn hoặc tuyển sinh BKHN...")
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
                if response_mode == "Mock demo (CP5)":
                    mock_results = build_mock_response(query, top_k)["sources"]
                    with dense_col:
                        render_retrieval_results("Semantic Search (mock)", mock_results)
                    with lexical_col:
                        render_retrieval_results("BM25 Lexical Search (mock)", mock_results[::-1])
                else:
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
                if response_mode == "Mock demo (CP5)":
                    response = build_mock_response(query, top_k=top_k)
                else:
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
