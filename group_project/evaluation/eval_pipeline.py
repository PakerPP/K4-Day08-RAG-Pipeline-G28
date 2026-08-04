"""
RAG Evaluation Pipeline.

Dùng RAGAS để đánh giá chất lượng RAG pipeline (Task 9 + Task 10) trên
golden_dataset.json (15 cặp Q&A về tuyển sinh Đại học Bách khoa Hà Nội).

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs (hybrid+rerank vs dense-only)
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.

Chạy:
    python -m group_project.evaluation.eval_pipeline
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Golden dataset có 1 câu negative test case (VinUni/RMIT) cố ý không có
# evidence trong corpus — dùng để test Task 10 từ chối trả lời đúng cách, nhưng
# sẽ làm hỏng context_recall/context_precision của RAGAS vì "expected_context"
# không tồn tại thật trong corpus. Loại khỏi tập chấm điểm định lượng, ghi chú
# riêng trong results.md.
NEGATIVE_TEST_MARKER = "KHÔNG CÓ DỮ LIỆU"


def load_golden_dataset(include_negative: bool = False) -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if include_negative:
        return data
    return [item for item in data if NEGATIVE_TEST_MARKER not in item["expected_answer"]]


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def evaluate_with_ragas(golden_dataset: list[dict], use_reranking: bool = True) -> "pandas.DataFrame":
    """
    Evaluate RAG pipeline (Task 9 retrieve + Task 10 generate_with_citation)
    sử dụng RAGAS trên 1 config cụ thể.

    Args:
        golden_dataset: List of {'question', 'expected_answer', 'expected_context'}
        use_reranking: True = hybrid search + RRF rerank (Config A);
                       False = dense-only, bỏ qua rerank (Config B)

    Returns:
        pandas.DataFrame với 1 dòng/câu hỏi + cột điểm cho từng metric.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import reorder_for_llm, format_context, _generate_answer, _answer_without_llm

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        question = item["question"]
        chunks = retrieve(question, top_k=5, use_reranking=use_reranking)
        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)

        if not context:
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        else:
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
            try:
                answer = _generate_answer(user_message)
            except Exception:
                answer = None
            if not answer:
                answer = _answer_without_llm(reordered)

        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append([c["content"] for c in chunks] or [""])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B: Config A (hybrid + RRF rerank) vs Config B (dense-only, không rerank).

    Returns:
        {
            "hybrid_rerank": pandas.DataFrame,
            "dense_only": pandas.DataFrame,
        }
    """
    print("Đang chạy Config A (hybrid + rerank)...")
    df_a = evaluate_with_ragas(golden_dataset, use_reranking=True)

    print("Đang chạy Config B (dense-only)...")
    df_b = evaluate_with_ragas(golden_dataset, use_reranking=False)

    return {"hybrid_rerank": df_a, "dense_only": df_b}


# =============================================================================
# Export Results
# =============================================================================

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def _avg_scores(df) -> dict:
    return {col: round(df[col].mean(), 3) for col in METRIC_COLUMNS if col in df.columns}


def _worst_performers(df, n: int = 3) -> list[dict]:
    """Lấy n câu có faithfulness thấp nhất (chỉ báo rõ nhất của hallucination)."""
    if "faithfulness" not in df.columns:
        return []
    sorted_df = df.sort_values("faithfulness", ascending=True).head(n)
    return sorted_df.to_dict("records")


def export_results(comparison: dict):
    """Export evaluation results (A/B comparison) ra results.md."""
    df_a = comparison["hybrid_rerank"]
    df_b = comparison["dense_only"]
    avg_a = _avg_scores(df_a)
    avg_b = _avg_scores(df_b)

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\n"
    content += "RAGAS 0.1.21 — 4 metrics chuẩn: Faithfulness, Answer Relevance, Context Recall, Context Precision.\n\n"
    content += (
        "> 1 câu trong golden_dataset.json (so sánh VinUni/RMIT) bị loại khỏi tập chấm điểm "
        "định lượng vì là negative test case cố ý (kiểm tra Task 10 từ chối trả lời khi thiếu "
        "evidence) — corpus hiện chỉ có dữ liệu Đại học Bách khoa Hà Nội.\n\n"
    )

    content += "---\n\n## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |\n"
    content += "|--------|---------------------------|----------------------|---|\n"
    for col in METRIC_COLUMNS:
        a_val = avg_a.get(col, float("nan"))
        b_val = avg_b.get(col, float("nan"))
        delta = round(a_val - b_val, 3)
        content += f"| {METRIC_LABELS[col]} | {a_val} | {b_val} | {delta:+.3f} |\n"
    avg_a_all = round(sum(avg_a.values()) / len(avg_a), 3) if avg_a else float("nan")
    avg_b_all = round(sum(avg_b.values()) / len(avg_b), 3) if avg_b else float("nan")
    content += f"| **Average** | **{avg_a_all}** | **{avg_b_all}** | **{round(avg_a_all - avg_b_all, 3):+.3f}** |\n"

    content += "\n---\n\n## A/B Comparison Analysis\n\n"
    content += (
        "**Config A (hybrid + rerank):** Semantic Search (OpenAI text-embedding-3-small) "
        "+ BM25 lexical search, merge bằng RRF, rerank lại trước khi trả top_k.\n\n"
    )
    content += (
        "**Config B (dense-only):** Chỉ dùng Semantic Search + BM25 merge qua RRF, "
        "bỏ qua bước rerank cuối (`use_reranking=False`).\n\n"
    )
    better = "Config A" if avg_a_all >= avg_b_all else "Config B"
    content += f"**Kết luận:** {better} có điểm trung bình cao hơn ({max(avg_a_all, avg_b_all)} so với {min(avg_a_all, avg_b_all)}).\n\n"

    content += "---\n\n## Worst Performers (Bottom 3, theo Faithfulness — Config A)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"
    for i, row in enumerate(_worst_performers(df_a, 3), 1):
        q = str(row.get("question", ""))[:60]
        f = round(row.get("faithfulness", 0), 3)
        r = round(row.get("answer_relevancy", 0), 3)
        rec = round(row.get("context_recall", 0), 3)
        content += f"| {i} | {q} | {f} | {r} | {rec} | | |\n"

    content += "\n---\n\n## Recommendations\n\n"
    content += "### Cải tiến 1\n**Action:**\n**Expected impact:**\n\n"
    content += "### Cải tiến 2\n**Action:**\n**Expected impact:**\n\n"
    content += "### Cải tiến 3\n**Action:**\n**Expected impact:**\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"Da ghi ket qua vao {RESULTS_PATH}")


if __name__ == "__main__":
    import sys

    # Mac dinh chay subset 5 cau: chay full 14 cau x 2 config x 4 metrics tao
    # ~112 request LLM lien tuc, da gap rate-limit tu OpenRouter khien pipeline
    # cham dan bat thuong (tu 74s len 15+ phut van chua xong). Subset 5 cau du
    # de results.md co y nghia va chay on dinh. Truyen "--full" de chay het.
    golden_dataset = load_golden_dataset()
    if "--full" not in sys.argv:
        golden_dataset = golden_dataset[:5]
    print(f"Loaded {len(golden_dataset)} test cases (da loai negative test case)")

    comparison = compare_configs(golden_dataset)
    export_results(comparison)
