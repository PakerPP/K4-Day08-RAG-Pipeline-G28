# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 — 4 metrics chuẩn: Faithfulness, Answer Relevance, Context Recall, Context Precision.

> 1 câu trong golden_dataset.json (so sánh VinUni/RMIT) bị loại khỏi tập chấm điểm định lượng vì là negative test case cố ý (kiểm tra Task 10 từ chối trả lời khi thiếu evidence) — corpus hiện chỉ có dữ liệu Đại học Bách khoa Hà Nội.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.7 | 0.7 | +0.000 |
| Answer Relevance | 0.696 | 0.697 | -0.001 |
| Context Recall | 1.0 | 1.0 | +0.000 |
| Context Precision | 1.0 | 1.0 | +0.000 |
| **Average** | **0.849** | **0.849** | **+0.000** |

---

## A/B Comparison Analysis

**Config A (hybrid + rerank):** Semantic Search (OpenAI text-embedding-3-small) + BM25 lexical search, merge bằng RRF, rerank lại trước khi trả top_k.

**Config B (dense-only):** Chỉ dùng Semantic Search + BM25 merge qua RRF, bỏ qua bước rerank cuối (`use_reranking=False`).

**Kết luận:** Trên subset 5 câu, Config A và Config B cho điểm trung bình gần như bằng nhau (0.849 vs 0.849, chênh lệch không đáng kể ở cả 4 metric). Bước rerank RRF không tạo khác biệt rõ rệt ở quy mô này — hợp lý vì với top_k=5 và chỉ 2 ranker (dense + BM25), RRF chủ yếu sắp xếp lại thứ hạng trong cùng một tập ứng viên nhỏ, ít thay đổi nội dung context đưa vào LLM. Cần chạy full 14 câu (`python -m group_project.evaluation.eval_pipeline --full`) để kết luận chắc chắn hơn, vì corpus đa dạng hơn có thể bộc lộ khác biệt rõ hơn giữa 2 config.

---

## Worst Performers (Bottom 3, theo Faithfulness — Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Lệ phí đăng ký Xét tuyển tài năng (XTTN) năm 2026 cho từng diện là bao nhiêu? | 0.0 | 0.0 | 1.0 | Retrieval (chunking) | Số liệu lệ phí (200k/300k/500k/100k) nằm ở `article_04` chunk_index=4, nhưng top-5 retrieval chỉ lấy được chunk_index=3 (đoạn mở đầu "CÁC LƯU Ý TRƯỚC KHI ĐĂNG KÝ" liền trước, không chứa số liệu). Model đúng theo system prompt (không bịa) nên trả lời "không thể xác minh" — faithfulness=0 vì answer không chứa statement nào để đối chiếu, không phải hallucination. |
| 2 | Điều kiện chứng chỉ IELTS để được miễn/quy đổi điều kiện tiếng Anh khi xét tuyển vào chương trình giảng dạy bằng tiếng Anh của BKHN 2026 là gì? | 0.5 | 0.88 | 1.0 | Generation | 3/5 chunks retrieval được có chứa "IELTS" (đủ evidence, recall=1.0), nhưng câu hỏi có 3 điều kiện thay thế nhau (VSTEP B1 / IELTS 5.0 / điểm THPT ≥6.5) — model có thể chỉ nêu 1-2 trong 3 điều kiện, khiến một phần câu trả lời không được context hỗ trợ đầy đủ. |
| 3 | Đại học Bách Khoa Hà Nội tuyển sinh năm 2026 dự kiến bao nhiêu chỉ tiêu và có mấy phương thức xét tuyển chính? | 1.0 | 0.864 | 1.0 | — | Không có lỗi — điểm answer_relevancy dưới 1.0 chỉ do RAGAS đo độ tương đồng ngữ nghĩa giữa câu hỏi sinh lại từ answer và câu hỏi gốc, không phản ánh sai sót thực tế. |

---

## Recommendations

### Cải tiến 1 — Tăng chunk_overlap hoặc dùng MarkdownHeaderTextSplitter cho danh sách liệt kê
**Action:** Worst performer #1 cho thấy `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) cắt đứt một danh sách 4 mục liền mạch (lệ phí theo diện) giữa 2 chunk liên tiếp. Tăng `CHUNK_OVERLAP` lên 150-200, hoặc chuyển sang `MarkdownHeaderTextSplitter` để giữ nguyên các block liệt kê dưới cùng 1 heading.
**Expected impact:** Giảm số câu hỏi có faithfulness=0 do retrieval thiếu evidence (không phải do model bịa), đặc biệt với các câu hỏi tra cứu số liệu cụ thể (lệ phí, mốc thời gian, điểm số).

### Cải tiến 2 — Tăng top_k cho các câu hỏi liệt kê nhiều điều kiện
**Action:** Worst performer #2 cho thấy câu hỏi có nhiều điều kiện thay thế nhau (3 loại chứng chỉ) dễ bị model chỉ nêu một phần dù đủ evidence. Cân nhắc tăng top_k mặc định từ 5 lên 7-8 khi câu hỏi chứa các từ khóa liệt kê ("các", "những", "gồm những gì"), hoặc thêm hướng dẫn trong system prompt yêu cầu liệt kê đầy đủ mọi điều kiện tìm thấy trong context.
**Expected impact:** Tăng answer completeness cho câu hỏi dạng liệt kê, cải thiện faithfulness khi context đã đủ nhưng answer chưa khai thác hết.

### Cải tiến 3 — Chạy full 14 câu để so sánh Config A/B đáng tin cậy hơn
**Action:** Subset 5 câu hiện tại chưa đủ để phân biệt rõ hiệu quả của reranking (RRF) — điểm trung bình 2 config gần như bằng nhau. Chạy `python -m group_project.evaluation.eval_pipeline --full` với toàn bộ 14 câu (rải đều theo thời gian để tránh rate-limit OpenRouter, ví dụ chia 2 lần chạy 7 câu) để có kết luận A/B đáng tin cậy hơn cho báo cáo cuối.
**Expected impact:** Kết luận A/B chính xác hơn, tránh đưa ra nhận định "rerank không có tác dụng" chỉ dựa trên mẫu quá nhỏ.
