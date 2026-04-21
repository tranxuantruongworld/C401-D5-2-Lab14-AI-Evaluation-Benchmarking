# Cá nhân Reflection - Lab Day 14: AI Evaluation & Benchmarking

**Họ và tên:** Hồ Sỹ Minh Hà
**Mã số sinh viên:** 2A202600060
**Nhóm:** D5-2

---

## 1. Engineering Contribution

Trong dự án này, tôi đã đóng góp trực tiếp vào các module lõi của hệ thống đánh giá tự động (Evaluation Pipeline), tập trung vào tính hiệu quả, độ tin cậy và khả năng mở rộng:

- **Async & Resilience (Module `engine/utils.py`):**
    - Thiết kế và triển khai decorator `retry_with_exponential_backoff`. Đây là yếu tố sống còn khi làm việc với các LLM API (như Gemini/OpenAI) vốn thường xuyên gặp lỗi `429 Resource Exhausted`. 
    - Cơ chế này sử dụng hàm mũ để tăng dần thời gian chờ kết hợp với `jitter` (nhiễu ngẫu nhiên) để tránh tình trạng "thundering herd", giúp pipeline chạy ổn định ngay cả khi bị rate limit.

- **Multi-Judge Consensus (Module `engine/llm_judge.py`):**
    - Triển khai `MultiModelJudge` cho phép chạy song song nhiều model Judge khác nhau (ví dụ: `gemini-3.1-flash-lite` đóng vai trò Senior và `gemma-27b` đóng vai trò Junior).
    - Xây dựng logic tính toán **Agreement Rate** giữa các Judge để định lượng mức độ tin cậy của kết quả đánh giá.

- **Metrics & Retrieval Evaluation (Module `engine/retrieval_eval.py`):**
    - Chịu trách nhiệm chính trong việc hiện thực hóa các metrics đo lường chất lượng Retrieval: **Hit Rate** và **MRR (Mean Reciprocal Rank)**.
    - Viết logic `extract_ids` sử dụng Regex để bóc tách ID các đoạn văn bản (chunks) từ câu trả lời của Agent, cho phép mapping chính xác với Ground Truth.

- **Agent V2 & Ingestion (Module `agent/main_agent_v2.py` & `data/beir_ingestion.py`):**
    - Nâng cấp Agent lên phiên bản V2 sử dụng SDK mới nhất, tối ưu prompt để Agent cung cấp trích dẫn nguồn (citations) rõ ràng hơn.

---

## 2. Technical Depth

Trong quá trình thực hiện, tôi đã nghiên cứu và áp dụng các khái niệm quan trọng sau:

- **MRR (Mean Reciprocal Rank):** Khác với Hit Rate (chỉ quan tâm có tìm thấy hay không), MRR đánh giá xem tài liệu đúng nằm ở vị trí thứ mấy. Công thức $1/rank$ trừng phạt nặng nề nếu tài liệu đúng nằm ở cuối danh sách. Điều này cực kỳ quan trọng trong RAG vì context window có hạn và các model bị ảnh hưởng bởi hiện tượng "Lost in the Middle".
- **Position Bias:** Tôi nhận thấy các LLM thường có xu hướng ưu tiên các thông tin xuất hiện ở đầu hoặc cuối context. Để giảm thiểu điều này, tôi đã triển khai Multi-Judge để lấy trung bình cộng điểm số, giúp giảm thiểu sai số hệ thống của một model đơn lẻ.
- **Cohen's Kappa & Agreement:** Mặc dù dự án sử dụng công thức khoảng cách điểm đơn giản cho Agreement Rate, tôi hiểu rằng Cohen's Kappa là tiêu chuẩn để loại bỏ xác suất đồng thuận ngẫu nhiên. Việc theo dõi chỉ số này giúp chúng tôi biết khi nào cần điều chỉnh prompt của Judge hoặc thay đổi model Judge mạnh hơn.
- **Trade-off Chi phí và Chất lượng:** Việc sử dụng các model Flash/Lite cho các task đơn giản (như extract ID) và model Pro cho task phức tạp (như Judge) giúp tối ưu hóa Cost/Performance của toàn bộ hệ thống.

---

## 3. Problem Solving

Một số vấn đề kỹ thuật phức tạp tôi đã giải quyết:

1. **Vấn đề Rate Limit:** Khi chạy 50 cases song song, API liên tục trả về lỗi 429. Thay vì giảm số lượng concurrency (làm chậm hệ thống), tôi đã tối ưu logic retry trong `utils.py` để "thích nghi" với giới hạn của API, giúp hệ thống vẫn hoàn thành 50 cases dưới 2 phút.
2. **Trích xuất Ground Truth ID:** Ban đầu, Agent trả về câu trả lời tự do khiến việc so sánh với IDs trong Golden Set gặp khó khăn. Tôi đã giải quyết bằng cách:
    - Một mặt, yêu cầu Agent format output theo chuẩn nhất định.
    - Mặt khác, viết Regex linh hoạt trong `RetrievalEvaluator` để bắt được các dạng "Trích dẫn 1", "Nguồn [2]", v.v., đảm bảo tính khách quan cho kết quả đo lường.
3. **Xử lý xung đột giữa các Judge:** Khi hai model Judge cho điểm quá lệch nhau (Agreement Rate thấp), hệ thống sẽ đánh dấu các case này để con người (Expert) có thể vào review thủ công, thay vì tin tưởng mù quáng vào AI Judge.

---
*Tôi cam đoan các thông tin trên là đúng sự thật và phản ánh đúng đóng góp của tôi trong dự án.*
