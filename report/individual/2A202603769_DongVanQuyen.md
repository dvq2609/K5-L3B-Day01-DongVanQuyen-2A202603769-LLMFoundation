# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603769
- **Họ và tên:** Đồng Văn Quyến
- **SEED** (in ở ô đầu notebook): 5035

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.985   0.761      0.859        0.824   
2. BoW + MLP (deep learning)         0.974   0.864      0.916        0.919   
3a. Encoder (MiniLM) + kNN           0.868   0.898      0.883        0.947   
3b. Encoder (MiniLM) + LogReg        0.910   0.920      0.915        0.919   
3c. Encoder (MiniLM) + MLP head      0.910   0.920      0.915        0.919   
4a. LLM nhỏ zero-shot                1.000   0.950      0.974        0.974   
4b. LLM lớn zero-shot                1.000   0.900      0.947        0.947   
4c. LLM nhỏ few-shot                 1.000   0.850      0.919        0.919   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           10.116       7950.289       0.058   
2. BoW + MLP (deep learning)        11.292      25156.727       0.292   
3a. Encoder (MiniLM) + kNN         108.093         23.534      46.793   
3b. Encoder (MiniLM) + LogReg      108.513         26.742      29.264   
3c. Encoder (MiniLM) + MLP head    109.092         26.308      45.994   
4a. LLM nhỏ zero-shot                0.000          1.559     372.851   
4b. LLM lớn zero-shot                0.000          2.592     315.572   
4c. LLM nhỏ few-shot                 0.000          1.579     436.837   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.178           0.0  
2. BoW + MLP (deep learning)          0.502           0.0  
3a. Encoder (MiniLM) + kNN           63.026           0.0  
3b. Encoder (MiniLM) + LogReg        43.082           0.0  
3c. Encoder (MiniLM) + MLP head      64.654           0.0  
4a. LLM nhỏ zero-shot              1463.779           0.0  
4b. LLM lớn zero-shot               505.703           0.0  
4c. LLM nhỏ few-shot               1635.350           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Model tụt giảm nhiều nhất: Nhóm BoW + MLP (và theo sát là BoW + LogReg) là những mô hình bị tụt giảm hiệu năng nặng nề nhất khi chuyển từ tập test thông thường sang 50 case khó (điểm F1 lao dốc mạnh xuống mức 0.333 – 0.350).

Nhóm case tụt nhiều nhất: Hai nhóm khiến các mô hình Bag-of-Words (BoW) thất bại thảm hại nhất là:

Spam giọng tự nhiên: Mô hình BoW + MLP chỉ đạt độ chính xác 0% (đoán đúng 0/10 case).

Thao túng / lừa đảo: Mô hình BoW + LogReg chỉ đạt 12% và BoW + MLP đạt 25%.


Case 1 (Thuộc nhóm Spam giọng tự nhiên):
Nội dung tin nhắn: "Mate you have to see this, I doubled my savings in 2 weeks with this app, use my code and we both get a bonus" (Thật = spam).

Cơ chế khiến model BoW sai:

Mô hình BoW (Bag-of-Words) nhận diện văn bản thông qua tập từ điển chứa các từ khóa đặc trưng của tin rác rõ ràng (như free, win, prize, cash, 150p).

Trong tin nhắn này, kẻ gian cố tình sử dụng giọng điệu trò chuyện tự nhiên của bạn bè ("Mate you have to see this"), hoàn toàn vắng bóng các từ khóa spam kinh điển. Do đó, vector đầu vào của BoW không kích hoạt các trọng số đẩy về phía spam, khiến mô hình nhầm lẫn đây là một tin nhắn bình thường (ham).

Case 2 (Thuộc nhóm Né từ khóa):
Nội dung tin nhắn: "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m" (Thật = spam).

Cơ chế khiến model BoW sai:

Kẻ lừa đảo sử dụng kỹ thuật né từ khóa (obfuscation) bằng cách thay thế các chữ cái bằng chữ số và ký tự tương đương hình ảnh (như dùng số 0 thay cho chữ o, số 4 thay cho a).

Khi hàm tokenize chạy, các từ biến dạng này không khớp với các từ có sẵn trong từ điển vocab (được xây dựng từ tập train thông thường). Kết quả là các token này bị bộ lọc bỏ qua hoàn toàn, tạo ra một vector toàn số 0, làm mô hình hoàn toàn bất lực và dự đoán sai lệch.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #2: ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.
Cách làm: thử max_size với 100 và 5000
Kết quả: Vocab = 100 -> F1 Test: 0.808  Vocab = 5000 -> F1 Test: 0.859
Giải thích: Khi giảm Vocab xuống 100 từ (max_size = 100): F1 test sẽ giảm sút rõ rệt. Vì Từ điển quá nhỏ khiến mô hình bỏ sót rất nhiều từ khóa quan trọng mang tính đặc trưng của tin nhắn spam (ví dụ: các từ hiếm nhưng có giá trị phân loại cao bị loại bỏ). Mô hình bị "thiếu thông tin" (underfitting), dẫn đến khả năng bắt spam (recall) kém đi.
Khi tăng Vocab lên 5000 từ (max_size = 5000): F1 test có thể cải thiện nhẹ hoặc bắt đầu chững lại/giảm nhẹ so với mốc chuẩn (2000 từ). Vì thêm từ giúp mô hình bao quát được nhiều từ vựng ngách hơn. Tuy nhiên, nếu từ điển quá lớn sẽ xuất hiện nhiều từ hiếm (typo, từ xuất hiện 1-2 lần), làm tăng độ nhiễu, khiến mô hình dễ học vẹt (overfitting) các đặc trưng không mang tính tổng quát trên tập test.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Đạt cả độ trễ lẫn chất lượng tốt nhất: 2. BoW + MLP (độ trễ p95 = 5.67 ms, F1 test = 0.916) hoặc 3b. Encoder + LogReg (p95 = 40.04 ms, F1 test = 0.915).

ẫn số liệu thực tế từ bảng tổng kết LabĐộ trễ (Latency p95):BoW + MLP: 0.50 ms / tin (rất nhanh, chạy hoàn toàn trên CPU/GPU với ma trận thưa).Encoder + LogReg: 43.08 ms / tin (đạt ngưỡng an toàn dưới SLA 50 ms, nhưng sát giới hạn hơn do tốn thêm thời gian bước encode câu qua MiniLM).Chất lượng phân loại (F1 Score):BoW + MLP: F1 test đạt 0.916 (F1 tập SUB cân bằng đạt 0.919).Encoder + LogReg: F1 test đạt 0.915 (F1 tập SUB đạt 0.919).Khả năng đáp ứng tải 10 triệu SMS/ngày:10 triệu tin/ngày tương đương với throughput trung bình khoảng $\approx 116$ tin/giây.Cả hai mô hình trên đều có tốc độ xử lý cả lô (batch throughput) cực kỳ cao (BoW + MLP đạt ~25.156 tin/s; Encoder + LogReg đạt ~26 tin/s), hoàn toàn dư sức gánh tải hệ thống lớn mà không cần đến các cụm máy chủ GPU đắt đỏ.

Đánh đổi về khả năng chống chịu "Case khó" (Robustness)

Đánh đổi về chi phí huấn luyện & tài nguyên bộ nhớ
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

Hỏi : Nhờ AI hỗ trợ cấu trúc hàm gọi API call_openai và viết hàm xử lý định dạng JSON/Text phản hồi (parse_label) theo đúng chuẩn OpenAI-compatible.

Nhận: Đoạn code mẫu sử dụng thư viện openai và vòng lặp try-except cơ bản.

Sửa : Bổ sung tham số provider_kwargs để truyền chính xác các giới hạn token và tham số điều khiển mức độ suy luận (reasoning_effort="none"), tránh lỗi 400 đối với các model thế hệ mới.

Điều hiểu ra hôm nay mà trước đó chưa hiểu
Trước đây, tôi thường xem các mô hình AI hoặc các phương pháp học máy (như Bag-of-Words hay Embedding) chỉ qua con số độ chính xác (Accuracy/F1) trên tập test chuẩn. Hôm nay, qua bộ 50 case stress test, tôi hiểu ra rằng các mô hình truyền thống (như BoW) thực chất chỉ đang "khớp từ khóa" bề nổi chứ không hề hiểu ngữ nghĩa, khiến chúng sập bẫy hoàn toàn khi gặp văn bản ngụy trang hoặc giọng điệu tự nhiên. Ngược lại, dù LLM thông minh và xử lý ngữ nghĩa xuất sắc, chúng lại dễ dàng bị đánh lừa bởi các thủ thuật Prompt Injection nếu không được cô lập dữ liệu đầu vào cẩn thận.
