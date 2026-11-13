# So sánh 10 thuật toán phân loại trong bài toán nhận diện khuôn mặt

Báo cáo chính của đề tài *Ứng dụng thuật toán KNN vào bài toán nhận diện người trong ảnh*, theo yêu cầu của giảng viên: so sánh 10 thuật toán
khác nhau, demo 2 thuật toán, dùng 5–6 độ đo, đo tốc độ phản hồi (FPS) và phân tích các trường hợp nhận diện sai.

> **Trạng thái: đã chạy xong một lần (2026-11-05) trên tập con 109 người.** Mọi số liệu dưới đây lấy từ `results/algorithms/*.csv` và
> `run_info.json` do `python benchmark.py` và `python eval_impostors_algos.py` tạo ra. Chưa chạy trên toàn bộ 1.774 người (mục 9).
> Phần so sánh 9 bộ embedding nằm ở [phụ lục](SO_SANH_MO_HINH.md); nó **không** phải so sánh 10 thuật toán vì cả 9 mô hình trong đó cùng dùng KNN.

## Tóm tắt

- **Sáu thuật toán đầu không phân biệt chắc chắn được.** Theo F1 macro của lần chia 80/20: SVM tuyến tính (0,9886), SVM RBF (0,9831),
  KNN (0,9792), Logistic Regression (0,9788), MLP (0,9786), LDA (0,9718); cả sáu nằm trong khoảng 0,0168 F1. Xếp theo F1 của 5-fold CV thì các thuật toán liền kề trong nhóm này cách nhau ít hơn độ lệch chuẩn
  (chỉ đầu và cuối nhóm, SVM tuyến tính 0,9812 và LDA 0,9687, cách nhau khoảng 2 độ lệch chuẩn), nên thứ hạng giữa chúng không chắc chắn (mục 4). Kém rõ ràng hơn: Gaussian NB (0,96), Extra Trees và Random Forest (0,92 và 0,91)
  và Decision Tree (0,35).
- **SVM tuyến tính nhỉnh hơn KNN một chút nhưng nhất quán:** F1 CV cao hơn trung bình 0,0074 và cao hơn ở 5/5 fold; trên tập test chỉ SVM tuyến tính đúng ở
  10 mẫu, chỉ KNN đúng ở 1 mẫu. Với 5 fold, chưa có kiểm định thống kê.
- **KNN đứng hạng 3 theo F1 và không lọt top 2 theo quy tắc đã đặt trước.** Bù lại KNN có thời gian huấn luyện gần bằng 0 (0,003 s), nhưng chậm hơn SVM tuyến tính và MLP khi dự đoán
  (6,3 ms so với 0,84 và 0,13 ms mỗi vector, vì phải so với mọi vector train; thời gian này tăng theo số mẫu train) và phải lưu toàn bộ vector train (9,0 MB).
- **Hai thuật toán demo trong app: SVM tuyến tính và KNN.** Quy tắc đặt trước cho ra SVM tuyến tính và SVM RBF; đây là lựa chọn của người làm đề tài (mục 5).
- **Khả năng loại "người lạ" khác nhau rất xa và không theo thứ hạng accuracy** (mục 7.3): MLP chỉ để lọt 1,1% người lạ và Logistic Regression 3,3%,
  trong khi SVM tuyến tính để lọt 17,4% và KNN 23,7% ở cùng mức giữ lại ~94% người quen.
- **Tốc độ end-to-end là 0,35 FPS cho cả 10 thuật toán**, vì RetinaFace tốn ~2,7 giây mỗi ảnh, gấp ~238 lần thời gian predict của thuật toán chậm nhất (mục 6).
- **7 mẫu test bị cả 10 thuật toán cùng sai.** Đã xem cả 7 ảnh: đều là ảnh khó (mặt nghiêng, hóa trang hoặc trang phục nặng, mắt nhắm khi cười, thiếu sáng, hai người trông giống nhau),
  không phải crop hỏng và không thấy ảnh gán sai nhãn (mục 7.2).

## 1. Yêu cầu của giảng viên và nơi thực hiện

| Yêu cầu | Thực hiện | Mục |
|---|---|---|
| Làm nhiều mô hình | 10 thuật toán phân loại, cùng một embedding (`classifiers.py`) | 3 |
| So sánh giữa 10 mô hình khác nhau | `python benchmark.py` → `results/algorithms/` | 4 |
| Demo 2 mô hình | `app.py` chọn giữa SVM tuyến tính và KNN (`python train_model.py`) | 5, 8 |
| 8 mô hình còn lại chỉ cần code | Có trong `classifiers.py`, chạy trong benchmark; train được bằng `train_model.py --algo` | 3 |
| 5–6 độ đo | 6 độ đo, kèm 5-fold CV | 4 |
| Tốc độ phản hồi (FPS) | thời gian huấn luyện, predict, scores, kích thước mô hình, FPS embed + phân loại và end-to-end | 6 |
| Các trường hợp nhận diện sai | `results/algorithms/failures_<thuật toán>.csv`, ảnh minh họa, phân tích chéo | 7 |

## 2. Phương pháp chung

**Pipeline** (`face_pipeline.py`, dùng chung cho app, train và benchmark):

1. RetinaFace detect và căn chỉnh khuôn mặt, lấy khuôn mặt lớn nhất. Làm một lần và cache ở `output/faces/`.
2. Embedding **ArcFace** (512 chiều) biến khuôn mặt thành vector, cache ở `output/embeddings/ArcFace.npz`.
3. Chuẩn hóa L2 vector.
4. Bộ phân loại (1 trong 10) dự đoán tên người. Cả 10 thuật toán nhận **cùng ma trận embedding** và **cùng chỉ số train/test**.

Vì mọi thứ trước bước 4 giống nhau, chênh lệch giữa các thuật toán là do **bộ phân loại**, không phải do embedding hay detector.

**Vì sao ArcFace.** ArcFace có F1 macro cao nhất (0,9853), ngang Facenet512 (0,9845), ở lần so sánh 9 embedding (xem [phụ lục](SO_SANH_MO_HINH.md)).
Lần so sánh đó đo trên 62 người LFW và **chưa được lặp lại trên tập 109 người** của báo cáo này. Kết luận về thuật toán áp dụng cho ArcFace, chưa chắc đúng với embedding khác.
Embedding khai báo ở `config.BENCH_EMBEDDING`.

### 2.1. Dữ liệu và cách chia

- **Tập so sánh:** tập con cố định gồm các người có **≥ 30 ảnh trong cache khuôn mặt**: **109 người, 5.459 ảnh**
  (30–215 ảnh mỗi người, 0 ảnh không detect được mặt). Nguồn: **105 người Pins** và
  **4 người LFW** (Alejandro Toledo, Alvaro Uribe, Andre Agassi, Arnold Schwarzenegger). Danh sách người được ghi cố định ở `results/algorithms/subset_people.txt`
  (do `face_pipeline.load_bench_subset` tạo một lần từ cache, sau đó luôn đọc lại file này). Số ảnh của mỗi người là mọi ảnh đang có trong cache, nên con số
  5.459 sẽ tăng nếu detect thêm ảnh của những người này.
- **Chia 80/20 phân tầng theo người**, `seed = 42`: **4.367 ảnh train / 1.092 ảnh test**, dùng chung cho cả 10 thuật toán.
- **5-fold cross-validation phân tầng** (cùng seed) trên toàn bộ 5.459 ảnh: trung bình ± độ lệch chuẩn mẫu (ddof = 1) của 5 giá trị. Cần thiết vì khi embedding đã tốt,
  nhiều thuật toán chỉ khác nhau vài mẫu test.
- **Mất cân bằng lớp:** 87 trong 109 người có đúng 30 ảnh, 22 người còn lại có từ 31 đến 215 ảnh và chiếm 2.849/5.459 ảnh (khoảng 52%).
  Accuracy (tính theo ảnh) nghiêng về nhóm 22 người này; Precision, Recall, F1 macro tính trung bình theo người nên không bị ảnh hưởng. Báo cáo cả hai.
- **Chưa làm:** chạy lại trên toàn bộ 1.774 người ([KE_HOACH_DATASET_LON.md](KE_HOACH_DATASET_LON.md)); cần detect thêm ~17.700 ảnh (vài giờ trên CPU).

**Hạn chế của dữ liệu.** Ảnh Pins lấy từ Pinterest nên một số ảnh có thể bị gán sai người (chưa được làm sạch); LFW là ảnh báo chí. Cả hai khác ảnh webcam của người dùng.
Số liệu không nên được hiểu là độ chính xác của ứng dụng trên dữ liệu riêng. Có ít nhất một cặp ảnh trùng hệt trong tập (Bill Gates, `Bill Gates193_541.jpg` và `Bill Gates196_544.jpg`);
nếu một ảnh rơi vào train và ảnh kia vào test thì có một mẫu test "dễ giả", ảnh hưởng tối đa 1/1.092. Chưa kiểm tra ảnh gần trùng (không giống hệt từng bit).

## 3. Mười thuật toán

Siêu tham số **cố định trước khi chạy** và không chỉnh theo kết quả test (không tinh chỉnh riêng cho từng thuật toán): đây là so sánh ở cấu hình hợp lý, **không phải cấu hình tối ưu của từng thuật toán**.
Giá trị dưới đây là giá trị trong [`classifiers.py`](../classifiers.py) và được ghi vào `results/algorithms/run_info.json`. Mọi thuật toán dùng `n_jobs = 1` để thời gian đo lặp lại được
và giống cách ứng dụng chạy (một ảnh mỗi lần). Không thuật toán nào báo `ConvergenceWarning`.

| # | Thuật toán | Cài đặt | Tham số | Điểm dùng cho Top-3 và độ tin cậy |
|---|---|---|---|---|
| 1 | KNN | **tự cài** `knn_func.py` | K = 3, Euclid, bỏ phiếu đa số, hòa phiếu chọn láng giềng gần nhất | −(khoảng cách nhỏ nhất tới từng người) |
| 2 | SVM tuyến tính | `SVC(kernel="linear")` | C = 1 | `decision_function` |
| 3 | SVM RBF | `SVC(kernel="rbf")` | C = 10, gamma = "scale" | `decision_function` |
| 4 | Logistic Regression | `LogisticRegression` | multinomial (mặc định), C = 1, `max_iter = 1000` | `predict_proba` |
| 5 | Random Forest | `RandomForestClassifier` | 300 cây, `random_state = 42` | `predict_proba` |
| 6 | Extra Trees | `ExtraTreesClassifier` | 300 cây, `random_state = 42` | `predict_proba` |
| 7 | Decision Tree | `DecisionTreeClassifier` | mặc định, `random_state = 42` | `predict_proba` |
| 8 | Gaussian Naive Bayes | `GaussianNB` | mặc định | `predict_proba` |
| 9 | LDA | `LinearDiscriminantAnalysis` | solver = svd | `predict_proba` |
| 10 | MLP | `MLPClassifier` | 1 lớp ẩn 256 nơ-ron, `max_iter = 500`, `random_state = 42` | `predict_proba` |

**Đối chiếu KNN tự cài:** `KNeighborsClassifier` của scikit-learn (K = 3, Euclid) lệch KNN tự cài ở 15/1.092 mẫu test, và cả 15 ca đều là hòa phiếu 1-1-1
(3 láng giềng thuộc 3 người khác nhau: scikit-learn chọn theo thứ tự chữ cái, KNN tự cài chọn láng giềng gần nhất). Không thấy dấu hiệu lỗi cài đặt.

**Không có Gradient Boosting / XGBoost** vì với 109 lớp và 512 chiều thời gian huấn luyện rất lớn so với lợi ích.

## 4. Sáu độ đo

| Độ đo | Ý nghĩa |
|---|---|
| Accuracy | Tỉ lệ mẫu dự đoán đúng |
| Precision (macro) | Trung bình theo từng người của: trong các ảnh bị dự đoán là người đó, bao nhiêu phần đúng |
| Recall (macro) | Trung bình theo từng người của: trong các ảnh thật của người đó, bao nhiêu phần được nhận ra |
| F1 (macro) | Trung bình điều hòa của Precision và Recall, rồi lấy trung bình theo người |
| Top-3 accuracy | Tỉ lệ mẫu mà nhãn đúng nằm trong 3 người có điểm cao nhất (điểm ở bảng mục 3). KNN dùng 3 người **khác nhau** gần nhất |
| Cohen's Kappa | Mức đồng thuận giữa dự đoán và nhãn thật sau khi trừ phần trùng hợp ngẫu nhiên |

Top-3 so sánh được giữa các thuật toán chỉ ở mức tương đối vì điểm của mỗi thuật toán có thang riêng.

### 4.1. Kết quả một lần chia 80/20 (1.092 mẫu test, xếp theo F1)

| Hạng | Thuật toán | Accuracy | Precision | Recall | F1 | Top-3 | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | SVM tuyến tính | 0,9908 | 0,9938 | 0,9847 | 0,9886 | 0,9936 | 0,9907 |
| 2 | SVM RBF | 0,9863 | 0,9919 | 0,9771 | 0,9831 | 0,9918 | 0,9860 |
| 3 | KNN (tự cài) | 0,9826 | 0,9878 | 0,9745 | 0,9792 | 0,9908 | 0,9823 |
| 4 | Logistic Regression | 0,9817 | 0,9912 | 0,9694 | 0,9788 | 0,9936 | 0,9814 |
| 5 | MLP | 0,9835 | 0,9857 | 0,9748 | 0,9786 | 0,9945 | 0,9832 |
| 6 | LDA | 0,9799 | 0,9830 | 0,9664 | 0,9718 | 0,9853 | 0,9795 |
| 7 | Gaussian NB | 0,9670 | 0,9745 | 0,9580 | 0,9616 | 0,9863 | 0,9665 |
| 8 | Extra Trees | 0,9295 | 0,9817 | 0,8860 | 0,9226 | 0,9707 | 0,9281 |
| 9 | Random Forest | 0,9222 | 0,9755 | 0,8737 | 0,9123 | 0,9734 | 0,9207 |
| 10 | Decision Tree | 0,4863 | 0,3886 | 0,3526 | 0,3536 | 0,5082 | 0,4772 |

Biểu đồ: [`f1_ranking.png`](../results/algorithms/f1_ranking.png), [`metrics_heatmap.png`](../results/algorithms/metrics_heatmap.png).

### 4.2. Kết quả 5-fold cross-validation (trung bình ± độ lệch chuẩn)

| Thuật toán | Accuracy (TB ± ĐLC) | F1 macro (TB ± ĐLC) | Fit trung bình mỗi fold (s) |
|---|---:|---:|---:|
| SVM tuyến tính | 0,9830 ± 0,0065 | 0,9812 ± 0,0062 | 0,816 |
| SVM RBF | 0,9789 ± 0,0065 | 0,9776 ± 0,0070 | 1,565 |
| KNN (tự cài) | 0,9780 ± 0,0064 | 0,9738 ± 0,0081 | 0,003 |
| Logistic Regression | 0,9756 ± 0,0057 | 0,9726 ± 0,0053 | 0,363 |
| MLP | 0,9830 ± 0,0030 | 0,9792 ± 0,0031 | 5,797 |
| LDA | 0,9745 ± 0,0054 | 0,9687 ± 0,0044 | 0,262 |
| Gaussian NB | 0,9659 ± 0,0063 | 0,9603 ± 0,0072 | 0,016 |
| Extra Trees | 0,9258 ± 0,0039 | 0,9182 ± 0,0057 | 7,956 |
| Random Forest | 0,9156 ± 0,0063 | 0,9049 ± 0,0077 | 67,903 |
| Decision Tree | 0,4847 ± 0,0102 | 0,3577 ± 0,0081 | 13,163 |

### 4.3. Cách đọc kết quả

- **Một mẫu test tương đương 0,09 điểm phần trăm accuracy.** Về Accuracy, sáu thuật toán đầu cách nhau chỉ khoảng 12 mẫu test.
- **So sánh ghép cặp** (cùng 5 fold, cùng tập test) nhạy hơn việc so độ lệch chuẩn riêng lẻ. SVM tuyến tính so với các thuật toán kế:

| Cặp so sánh | Chênh F1 TB qua 5 fold | Số fold cặp đầu cao hơn | Mẫu test chỉ thuật toán đầu đúng | Mẫu test chỉ thuật toán sau đúng |
|---|---:|---:|---:|---:|
| SVM tuyến tính − SVM RBF | 0,0036 | 4/5 | 5 | 0 |
| SVM tuyến tính − MLP | 0,0020 | 3/5 | 9 | 1 |
| SVM tuyến tính − KNN (tự cài) | 0,0074 | 5/5 | 10 | 1 |
| SVM tuyến tính − Logistic Regression | 0,0086 | 5/5 | 10 | 0 |
| SVM tuyến tính − LDA | 0,0125 | 5/5 | 14 | 2 |

  SVM tuyến tính cao hơn KNN ở cả 5 fold; so với MLP chỉ cao hơn ở 3/5 fold, so với SVM RBF ở 4/5 fold. Những cặp này **chưa đủ để kết luận** vì chỉ có 5 fold và không có kiểm định thống kê.
- **Nhóm giữa và nhóm cuối thì rõ.** Gaussian NB (0,962) kém nhóm đầu vài điểm; Extra Trees và Random Forest kém đầu bảng 6,6 và 7,6 điểm F1; Decision Tree chỉ đúng 49% mẫu.
  Cây quyết định xét từng chiều riêng lẻ, trong khi thông tin của embedding ArcFace rải đều trên 512 chiều; đây là giải thích hợp lý nhưng **chưa được kiểm chứng riêng**.
- **Precision cao hơn Recall ở mọi thuật toán** (ví dụ Random Forest: 0,976 so với 0,874): nghĩa là một số người bị bỏ sót nhiều hơn là bị nhận nhầm.
  Có thể liên quan tới mất cân bằng lớp ở mục 2.1, nhưng **chưa được kiểm chứng**.

## 5. Chọn 2 thuật toán demo

**Quy tắc đặt trước khi chạy:** xếp theo F1 macro của lần chia 80/20, hòa thì xét Accuracy, rồi thời gian predict. Kết quả (`best_algorithms.json`): **SVM tuyến tính** và **SVM RBF**.
Chênh lệch F1 trung bình CV giữa hai thuật toán này là 0,0036, nhỏ hơn độ lệch chuẩn CV lớn hơn (0,0070), nên **không phân biệt được** theo CV.
Ở vị trí thứ hai, MLP có F1 CV (0,9792) cao hơn SVM RBF (0,9776) dù thấp hơn ở một lần chia; vị trí này không vững.

**Hai thuật toán trong app: SVM tuyến tính và KNN** (`config.DEMO_ALGORITHMS`). SVM RBF được thay bằng KNN, vì KNN là thuật toán của đề tài và nằm trong nhóm sáu thuật toán không phân biệt được,
nên có thể so sánh trực tiếp trong app. **Đây là lựa chọn của người làm đề tài, không phải kết quả của quy tắc.**

**Một điều cần cân nhắc** (mục 7.3): nếu mục tiêu của app là *từ chối người lạ*, MLP và Logistic Regression làm tốt hơn hẳn SVM tuyến tính và KNN dù accuracy tương đương.

## 6. Tốc độ

Máy đo: Windows-10-10.0.19045-SP0, Intel64 Family 6 Model 191 Stepping 2, GenuineIntel, 16 luồng, Python 3.12.10, TensorFlow 2.21.0, scikit-learn 1.9.1, DeepFace 0.0.103. Không dùng GPU.
**Mọi số đo FPS phải được hiểu là của máy này.** Số FPS ở [phụ lục](SO_SANH_MO_HINH.md) đo trên máy khác (i5-1135G7, Windows 11) và không so được với bảng dưới.

**Cách đo.** Detect và embed là chi phí **chung** cho cả 10 thuật toán nên chỉ đo một lần (50 ảnh gốc chọn ngẫu nhiên từ tập test, ảnh trung bình
243×226 px, lớn nhất 668×550): **detect 2727 ms, embed 95,1 ms**.
Thời gian phân loại đo riêng cho từng thuật toán trên một vector mỗi lần (200 vector × 3 lượt, bỏ lượt khởi động, không gom batch).
FPS embed + predict = 1/(embed + predict); FPS end-to-end = 1/(detect + embed + predict). Đo thành phần rồi ghép như vậy tránh 10 lần detect dư thừa và nhiễu giữa các thuật toán,
nhưng khác với việc chạy cả pipeline cho từng thuật toán.

| Thuật toán | Huấn luyện (s) | predict (ms) | scores (ms) | Kích thước (MB) | FPS embed + predict | FPS end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| SVM tuyến tính | 0,905 | 0,84 ± 0,08 | 63,45 | 19,9 | 10,4 | 0,354 |
| SVM RBF | 1,638 | 1,15 ± 0,01 | 63,59 | 20,2 | 10,4 | 0,354 |
| KNN (tự cài) | 0,003 | 6,33 ± 0,18 | 6,21 | 9,0 | 9,9 | 0,354 |
| Logistic Regression | 0,441 | 0,08 ± 0,00 | 0,07 | 0,2 | 10,5 | 0,354 |
| MLP | 5,962 | 0,13 ± 0,00 | 0,08 | 2,6 | 10,5 | 0,354 |
| LDA | 0,414 | 0,08 ± 0,00 | 0,07 | 0,7 | 10,5 | 0,354 |
| Gaussian NB | 0,036 | 1,37 ± 0,04 | 1,40 | 0,5 | 10,4 | 0,354 |
| Extra Trees | 8,449 | 11,45 ± 0,24 | 11,29 | 1262,9 | 9,4 | 0,353 |
| Random Forest | 68,799 | 10,96 ± 0,11 | 11,04 | 670,7 | 9,4 | 0,353 |
| Decision Tree | 13,079 | 0,07 ± 0,02 | 0,06 | 2,6 | 10,5 | 0,354 |

`predict`: chỉ lấy nhãn. `scores`: lấy thêm độ tin cậy, cần để loại "Unknown". FPS tính từ `predict`. Biểu đồ: [`time_and_size.png`](../results/algorithms/time_and_size.png),
[`speed_vs_f1.png`](../results/algorithms/speed_vs_f1.png).

- **RetinaFace là nút thắt:** FPS end-to-end chỉ 0,353–0,354 cho cả 10 thuật toán, khác nhau dưới 1%.
- **Thời gian detect không giải thích được hoàn toàn.** Phụ lục đo ~1,2 giây/ảnh, ở đây 2,7 giây. Giả thuyết "ảnh Pins lớn hơn LFW" **sai**: kiểm tra riêng 10 ảnh LFW (250×250) và 10 ảnh Pins
  trên máy này cho 2,4–2,6 giây và 2,6–2,8 giây (hai lượt), tức gần như nhau. Nguyên nhân còn lại có thể là máy khác, DeepFace 0.0.103 so với 0.0.101, hoặc cách TensorFlow dùng luồng; **chưa kiểm chứng**.
  Hai lượt đo cùng một nhóm ảnh lệch nhau khoảng 8%.
- **`scores` của SVM chậm gấp 55–76 lần `predict`** (63 ms so với 0,84 ms), nhiều khả năng do cách scikit-learn đổi điểm "ovr" cho 109 lớp (**chưa được kiểm chứng riêng**). Nếu app cần loại Unknown
  thì chi phí thực tế của SVM là cột `scores`, vẫn nhỏ so với 2727 ms detect.
- **Kích thước:** Random Forest 670 MB và Extra Trees 1.262 MB (cây mọc hết cỡ), còn Logistic Regression chỉ 0,23 MB.
  KNN phải giữ toàn bộ vector train nên lớn dần theo dữ liệu.
- **Thời gian huấn luyện:** Random Forest 69 s, Decision Tree 13 s, các thuật toán còn lại dưới 9 s.
  KNN không có bước huấn luyện thực sự (chỉ lưu dữ liệu).

## 7. Các trường hợp nhận diện sai

Một mẫu test là **lỗi** nếu dự đoán sai, hoặc dự đoán đúng nhưng bị loại do dưới ngưỡng Unknown. Mỗi lỗi xếp vào một nhóm (theo thứ tự ưu tiên):

| Nguyên nhân | Định nghĩa |
|---|---|
| `no_face_detected` | RetinaFace không tìm thấy mặt (`confidence = 0`). **Bằng 0 ở mọi thuật toán** vì cả 5.459 ảnh đều có mặt được detect |
| `bad_crop` | Crop khuôn mặt có hơn 15% pixel đen: bước detect/căn chỉnh tạo crop hỏng. Có 18/1.092 crop như vậy trong tập test |
| `misidentified` | Crop bình thường nhưng bộ phân loại nhận nhầm người |
| `rejected_unknown` | Nhận đúng nhưng độ tin cậy dưới ngưỡng nên bị loại thành Unknown |

### 7.1. Số liệu (1.092 mẫu test)

| Thuật toán | Đúng | Nhận nhầm | Crop hỏng | Bị loại Unknown | Ngưỡng Unknown |
|---|---:|---:|---:|---:|---:|
| SVM tuyến tính | 1082 | 10 | 2 | 53 | 108,3294 |
| SVM RBF | 1077 | 15 | 2 | 52 | 108,3285 |
| KNN (tự cài) | 1073 | 19 | 5 | 49 | -0,9795 |
| Logistic Regression | 1072 | 20 | 2 | 52 | 0,1633 |
| MLP | 1074 | 18 | 2 | 52 | 0,9247 |
| LDA | 1070 | 22 | 1 | 9 | 1,0000 |
| Gaussian NB | 1056 | 36 | 0 | 4 | 1,0000 |
| Extra Trees | 1015 | 76 | 4 | 42 | 0,0833 |
| Random Forest | 1007 | 83 | 3 | 43 | 0,0833 |
| Decision Tree | 531 | 548 | 13 | 0 | 1,0000 |

Biểu đồ: [`failure_breakdown.png`](../results/algorithms/failure_breakdown.png), ma trận nhầm lẫn của hai thuật toán demo
([SVM tuyến tính](../results/algorithms/confusion_SVM-linear.png), [SVM RBF](../results/algorithms/confusion_SVM-RBF.png)). Chi tiết từng ca (ảnh thật, nhãn đúng, nhãn dự đoán,
độ tin cậy, nguyên nhân): `failures_<thuật toán>.csv`; ảnh minh họa (trái: ảnh test, phải: mẫu train gần nhất của người bị nhận nhầm) trong `failures/<thuật toán>/`.

**Ngưỡng Unknown.** Mỗi thuật toán dùng độ tin cậy cao nhất của nó và ngưỡng là **percentile 5 của độ tin cậy trên các mẫu test được nhận đúng** (giữ lại 95%). Với KNN tương đương
percentile 95 của khoảng cách láng giềng gần nhất. Ngưỡng tính trên chính tập test nên số "bị loại" luôn xấp xỉ 5% mẫu đúng (khoảng 52 mẫu); đây là hệ quả của quy tắc, không phải lỗi của thuật toán.
Các thang điểm **không so sánh được giữa các thuật toán**. Với **Decision Tree, Gaussian NB và LDA** xác suất đầu ra gần như luôn bằng 1 (bão hòa) nên ngưỡng không loại được mẫu nào
(0, 4 và 9 ca): cột "bị loại Unknown" của ba thuật toán này không dùng để so sánh.
Điểm của SVM nằm sát ~108,33 do cách scikit-learn đổi điểm "ovr" (số phiếu cộng một phần độ tin cậy); thứ tự vẫn đúng nên ngưỡng vẫn có tác dụng (mục 7.3).

### 7.2. Phân tích chéo: một mẫu bị sai bởi bao nhiêu thuật toán

| Số thuật toán sai mẫu đó | Cả 10 thuật toán | Bỏ Decision Tree (9) |
|---:|---:|---:|
| 0 | 517 | 970 |
| 1 | 461 | 52 |
| 2 | 48 | 34 |
| 3 | 31 | 12 |
| 4 | 12 | 8 |
| 5 | 7 | 6 |
| 6 | 6 | 1 |
| 7 | 1 | 1 |
| 8 | 1 | 1 |
| 9 | 1 | 7 |
| 10 | 7 | 0 |

Biểu đồ: [`overlap_hist.png`](../results/algorithms/overlap_hist.png). Decision Tree sai 561 mẫu (51%) nên làm phồng cột "sai bởi đúng 1 thuật toán"; cột "Bỏ Decision Tree" phản ánh tốt hơn.

- **Cả 10 thuật toán cùng sai: 7 mẫu** (0,64%). Đã xem cả 7 ảnh (`failures/_all_wrong/`): mặt nghiêng (Henry Cavil), hóa trang hoặc trang phục nặng
  (Zoe Saldana, Robert De Niro, scarlett johansson), mắt nhắm khi cười lớn (Megan Fox, nhầm với Brie Larson), ảnh thiếu sáng (Tom Hardy), và cặp trông giống nhau (amber heard nhầm với Adriana Lima).
  Không ảnh nào là crop hỏng (tỉ lệ pixel đen ≤ 0,03) và không thấy ảnh gán sai nhãn, nhưng nhãn sai ở nơi khác trong Pins thì chưa kiểm tra. Đây là **lỗi của dữ liệu khó**, không phải lỗi của một thuật toán.
- **Chỉ KNN sai: 1 mẫu** (Brie Larson nhầm với Richard Harmon, khoảng cách 0,95). Tổng cộng KNN sai 19 mẫu.
- **Crop hỏng:** 18/1.092 crop test có hơn 15% pixel đen. Ở nhóm sáu thuật toán đầu chỉ 1–5 mẫu trong số đó bị sai hoặc bị loại (Decision Tree: 13), ít nổi bật hơn ở phụ lục (3 trong 5 lỗi của ArcFace là crop hỏng, trên tập LFW khác).

### 7.3. Người lạ có bị nhận nhầm thành người quen không?

`python eval_impostors_algos.py` huấn luyện lại từng thuật toán trên tập train và chấm 1.000 khuôn mặt "người lạ" (crop từ bộ Kaggle face-detection, không có danh tính,
do `build_impostors.py` tạo) ở **đúng ngưỡng Unknown của mục 7.1**. FAR là tỉ lệ người lạ vẫn bị nhận thành một người quen.

| Thuật toán | Người lạ bị nhận nhầm (FAR) | Người quen được giữ lại | Người quen được giữ lại khi FAR = 5% |
|---|---:|---:|---:|
| SVM tuyến tính | 17,4% | 94,3% | 92,4% |
| SVM RBF | 22,6% | 94,0% | 82,7% |
| KNN (tự cài) | 23,7% | 94,2% | 7,0% |
| Logistic Regression | 3,3% | 93,7% | 95,0% |
| MLP | 1,1% | 93,5% | 96,3% |
| LDA | 25,1% | 97,4% | 97,4% |
| Gaussian NB | 73,7% | 98,4% | 98,4% |
| Extra Trees | 22,5% | 91,1% | 75,0% |
| Random Forest | 28,8% | 91,4% | 83,2% |
| Decision Tree | 100,0% | 100,0% | 100,0% |

Cột cuối không phụ thuộc cách đặt ngưỡng: chọn ngưỡng sao cho 5% người lạ lọt qua, rồi xem còn giữ được bao nhiêu người quen (cao hơn = độ tin cậy phân biệt quen/lạ tốt hơn).

- **Hạng theo accuracy không báo trước khả năng loại người lạ.** MLP (1,1%) và Logistic Regression (3,3%) lọt ít nhất; SVM tuyến tính (17,4%), SVM RBF, LDA, Extra Trees,
  KNN (23,7%) và Random Forest (28,8%) lọt từ 17% đến 29%; Gaussian NB lọt 74% và Decision Tree 100% (ngưỡng bão hòa, không loại được gì).
- **KNN yếu nhất ở cột cuối:** chỉ giữ 7,0% người quen khi chặn người lạ ở mức 5%. Nguyên nhân đã được tìm hiểu một phần: 143/1.000 người lạ có láng giềng gần nhất cách dưới 0,5
  (trung vị 1,08), và trong 2 cặp đã xem, người lạ là mặt nghiêng/cúi còn mẫu quen gần nhất cũng là mặt nghiêng/xoay (một ảnh Anne Hathaway xoay làm "điểm hút" cho nhiều người lạ). Nghĩa là embedding gom các khuôn mặt
  cùng góc nghiêng lại gần nhau bất kể danh tính, và khoảng cách láng giềng gần nhất bị ảnh hưởng nặng hơn các bộ phân loại học trên toàn bộ dữ liệu. Chỉ mới xem 2 cặp nên đây là **giả thuyết có cơ sở, chưa kiểm chứng đầy đủ**.
- Bộ crop người lạ khá nhiễu (mặt nghiêng, tối, mờ) giống người lạ thật, nhưng con số phụ thuộc vào bộ crop này và vào số người quen trong mô hình (109).

## 8. Demo

`python train_model.py` train SVM tuyến tính và KNN trên ArcFace (toàn bộ 5.459 ảnh của tập con, ngưỡng Unknown tính bằng dự đoán out-of-fold 5-fold) và lưu `output/model_ArcFace_SVM-linear.joblib`, `output/model_ArcFace_KNN.npz`.
`streamlit run app.py` → **Nhận diện** → nguồn **Thuật toán phân loại (ArcFace)**: ô chọn liệt kê **đủ 10 thuật toán**, hai thuật toán demo (⭐ SVM tuyến tính, ⭐ KNN) ở đầu danh sách.
8 thuật toán còn lại cũng dùng được trong app sau khi train bằng `python train_model.py --algo SVM-RBF LogisticRegression RandomForest ExtraTrees DecisionTree GaussianNB LDA MLP`
(Random Forest ~0,8 GB và Extra Trees ~1,6 GB); chưa train thì app ghi "(chưa train)" kèm lệnh. Với Decision Tree, Gaussian NB và LDA app cảnh báo rằng ngưỡng Unknown gần như vô hiệu (mục 7.1).
Trang **So sánh mô hình** hiển thị các bảng và biểu đồ của báo cáo này. Kho vector PostgreSQL + pgvector chỉ có ý nghĩa với KNN và không áp dụng cho các thuật toán khác.

**Kiểm tra thủ công một lần** (script `output/phase5_check.py`, **không nằm trong repo nên không tái lập bằng lệnh có sẵn**): 40 ảnh của người đã quen nhưng chưa từng nằm trong tập train
(`external/pins_demo_test/`, 1 ảnh mỗi người), đi qua đúng hàm nhận diện của app. SVM tuyến tính nhận đúng 38/40 (95,0%), 2 ảnh bị loại Unknown, 0 nhận nhầm; KNN đúng 36/40 (90,0%), 3 ảnh Unknown, 1 nhận nhầm.
Mẫu quá nhỏ (1 ảnh = 2,5 điểm phần trăm) nên **không dùng để kết luận thuật toán nào tốt hơn**. Chạy thật trong app: tải một ảnh Robert De Niro chưa từng thấy, SVM tuyến tính nhận đúng, KNN trả Unknown.

## 9. Hạn chế

- **Một embedding duy nhất (ArcFace)**, chưa xác nhận lại trên tập 109 người.
- **Siêu tham số không tinh chỉnh** (ví dụ KNN chỉ thử K = 3): kết quả của từng thuật toán có thể được cải thiện.
- **Một lần chia và 5 fold:** độ lệch chuẩn tính từ 5 giá trị, chưa có kiểm định thống kê; thứ hạng trong nhóm sáu thuật toán đầu không chắc chắn.
- **Tập 109 người, mất cân bằng lớp, nhãn Pins chưa được làm sạch**; chưa chạy trên 1.774 người.
- **Ngưỡng Unknown tính trên chính tập test** (benchmark) nên tỉ lệ người quen được giữ lại hơi lạc quan; ngưỡng dùng trong app tính bằng dự đoán out-of-fold.
- **Ngưỡng Unknown không so sánh được giữa các thuật toán** và vô hiệu với Decision Tree, Gaussian NB, LDA.
- **FAR người lạ** phụ thuộc bộ crop 1.000 ảnh và số người quen trong mô hình.
- **FPS chỉ trên CPU, một máy**; thời gian detect cao gấp đôi phụ lục chưa được giải thích; đo thành phần rồi ghép chứ không chạy cả pipeline từng thuật toán.
- **Người mới chụp trong app** chưa nằm trong mô hình `--algo` cho đến khi có ảnh của họ trong cache khuôn mặt và train lại.

## 10. Tái lập

```bash
python prepare_data.py && python prepare_pins.py     # dữ liệu vào dataset/
# cache khuôn mặt output/faces/ và embedding output/embeddings/ArcFace.npz: tạo bởi ensure_face_cache() / embed_dataset
python benchmark.py                                  # 10 thuật toán, ~13 phút -> results/algorithms/ (python benchmark.py --quick để thử nhanh)
python eval_impostors_algos.py                       # FAR người lạ cho 10 thuật toán, ~5 phút (cần build_impostors.py đã tạo crop người lạ)
python plot_results.py                               # 8 biểu đồ
python train_model.py                                # SVM tuyến tính + KNN cho app
streamlit run app.py
```

Benchmark đọc cache khuôn mặt chứ **không detect**; nếu cache thiếu ảnh của 109 người trong `subset_people.txt` thì chỉ dùng phần có sẵn. `benchmark_embeddings.py` và `train_model.py --model`
thì gọi `ensure_face_cache()`, detect mọi ảnh chưa có trong cache (hiện ~17.700 ảnh). Thời gian trên máy phát triển (xem mục 6): benchmark ~13 phút, `eval_impostors_algos.py` ~5 phút (lần đầu thêm ~100 giây embed 1.000 crop).

## 11. File liên quan

| File | Vai trò |
|---|---|
| `config.py` | `BENCH_EMBEDDING`, `BENCH_MIN_IMAGES`, `CLASSIFIER_NAMES`, `DEMO_ALGORITHMS`, đường dẫn, seed |
| `classifiers.py` | Bảng 10 thuật toán, giao diện chung `fit`/`predict`/`scores`, lưu/nạp |
| `knn_func.py` | KNN tự cài |
| `face_pipeline.py` | detect, embed, cache; `load_bench_subset` đọc tập con mà không detect |
| `benchmark.py` | 6 độ đo, CV, tốc độ, ca lỗi, phân tích chéo, chọn 2 thuật toán |
| `eval_impostors_algos.py` | FAR người lạ cho 10 thuật toán |
| `plot_results.py` | 8 biểu đồ (`--kind embeddings` cho phụ lục) |
| `train_model.py` | Train thuật toán demo (`--algo`), hoặc KNN theo embedding (`--model`) |
| `app.py` | Giao diện: chọn thuật toán, trang so sánh |
| `results/algorithms/` | `metrics.csv`, `cv.csv`, `cv_folds.csv`, `timing.csv`, `failure_summary.csv`, `failures_*.csv`, `predictions.csv`, `overlap*.csv`, `impostor_eval.csv`, `best_algorithms.json`, `subset_people.txt`, `run_info.json`, biểu đồ |
