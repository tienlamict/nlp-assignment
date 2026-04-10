# Hướng Dẫn Thực Hành: Dịch Máy Anh - Việt (NMT)

Tài liệu này mô tả các công việc đã thực hiện để sửa đổi và hoàn thiện file `NTM.ipynb` cho bài toán dịch máy Anh - Việt sử dụng mô hình Seq2Seq với cơ chế Bahdanau Attention.

## 1. Các công việc đã thực hiện

### Sửa lỗi dữ liệu (Dataset)
*   **Cập nhật nguồn tải**: Do các link tải dữ liệu từ Stanford trong file gốc bị lỗi (403 Forbidden), mã nguồn đã được cập nhật để tải từ mirror GitHub uy tín (`stefan-it/nmt-en-vi`).
*   **Tự động giải nén**: Thêm thư viện `tarfile` để tự động giải nén các tệp `.tgz` sau khi tải về, đảm bảo quy trình chạy liền mạch.

### Cải thiện Tiền xử lý (Preprocessing)
*   Chuẩn hóa văn bản: Chuyển về chữ thường, xử lý khoảng cách giữa các dấu câu đặc biệt (như `?`, `!`, `.`) để bộ tách từ (tokenizer) hoạt động chính xác hơn cho cả tiếng Anh và tiếng Việt.

### Hoàn thiện Kiến trúc Mô hình
*   **Encoder**: Xây dựng lớp Encoder sử dụng GRU và Embedding.
*   **Bahdanau Attention**: Triển khai cơ chế chú ý để giúp mô hình tập trung vào các từ quan trọng trong câu gốc khi dịch.
*   **Decoder**: Xây dựng lớp Decoder kết hợp context vector từ Attention để dự đoán từ tiếp theo.

### Huấn luyện và Đánh giá
*   **Vòng lặp huấn luyện**: Triển khai `train_step` với kỹ thuật **Teacher Forcing** để tăng tốc độ hội tụ.
*   **Hàm dịch (Inference)**: Viết hàm `evaluate` và `translate` để thực hiện dự đoán trên các câu mới.
*   **Đo lường BLEU**: Tích hợp tính toán điểm BLEU (Bilingual Evaluation Understudy) để đánh giá định lượng chất lượng bản dịch.

---

## 2. Hướng dẫn chạy chương trình

### Yêu cầu hệ thống
*   Python 3.x
*   Các thư viện: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`, `nltk`.

### Các bước thực hiện

1.  **Mở Notebook**:
    Mở file `NTM.ipynb` bằng Jupyter Notebook, JupyterLab hoặc Google Colab.

2.  **Cài đặt thư viện (nếu cần)**:
    Nếu chạy trên máy cục bộ chưa có thư viện, hãy chạy lệnh:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn nltk
    ```

3.  **Cấu hình tham số (Tùy chọn)**:
    Trong cell chứa biến `num_examples` và `EPOCHS`:
    *   Mặc định đang để `num_examples = 50000` và `EPOCHS = 10` để đạt kết quả tốt.
    *   Nếu muốn kiểm tra nhanh (chạy trên CPU), bạn có thể giảm xuống `num_examples = 2000` và `EPOCHS = 1`.

4.  **Chạy tất cả các Cell (Run All)**:
    *   Notebook sẽ tự động tải bộ dữ liệu IWSLT15 (khoảng 133k cặp câu).
    *   Quá trình huấn luyện sẽ bắt đầu và hiển thị giá trị Loss sau mỗi 100 batch.
    *   Sau khi huấn luyện xong, mô hình sẽ thực hiện dịch thử các câu ví dụ như "how are you ?" và tính điểm BLEU trên tập test.

### Kết quả mong đợi
*   **Loss**: Giá trị loss giảm dần qua các epoch.
*   **Translation**: Câu tiếng Việt dự đoán có ý nghĩa tương đương với câu tiếng Anh đầu vào.
*   **BLEU Score**: Điểm số BLEU sẽ phản ánh độ chính xác của mô hình (thường đạt kết quả khả quan sau 10-20 epoch huấn luyện với dữ liệu lớn).

---
*Lưu ý: Khuyến khích sử dụng GPU (như trên Google Colab) để quá trình huấn luyện 50,000 ví dụ hoàn thành trong thời gian ngắn.*
