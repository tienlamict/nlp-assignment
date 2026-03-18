# Assignment 2: POS Tagging Evaluation on Brown Corpus

## Tổng quan

Bài tập này thực hiện gán nhãn từ loại (POS tagging) trên Brown Corpus và đánh giá hiệu suất của hai bộ tagger khác nhau.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy chương trình

```bash
python pos_tagging_evaluation.py
#python assignment-2/pos_tagging_evaluation.py
```

## Mô tả các bước thực hiện

### 1. Tải Brown Corpus
- Sử dụng NLTK để tải Brown Corpus với universal tagset
- Tổng số câu: 57,340
- Chia dữ liệu: 80% training (45,872 câu), 20% testing (11,468 câu)

### 2. Hai bộ POS Tagger

#### Tagger 1: Unigram Tagger với Default Backoff
- **Cấu trúc**: UnigramTagger → DefaultTagger (NOUN)
- **Nguyên lý**: Gán nhãn dựa trên từ đơn lẻ, nếu không tìm thấy thì gán nhãn mặc định là NOUN
- **Ưu điểm**: Đơn giản, nhanh
- **Nhược điểm**: Không xét ngữ cảnh

#### Tagger 2: Trigram Tagger với Backoff Chain
- **Cấu trúc**: TrigramTagger → BigramTagger → UnigramTagger → DefaultTagger (NOUN)
- **Nguyên lý**: Gán nhãn dựa trên 3 từ liên tiếp, nếu không có thì thử 2 từ, rồi 1 từ, cuối cùng là mặc định
- **Ưu điểm**: Xét ngữ cảnh tốt hơn, độ chính xác cao hơn
- **Nhược điểm**: Phức tạp hơn, cần nhiều dữ liệu training hơn

### 3. Các độ đo đánh giá

Chương trình tính toán các độ đo sau cho mỗi tagger:

- **Precision (Độ chính xác)**: Tỷ lệ các nhãn được gán đúng trong số các nhãn đã gán
- **Recall (Độ phủ)**: Tỷ lệ các nhãn đúng được tìm thấy trong tổng số nhãn đúng
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **Macro-F1**: Trung bình F1-Score của tất cả các lớp (không tính trọng số)
- **Accuracy**: Tỷ lệ tổng thể các từ được gán nhãn đúng

## Kết quả

### So sánh tổng quan

| Độ đo | Tagger 1 (Unigram) | Tagger 2 (Trigram) |
|-------|--------------------|--------------------|
| Accuracy | 93.66% | 94.41% |
| Macro Precision | 0.8973 | 0.9026 |
| Macro Recall | 0.8646 | 0.8631 |
| Macro F1-Score | 0.8660 | 0.8674 |

### Kết luận

**Tagger 2 (Trigram)** hoạt động tốt hơn với:
- Macro F1-Score: 0.8674 (so với 0.8660 của Unigram)
- Accuracy: 94.41% (so với 93.66% của Unigram)

Trigram tagger có hiệu suất tốt hơn vì nó xét ngữ cảnh của 3 từ liên tiếp, giúp phân biệt tốt hơn các trường hợp từ có thể có nhiều từ loại khác nhau tùy thuộc vào vị trí trong câu.

## Cấu trúc file

```
assignment-2/
├── pos_tagging_evaluation.py  # Script chính
├── requirements.txt            # Dependencies
└── README.md                   # Tài liệu này
```

## Universal POS Tags

Brown Corpus sử dụng 12 nhãn từ loại phổ quát:
- `.` - Dấu câu
- `ADJ` - Tính từ
- `ADP` - Giới từ
- `ADV` - Trạng từ
- `CONJ` - Liên từ
- `DET` - Từ hạn định
- `NOUN` - Danh từ
- `NUM` - Số
- `PRON` - Đại từ
- `PRT` - Tiểu từ
- `VERB` - Động từ
- `X` - Khác
