"""
[BT 01] Xây dựng mô hình ngôn ngữ n-gram âm tiết cho tiếng Việt
- Mô hình bigram cấp độ âm tiết
- Tính xác suất câu
- Sinh câu từ mô hình
"""

import sys
import io
import os
import re
import random
import math
import json
import urllib.request
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")

HF_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=phamson02%2Fvietnamese-poetry-corpus"
    "&config=default&split=train"
)


def load_corpus(file_path=None):
    """Tải corpus từ file. Mặc định đọc từ corpus.txt cùng thư mục."""
    path = file_path or DEFAULT_CORPUS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _fetch_hf_page(offset, length=100):
    """Lấy một trang (tối đa 100 dòng) từ HuggingFace Datasets API."""
    url = f"{HF_API}&offset={offset}&length={length}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["rows"]


def load_corpus_from_hf(num_rows=500):
    """
    Tải num_rows bài thơ từ HuggingFace (phamson02/vietnamese-poetry-corpus).
    Trả về chuỗi văn bản, mỗi dòng thơ là một dòng riêng.
    """
    print(f"Đang tải {num_rows} bài thơ từ HuggingFace Dataset...")
    all_lines = []
    batch = 100
    fetched = 0

    for offset in range(0, num_rows, batch):
        length = min(batch, num_rows - offset)
        rows = _fetch_hf_page(offset, length)
        for row in rows:
            content = row["row"]["content"]
            # Chuẩn hóa dấu phân cách dòng thơ "<\n>" thành newline thực
            content = re.sub(r"\s*<\s*\\?n\s*>\s*", "\n", content)
            all_lines.append(content.strip())
        fetched += len(rows)
        print(f"  [{fetched}/{num_rows}] bài đã tải", end="\r")

    print()
    return "\n".join(all_lines)


# ============================================================
# TIỀN XỬ LÝ VĂN BẢN
# ============================================================


def preprocess(text):
    """
    Tách văn bản thành danh sách câu.
    Mỗi câu là danh sách âm tiết bao bởi <s> và </s>.
    """
    text = re.sub(r"\s+", " ", text)
    # Tách câu theo dấu câu hoặc xuống dòng
    raw_sentences = re.split(r"[.!?\n]+", text)

    processed = []
    for sent in raw_sentences:
        sent = sent.strip().lower()
        # Loại bỏ ký tự đặc biệt, giữ lại chữ cái tiếng Việt và khoảng trắng
        sent = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", "", sent)
        sent = sent.strip()
        if not sent:
            continue
        syllables = sent.split()
        if len(syllables) < 2:
            continue
        processed.append(["<s>"] + syllables + ["</s>"])

    return processed


# ============================================================
# MÔ HÌNH BIGRAM
# ============================================================


class BigramLanguageModel:
    """Mô hình ngôn ngữ bigram cấp âm tiết cho tiếng Việt."""

    def __init__(self, smoothing=1.0):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.smoothing = smoothing  # Laplace (add-k) smoothing

    def train(self, sentences):
        """Huấn luyện mô hình từ danh sách câu đã tiền xử lý."""
        for sent in sentences:
            for syllable in sent:
                self.vocab.add(syllable)
                self.unigram_counts[syllable] += 1
            for i in range(len(sent) - 1):
                self.bigram_counts[sent[i]][sent[i + 1]] += 1

        print(f"Tổng quan mô hình:")
        print(f"  - Số câu:       {len(sentences)}")
        print(f"  - Kích thước V:  {len(self.vocab)}")
        print(f"  - Tổng unigram:  {sum(self.unigram_counts.values())}")
        total_bigrams = sum(
            count for counter in self.bigram_counts.values()
            for count in counter.values()
        )
        print(f"  - Tổng bigram:   {total_bigrams}")

    def prob(self, w_prev, w_curr):
        """
        P(w_curr | w_prev) với Laplace smoothing.
        P(w_i | w_{i-1}) = (C(w_{i-1}, w_i) + k) / (C(w_{i-1}) + k * |V|)
        """
        count_bigram = self.bigram_counts[w_prev][w_curr]
        count_prev = self.unigram_counts[w_prev]
        V = len(self.vocab)
        return (count_bigram + self.smoothing) / (count_prev + self.smoothing * V)

    def sentence_log_probability(self, sentence_str):
        """
        Tính log xác suất (log2) của một câu.
        P(S) = P(w1|<s>) * P(w2|w1) * ... * P(</s>|wn)
        """
        syllables = ["<s>"] + sentence_str.strip().lower().split() + ["</s>"]
        log_prob = 0.0
        details = []

        for i in range(len(syllables) - 1):
            w_prev = syllables[i]
            w_curr = syllables[i + 1]
            p = self.prob(w_prev, w_curr)
            log_p = math.log2(p)
            log_prob += log_p
            details.append((w_prev, w_curr, p, log_p))

        return log_prob, 2 ** log_prob, details

    def generate_sentence(self, max_length=20, seed=None):
        """Sinh một câu ngẫu nhiên từ mô hình bigram."""
        if seed is not None:
            random.seed(seed)

        words = []
        current = "<s>"

        for _ in range(max_length):
            candidates = list(self.bigram_counts[current].keys())
            if not candidates:
                break

            weights = [self.bigram_counts[current][c] for c in candidates]
            next_word = random.choices(candidates, weights=weights, k=1)[0]

            if next_word == "</s>":
                break

            words.append(next_word)
            current = next_word

        return " ".join(words)

    def top_bigrams(self, n=10):
        """Hiển thị n bigram phổ biến nhất."""
        all_bigrams = []
        for w1, counter in self.bigram_counts.items():
            for w2, count in counter.items():
                all_bigrams.append((w1, w2, count))
        all_bigrams.sort(key=lambda x: x[2], reverse=True)
        return all_bigrams[:n]

    def perplexity(self, sentence_str):
        """Tính perplexity của câu."""
        syllables = ["<s>"] + sentence_str.strip().lower().split() + ["</s>"]
        N = len(syllables) - 1
        log_prob, _, _ = self.sentence_log_probability(sentence_str)
        return 2 ** (-log_prob / N)


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================


def main():
    print("=" * 65)
    print("  MÔ HÌNH NGÔN NGỮ BIGRAM ÂM TIẾT TIẾNG VIỆT")
    print("=" * 65)

    # --- 1. Tải và tiền xử lý corpus ---
    print("\n[1] TẢI VÀ TIỀN XỬ LÝ CORPUS")
    print("-" * 40)

    use_hf = "--hf" in sys.argv
    num_rows = 500
    if "--num-rows" in sys.argv:
        idx = sys.argv.index("--num-rows")
        try:
            num_rows = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Cảnh báo: --num-rows cần kèm số nguyên. Dùng mặc định 500.")

    if use_hf:
        corpus_text = load_corpus_from_hf(num_rows)
    else:
        corpus_text = load_corpus()

    sentences = preprocess(corpus_text)
    print(f"Số câu sau tiền xử lý: {len(sentences)}")

    # --- 2. Xây dựng mô hình bigram ---
    print(f"\n[2] XÂY DỰNG MÔ HÌNH BIGRAM")
    print("-" * 40)
    model = BigramLanguageModel(smoothing=1.0)
    model.train(sentences)

    print(f"\nTop 15 bigram phổ biến nhất:")
    print(f"  {'Bigram':<30} {'Tần suất':>8}")
    print(f"  {'-'*38}")
    for w1, w2, count in model.top_bigrams(15):
        print(f"  ({w1}, {w2}){' '*(26-len(w1)-len(w2))} {count:>8}")

    # --- 3. Tính xác suất câu ---
    print(f"\n[3] TÍNH XÁC SUẤT CÂU")
    print("-" * 40)

    test_sentence = "Hôm nay trời đẹp lắm"
    log_prob, prob, details = model.sentence_log_probability(test_sentence)

    print(f'\nCâu: "{test_sentence}"')
    print(f"\nChi tiết tính toán P(câu) = ∏ P(w_i | w_{{i-1}}):\n")
    print(f"  {'Bigram':<30} {'P(w_i|w_{{i-1}})':>15} {'log2(P)':>12}")
    print(f"  {'-'*57}")
    for w_prev, w_curr, p, log_p in details:
        bigram_str = f"P({w_curr} | {w_prev})"
        print(f"  {bigram_str:<30} {p:>15.8f} {log_p:>12.4f}")

    print(f"\n  Log2 P(câu)  = {log_prob:.6f}")
    print(f"  P(câu)       = {prob:.2e}")
    print(f"  Perplexity   = {model.perplexity(test_sentence):.4f}")

    # So sánh với một vài câu khác
    print(f"\n--- So sánh xác suất với các câu khác ---\n")
    other_sentences = [
        "Tôi đi học ở trường đại học",
        "Hôm nay trời đẹp lắm",
        "Việt Nam là đất nước xinh đẹp",
        "Cà phê sách bóng đá thể thao",
        "Xanh trời tôi phở ăn bơi",
    ]
    print(f"  {'Câu':<42} {'log2 P(S)':>12} {'Perplexity':>12}")
    print(f"  {'-'*66}")
    for s in other_sentences:
        lp, _, _ = model.sentence_log_probability(s)
        pp = model.perplexity(s)
        print(f"  {s:<42} {lp:>12.4f} {pp:>12.2f}")

    # --- 4. Sinh câu ---
    print(f"\n[4] SINH CÂU TỪ MÔ HÌNH BIGRAM")
    print("-" * 40)
    print()
    for i in range(10):
        generated = model.generate_sentence(max_length=15)
        print(f"  Câu {i+1:>2}: {generated}")

    print()
    print("=" * 65)
    print("  HOÀN TẤT!")
    print("=" * 65)


if __name__ == "__main__":
    main()
