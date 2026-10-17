# prepare_data.py
import argparse
import glob
import json
import random
import re
import unicodedata
from pathlib import Path


ALLOWED_PUNCTUATION = set("，。！？；、：—…·")
NOISE_MARKERS = set("□■�囗〓")


def is_cjk(char):
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def clean_poem(paragraphs):
    """删除校勘括注，并拒绝含缺字、拉丁字符或编辑标记的文本。"""
    text = "".join(str(line) for line in paragraphs)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"（[^（）]*）|\([^()]*\)", "", text)
    text = text.translate(str.maketrans({
        ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："
    }))
    text = re.sub(r"\s+", "", text)

    if not (10 <= len(text) <= 200) or any(c in text for c in NOISE_MARKERS):
        return None
    if any(not (is_cjk(c) or c in ALLOWED_PUNCTUATION) for c in text):
        return None
    if sum(is_cjk(c) for c in text) < 10:
        return None
    return text


def write_lines(path, poems):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for poem in poems:
            f.write(poem + "\n")


def build_corpus(poetry_dir, out_dir, file_pattern="poet.tang.*.json",
                 val_ratio=0.1, seed=42, max_poems=None):
    """清洗、去重并按整首诗固定随机划分训练集和验证集。"""
    seen = set()
    poems = []
    rejected = 0
    pattern = str(Path(poetry_dir) / file_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到数据文件：{pattern}")
    print(f"找到 {len(files)} 个数据文件")

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            records = json.load(f)
        for poem in records:
            text = clean_poem(poem.get("paragraphs", []))
            if text is None or text in seen:
                rejected += 1
                continue
            seen.add(text)
            poems.append(text)
            if max_poems and len(poems) >= max_poems:
                break
        if max_poems and len(poems) >= max_poems:
            break

    if len(poems) < 2:
        raise ValueError("清洗后诗歌数量不足，无法划分数据集")
    rng = random.Random(seed)
    rng.shuffle(poems)
    val_count = max(1, round(len(poems) * val_ratio))
    val_poems = poems[:val_count]
    train_poems = poems[val_count:]

    out_dir = Path(out_dir)
    write_lines(out_dir / "corpus_train.txt", train_poems)
    write_lines(out_dir / "corpus_val.txt", val_poems)
    write_lines(out_dir / "corpus.txt", train_poems + val_poems)
    print(f"保留 {len(poems):,} 首，过滤/去重 {rejected:,} 首")
    print(f"训练 {len(train_poems):,} 首，验证 {len(val_poems):,} 首 → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗并划分全唐诗语料")
    parser.add_argument("--poetry-dir", default="chinese-poetry/全唐诗")
    parser.add_argument("--out-dir", default="data/tang")
    parser.add_argument("--file-pattern", default="poet.tang.*.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-poems", type=int)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio 必须在 0 和 1 之间")
    build_corpus(**vars(args))
