import argparse
import numpy as np
from tokenizers import Tokenizer


def pack_file(tokenizer, input_path, output_path, eos_id):
    all_ids = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            all_ids.extend(tokenizer.encode(text).ids)
            all_ids.append(eos_id)
    array = np.asarray(all_ids, dtype=np.uint16)
    np.save(output_path, array)
    print(f"{output_path}: {len(array):,} tokens")


def main():
    parser = argparse.ArgumentParser(description="分别打包训练集与验证集")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--train-input", default="data/tang/corpus_train.txt")
    parser.add_argument("--val-input", default="data/tang/corpus_val.txt")
    parser.add_argument("--train-output", default="data/tang/train_ids.npy")
    parser.add_argument("--val-output", default="data/tang/val_ids.npy")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("词表过大，uint16 无法保存 token id")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer 缺少 <|endoftext|>")
    pack_file(tokenizer, args.train_input, args.train_output, eos_id)
    pack_file(tokenizer, args.val_input, args.val_output, eos_id)


if __name__ == "__main__":
    main()
