import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    parser = argparse.ArgumentParser(description="只使用训练集训练 BBPE tokenizer")
    parser.add_argument("--input", default="data/tang/corpus_train.txt")
    parser.add_argument("--output", default="tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([args.input], trainer)
    tokenizer.save(args.output)
    print("词表大小：", tokenizer.get_vocab_size())
    out = tokenizer.encode("床前明月光")
    print("编码：", out.ids)
    print("解码：", tokenizer.decode(out.ids))


if __name__ == "__main__":
    main()
