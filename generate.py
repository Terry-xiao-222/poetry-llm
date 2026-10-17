import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import MiniLLM


def top_p_filter(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits[remove] = -float("inf")
    return torch.full_like(logits, -float("inf")).scatter(1, sorted_indices, sorted_logits)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.85,
             top_k=50, top_p=0.9, repetition_penalty=1.1, device="cpu"):
    model.eval()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt 编码后为空")
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(x[:, -model.cos.shape[0]:])[:, -1, :]
        if repetition_penalty != 1.0:
            used_ids = torch.unique(x[0])
            used_logits = logits[:, used_ids]
            logits[:, used_ids] = torch.where(used_logits < 0,
                                               used_logits * repetition_penalty,
                                               used_logits / repetition_penalty)
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            if 0 < top_p < 1:
                logits = top_p_filter(logits, top_p)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == eos_id:
            break
    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="使用最佳 checkpoint 续写古诗")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--prompt", action="append", help="可重复传入")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MiniLLM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])

    for prompt in args.prompt or ["春", "白日", "床前明月光，"]:
        poem = generate(model, tokenizer, prompt, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p,
                        args.repetition_penalty, device)
        print(f"开头「{prompt}」→ {poem}\n")


if __name__ == "__main__":
    main()
