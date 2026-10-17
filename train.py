import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from model import MiniLLM


MODEL_CONFIG = dict(vocab_size=8000, dim=512, n_layers=8, n_heads=8,
                    n_kv_heads=2, ffn_hidden=1536, max_seq_len=256)
FORMAT_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(description="训练古诗 MiniLLM（最佳模型与早停）")
    parser.add_argument("--train-data", default="data/tang/train_ids.npy")
    parser.add_argument("--val-data", default="data/tang/val_ids.npy")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--total-steps", type=int, default=8000)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--decay-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5,
                        help="连续多少次验证未改善后早停；0 表示禁用")
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--save-every", type=int, default=0,
                        help="里程碑保存间隔；0 表示不保存")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", nargs="?", const="latest",
                        help="从指定路径恢复；省略路径时使用 checkpoint-dir/latest.pt")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args):
    if args.block_size > MODEL_CONFIG["max_seq_len"]:
        raise ValueError("block-size 不能大于模型 max_seq_len")
    if min(args.total_steps, args.batch_size, args.eval_every, args.eval_batches) <= 0:
        raise ValueError("训练步数、batch size 和评估间隔必须为正数")
    if args.warmup_steps < 0 or args.decay_steps < 0:
        raise ValueError("warmup/decay 步数不能为负数")
    if args.warmup_steps + args.decay_steps > args.total_steps:
        raise ValueError("warmup-steps + decay-steps 不能超过 total-steps")


def load_tokens(path, block_size, label):
    data = np.load(path, mmap_mode="r")
    if data.ndim != 1 or len(data) <= block_size + 1:
        raise ValueError(f"{label}数据必须是一维且长度大于 block-size：{path}")
    return data


def make_fixed_indices(data_len, block_size, batch_size, n_batches, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randint(data_len - block_size, (batch_size,), generator=generator)
            for _ in range(n_batches)]


def batch_from_indices(data, indices, block_size, device):
    starts = indices.tolist()
    x = np.stack([data[i:i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def capture_rng_state(train_generator):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "train_generator": train_generator.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, train_generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, step, model, optimizer, scheduler, args,
                    best_val_loss, patience_best_val_loss, bad_evals, train_generator):
    torch.save({
        "format_version": FORMAT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": MODEL_CONFIG,
        "train_args": vars(args),
        "best_val_loss": best_val_loss,
        "patience_best_val_loss": patience_best_val_loss,
        "bad_evals": bad_evals,
        "rng_state": capture_rng_state(train_generator),
    }, path)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_tokens(args.train_data, args.block_size, "训练")
    val_data = load_tokens(args.val_data, args.block_size, "验证")
    print(f"设备：{device} | 训练 token：{len(train_data):,} | 验证 token：{len(val_data):,}")

    model = MiniLLM(**MODEL_CONFIG).to(device)
    print(f"参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    stable_steps = args.total_steps - args.warmup_steps - args.decay_steps

    def wsd_multiplier(step):
        if args.warmup_steps and step < args.warmup_steps:
            return max(step, 1) / args.warmup_steps
        if step < args.warmup_steps + stable_steps or args.decay_steps == 0:
            return 1.0
        progress = (step - args.warmup_steps - stable_steps) / args.decay_steps
        return max(0.1, 1.0 - 0.9 * min(progress, 1.0))

    scheduler = LambdaLR(optimizer, wsd_multiplier)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    fixed_train = make_fixed_indices(len(train_data), args.block_size, args.batch_size,
                                     args.eval_batches, args.seed + 2)
    fixed_val = make_fixed_indices(len(val_data), args.block_size, args.batch_size,
                                   args.eval_batches, args.seed + 3)

    @torch.no_grad()
    def evaluate(data, fixed_indices):
        model.eval()
        losses = []
        for indices in fixed_indices:
            x, y = batch_from_indices(data, indices, args.block_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, MODEL_CONFIG["vocab_size"]),
                                   y.reshape(-1))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    start_step, best_val_loss, patience_best_val_loss, bad_evals = 0, math.inf, math.inf, 0
    if args.resume:
        checkpoint_path = (checkpoint_dir / "latest.pt"
                           if args.resume == "latest" else Path(args.resume))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("旧 checkpoint 与新初始化/数据流程不兼容，请从头训练")
        if checkpoint.get("config") != MODEL_CONFIG:
            raise ValueError("checkpoint 模型配置与当前配置不一致")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        patience_best_val_loss = float(
            checkpoint.get("patience_best_val_loss", best_val_loss)
        )
        bad_evals = int(checkpoint.get("bad_evals", 0))
        restore_rng_state(checkpoint.get("rng_state"), train_generator)
        print(f"从 {checkpoint_path} 的 step {start_step} 继续")

    log_path = checkpoint_dir / "metrics.csv"
    if start_step == 0 or not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_ppl", "learning_rate"])

    model.train()
    last_step = start_step - 1
    for step in range(start_step, args.total_steps):
        indices = torch.randint(len(train_data) - args.block_size,
                                (args.batch_size,), generator=train_generator)
        x, y = batch_from_indices(train_data, indices, args.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, MODEL_CONFIG["vocab_size"]), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"step {step}: 梯度出现 NaN/Inf，训练终止")
        optimizer.step()
        scheduler.step()
        last_step = step

        if step == 0 or (step + 1) % args.eval_every == 0:
            train_loss = evaluate(train_data, fixed_train)
            val_loss = evaluate(val_data, fixed_val)
            val_ppl = math.exp(min(val_loss, 50.0))
            cur_lr = scheduler.get_last_lr()[0]
            print(f"step {step:5d} | 训练loss {train_loss:.3f} | 验证loss {val_loss:.3f} | "
                  f"困惑度 {val_ppl:.1f} | lr {cur_lr:.2e}")
            improved = val_loss < best_val_loss
            significant_improvement = val_loss < patience_best_val_loss - args.min_delta
            if improved:
                best_val_loss = val_loss
            if significant_improvement:
                patience_best_val_loss, bad_evals = val_loss, 0
            else:
                bad_evals += 1

            save_checkpoint(checkpoint_dir / "latest.pt", step, model, optimizer,
                            scheduler, args, best_val_loss, patience_best_val_loss,
                            bad_evals, train_generator)
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", step, model, optimizer,
                                scheduler, args, best_val_loss, patience_best_val_loss,
                                bad_evals, train_generator)
                print(f"  已更新最佳模型：val_loss={best_val_loss:.3f}")
            with log_path.open("a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([step, train_loss, val_loss, val_ppl, cur_lr])

            if args.patience and bad_evals >= args.patience:
                print(f"验证集连续 {bad_evals} 次未改善，提前停止。")
                break

        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"step_{step}.pt", step, model, optimizer,
                            scheduler, args, best_val_loss, patience_best_val_loss,
                            bad_evals, train_generator)

    if last_step >= 0:
        save_checkpoint(checkpoint_dir / "latest.pt", last_step, model, optimizer,
                        scheduler, args, best_val_loss, patience_best_val_loss,
                        bad_evals, train_generator)
    print(f"训练结束。最佳验证 loss：{best_val_loss:.3f}")
    print(f"生成时使用：{checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
