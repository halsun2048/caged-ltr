"""Train a real multilingual text-encoder student on MIND-derived clicks.

This deliberately replaces the TF-IDF efficiency baseline with a trainable
Transformer bi-encoder.  It is a text-ranking experiment, not an industrial
ad-log claim: the public community mirror is a small MIND-derived subset.
"""
from __future__ import annotations

import argparse, ast, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def parse_list(value):
    if isinstance(value, list): return value
    try:
        out = ast.literal_eval(str(value))
        return out if isinstance(out, list) else [str(value)]
    except Exception:
        return [x.strip() for x in str(value).split("\n") if x.strip()]


class BiEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__(); self.encoder = AutoModel.from_pretrained(name)
    def embed(self, **batch):
        out = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        return nn.functional.normalize((out * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)


def encode(tokenizer, texts, device, max_length):
    return {k: v.to(device) for k, v in tokenizer(texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt").items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/external/mind/sproos_mindsmall_tr/train.parquet")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--output", default="reports/experiments/mind_text_encoder_student.json")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(args.data)
    rows = []
    for _, r in df.iterrows():
        history = parse_list(r["query"]); q = " [SEP] ".join(map(str, history[-10:]))
        pos = str(r["positive"]); negs = parse_list(r["negative"])
        rows.append((q, pos, negs[0] if negs else ""))
    random.Random(args.seed).shuffle(rows); split = max(1, int(len(rows) * .8)); train, valid = rows[:split], rows[split:]
    tok = AutoTokenizer.from_pretrained(args.model); model = BiEncoder(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); best = -1.; best_epoch = 0
    def score(batch_rows):
        qs, ps, ns = zip(*batch_rows)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            qv = model.embed(**encode(tok, list(qs), device, args.max_length)); pv = model.embed(**encode(tok, list(ps), device, args.max_length)); nv = model.embed(**encode(tok, list(ns), device, args.max_length))
        return (qv * pv).sum(-1), (qv * nv).sum(-1)
    def evaluate(rows_):
        model.eval(); hit = []
        with torch.no_grad():
            for i in range(0, len(rows_), args.batch_size):
                p, n = score(rows_[i:i+args.batch_size]); hit.extend((p > n).float().cpu().tolist())
        return float(np.mean(hit))
    start = time.time()
    for ep in range(args.epochs):
        model.train(); random.shuffle(train); losses=[]
        for i in range(0, len(train), args.batch_size):
            p, n = score(train[i:i+args.batch_size]); loss = -torch.log(torch.sigmoid(p-n).clamp_min(1e-7)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            if args.progress and (i // args.batch_size) % 10 == 0: print(f"epoch={ep+1}/{args.epochs} batch={min(i+args.batch_size,len(train))}/{len(train)} loss={losses[-1]:.4f}", flush=True)
        acc = evaluate(valid); print(f"[done] epoch={ep+1}/{args.epochs} train_loss={np.mean(losses):.4f} valid_pair_acc={acc:.4f}", flush=True)
        if acc > best: best, best_epoch = acc, ep + 1
    payload = {"schema":"mind_text_encoder_student_v1", "data":"MIND-derived sproos/mindsmall-tr", "rows":len(rows), "train_rows":len(train), "valid_rows":len(valid), "model":args.model, "device":str(device), "epochs":args.epochs, "best_epoch":best_epoch, "valid_pair_accuracy":best, "elapsed_seconds":round(time.time()-start,2), "tfidf_replacement":True}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    Path(args.output).with_suffix('.md').write_text("# MIND text encoder student\n\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False))
if __name__ == "__main__": main()
