#!/usr/bin/env python3
"""LoRA smoke: pretrain on task A -> adapt to shifted task B (LoRA rank sweep vs full FT) -> merge -> results/."""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np

from lora import accuracy, forward, init_lora, init_mlp, merge_lora, n_params, train
from smoke_plots import make_plots, write_results_md
from tasks import D_IN, N_CLASSES, make_task, make_teachers

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "d_in": D_IN, "d_hidden": 64, "n_classes": N_CLASSES, "n_pretrain": 8000, "n_adapt": 200, "n_test": 2000,
    "pretrain_epochs": 80, "pretrain_lr": 3e-3, "adapt_epochs": 60, "lora_lr": 1e-2, "full_ft_lr": 3e-3,
    "batch_pretrain": 64, "batch_adapt": 32, "ranks": [1, 2, 4, 8], "alpha_equals_rank": True, "shift_rank": 2,
    "shift_scale": 0.7,
}


def _compact(js: str) -> str:
    """Put scalar lists (curves, ranks) on one line."""
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def main() -> None:
    t0 = time.perf_counter()
    T = make_teachers(SEED, shift_rank=CFG["shift_rank"], shift_scale=CFG["shift_scale"])
    XA, yA = make_task(T, "A", CFG["n_pretrain"], 1)
    XAt, yAt = make_task(T, "A", CFG["n_test"], 2)
    XB, yB = make_task(T, "B", CFG["n_adapt"], 3)
    XBt, yBt = make_task(T, "B", CFG["n_test"], 4)

    # 1) pretrain base on task A
    rng = np.random.default_rng(SEED)
    base = init_mlp(CFG["d_in"], CFG["d_hidden"], CFG["n_classes"], rng)
    pre_hist = train(base, XA, yA, epochs=CFG["pretrain_epochs"], lr=CFG["pretrain_lr"], batch=CFG["batch_pretrain"], rng=rng)
    base_snapshot = copy.deepcopy(base)
    pre_acc_A = accuracy(base, XAt, yAt)
    zero_shot_B = accuracy(base, XBt, yBt)
    total_params = n_params(base)

    # 2) LoRA rank sweep (base frozen)
    lora_runs = {}
    for r in CFG["ranks"]:
        ad = init_lora(base, r, np.random.default_rng(100 + r))
        scale = 1.0  # alpha = r  ->  alpha / r = 1
        hist = train(base, XB, yB, epochs=CFG["adapt_epochs"], lr=CFG["lora_lr"], batch=CFG["batch_adapt"],
                     rng=np.random.default_rng(7), lora=ad, scale=scale, train_base=False, X_val=XBt, y_val=yBt)
        merged = merge_lora(base, ad, scale)
        lg_adapter = forward(base, XBt, ad, scale)[0]
        lg_merged = forward(merged, XBt)[0]
        lora_runs[f"lora_r{r}"] = {
            "rank": r, "trainable_params": n_params(ad), "trainable_frac": round(n_params(ad) / total_params, 4),
            "acc_B": accuracy(base, XBt, yBt, ad, scale), "acc_A_with_adapter": accuracy(base, XAt, yAt, ad, scale),
            "acc_A_adapter_off": accuracy(base, XAt, yAt),
            "merged_acc_B": accuracy(merged, XBt, yBt), "merge_max_abs_logit_diff": float(np.abs(lg_adapter - lg_merged).max()),
            "delta_W1_rank": int(np.linalg.matrix_rank(ad["A1"] @ ad["B1"])), "curve": [h["val_acc"] for h in hist],
            "final_train_loss": hist[-1]["train_loss"],
        }
    base_unchanged = all(np.array_equal(base[k], base_snapshot[k]) for k in base)

    # 3) full fine-tune baseline + train-from-scratch baseline
    full = copy.deepcopy(base_snapshot)
    fh = train(full, XB, yB, epochs=CFG["adapt_epochs"], lr=CFG["full_ft_lr"], batch=CFG["batch_adapt"],
               rng=np.random.default_rng(7), X_val=XBt, y_val=yBt)
    scratch = init_mlp(CFG["d_in"], CFG["d_hidden"], CFG["n_classes"], np.random.default_rng(5))
    sh = train(scratch, XB, yB, epochs=CFG["adapt_epochs"], lr=CFG["full_ft_lr"], batch=CFG["batch_adapt"],
               rng=np.random.default_rng(7), X_val=XBt, y_val=yBt)
    baselines = {
        "full_ft": {"trainable_params": total_params, "trainable_frac": 1.0, "acc_B": accuracy(full, XBt, yBt),
                    "acc_A_after": accuracy(full, XAt, yAt), "curve": [h["val_acc"] for h in fh]},
        "scratch_on_B": {"trainable_params": total_params, "trainable_frac": 1.0, "acc_B": accuracy(scratch, XBt, yBt),
                         "curve": [h["val_acc"] for h in sh]},
    }
    best = max(lora_runs.values(), key=lambda d: d["acc_B"])
    runtime = time.perf_counter() - t0

    metrics = {
        "project": "ai-learn-11-lora-scratch", "seed": SEED, "config": CFG, "total_base_params": total_params,
        "pretrain": {"acc_A": pre_acc_A, "final_loss": pre_hist[-1]["train_loss"], "zero_shot_acc_B": zero_shot_B},
        "lora": lora_runs, "baselines": baselines, "base_weights_unchanged_after_lora": bool(base_unchanged),
        "best_lora": {"rank": best["rank"], "acc_B": best["acc_B"], "trainable_params": best["trainable_params"]},
        "runtime_s": round(runtime, 3),
    }
    RESULTS.mkdir(exist_ok=True)
    plots = make_plots(RESULTS, metrics)
    metrics["plots"] = plots
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(metrics, indent=1)), encoding="utf-8")
    shot = {
        "project": metrics["project"], "seed": SEED, "config": CFG,
        "pretrain_acc_A": pre_acc_A, "zero_shot_acc_B": zero_shot_B,
        "lora_acc_B": {k: v["acc_B"] for k, v in lora_runs.items()},
        "lora_trainable_params": {k: v["trainable_params"] for k, v in lora_runs.items()},
        "full_ft_acc_B": baselines["full_ft"]["acc_B"], "full_ft_trainable_params": total_params,
        "scratch_acc_B": baselines["scratch_on_B"]["acc_B"],
        "max_merge_logit_diff": max(v["merge_max_abs_logit_diff"] for v in lora_runs.values()),
        "base_weights_unchanged_after_lora": bool(base_unchanged), "runtime_s": metrics["runtime_s"],
        "pass": bool(best["acc_B"] > zero_shot_B and base_unchanged and all(v["merge_max_abs_logit_diff"] < 1e-9 for v in lora_runs.values())),
    }
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    write_results_md(RESULTS, metrics, plots)

    print(f"pretrain acc A={pre_acc_A:.4f} | zero-shot on B={zero_shot_B:.4f} | base params={total_params}")
    for k, v in lora_runs.items():
        print(f"  {k}: params={v['trainable_params']:5d} ({v['trainable_frac']:.1%}) acc_B={v['acc_B']:.4f} "
              f"merged={v['merged_acc_B']:.4f} merge_diff={v['merge_max_abs_logit_diff']:.1e} A(adapter off)={v['acc_A_adapter_off']:.4f}")
    print(f"  full_ft: params={total_params} acc_B={baselines['full_ft']['acc_B']:.4f} acc_A_after={baselines['full_ft']['acc_A_after']:.4f}")
    print(f"  scratch_on_B: acc_B={baselines['scratch_on_B']['acc_B']:.4f}")
    print(f"runtime {runtime:.2f}s | wrote {RESULTS}")


if __name__ == "__main__":
    main()
