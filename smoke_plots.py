"""Matplotlib SVG plots + RESULTS.md writer for the LoRA smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-11", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
_COLORS = {"lora_r1": "#8ecae6", "lora_r2": "#219ebc", "lora_r4": "#126782", "lora_r8": "#023047",
           "full_ft": "#e76f51", "scratch_on_B": "#adb5bd"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names = []
    # 1) accuracy vs trainable params
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in m["lora"].items():
        ax.scatter(v["trainable_params"], v["acc_B"], s=60, color=_COLORS[k], zorder=3)
        ax.annotate(f"r={v['rank']}", (v["trainable_params"], v["acc_B"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    fb = m["baselines"]["full_ft"]
    ax.scatter(fb["trainable_params"], fb["acc_B"], s=80, marker="s", color=_COLORS["full_ft"], label="full fine-tune", zorder=3)
    ax.axhline(m["pretrain"]["zero_shot_acc_B"], color="#666", ls="--", lw=1, label="base, no adaptation")
    ax.axhline(m["baselines"]["scratch_on_B"]["acc_B"], color="#adb5bd", ls=":", lw=1.2, label="train from scratch on B")
    ax.set_xscale("log")
    ticks = sorted([v["trainable_params"] for v in m["lora"].values()] + [fb["trainable_params"]])
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("trainable parameters (log)")
    ax.set_ylabel("task B test accuracy")
    ax.set_title("LoRA rank sweep vs full fine-tune")
    ax.legend(fontsize=8, loc="lower right")
    names.append(_save(fig, out / "accuracy_vs_params.svg"))
    # 2) validation curves
    fig, ax = plt.subplots(figsize=(6, 4))
    curves = {**{k: v["curve"] for k, v in m["lora"].items()}, "full_ft": fb["curve"], "scratch_on_B": m["baselines"]["scratch_on_B"]["curve"]}
    for k, c in curves.items():
        ax.plot(range(1, len(c) + 1), c, color=_COLORS[k], lw=1.4, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("task B test accuracy")
    ax.set_title("Adaptation curves")
    ax.legend(fontsize=7, ncol=2)
    names.append(_save(fig, out / "adaptation_curves.svg"))
    # 3) retention of task A
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = ["pretrained", "LoRA r4 (adapter off)", "LoRA r4 (adapter on)", "full FT"]
    r4 = m["lora"]["lora_r4"]
    vals = [m["pretrain"]["acc_A"], r4["acc_A_adapter_off"], r4["acc_A_with_adapter"], fb["acc_A_after"]]
    ax.bar(labels, vals, color=["#2a9d8f", "#126782", "#8ecae6", "#e76f51"], edgecolor="#222")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("task A test accuracy")
    ax.set_title("Task A retention: frozen base + removable adapter")
    ax.tick_params(axis="x", labelsize=8)
    names.append(_save(fig, out / "task_a_retention.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c, p, fb, sc = m["config"], m["pretrain"], m["baselines"]["full_ft"], m["baselines"]["scratch_on_B"]
    L = ["# Results -- ai-learn-11-lora-scratch", "",
         f"**Seed:** `{m['seed']}` | MLP {c['d_in']}->{c['d_hidden']}->{c['n_classes']} ({m['total_base_params']} params) | "
         f"pretrain n={c['n_pretrain']} | adapt n={c['n_adapt']} | test n={c['n_test']} | shift = rank-{c['shift_rank']} teacher update", "",
         "## Pretraining (task A)", "", "| Metric | Value |", "|---|---:|",
         f"| Task A test accuracy | {p['acc_A']:.4f} |", f"| Zero-shot accuracy on shifted task B | {p['zero_shot_acc_B']:.4f} |", "",
         "## Adaptation to task B (real smoke run)", "",
         "| Method | Trainable params | % of base | Task B acc | Task A acc after | Merged acc | max merge logit diff |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in m["lora"].items():
        L.append(f"| LoRA r={v['rank']} | {v['trainable_params']} | {100 * v['trainable_frac']:.1f}% | {v['acc_B']:.4f} | "
                 f"{v['acc_A_with_adapter']:.4f} (adapter off: {v['acc_A_adapter_off']:.4f}) | {v['merged_acc_B']:.4f} | {v['merge_max_abs_logit_diff']:.1e} |")
    L += [f"| Full fine-tune | {fb['trainable_params']} | 100.0% | {fb['acc_B']:.4f} | {fb['acc_A_after']:.4f} | - | - |",
          f"| Train from scratch on B | {sc['trainable_params']} | 100.0% | {sc['acc_B']:.4f} | - | - | - |", "",
          f"- Best LoRA: **r={m['best_lora']['rank']}** -> task B acc **{m['best_lora']['acc_B']:.4f}** with {m['best_lora']['trainable_params']} trainable params "
          f"vs full FT {fb['acc_B']:.4f} with {fb['trainable_params']}.",
          f"- Base weights bit-identical after all LoRA runs: **{m['base_weights_unchanged_after_lora']}**, so switching the adapter off restores task A exactly.",
          "- Merging `W' = W + (alpha/r) A B` reproduces the adapter's logits to float precision (see the max diff column), so there's no inference overhead.", "",
          "## Plots", ""] + [f"![{n}]({n})" for n in plots] + ["", f"Wall time: {m['runtime_s']:.2f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
