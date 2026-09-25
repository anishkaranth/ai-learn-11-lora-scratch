# AI Learn 11 — LoRA from Scratch (NumPy)

Implement **Low-Rank Adaptation (LoRA)** by hand. Pretrain a tiny MLP on task A, then adapt it to a *shifted* task B by freezing the base weights and training only low-rank adapters `A` and `B`. Sweep the rank (1, 2, 4, 8), compare against a full fine-tune on accuracy and trainable-parameter count, and merge the adapter back into the weights.

Phase B (AI components) follows the eval harness (`ai-learn-10`). Everything is NumPy with manual backprop and a hand-written Adam. There's no autograd and no network access.

## Learning goals

- **LoRA forward**: `y = xW + b + (alpha/r) · (xA)B`, with `W` frozen, `A ∈ R^{d_in×r}` random, and `B ∈ R^{r×d_out}` zero, so the adapted model starts exactly at the base
- **LoRA backward**: `dB = s·(xA)ᵀ dY` and `dA = s·xᵀ (dY Bᵀ)`, and the base weights get no update
- **Parameter efficiency**: `r·(d_in + d_out)` trainable params per layer instead of `d_in·d_out`
- **Rank sweep**: accuracy vs rank vs trainable params, compared with a full fine-tune and with training from scratch on the small task-B set
- **Merging**: `W' = W + (alpha/r)·A·B` gives identical logits (to float precision) with zero inference overhead
- **No forgetting**: the base stays bit-identical, so turning the adapter off restores task A exactly

## Brief architecture

```
task A (n=8000) ──► pretrain MLP 16→64→4 (all weights) ──► frozen base W1,b1,W2,b2
                                                              │
task B (n=200, rank-2 shifted teacher) ──► train A1,B1,A2,B2 only (r ∈ {1,2,4,8})
                                                              │
                        ┌─────────────────────────────────────┼──────────────────────┐
                        ▼                                     ▼                      ▼
                eval on task B                    merge W' = W + A·B        adapter off → task A
                        ▲
full fine-tune / train-from-scratch baselines on the same 200 task-B samples
```

The distribution shift is itself low rank: task B's teacher = task A's teacher plus a rank-2 update of its first layer. That's the setting the LoRA "low intrinsic rank" hypothesis is about.

> Note: the MLP is tiny (1,348 params), so r=8 already costs about 88% of a full fine-tune. On real transformer layers (d = 4096), rank 8 is about 0.4% of the layer's parameters. The formula is the same; only the ratio changes.

## Layout

```
tasks.py               # teacher MLPs, task A and low-rank-shifted task B
lora.py                # MLP, LoRA adapters, manual backprop, Adam, train, merge_lora
smoke_plots.py         # matplotlib SVG plots + RESULTS.md
svg_utils.py           # small SVG minifier (keeps plots text-friendly)
run_smoke.py           # pretrain -> rank sweep -> full FT -> merge check -> results/
notebooks/lora_scratch.ipynb
results/               # committed RESULTS.md, metrics.json, JSON.shot, *.svg
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in about 1–2 seconds with seed 42. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next

A toy CLIP-style multimodal model (`ai-learn-12`), guardrails and structured output (`ai-learn-13`), and then the Phase C end-to-end assistant milestone (`ai-learn-14`).
