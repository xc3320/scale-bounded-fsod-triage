# Scale-Bounded Triage for Few-Shot Object Detection

Official code and result archives for the paper under review:

> **Decomposing Novel-Class Background Bias in Few-Shot Object Detection: Scale-Dependent Predictability, One-Size-Fits-All Collapse, and Triage-Guided Semantic Calibration**

*(Under review at Pattern Recognition. Citation entry will be added upon acceptance.)*

Fine-tuned few-shot object detectors systematically misclassify novel-class instances as background. This repository contains the code and frozen result archives behind three findings:

1. **A two-component decomposition with a predictability boundary.** The background-decision rate decomposes into a scale-general effect F(s) and a few-shot bias G. The decomposition predicts novel background rates on unseen splits for **medium/large objects (MAE 5.62 pp)** but is statistically indeterminate for **small objects** (n ≤ 44; confidence half-width 11–24 pp exceeds the pre-registered 8 pp criterion) — a characterized *boundary* on where repair is justified.

2. **One-size-fits-all margin injection collapses novel mAP.** A uniform training-time margin, tuned on base-class validation, raises per-class variance and degrades novel mAP from **25.25 → 14.96** — evidence that the two failure components demand different treatments.

3. **Triage-guided, training-free CLIP semantic calibration.** Confining the semantic-side repair to mid/large objects, a training-free CLIP-gated calibration on proposal class logits improves novel mAP on all three VOC splits with fixed λ = 10:

| Split | Baseline novel mAP | + CLIP gate (λ=10) | Gain |
|-------|-------------------:|-------------------:|------|
| 1 | 24.06 | 32.26 | **+8.20 pp** |
| 2 | 38.47 | 40.64 | **+2.17 pp** |
| 3 | 31.32 | 47.34 | **+16.02 pp** |
| mean | 31.28 | 40.08 | **+9.47 ± 4.91 pp** |

Base-class mAP is never degraded on any split. Small objects are honestly excluded as a classification-plus-localization dual bottleneck (oracle ceiling 30.10 vs. 64.90 for large). A COCO probe reproduces the scale gradient and the separability-without-repairability structure, confirming these mechanisms are not VOC artefacts.

## Repository layout

```
fsod/                     detection codebase (torchvision Faster R-CNN + FPN)
  train_base.py           base-class training
  finetune.py             few-shot fine-tuning (expand_classifier)
  eval_clip_fusion.py     forward+CLIP caching and semantic-gate evaluation
  eval_phase5_triage.py   ★ triage vs. one-size-fits-all comparison + label-free λ selection
  probe_phase1_decomposition.py   two-component decomposition (F(s), G) + predictability test
  probe_misalignment.py / probe_misalign_byconfig.py  score-localization misalignment diagnosis
  probe_oracle_cls.py     oracle ceiling (30.10 small / 64.90 large)
  margin_loss.py          uniform margin injection (collapse experiment)
  probe_phase3_coco.py    COCO cross-dataset probe
data/                     frozen result JSONs (numbers cited in the paper)
  phase5_triage_s{1,2,3}.json     triage comparison per split
  phase5_fixed_lambda.json        fixed-λ cross-split robustness check
figures/figure1_mechanism.png     mechanism overview figure
scripts/draw_figure1.py           script that renders the figure above
```

## Requirements

- Python 3.10+
- PyTorch (CUDA), torchvision
- numpy, Pillow
- OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`)

A single consumer GPU with ≥ 12 GB VRAM is sufficient; all reported numbers were produced on an RTX 3060 12 GB.

## Reproducing the triage experiment

```bash
# 1. Base training and 5-shot fine-tuning per split (split ∈ {1,2,3})
PYTHONPATH=. python -m fsod.train_base      --data 0712 --split {1,2,3}
PYTHONPATH=. python -m fsod.finetune        --data 0712 --split {1,2,3} --shot 5 ...

# 2. Build the forward+CLIP cache (one pass, shared by all λ/mode combos)
PYTHONPATH=. python -m fsod.eval_clip_fusion \
    --ckpt runs/ft_s{S}_ctrl_base/ft.pth --data 0712 --split {S} --shot 5 \
    --out data/clip_fusion_cache_full.pt

# 3. Triage vs. one-size-fits-all comparison + label-free λ selection
PYTHONPATH=. python -m fsod.eval_phase5_triage \
    --ckpt runs/ft_s{S}_ctrl_base/ft.pth --data 0712 --split {S} --shot 5 \
    --cache data/clip_fusion_cache_full.pt --out data/phase5_triage_s{S}.json
```

Pre-registered acceptance criteria (frozen before the run, 2026-11-24):

- **D1 (triage validity):** triaged novel mAP ≥ gate baseline at the same λ, and small-object novel AP not increased; any row worse than −1 pp ⇒ FAIL.
- **D2 (label-free λ):** novel mAP drop vs. manually tuned λ ≤ 2 pp, base mAP not degraded.

Reported outcomes (honest reporting includes the failures):

- `tri_hard` (hard scale binning): PASS on D1 (mean +9.57 pp novel, base never degraded).
- `tri_soft` (continuous soft weighting): FAIL on D1 — small-object AP unexpectedly degraded on split 1's tail; reported as-is.
- `lambda_auto` (λ chosen on base classes only): FAIL on D2 — base-optimal λ does not align with novel-optimal λ on all splits; fixed λ = 10 is the reported robust operating point (all gains positive, mean gap to per-split optimum 0.67 pp).

## License

Released under the MIT License. The frozen result JSONs in `data/` are the numeric source for the paper and must not be modified.
