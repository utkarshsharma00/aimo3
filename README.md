# Every Improvement Failed: gpt-oss-120b on AIMO3

**Writeup Prize submission for AI Mathematical Olympiad — Progress Prize 3**

**Team:** flamingice (solo) · **Public LB:** X / X (TBD) · **Mean:** 40.78/50 (σ=1.20, n=9, best 43)

## What this is

A negative results report documenting 25 submissions across 11 notebook configurations, 15 offline research branches, and a $300 GRPO fine-tuning experiment. The only intervention that produced a statistically defensible improvement above run-to-run noise was a 5-component weighted entropy voting function for self-consistency.

- **Writeup:** [`Kaggle Writeup link`](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/every-improvement-failed-aimo3)
- **Paper:** `AIMO3_Writeup_flamingice.md` in this repo

## Model

This submission uses the **unmodified** base model [`danielhanchen/gpt-oss-120b`](https://huggingface.co/danielhanchen/gpt-oss-120b) with **no fine-tuning, no LoRA, no weight patches**. The entire contribution is inference-side.

## Quick start (standalone, any H100)

```bash
# 1. Install dependencies
pip install vllm openai openai_harmony transformers jupyter_client

# 2. Download model (~60GB)
huggingface-cli download danielhanchen/gpt-oss-120b --local-dir ./gpt-oss-120b

# 3. Run on a CSV of problems (columns: id, problem)
python run_inference.py --model_path ./gpt-oss-120b --input problems.csv --output answers.csv

# 4. Or run on a single problem
python run_inference.py --model_path ./gpt-oss-120b \
    --problem "Find the remainder when 2^2025 is divided by 17."
```

### Requirements
- 1× H100 80GB GPU (or equivalent with ≥80GB VRAM)
- ~60GB disk for model weights
- Python 3.10+

## Repository structure

```
├── run_inference.py              # Standalone inference script (Kaggle-agnostic)
├── final-submission.ipynb        # Original Kaggle submission notebook
├── AIMO3_Writeup_flamingice.md   # Full writeup (4,987 words + appendices)
├── figures/                      # All 9 writeup figures
│   ├── fig1_leaderboard_trajectory.png
│   ├── fig2_score_distribution.png
│   ├── fig3_grpo_reward_trajectory.png
│   ├── fig4_grpo_cost.png
│   ├── fig5_grpo_reward_histogram.png
│   ├── fig6_trl_collapse.png
│   ├── fig7_feature_comparison.png
│   ├── fig8_ablation_waterfall.png
│   └── fig9_branch_b_hard50.png
└── README.md
```

## Key technical details

### 5-component weighted entropy voting

The core contribution. Each of 8 parallel attempts produces an answer; we weight each attempt by inverse composite entropy:

```python
weighted_entropy = (
    0.30 * mean_entropy
  + 0.40 * position_weighted_entropy   # decay 0.995, recent tokens dominate
  + 0.20 * std_dev_of_entropy          # variance penalty
  + 0.30 * 3.0 * high_entropy_ratio    # fraction with H > 2.0 bits
  - 0.10 * (max_low_entropy_streak / n) # bonus for sustained confidence
)
```

Full source in `run_inference.py` → `AIMO3Solver._compute_weighted_entropy()`.

### Configuration

| Parameter | Value |
|---|---|
| Model | `danielhanchen/gpt-oss-120b` (MXFP4, no fine-tuning) |
| Attempts | 8 parallel |
| Temperature | 1.0 |
| min_p | 0.02 |
| Early stop | 4 agreements |
| Context | 65,536 tokens |
| KV cache | fp8_e4m3 |
| Time budget | 17,400s (4h50m), adaptive per-problem |
| Reasoning effort | HIGH |

### Provenance

The inverse-entropy voting pattern was inspired by [pawanmali's "Chasing 47/50" notebook](https://www.kaggle.com/code/pawanmali/chasing-47-50-aimo3-journey-of-60-experiments). The 5-component composite formula was developed independently during this competition.

## Supplementary data

- `grpo_v6_steps.csv` — GRPO training telemetry (190 rows × 14 columns)

## Acknowledgements

- XTX Markets and the AIMO Prize for sponsoring the competition
- Simon Frieder for the writeup prize structure
- pawanmali for the entropy-weighting notebook that inspired the voting function
- zfturbo for the hard50 benchmark dataset used for offline evaluation
- The AIMO3 Kaggle discussion community

## License

MIT
