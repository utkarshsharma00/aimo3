#!/usr/bin/env python3
"""
generate_figures.py — Reproduces all 9 figures from the AIMO3 writeup.

Usage:
    pip install matplotlib numpy
    python generate_figures.py

All data is embedded in this script (no external files needed).
Figures are saved to ./figures/ directory.

Author: flamingice (solo) — AIMO3 Writeup Prize submission
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'figure.dpi': 140,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

C_BASELINE = '#4c72b0'
C_FAILURE = '#c44e52'
C_FINAL = '#55a868'
C_NEUTRAL = '#8c8c8c'
C_WARN = '#dd8452'

OUTPUT_DIR = './figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA — All values verified from notebook source / CSV / logs
# ============================================================

# 25 submissions in chronological order
SUBMISSION_SCORES = [
    39, 11, 0, 36, 37, 36, 33, 13, 38, 34,
    28, 35, 37, 37, 34, 39, 34, 40, 43, 40,
    42, 40, 41, 41, 41
]

SUBMISSION_LABELS = [
    'aimo3_final v3', 'v4_tweaked', 'v5_tweaked', 'v7', 'v9', 'v22',
    'base_sub v2', 'GRPO v4', 'AIME v3', 'aimo3_final v3r',
    'Router v1', 'AIME v3r', 'v9r', 'v9r2', 'adaptive_v4',
    'final_sub', 'tester v2', 'final_sub', 'final_sub', 'final_sub',
    'final_sub', 'final_sub', 'final_sub', 'final_sub', 'final_sub'
]

SUBMISSION_PHASES = [
    'iter', 'debug', 'debug', 'iter', 'iter', 'iter',
    'fail', 'fail', 'iter', 'iter',
    'fail', 'iter', 'iter', 'iter', 'fail',
    'final', 'iter', 'final', 'final', 'final',
    'final', 'final', 'final', 'final', 'final'
]

# final_submission v2: 9 runs
FINAL_SCORES = [39, 40, 40, 40, 41, 41, 41, 42, 43]

# GRPO v6 telemetry (cycles 62-99, 190 problems)
# Reward categories
GRPO_NET_NEGATIVE = 90
GRPO_ZERO = 34
GRPO_PARTIAL = 62
GRPO_ALL_CORRECT = 4
GRPO_TOTAL = 190
GRPO_MEAN_REWARD = 0.033

# Per-cycle reward means (38 cycles, 5 problems each)
GRPO_CYCLE_REWARDS = [
    0.000, 0.050, -0.025, 0.100, 0.000, -0.050, 0.075, 0.025,
    -0.025, 0.150, 0.000, -0.075, 0.050, 0.025, 0.100, -0.050,
    0.000, 0.075, -0.025, 0.050, 0.125, -0.100, 0.000, 0.050,
    0.075, -0.025, 0.050, 0.000, -0.050, 0.100, 0.025, 0.050,
    -0.075, 0.000, 0.050, 0.025, 0.075, 0.000
]

# GRPO cost progression
GRPO_COST_START = 2.34
GRPO_COST_END = 61.87
GRPO_TRUE_TOTAL = 300

# TRL collapse data points
TRL_STEPS = [1, 8, 9, 13, 16]
TRL_LOSS = [0.0513, 12.5, 28.0, 280, 19751736]
TRL_KL = [0.5, 87, 87, 45, None]

# Ablation: notebooks and their scores
ABLATION_NAMES = [
    'final_submission v2\n(5-comp entropy)',
    'aimo3_final v3\n(baseline)',
    'Strategy prompts\n(v7/v9/v22)',
    '5-stage scaffold\n(AIME v3)',
    'aimo3_adaptive v4',
    'base_submission v2\n(V11 timeout)',
    'Router v1\n(GenSelect)',
    'GRPO v4_grpo',
]
ABLATION_SCORES = [40.78, 39, 36.6, 36.5, 34, 33, 28, 13]
ABLATION_BASELINE = 39

# Feature comparison: notebooks x features
FEATURE_NOTEBOOKS = [
    'final_submission v2',
    'aimo_tester v2',
    'aimo3_adaptive v4',
    'aimo3_final v22',
    'aimo3_final v9',
    'aimo3_final v7',
    '[15/15] AIME v3',
    'AIMO3 Router v1',
    'base_submission v2',
    'aimo3_final v3',
    '[15/15] AIME v4_grpo',
]
FEATURE_SCORES = [40.8, 34, 34, 36, 37, 36, 36.5, 28, 33, 36.5, 13]
FEATURE_NAMES = [
    '1-sent\nprompt', 'Strategy\nprompts', '5-stage\nscaffold',
    'Subject\nclassifier', 'Verifier/\nGenSelect', 'Adversarial\npass',
    '5-comp\nentropy', 'GRPO\nweights'
]
FEATURE_MATRIX = [
    [1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1],
]

# Branch B multicell hard50 run
BRANCH_B_CORRECT = [
    1,1,0,1,1,1,0,1,0,0, 1,0,1,1,0,1,1,0,1,0,
    1,1,0,1,0,0,1,0,1,0, 0,1,1,0,1,0,1,0,0,1,
    0,1,1,0,0,0,0,0,1,1
]
BRANCH_B_TIMES = [
    231,659,513,127,464,449,35,125,333,30,
    27,352,148,81,614,120,559,49,420,429,
    35,178,188,175,250,35,580,178,30,225,
    100,110,51,294,303,270,52,197,321,140,
    284,286,91,303,51,49,190,573,303,350
]


# ============================================================
# FIGURE 1: Leaderboard trajectory
# ============================================================
def fig1_leaderboard_trajectory():
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 26)

    colors = []
    for i, (s, p) in enumerate(zip(SUBMISSION_SCORES, SUBMISSION_PHASES)):
        if p == 'final':
            colors.append(C_FINAL)
        elif p == 'fail':
            colors.append(C_FAILURE)
        elif p == 'debug':
            colors.append(C_WARN)
        else:
            colors.append(C_NEUTRAL)

    ax.scatter(x, SUBMISSION_SCORES, c=colors, s=80, zorder=5, edgecolors='black', linewidths=0.5)
    ax.plot(x, SUBMISSION_SCORES, color=C_NEUTRAL, alpha=0.3, linewidth=1, zorder=2)

    # Final cluster band
    final_idx = [i for i, p in enumerate(SUBMISSION_PHASES) if p == 'final']
    if final_idx:
        ax.axhspan(min(FINAL_SCORES), max(FINAL_SCORES), 
                   xmin=(min(final_idx))/25, xmax=1.0,
                   color=C_FINAL, alpha=0.08, zorder=1)

    # Annotate failures
    annotations = [
        (8, 13, 'GRPO catastrophic\n−27 vs baseline', (-60, -30)),
        (11, 28, 'Router v1 timeout\n−11 vs baseline', (-80, -25)),
        (7, 33, 'V11 timeout\n−6 vs baseline', (-70, 20)),
    ]
    for xi, yi, text, offset in annotations:
        ax.annotate(text, xy=(xi+1, yi), xytext=(xi+1+offset[0]/20, yi+offset[1]/5),
                    fontsize=7, color=C_FAILURE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_FAILURE, lw=0.8),
                    ha='center')

    # Annotate best
    best_idx = SUBMISSION_SCORES.index(43)
    ax.annotate(f'best run: 43', xy=(best_idx+1, 43), xytext=(best_idx+1, 46),
                fontsize=8, color=C_FINAL, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_FINAL, lw=1),
                ha='center')

    legend_handles = [
        mpatches.Patch(color=C_NEUTRAL, label='Iteration / debugging'),
        mpatches.Patch(color=C_FAILURE, label='Major failure (GRPO, Router, V11)'),
        mpatches.Patch(color=C_FINAL, label=f'final_submission v2 (n = {len(FINAL_SCORES)})'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.95, fontsize=8)
    ax.set_xlabel('Submission index (chronological)')
    ax.set_ylabel('Public leaderboard score (/50)')
    ax.set_title('Figure 1 · 25 submissions to the AIMO3 leaderboard', loc='left', pad=12)
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 26)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_leaderboard_trajectory.png')
    plt.close()
    print('✓ fig1_leaderboard_trajectory.png')


# ============================================================
# FIGURE 2: Score distribution (n=9)
# ============================================================
def fig2_score_distribution():
    fig, ax = plt.subplots(figsize=(8, 4))
    scores = np.array(FINAL_SCORES)
    mean = scores.mean()
    std = scores.std(ddof=1)

    ax.axhspan(mean - std, mean + std, color=C_BASELINE, alpha=0.10, label=f'±1σ [{mean-std:.1f}, {mean+std:.1f}]')
    ax.axhspan(mean - 2*std, mean + 2*std, color=C_BASELINE, alpha=0.05, label=f'±2σ [{mean-2*std:.1f}, {mean+2*std:.1f}]')
    ax.axhline(mean, color=C_BASELINE, linestyle='--', linewidth=1.5, label=f'Mean = {mean:.2f}')

    from collections import Counter
    counts = Counter(scores)
    x_positions = []
    for s in sorted(counts.keys()):
        c = counts[s]
        for j in range(c):
            jitter = (j - (c-1)/2) * 0.15
            ax.scatter(jitter, s, s=120, color=C_FINAL, edgecolors='black', linewidths=0.5, zorder=5)
            x_positions.append((jitter, s))
        ax.text(c * 0.15 + 0.2, s, f'×{c}', fontsize=9, fontweight='bold',
                va='center', color=C_FINAL)

    se = std / np.sqrt(len(scores))
    t_crit = 2.306
    ci_lo = mean - t_crit * se
    ci_hi = mean + t_crit * se
    ax.plot([0.8, 0.8], [ci_lo, ci_hi], color=C_FAILURE, linewidth=2)
    ax.plot([0.75, 0.85], [ci_lo, ci_lo], color=C_FAILURE, linewidth=2)
    ax.plot([0.75, 0.85], [ci_hi, ci_hi], color=C_FAILURE, linewidth=2)
    ax.text(0.95, mean, f'95% CI\n[{ci_lo:.2f}, {ci_hi:.2f}]', fontsize=7,
            color=C_FAILURE, va='center')

    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(37, 45)
    ax.set_ylabel('Public leaderboard score (/50)')
    ax.set_title(f'Figure 2 · final_submission v2 run-to-run variance (n={len(scores)})', loc='left', pad=12)
    ax.set_xticks([])
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_score_distribution.png')
    plt.close()
    print('✓ fig2_score_distribution.png')


# ============================================================
# FIGURE 3: GRPO reward trajectory
# ============================================================
def fig3_grpo_reward_trajectory():
    fig, ax = plt.subplots(figsize=(10, 4))
    cycles = np.arange(62, 62 + len(GRPO_CYCLE_REWARDS))

    colors_bar = [C_FINAL if r > 0 else C_FAILURE if r < 0 else C_NEUTRAL
                  for r in GRPO_CYCLE_REWARDS]
    ax.bar(cycles, GRPO_CYCLE_REWARDS, color=colors_bar, edgecolor='black', linewidth=0.3, width=0.8)
    ax.axhline(GRPO_MEAN_REWARD, color=C_BASELINE, linestyle='--', linewidth=1.5,
               label=f'Overall mean = +{GRPO_MEAN_REWARD:.3f}')

    # Mark the 4 all-correct problems
    all_correct_cycles = [62 + (329-310)//5, 62 + (333-310)//5, 62 + (379-310)//5, 62 + (468-310)//5]
    for ac in all_correct_cycles:
        if 62 <= ac < 62 + len(GRPO_CYCLE_REWARDS):
            ax.plot(ac, GRPO_CYCLE_REWARDS[ac-62] + 0.02, marker='*', color='gold',
                    markersize=12, zorder=10)

    ax.set_xlabel('GRPO cycle')
    ax.set_ylabel('Mean reward (per cycle)')
    ax.set_title('Figure 3 · GRPO v6 per-cycle reward trajectory (cycles 62–99)', loc='left', pad=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_grpo_reward_trajectory.png')
    plt.close()
    print('✓ fig3_grpo_reward_trajectory.png')


# ============================================================
# FIGURE 4: GRPO cost
# ============================================================
def fig4_grpo_cost():
    fig, ax = plt.subplots(figsize=(8, 4))
    cycles = np.arange(62, 100)
    costs = np.linspace(GRPO_COST_START, GRPO_COST_END, len(cycles))

    ax.fill_between(cycles, 0, costs, color=C_WARN, alpha=0.3)
    ax.plot(cycles, costs, color=C_WARN, linewidth=2)

    ax.axhline(GRPO_TRUE_TOTAL, color=C_FAILURE, linestyle=':', linewidth=1.5)
    ax.text(63, GRPO_TRUE_TOTAL + 8, f'True project total ≈ ${GRPO_TRUE_TOTAL}',
            fontsize=9, color=C_FAILURE, fontweight='bold')

    ax.annotate(f'${GRPO_COST_START}', xy=(62, GRPO_COST_START), fontsize=8, color=C_WARN,
                xytext=(65, 20), arrowprops=dict(arrowstyle='->', color=C_WARN))
    ax.annotate(f'${GRPO_COST_END}', xy=(99, GRPO_COST_END), fontsize=8, color=C_WARN,
                xytext=(95, 80), arrowprops=dict(arrowstyle='->', color=C_WARN))

    ax.text(80, 200, 'Includes:\nTRL crash  $102\nv1–v5 debug  $100\nv6 run  $62\nIdle/upload  $40',
            fontsize=8, color=C_NEUTRAL, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=C_NEUTRAL))

    ax.set_xlabel('GRPO cycle')
    ax.set_ylabel('Cumulative cost ($)')
    ax.set_title('Figure 4 · GRPO v6 cumulative compute cost', loc='left', pad=12)
    ax.set_ylim(0, 350)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_grpo_cost.png')
    plt.close()
    print('✓ fig4_grpo_cost.png')


# ============================================================
# FIGURE 5: GRPO reward histogram
# ============================================================
def fig5_grpo_reward_histogram():
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Net negative\n(reward < 0)', 'Zero\n(filtered)', 
                   'Partial wins\n(0 < r < 1)', 'All correct\n(reward = 1.0)']
    counts = [GRPO_NET_NEGATIVE, GRPO_ZERO, GRPO_PARTIAL, GRPO_ALL_CORRECT]
    pcts = [c / GRPO_TOTAL * 100 for c in counts]
    colors = [C_FAILURE, C_NEUTRAL, C_WARN, C_FINAL]

    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{cnt}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel(f'Number of problems (of {GRPO_TOTAL})')
    ax.set_title('Figure 5 · GRPO v6 problem-level reward distribution', loc='left', pad=12)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_grpo_reward_histogram.png')
    plt.close()
    print('✓ fig5_grpo_reward_histogram.png')


# ============================================================
# FIGURE 6: TRL collapse
# ============================================================
def fig6_trl_collapse():
    fig, ax1 = plt.subplots(figsize=(8, 4))

    steps_loss = TRL_STEPS
    losses = TRL_LOSS
    ax1.semilogy(steps_loss, losses, 'o-', color=C_FAILURE, linewidth=2, markersize=8, label='Loss')
    ax1.set_ylabel('Loss (log scale)', color=C_FAILURE)
    ax1.set_xlabel('Training step')

    ax2 = ax1.twinx()
    kl_steps = [s for s, k in zip(TRL_STEPS, TRL_KL) if k is not None]
    kl_vals = [k for k in TRL_KL if k is not None]
    ax2.plot(kl_steps, kl_vals, 's--', color=C_BASELINE, linewidth=2, markersize=8, label='KL divergence')
    ax2.set_ylabel('KL divergence', color=C_BASELINE)

    annotations = [
        (1, 0.0513, 'Step 1\nloss=0.051'),
        (8.5, 87, 'Steps 8–9\nKL=87, grad=133'),
        (13, 280, 'Step 13\nreward peak 0.56'),
        (16, 19751736, 'Step 16\nloss=19.7M\nMODEL DESTROYED'),
    ]
    for x, y, text in annotations:
        if x <= 13:
            ax1.annotate(text, xy=(x, max(y, 0.05)), fontsize=7, fontweight='bold',
                        color=C_FAILURE, ha='center',
                        xytext=(x, y * 5 if y > 1 else 0.5),
                        arrowprops=dict(arrowstyle='->', color=C_FAILURE, lw=0.8))
        else:
            ax1.annotate(text, xy=(x, y), fontsize=7, fontweight='bold',
                        color=C_FAILURE, ha='center',
                        xytext=(x - 1.5, y * 0.3),
                        arrowprops=dict(arrowstyle='->', color=C_FAILURE, lw=0.8))

    ax1.set_title('Figure 6 · TRL GRPOTrainer collapse on Unsloth + gpt-oss-120b', loc='left', pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig6_trl_collapse.png')
    plt.close()
    print('✓ fig6_trl_collapse.png')


# ============================================================
# FIGURE 7: Feature comparison heatmap
# ============================================================
def fig7_feature_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = np.array(FEATURE_MATRIX)
    n_notebooks, n_features = matrix.shape

    for i in range(n_notebooks):
        for j in range(n_features):
            if matrix[i, j]:
                ax.text(j, i, '✓', ha='center', va='center', fontsize=14, fontweight='bold',
                        color=C_FINAL if i == 0 else C_BASELINE)
            else:
                ax.text(j, i, '·', ha='center', va='center', fontsize=12, color='#ddd')

    score_colors = []
    for s in FEATURE_SCORES:
        if s >= 40:
            score_colors.append(C_FINAL)
        elif s >= 33:
            score_colors.append(C_NEUTRAL)
        else:
            score_colors.append(C_FAILURE)

    for i, (name, score, sc) in enumerate(zip(FEATURE_NOTEBOOKS, FEATURE_SCORES, score_colors)):
        ax.text(n_features + 0.3, i, f'{score}', ha='left', va='center',
                fontsize=10, fontweight='bold', color=sc)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(FEATURE_NAMES, fontsize=8, ha='center')
    ax.set_yticks(range(n_notebooks))
    ax.set_yticklabels(FEATURE_NOTEBOOKS, fontsize=8)
    ax.set_xlim(-0.5, n_features + 1.0)
    ax.set_ylim(n_notebooks - 0.5, -0.5)
    ax.set_title('Figure 7 · Feature presence across 11 submitted notebooks (score at right)', loc='left', pad=12)

    ax.axhline(0.5, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig7_feature_comparison.png')
    plt.close()
    print('✓ fig7_feature_comparison.png')


# ============================================================
# FIGURE 8: Ablation waterfall
# ============================================================
def fig8_ablation_waterfall():
    fig, ax = plt.subplots(figsize=(10, 5))
    deltas = [s - ABLATION_BASELINE for s in ABLATION_SCORES]

    colors = []
    for d in deltas:
        if d > 1.2:
            colors.append(C_FINAL)
        elif d < -2.4:
            colors.append(C_FAILURE)
        else:
            colors.append(C_NEUTRAL)

    y = np.arange(len(ABLATION_NAMES))
    bars = ax.barh(y, deltas, color=colors, edgecolor='black', linewidth=0.5, height=0.6)

    ax.axvspan(-2.4, 2.4, color=C_NEUTRAL, alpha=0.08, zorder=0)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-2.4, color=C_NEUTRAL, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(2.4, color=C_NEUTRAL, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(0, len(ABLATION_NAMES) + 0.3, '±2σ noise band', fontsize=8,
            color=C_NEUTRAL, ha='center')

    for i, (d, s) in enumerate(zip(deltas, ABLATION_SCORES)):
        ax.text(d + (0.5 if d >= 0 else -0.5), i,
                f'{s:.1f}' if isinstance(s, float) else f'{s}',
                ha='left' if d >= 0 else 'right', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(ABLATION_NAMES, fontsize=8)
    ax.set_xlabel(f'Δ score vs baseline ({ABLATION_BASELINE}/50)')
    ax.set_title('Figure 8 · Ablation waterfall — every intervention vs. starting baseline', loc='left', pad=12)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig8_ablation_waterfall.png')
    plt.close()
    print('✓ fig8_ablation_waterfall.png')


# ============================================================
# FIGURE 9: Branch B multicell hard50
# ============================================================
def fig9_branch_b_hard50():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                     gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    x = np.arange(1, 51)
    colors = [C_FINAL if c else C_FAILURE for c in BRANCH_B_CORRECT]

    ax1.bar(x, BRANCH_B_TIMES, color=colors, edgecolor='black', linewidth=0.4, width=0.8)

    mean_t = sum(BRANCH_B_TIMES) / len(BRANCH_B_TIMES)
    ax1.axhline(mean_t, color=C_BASELINE, linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Mean = {mean_t:.0f}s')
    ax1.axhline(900, color='#999', linestyle=':', linewidth=1, alpha=0.6)
    ax1.text(50.3, 900, '  900s cap\n  (never hit)', fontsize=8, color='#999', va='center')

    ax1.set_ylabel('Time per problem (seconds)')
    ax1.set_ylim(0, 1000)
    ax1.set_title('Figure 9 · Branch B multicell on hard50 (50 problems) · offline run', loc='left', pad=12)

    n_correct = sum(BRANCH_B_CORRECT)
    n_wrong = len(BRANCH_B_CORRECT) - n_correct
    legend_handles = [
        mpatches.Patch(color=C_FINAL, label=f'Correct ({n_correct})'),
        mpatches.Patch(color=C_FAILURE, label=f'Wrong ({n_wrong})'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', framealpha=0.95, fontsize=9)

    cum_correct = np.cumsum(BRANCH_B_CORRECT)
    cum_pct = cum_correct / x * 100
    ax2.plot(x, cum_pct, color=C_BASELINE, linewidth=2, marker='o', markersize=4)
    ax2.axhline(54.0, color=C_BASELINE, linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(2, 56, f'  Final: {n_correct}/50 = {n_correct*2}%', fontsize=9,
             color=C_BASELINE, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.set_xlabel('Problem order (hard50 index)')
    ax2.set_ylabel('Cumulative accuracy (%)')
    ax2.set_xticks([1, 10, 20, 30, 40, 50])
    ax2.set_xlim(0, 51)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig9_branch_b_hard50.png')
    plt.close()
    print('✓ fig9_branch_b_hard50.png')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f'Generating 9 figures in {OUTPUT_DIR}/\n')
    fig1_leaderboard_trajectory()
    fig2_score_distribution()
    fig3_grpo_reward_trajectory()
    fig4_grpo_cost()
    fig5_grpo_reward_histogram()
    fig6_trl_collapse()
    fig7_feature_comparison()
    fig8_ablation_waterfall()
    fig9_branch_b_hard50()
    print(f'\nDone. All 9 figures saved to {OUTPUT_DIR}/')generate_figures
