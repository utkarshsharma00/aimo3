#!/usr/bin/env python3
"""
Sensitivity analysis v2 — calibrated to AIMO3 difficulty distribution.
Models the KEY cases where entropy voting differs from majority vote.
"""
import math, random, numpy as np
from collections import defaultdict, Counter

random.seed(42)
np.random.seed(42)

def compute_weighted_entropy(entropies, weights):
    w_mean, w_pos, w_std, w_her, w_streak = weights
    n = len(entropies)
    if n == 0: return float('inf')
    mean_ent = sum(entropies) / n
    std_dev = math.sqrt(sum((e - mean_ent)**2 for e in entropies) / (n - 1)) if n > 1 else 0.0
    decay = 0.995
    ws = [decay ** (n - 1 - i) for i in range(n)]
    tw = sum(ws)
    pos_weighted = sum(w * e for w, e in zip(ws, entropies)) / tw if tw > 0 else mean_ent
    high_ent_ratio = sum(1 for e in entropies if e > 2.0) / n
    ms, cs = 0, 0
    for e in entropies:
        if e < 1.0: cs += 1; ms = max(ms, cs)
        else: cs = 0
    streak = w_streak * (ms / n)
    return max(w_mean*mean_ent + w_pos*pos_weighted + w_std*std_dev + w_her*high_ent_ratio + streak, 1e-9)

def select_answer(attempts, weights):
    aw = defaultdict(float)
    for ans, eseq in attempts:
        if ans is not None:
            aw[ans] += 1.0 / max(compute_weighted_entropy(eseq, weights), 1e-9)
    return max(aw.items(), key=lambda x: x[1])[0] if aw else None

def majority_vote(attempts):
    answers = [a for a, _ in attempts if a is not None]
    return Counter(answers).most_common(1)[0][0] if answers else None

def gen_ent(profile, n=300):
    """Generate per-token entropy sequences for different solver profiles."""
    if profile == 'low_flat':      # confident throughout
        return list(np.clip(np.random.normal(0.35, 0.10, n), 0.05, 2.0))
    elif profile == 'low_converge':  # explores then converges to low
        e = np.concatenate([np.random.normal(1.2, 0.3, n//2), np.random.normal(0.3, 0.08, n-n//2)])
        return list(np.clip(e, 0.05, 3.0))
    elif profile == 'medium':
        return list(np.clip(np.random.normal(0.75, 0.25, n), 0.05, 3.0))
    elif profile == 'high_noisy':    # uncertain, spiky
        b = np.random.normal(1.3, 0.4, n)
        b[np.random.choice(n, n//8, replace=False)] += 1.0
        return list(np.clip(b, 0.05, 4.0))
    elif profile == 'high_diverge':  # starts ok, diverges at end
        e = np.concatenate([np.random.normal(0.6, 0.2, n//2), np.random.normal(1.8, 0.5, n-n//2)])
        return list(np.clip(e, 0.05, 4.0))

def gen_scenarios(n=1500):
    """
    Model 50 AIMO3 problems × 30 runs = 1500 scenarios.
    Difficulty calibrated to host's pass@8 data.
    KEY: include contested cases where entropy != majority.
    """
    scenarios, labels = [], []
    for _ in range(n):
        # Difficulty distribution: 25 easy, 15 medium, 10 hard per 50 problems
        diff = random.choices(['easy', 'medium', 'hard', 'tie', 'minority_correct'],
                             weights=[0.35, 0.20, 0.10, 0.20, 0.15])[0]
        labels.append(diff)
        C, W1, W2 = 100, 101, 999
        atts = []

        if diff == 'easy':
            # 6-2 or 7-1, correct has better entropy
            nc = random.choice([6, 7, 8])
            for _ in range(nc):
                atts.append((C, gen_ent(random.choice(['low_flat', 'low_converge']))))
            for _ in range(8 - nc):
                atts.append((W1, gen_ent(random.choice(['high_noisy', 'medium']))))

        elif diff == 'medium':
            # 5-3, correct slightly better entropy
            for _ in range(5):
                atts.append((C, gen_ent(random.choice(['low_converge', 'medium']))))
            for _ in range(3):
                atts.append((W1, gen_ent(random.choice(['medium', 'high_noisy']))))

        elif diff == 'hard':
            # 3-5 — correct answer is MINORITY (majority wrong)
            for _ in range(3):
                atts.append((C, gen_ent(random.choice(['low_converge', 'medium']))))
            for _ in range(5):
                atts.append((W1, gen_ent(random.choice(['medium', 'high_noisy']))))

        elif diff == 'tie':
            # 4-4 exact tie — entropy MUST break it
            for _ in range(4):
                atts.append((C, gen_ent(random.choice(['low_flat', 'low_converge']))))
            for _ in range(4):
                atts.append((W1, gen_ent(random.choice(['high_noisy', 'high_diverge']))))

        elif diff == 'minority_correct':
            # 3-3-2 three-way split. Correct has 3 but so does wrong1.
            # Correct attempts have lower entropy (more confident)
            for _ in range(3):
                atts.append((C, gen_ent('low_flat')))
            for _ in range(3):
                atts.append((W1, gen_ent('high_noisy')))
            for _ in range(2):
                atts.append((W2, gen_ent('medium')))

        random.shuffle(atts)
        scenarios.append((atts, C))
    return scenarios, labels

DEFAULT = [0.30, 0.40, 0.20, 0.90, -0.10]
NAMES = ['mean_entropy', 'position_weighted', 'std_dev', 'high_ent_ratio', 'streak_bonus']

print("=" * 70)
print("SENSITIVITY ANALYSIS v2: Calibrated to AIMO3 Difficulty")
print("=" * 70)

scenarios, labels = gen_scenarios(1500)
base_ans = [select_answer(a, DEFAULT) for a, _ in scenarios]
base_ok = sum(1 for i, (_, c) in enumerate(scenarios) if base_ans[i] == c)
base_acc = base_ok / len(scenarios) * 100

maj_ans = [majority_vote(a) for a, _ in scenarios]
maj_ok = sum(1 for i, (_, c) in enumerate(scenarios) if maj_ans[i] == c)
maj_acc = maj_ok / len(scenarios) * 100

print(f"\n5-comp entropy voting:  {base_ok}/{len(scenarios)} ({base_acc:.1f}%)")
print(f"Plain majority vote:    {maj_ok}/{len(scenarios)} ({maj_acc:.1f}%)")
print(f"Entropy advantage:      {base_acc - maj_acc:+.1f}%")

# Breakdown by difficulty
print(f"\n{'Difficulty':<20} {'N':>6} {'5-comp':>10} {'Majority':>10} {'Δ':>8}")
print("-" * 58)
for d in ['easy', 'medium', 'hard', 'tie', 'minority_correct']:
    idx = [i for i, l in enumerate(labels) if l == d]
    n = len(idx)
    ec = sum(1 for i in idx if base_ans[i] == scenarios[i][1])
    mc = sum(1 for i in idx if maj_ans[i] == scenarios[i][1])
    print(f"{d:<20} {n:>6} {ec/n*100:>9.1f}% {mc/n*100:>9.1f}% {(ec-mc)/n*100:>+7.1f}%")

# TEST 1: Individual weight perturbation
print(f"\n{'='*70}")
print("TEST 1: Individual weight perturbation")
print("="*70)
print(f"\n{'Component':<22} {'Factor':>8} {'Flips':>7} {'Flip%':>7} {'Acc':>7} {'ΔAcc':>7}")
print("-" * 62)

for wi, wn in enumerate(NAMES):
    for factor in [0.00, 0.50, 0.75, 1.25, 1.50, 2.00]:
        p = DEFAULT.copy()
        p[wi] = DEFAULT[wi] * factor
        fl, pc = 0, 0
        for i, (att, cor) in enumerate(scenarios):
            ans = select_answer(att, p)
            if ans != base_ans[i]: fl += 1
            if ans == cor: pc += 1
        a = pc / len(scenarios) * 100
        print(f"{wn:<22} {factor:>7.2f}x {fl:>7} {fl/len(scenarios)*100:>6.1f}% {a:>6.1f}% {a-base_acc:>+6.1f}%")
    print()

# Alternative methods
print(f"\n{'='*70}")
print("TEST 2: Alternative voting methods")
print("="*70)

methods = {
    '5-comp (default)': DEFAULT,
    'Position only': [0.0, 1.0, 0.0, 0.0, 0.0],
    'Mean entropy only': [1.0, 0.0, 0.0, 0.0, 0.0],
    'Mean + position': [0.30, 0.40, 0.0, 0.0, 0.0],
    'No streak bonus': [0.30, 0.40, 0.20, 0.90, 0.0],
    'No std penalty': [0.30, 0.40, 0.0, 0.90, -0.10],
    'Double position wt': [0.30, 0.80, 0.20, 0.90, -0.10],
    'Half position wt': [0.30, 0.20, 0.20, 0.90, -0.10],
    'No high_ent_ratio': [0.30, 0.40, 0.20, 0.00, -0.10],
}
print(f"\n{'Method':<25} {'Correct':>9} {'Acc':>7} {'ΔAcc':>7} {'Flips':>7}")
print("-" * 58)
for name, w in methods.items():
    ans_list = [select_answer(a, w) for a, _ in scenarios]
    ok = sum(1 for i, (_, c) in enumerate(scenarios) if ans_list[i] == c)
    fl = sum(1 for i in range(len(scenarios)) if ans_list[i] != base_ans[i])
    a = ok / len(scenarios) * 100
    print(f"{name:<25} {ok:>9} {a:>6.1f}% {a-base_acc:>+6.1f}% {fl:>7}")
print(f"{'Plain majority vote':<25} {maj_ok:>9} {maj_acc:>6.1f}% {maj_acc-base_acc:>+6.1f}% {sum(1 for i in range(len(scenarios)) if maj_ans[i] != base_ans[i]):>7}")

# TEST 3: Monte Carlo
print(f"\n{'='*70}")
print("TEST 3: Monte Carlo (200 random ±50% perturbations, all 5 weights)")
print("="*70)
mc_d, mc_f = [], []
for _ in range(200):
    p = [w * random.uniform(0.5, 1.5) for w in DEFAULT]
    fl, pc = 0, 0
    for i, (att, cor) in enumerate(scenarios):
        ans = select_answer(att, p)
        if ans != base_ans[i]: fl += 1
        if ans == cor: pc += 1
    mc_f.append(fl)
    mc_d.append(pc / len(scenarios) * 100 - base_acc)

mc_f, mc_d = np.array(mc_f), np.array(mc_d)
print(f"\nFlip rate:  mean={mc_f.mean()/len(scenarios)*100:.1f}%, median={np.median(mc_f)/len(scenarios)*100:.1f}%, max={mc_f.max()/len(scenarios)*100:.1f}%")
print(f"Acc delta:  mean={mc_d.mean():+.2f}%, std={mc_d.std():.2f}%, range=[{mc_d.min():+.2f}%, {mc_d.max():+.2f}%]")
print(f"Improved: {(mc_d > 0.01).sum()}/200  Degraded: {(mc_d < -0.01).sum()}/200  Stable: {((mc_d >= -0.01) & (mc_d <= 0.01)).sum()}/200")

# TEST 4: Where do flips concentrate?
print(f"\n{'='*70}")
print("TEST 4: Flips by scenario type (position weight zeroed)")
print("="*70)
p0 = DEFAULT.copy(); p0[1] = 0.0
tc, tf = defaultdict(int), defaultdict(int)
for i, (att, cor) in enumerate(scenarios):
    tc[labels[i]] += 1
    if select_answer(att, p0) != base_ans[i]: tf[labels[i]] += 1
print(f"\n{'Scenario':<22} {'N':>6} {'Flips':>7} {'Rate':>7}")
print("-" * 44)
for st in ['easy', 'medium', 'hard', 'tie', 'minority_correct']:
    print(f"{st:<22} {tc[st]:>6} {tf[st]:>7} {tf[st]/max(tc[st],1)*100:>6.1f}%")

print(f"\n{'='*70}")
print("COMPLETE")
print("="*70)
