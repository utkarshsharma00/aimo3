# Every Improvement Failed: A Negative-Results Report on Inference and Fine-Tuning Interventions for gpt-oss-120b on AIMO3

**Team:** flamingice (solo) · **Competition:** AI Mathematical Olympiad - Progress Prize 3 · **Submission for the Writeup Prize**

---

## Abstract

We report a negative-results-dominated investigation of inference-time and training-time interventions for `gpt-oss-120b` on the AIMO3 competition. Our private-leaderboard submission, `final_submission v2`, achieved a public-leaderboard mean of **40.78** (σ = 1.20, n = 9 identical runs, best 43) using parallel self-consistency over 8 attempts with a 5-component weighted-entropy voting function and a one-sentence system prompt. We submitted **25 successful runs across 11 distinct notebook configurations** over 25 days. With one exception, every architectural addition we attempted - multi-strategy prescriptive prompts, structured 5-stage scaffolding, subject-routed multi-stage arbitration with verifier/GenSelect/adversarial passes, and GRPO LoRA fine-tuning - either fell *within* run-to-run noise or fell *catastrophically* below it. The most expensive failure (≈$300 of B200 GPU time on GRPO experiments) is documented with 190 rows of training telemetry and traces to a known Unsloth `merged_4bit_forced` → vLLM MXFP4 deployment pipeline incompatibility. Our central methodological contribution is the **noise floor**: with σ = 1.20 across identical runs, any apparent improvement smaller than ≈2.4 points cannot be distinguished from running the same notebook twice. We argue that for inference-constrained math competitions, this noise dominates almost all engineering decisions, and that future participants should establish their own variance baselines *before* iterating on architecture.

---

## 1. Introduction

The AIMO3 competition required solving 50 private-set olympiad-level problems with non-negative integer answers in [0, 99999], inside a single Kaggle notebook constrained to **≈5 hours of H100 GPU time**. The runtime constraint transforms the optimization target: the goal is not "maximize per-problem accuracy" but "maximize the joint expectation of accuracy and successful completion under a wall-clock cliff." This distinction kills nearly every intuition borrowed from offline pass@*N* experiments.

`gpt-oss-120b` (released in 2025, 117B total parameters with ≈5.1B active per token via mixture-of-experts, native MXFP4 weights) is strong enough on olympiad math that the AIMO3 leaderboard is plausibly bottlenecked by inference orchestration rather than reasoning capacity. Public community notebooks built on `gpt-oss-120b` were reported to score in the high 30s and low 40s out of 50 with relatively simple self-consistency setups. Against that background, our research question was: **starting from a strong base model, where do additional engineering investments actually move the score?**

We tested four classes of intervention:

1. **Prompt engineering** - prescriptive multi-strategy prompts and structured 5-stage scaffolds (UNDERSTAND → EXPLORE → PLAN → EXECUTE → VERIFY).
2. **Inference-time orchestration** - adaptive routing, multi-stage solving, posterior voting, GenSelect, adversarial verification.
3. **More attempts** - pushing N from 8 to 12.
4. **Reinforcement learning** - GRPO LoRA fine-tuning with a custom vLLM + Unsloth training loop.

Every one of them either left the score within statistical noise of where we started, or made it dramatically worse. The single intervention that produced a stable mean above the noise floor was a **5-component weighted entropy formula** for self-consistency voting, combined with deliberate stripping of every other piece of scaffolding. This paper documents what we tried, why we tried it, and - more importantly - why the things that didn't work didn't work.

**Reproducibility note.** All code, prompts, and configurations are reported exactly as run. The full submission history (Appendix C), the GRPO training telemetry CSV (Appendix D), the offline research branch catalog (Appendix H), and the verbatim source of every notebook discussed here are available with this writeup.

---

## 2. The submission landscape

We made **25 successful submissions across 11 notebook configurations** between competition entry and the April 15 deadline. Figure 1 shows the chronological trajectory.

![Figure 1: Leaderboard trajectory across 25 submissions](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig1_leaderboard_trajectory.png)

Three structural features of this trajectory motivate the rest of the paper:

**The ramp-and-plateau shape.** Scores rise from the high 20s to the high 30s as obvious bugs are removed, then plateau in a narrow band centered around 40 with substantial variance. No single architectural change moves us *out* of this band on the upside.

**The catastrophic dips.** Three submissions fall well below the band: GRPO at 13, the adaptive router at 28, and the V11 timeout at 33. Each represents a different failure mode (deployment pipeline corruption, time-budget collapse from per-problem complexity, raw wall-clock overrun). All three are documented in detail below.

**The final cluster.** The last nine submissions on the right are repeated runs of `final_submission v2` - the same notebook, the same seed, the same code - submitted to measure run-to-run variance. They span 39 to 43 with a mean of 40.78 and σ = 1.20.

This last cluster is the load-bearing observation of the entire writeup. We turn to it next.

---

## 3. The noise floor

We submitted `final_submission v2` nine times without modification. The scores were:

> **39, 40, 43, 40, 42, 40, 41, 41, 41**

Mean = 40.778. Median = 41. σ = 1.202. Range = 4 points. 95% CI for the mean = [39.85, 41.70]. Figure 2 shows the distribution.

![Figure 2: Run-to-run variance establishes the noise floor](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig2_score_distribution.png)

This variance has two sources we cannot fully separate. First, the gpt-oss-120b decoder produces non-deterministic output even with a fixed seed, because the parallel attempts are issued asynchronously to a vLLM server whose batching schedule depends on real-time CUDA kernel timing and KV-cache state. Second, Kaggle's grading is deterministic but the scheduling of which problems hit which attempt order is not. Whatever the underlying mechanism, **the operationally relevant fact is that nine runs of identical code spanned four points**.

The implication for our research methodology is severe and one-directional:

> **Any single comparison between two notebooks that differs by less than ≈2σ ≈ 2.4 points is statistically indistinguishable from running the same notebook twice.**

In retrospect, the most important methodological mistake we made early in the competition was treating single-shot leaderboard scores as ground truth. When `aimo3_final v9` returned 37 three times in a row and `[15/15] AIME v3` returned 35 and 38, we read those as meaningful differences and chased the apparent winner. They are not meaningful differences. They are draws from the same noisy distribution.

We did not realize this until the final week, when a deliberate variance experiment on the converged notebook produced the n = 9 distribution above. Had we run such an experiment on day one, we would have saved ≈$300 of GPU spend and three weeks of architectural iteration. **This is recommendation #1 of the paper: future AIMO3-style participants should establish a noise floor before iterating on architecture.**

---

## 4. Offline evaluation on hard50

Throughout the competition we maintained an offline evaluation loop against the Kaggle community dataset **`hard_50_math_problems_set_v6.csv`** (published by user `zfturbo` as `hard-math-problems-for-aimo-3`) - a 50-problem benchmark of olympiad-level questions with known integer answers. Offline evaluation served as a debugging sanity check and as a relative-performance gauge for architectures we were not yet willing to spend a submission slot on.

We authored **15 offline research branches** during the competition, organized into five families, each targeting a specific failure class observed in earlier runs:

- **Branch A / A2 (contested resolver)** - trunk + candidate aggregation table + pairwise resolver. Motivation: hard50 showed that many wrong answers were *rerankable* - the correct answer was in the candidate pool but a weak plurality of wrong attempts shipped it. Heuristic constants were drawn from observed failure patterns (66 vs 69, 570 vs 285, 24 vs 25, 2000 vs 1999, 105 vs 106).
- **Branch A3 (closure-aware challenge + dual-check flip gate)** - collect *all* boxed integers in an attempt and use the last to defend against premature boxing; flips require two independent checks. Authors explicitly noted A3 was "*intended for AIMO3, not hard50-only optimization.*"
- **Branch B (pairwise-authorized arbitration)** - pairwise becomes the only component that can authorize a challenger flip; pressure and audit become veto layers only.
- **Branch C (controller + code rescue + provenance-aware weighting)** - state controller that stops spending budget on already-converged or broken pools; code rescue for broken pools; candidates supported by multiple evidence modes weigh more than plain-text singletons.
- **Branch C+ (trusted main-pool arbitration + routed micro-prompts + typed verifiers)** - the most elaborate research branch. Authors again stated it was "*intended for AIMO3-level problems, not hard50-only tuning.*"

**None of these 15 branches were ever submitted to the Kaggle leaderboard.** Of the 15, **nine were never executed to completion** due to infrastructure and wiring bugs. Two contested-resolver branches ran 50 problems on hard50 but never logged per-problem correctness. Three Branch B/C variants share a single stale truncated run (cumulative 19/29 at problem 30) that we cannot attribute to a specific architecture because the three notebooks have different code but identical saved outputs from one Colab session. **One notebook - `aimo3_branch_B_multicell` - completed a full, inline-graded 50-problem hard50 run.** That is the only offline number we can cite with full verification. Full per-branch execution status is catalogued in Appendix H.

**Branch B multicell on hard50: 27/50 (54.0%), mean 233 s/problem, total 194 min.** Figure 9 shows the per-problem timing (top) and cumulative accuracy (bottom). No problem hit the 900 s cap; the run finished comfortably inside a notional 5-hour budget at 3.24 h total.

![Figure 9: Branch B multicell full hard50 run](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig9_branch_b_hard50.png)

Accuracy begins at 100% after problem 2, collapses to the mid-60s by problem 10, and bleeds out to 54% by problem 50. The architecture looked competitive early and degraded on the harder back half. Since no problem timed out, 27/50 reflects answer quality, not budget exhaustion.

**Interpretation is constrained.** Branch B multicell's elaborate pairwise-arbitration architecture scored 27 on hard50; `final_submission v2` scored a mean of 40.78 on AIMO3. If hard50 difficulty were comparable to AIMO3, Branch B would project to ~27 on AIMO3 - 14 points below the simpler approach. But hard50 is not AIMO3: it is community-sourced, intentionally difficult, and may systematically underestimate AIMO3 performance - which is precisely why Branches A3 and C+ were explicitly designed "not for hard50-only tuning." We also never ran a variance experiment on hard50 analogous to our n = 9 experiment, so we cannot say whether 27/50 is itself ±2 points of noise.

Even with those caveats, the directional signal was clear: **our only offline-executed elaborate architecture predicted a losing Kaggle score, and the simpler baseline outperformed it.** The writeup-prize rubric asks *"did you use any internal benchmarks?"* - yes, but less effectively than we should have. Nine of our 15 research notebooks never finished a run; we never calibrated hard50 against AIMO3; and the one completed offline run told a story that should have killed Branch B's lineage earlier than it did. **Future participants: instrument your offline benchmark the way you instrument your leaderboard submissions, get all your research notebooks across the finish line, and run a variance experiment on the offline set before treating any single offline number as signal.** Appendix H lists all 15 branches with design intent and execution status.

---

## 5. The converged solution: `final_submission v2`

### 5.1 Architecture

`final_submission v2` is intentionally lean. It does not contain a 5-stage prompt, a router, a verifier, an adversarial pass, GenSelect, multi-strategy prescriptive prompts, or fine-tuned weights. It contains four things. First, the base model: `danielhanchen/gpt-oss-120b` served by local vLLM (KV-cache fp8_e4m3, gpu_memory_utilization 0.96, max_model_len 65,536). Second, a one-sentence system prompt: *"You are a world-class IMO competitor. The final answer must be a non-negative integer between 0 and 99999. Place your final answer inside `\boxed{}`."* The model is invoked through the openai-harmony format with `ReasoningEffort.HIGH`, the gpt-oss native reasoning control. Third, tool-augmented parallel self-consistency: 8 attempts dispatched in parallel against a pool of 16 long-lived Jupyter sandboxes (math, numpy, sympy, mpmath, itertools, collections; `mpmath.mp.dps = 64`), up to 128 harmony turns each, T = 1.0, min_p = 0.02, no top_p, per-attempt seed `(global_seed + attempt_index)²`. Streams are inspected character-by-character; an attempt short-circuits the moment a `\boxed{...}` appears. Fourth, 5-component weighted-entropy voting (Section 5.2); when 4 attempts agree, remaining attempts are cancelled.

The time budget is computed adaptively per problem: `min(max(time_remaining − 270 × problems_remaining, 270), 900)` seconds, with `notebook_limit = 17400` (4h50m, leaving safety margin inside Kaggle's 5h cliff). Appendix A contains the full configuration.

### 5.2 The 5-component weighted entropy

Inverse-entropy voting is a known technique: weight each attempt by 1/H, sum weights per answer, return the highest-weighted answer. The novelty in `final_submission v2` is that the entropy itself is a five-part composite over the attempt's logprob trajectory, rather than a single mean. Reading from the source:

```python
weighted_entropy = (
    0.30 * mean_entropy
  + 0.40 * position_weighted_entropy   # exponential decay 0.995, recent tokens dominate
  + 0.20 * std_dev_of_entropy          # variance penalty for inconsistent confidence
  + 0.30 * 3.0 * high_entropy_ratio    # fraction of tokens with H > 2.0 bits, scaled ×3
  - 0.10 * (max_low_entropy_streak / n)  # bonus for sustained confident stretches
)
```

The intuition: position weighting with decay 0.995 means tokens near the final boxed answer count exponentially more than tokens early in the chain of thought - a solution uncertain during exploration but confident at conclusion outranks one that is confident throughout but flinches at the end. The variance penalty punishes oscillating confidence ("the model kept changing its mind"). The high-entropy ratio counts tokens where the next-token distribution exceeds 2 bits of Shannon entropy, and the ×3 multiplier inside a 30% weighting makes this term dominant when an attempt has many uncertain stretches. The streak bonus rewards sustained runs of < 1 bit entropy as a marker of "the model is on a roll."

These weights and thresholds are not from a paper. They were tuned by hand on intermediate submissions in this competition. The core idea of inverse-entropy weighted voting was informed by [pawanmali's "Chasing 47/50" notebook](https://www.kaggle.com/code/pawanmali/chasing-47-50-aimo3-journey-of-60-experiments), which demonstrated entropy-based confidence weighting for self-consistency on AIMO3. We do not claim they are optimal - only that this composite produced a stable mean of 40.78 in `final_submission v2`, while simpler "mean entropy" voting in eight other notebooks we submitted produced means in the 33–37 region. Section 6 unpacks that comparison.

### 5.3 What is *not* in `final_submission v2`

For the avoidance of doubt: `final_submission v2` does not contain the 5-stage UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY scaffold (tested in `[15/15] AIME v3`, scored 35 and 38), the 8 prescriptive strategy prompts from the `aimo3_final` family (five submissions, mean ~36.6), the verifier/GenSelect/adversarial scaffolding from the same family, the subject-classifier routed prompts from `aimo3_adaptive v4` (scored 34), the full adaptive router with posterior voting (scored 28), the increased attempt count from `base_submission v2` (scored 33, timed out), or the GRPO fine-tuned weights from the v4_grpo submission (scored 13, merge format mismatch). Figures 7 and 8 visualize the same information at the feature and score levels respectively.

---

## 6. Ablation: every intervention vs. baseline

Figure 8 collects every intervention we tried as a horizontal bar chart, compared against the baseline of 39 (the first run of `aimo3_final v3`, which is the earliest non-debugging submission still in our history).

![Figure 8: Ablation waterfall — every intervention vs. starting baseline](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig8_ablation_waterfall.png)

Three regions: **inside the noise band** - early `aimo3_final` iterations, the multi-strategy prompt family, and the 5-stage scaffold sit within ±2σ of the 39 baseline and are indistinguishable from starting fresh (the strategy-prompts family was tested in five submissions and never broke out). **Below the noise band** - the structured-scaffold-plus-classifier branch (−5), V11 timeout (−6), the router with GenSelect/adversarial pass (−11), and GRPO (−26); all real failures with identifiable causes that Section 7 dissects. **Above the noise band** - only `final_submission v2` at +1.78, with the lower bound of its 95% CI (39.85) itself above baseline. It is the *only* intervention in the entire competition with statistically defensible upward movement.

Figure 7 makes the same point feature-by-feature. Each row is a notebook, each column a feature; checkmarks indicate presence.

![Figure 7: Feature presence across 11 submitted notebooks](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig7_feature_comparison.png)

`final_submission v2` (top row) has *fewer* features than every other notebook except the most basic, and is the only one with the 5-component weighted entropy. Notebooks with strategy prompts cluster around 36–37; notebooks with verifier/GenSelect/adversarial scaffolding cluster around 28–37; the router is at 28; GRPO at 13. Adding architecture moved us downward; subtracting it and replacing the voting function moved us upward. The honest causal claim we can make is that **stripping the scaffolding AND replacing the voting function with 5-component weighted entropy together produced a defensible improvement above noise**. We cannot say which change was load-bearing on its own - that would require an isolated ablation we did not run before the deadline.

---

## 7. Failed interventions, in detail

### 7.1 Multi-strategy prescriptive prompts (V10 family, scored 36–37)

The `aimo3_final` v7/v9/v22 family launches eight parallel attempts, each with a different prescriptive system prompt instructing the model to use a specific mathematical technique - *analytic, computational, small-cases, work-backwards, algebraic, modular, geometric, random-testing-and-verification*. The hypothesis was that prompt-induced strategy diversity would produce a more diverse candidate pool for self-consistency voting to exploit, extending the DIPPER intuition that framing diversity beats temperature diversity alone.

It failed because `gpt-oss-120b` at `ReasoningEffort.HIGH` complies with prescribed techniques even when they are poorly suited - a geometry problem given a "modular arithmetic" prompt produced attempts that contorted themselves toward modular methods and failed. The base model already selects appropriate techniques at T = 1.0; external prescription overrides this selection in the wrong direction about as often as the right one. Across five submissions, the mean was approximately 36.6 - Δ ≈ −2.4 vs. baseline, just inside the lower noise edge but directionally negative. The methodological distinction we carried forward: prescribing solution *content* hurts, prescribing reasoning *form* (Section 7.2) is roughly neutral.

### 7.2 5-stage structured scaffold (`[15/15] AIME v3`, scored 35 and 38)

The `[15/15] AIME 2026 I 120b in 20mins` notebook v3 uses the full 5-stage UNDERSTAND → EXPLORE → PLAN → EXECUTE → VERIFY scaffold popular in CoT prompting research, with a verbose ~1500-character system prompt detailing reasoning principles and verification requirements. The hypothesis was that even a strong base model would benefit from being forced to plan before executing.

The two submissions scored 35 and 38 - squarely inside the ±2σ noise band of baseline. We cannot reject the null hypothesis that the scaffold has zero effect on `gpt-oss-120b`. A plausible explanation is that `ReasoningEffort.HIGH` already imposes a reasoning structure that the prompt-level scaffold duplicates without adding signal. **Technique transfer from smaller-model prompting research does not automatically apply to gpt-oss-120b at high reasoning effort.**

### 7.3 The adaptive router (`AIMO3_Adaptive_Router_Confidence_Vote_v1`, scored 28)

A multi-stage system: a subject classifier routed each problem to one of four prompt families (geometry, number theory, combinatorics, algebra); a "quick stage" issued 2 cheap attempts (T = 0.55 and 0.80, max 2200 tokens, 24 turns); if those agreed with high posterior confidence, return immediately, otherwise an "explore stage" issued more attempts at higher temperature with longer budgets; weak candidates were routed through verifier, GenSelect, and adversarial passes; `high_problem_timeout = 840` s per problem. The hypothesis was the standard one: easy problems should consume little budget, hard problems more; an adaptive system should outperform a flat-budget system under heterogeneous difficulty.

It failed structurally. The first problem consumed 840 s. The second consumed approximately 825 s. The router had burned 28% of the entire 5-hour notebook budget on two problems, and the remaining 48 problems shared the rest. Many timed out or returned 0. Final score: 28 - fully twelve points below baseline, the second-largest single-intervention drop in the competition. The router's adaptive logic correctly identified problems 1 and 2 as hard and allocated them more time; it then found nothing useful in the extra time. The correct behavior under those conditions is to give up early and conserve budget for problems where extra time might actually help - we did not implement that. A v2 with reduced timeouts reached 19/30 on an offline slice but we did not submit it; the converged `final_submission v2` was already outperforming it. **Adaptive complexity is only valuable when the marginal value of extra time on a hard problem is positive. For hard olympiad problems in the AIMO3 distribution against gpt-oss-120b, that marginal value appears to be approximately zero past 5–6 minutes.**

### 7.4 More attempts on stock weights (`base_submission v2`, scored 33)

This is the team's fork of the public `AIMO 3 | GPT-OSS-120B (with tools)` notebook ("the 42-notebook" in community discussion), with three modifications: `attempts` raised from 8 to 12, `early_stop` raised from 4 to 5, and a "Be concise. Stop immediately after `\boxed{}`" instruction added. The notebook also uses the stock OpenAI `gpt-oss-120b` weights rather than Daniel Hanchen's variant. Hypothesis: more samples → stronger voting → higher score, up to the point where the time budget binds.

It exceeded the 5-hour budget. Twelve attempts × 50 problems with a verbose chain-of-thought trajectory pushed total runtime past the cliff; problems processed after timeout returned 0. Final score: 33. The experiment is **confounded** because we changed three variables at once and cannot cleanly attribute the −6 to "more attempts" alone. Operational lesson: **if you push N upward, run a wall-clock experiment first, and budget for the slowest model variant.**

### 7.5 GRPO LoRA fine-tuning ([15/15] AIME v4_grpo, scored 13)

This is the largest and most expensive negative result in the writeup, and the most generally useful for the community. We approached it in two phases.

**Phase A: TRL GRPOTrainer (collapsed at step 16, $101.73 sunk).** Our first attempt used Hugging Face TRL's `GRPOTrainer` with Unsloth on a RunPod B200 ($5.03/hr). Configuration: LR 5e-6, max_grad_norm 1.0, β = 0.0, 8 generations per problem, `use_vllm=False` (because TRL's vLLM integration crashed at startup against gpt-oss-120b). Each step took ≈76 minutes. The training collapsed:

- Step 1: loss 0.0513, reward 0.375.
- Steps 8–9: KL divergence to reference policy reached **87**, gradient norm reached **133**.
- Step 13: peak reward 0.5625 - but loss had already climbed to 280.
- Step 16: loss reached **19,751,736**. Model destroyed. Killed.

Figure 6 shows the trajectory.

![Figure 6: TRL GRPOTrainer collapse on Unsloth + gpt-oss-120b](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig6_trl_collapse.png)

The diagnosis is straightforward: LR 5e-6 was too aggressive for an MoE model at 117B parameters, and `max_grad_norm` 1.0 was too loose to clip the gradient explosions that began once the policy started drifting from the reference.

**Phase B: Custom vLLM + Unsloth loop (v6) - trained stably, deployed catastrophically.** We rebuilt the training loop from scratch to avoid TRL's vLLM integration: each cycle launched vLLM with the current adapter, generated 8 completions per problem for 5 problems, killed vLLM to free GPU memory, applied a GRPO update via Unsloth, saved the adapter, restarted. We tightened the configuration substantially (LR 2e-6 with cosine decay, max_grad_norm 0.3, Dr.GRPO advantage normalization, DAPO-style overlong filtering, binary reward, β = 0, off-policy generation with vLLM on the base model and adapter trained separately; see Appendix B for the full config).

We have telemetry for **190 problems across 38 cycles** (cycles 62–99, problems 310–499). The training was **stable** - zero loss collapses, loss values mostly in [0, 0.3], gradient norms cleanly bounded by the 0.3 clip. Figure 3 shows the per-cycle reward trajectory:

![Figure 3: GRPO v6 reward trajectory](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig3_grpo_reward_trajectory.png)

The reward signal tells the second half of the story. The overall mean reward across all 190 problems was **+0.033** - barely above zero, after 38 cycles of training. Only **4 problems out of 190 (2.1%)** achieved reward 1.0 (all 8 generations correct): problems 329, 333, 379, 468. Three of those four had near-zero completion length, suggesting trivially short correct answers. Figure 5 shows the full reward distribution:

![Figure 5: GRPO v6 problem-level reward distribution](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig5_grpo_reward_histogram.png)

Forty-seven percent of training problems received net negative reward. Eighteen percent received exactly zero (all 8 generations wrong, or all 8 right) - these are filtered from the gradient by Dr.GRPO's advantage normalization, contributing nothing. Thirty-three percent were partial wins. Only 2% were complete wins. This is the hardest possible shape for GRPO: the model is below 50% on the training distribution, and the reward signal is sparse and noisy. Even a stable training loop cannot extract much policy improvement from this.

**The deployment failure.** We merged the trained adapter using Unsloth's `merged_4bit_forced` method. The submission scored **13/50** - a 27-point drop from baseline. The root cause is a **quantization format mismatch**: `gpt-oss-120b`'s native weight format is **MXFP4** (E2M1 with E8M0 block scaling), but `merged_4bit_forced` produces **BitsAndBytes NF4**. vLLM routes the merged weights through its BitsAndBytes loader rather than the MXFP4 kernels, producing severely degraded output. The fix - which we discovered post-mortem from Unsloth documentation and vLLM GitHub issues #19361 and #21932 - is to merge with `merged_16bit` or save with `save_method="mxfp4"`. A community member elsewhere in the AIMO3 forum reportedly spent **$500** on SFT+RL with a similar pipeline and scored 36, attributing it to the same issue. We spent approximately **$300** total on GRPO across all phases (Figure 4): TRL crash $102, v1–v5 debugging $100, v6 $62, idle $40.

![Figure 4: GRPO v6 cumulative compute cost](https://raw.githubusercontent.com/utkarshsharma00/aimo3/main/figures/fig4_grpo_cost.png)

The actionable takeaway: **never use `merged_4bit_forced` to deploy a fine-tuned `gpt-oss-120b` LoRA to vLLM.** Use `merged_16bit`, `save_method="mxfp4"`, or NVIDIA's QAT pipeline.

---

## 8. Comparison with state of the art

We benchmarked exclusively against the open-weight setting because that is what AIMO3's hardware budget allows. The relevant SOTA points are:

- **AIMO2 winner (NemoSkills, 2025):** 34/50 on 4×L4 GPUs, using a Qwen2.5-14B model fine-tuned on 540K custom problems with 3.2M CoT solutions and 1.7M tool-integrated reasoning traces, plus GenSelect at inference time. Documented in arXiv:2504.16891.
- **Public community baselines on `gpt-oss-120b` for AIMO3:** Reported in the high 30s to low 40s with simple self-consistency setups.
- **Our submission:** 40.78 mean (n=9, σ=1.20, best 43) with no fine-tuning, no scaffolding, and a 5-component weighted entropy voting function.
- **The top of the AIMO3 public leaderboard at writeup time:** ≥46/50, presumably achieved through inference acceleration (EAGLE-3 speculative decoding has been discussed publicly), correct fine-tuning pipelines that avoid the MXFP4 mismatch, or both.

The structural shift from AIMO2 to AIMO3 is striking. AIMO2 was won by a team that invested heavily in training-time compute and produced a 14B-parameter model that scored 34. AIMO3 allows H100 hardware, which makes `gpt-oss-120b` runnable, and the *unmodified* base model with a sensible voting function reaches 40.78 - six points above the AIMO2 winner with zero training-time investment. **Base model capability dominates the AIMO3 landscape in a way it did not dominate AIMO2.** Our negative GRPO result is consistent with this: training-time interventions face an uphill battle against a base model that is already near the ceiling of the inference-constrained accuracy distribution.

We did not benchmark against commercial models (GPT-4, Claude, Gemini) because the AIMO3 rules require self-contained Kaggle notebook execution and commercial API calls are not permitted in the competition pipeline. For external context, the AIMO Prize organizers have published evidence that the gap between open-weight and commercial models on olympiad math has narrowed substantially since AIMO1.

---

## 9. Lessons and recommendations

The competition produced eight findings we believe will help future participants more than another point on the leaderboard would.

1. **Establish the noise floor first.** Submit the same notebook 5–10 times before iterating on architecture. σ = 1.20 means any 1–2 point "improvement" is invisible to single-shot comparison. Every other recommendation depends on this one.

2. **Per-problem time budgets are absolute.** The 5-hour cliff is hard-enforced. Any architecture that *can* spend more time on a problem must have a hard cap well below what the budget would technically allow - hard problems do not pay you back for extra time.

3. **Strategy-prompt diversity hurts a strong base model.** When `gpt-oss-120b` at `ReasoningEffort.HIGH` is told to use a specific technique, it complies even when the technique is wrong. Sampling-temperature diversity at T = 1.0 produces enough variance for self-consistency voting without prescriptive prompts.

4. **Confidence estimation matters more than sample count.** Our single defensible improvement above noise came from changing the *voting function*, not from generating more samples. Treat the voting function as a first-class optimization target.

5. **`merged_4bit_forced` is a trap for `gpt-oss-120b`.** The Unsloth → vLLM pipeline silently produces a BitsAndBytes-quantized model that vLLM cannot run efficiently. Use `merged_16bit` or `save_method="mxfp4"`.

6. **Stable GRPO training is necessary but not sufficient.** Our v6 loop trained stably for 38 cycles with no collapses, but the deployed model scored 13 because of the merge format issue, and the reward signal during training (mean +0.033) was too sparse for meaningful policy improvement.

7. **Instrument your offline benchmark like your leaderboard.** We ran 15 offline research branches but only one completed a verifiable end-to-end run. We never calibrated hard50 difficulty against AIMO3. Subtraction beats addition: every architectural addition we tested was neutral or harmful.

8. **Document the negative results.** We invested ~$300 of B200 GPU time and three weeks of architectural iteration in directions that did not work. If this writeup helps even one future team avoid the `merged_4bit_forced` trap or the strategy-prompt-diversity dead end, it has paid for itself many times over.

---

## 10. Conclusion

We did not win AIMO3. Our public-leaderboard mean of 40.78 places us 142nd out of 4,138 teams (top 3.4%), well above naive baselines but below the top of the leaderboard. What we contribute to the community is not a higher score but a documented, statistically defensible characterization of the AIMO3 inference problem with `gpt-oss-120b`: the base model is strong enough that most engineering interventions are dominated by run-to-run noise; the few interventions that move the score appreciably move it *downward*; and the single defensible upward move we found was a more careful confidence-estimation function for self-consistency voting. The expensive failure modes - strategy-prompt prescription, multi-stage routing, and most importantly GRPO with broken deployment - are all reproducible from the configurations and telemetry in the appendices. Every score in this paper is from an actual public-leaderboard submission, every configuration is from actual source code, and every causal claim is hedged to the strength of the evidence we have.

If you take one thing from this writeup, take this: **before you spend a week building a router, submit the same notebook many times and look at the variance.**

---
---

# Appendices

## Appendix A: Full configuration of `final_submission v2`

This is the verbatim configuration from the `CFG` class in the submitted notebook.

| Parameter | Value |
|---|---|
| **Base model** | `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1` |
| **Served model name (vLLM)** | `gpt-oss` |
| **Quantization** | Native MXFP4 |
| **Tensor parallel size** | 1 |
| **gpu_memory_utilization** | 0.96 |
| **kv_cache_dtype** | `fp8_e4m3` |
| **dtype** | `auto` |
| **context_tokens (max_model_len)** | 65,536 |
| **batch_size (max_num_seqs)** | 256 |
| **stream_interval** | 200 |
| **enable_prefix_caching** | True |
| **async_scheduling** | True |
| **System prompt** | *"You are a world-class IMO competitor. The final answer must be a non-negative integer between 0 and 99999. Place your final answer inside `\boxed{}`."* |
| **Tool prompt** | *"Use this tool to execute Python code. The environment is a stateful Jupyter notebook. Always use print() to output results."* |
| **Preference prompt** | *"You have access to math, numpy, sympy, and mpmath."* |
| **Reasoning effort** | `ReasoningEffort.HIGH` (set on harmony SystemContent) |
| **Tool format** | openai-harmony Conversation API |
| **Available libraries in sandbox** | math, numpy, sympy, mpmath, itertools, collections; `mpmath.mp.dps = 64` |
| **temperature** | 1.0 |
| **min_p** | 0.02 |
| **top_p** | not set (intentionally - top_p = 0.9 was tested and caused a regression in earlier iterations) |
| **top_logprobs** | 5 |
| **attempts** | 8 |
| **workers (parallel sandboxes)** | 16 |
| **early_stop** | 4 (cancel remaining attempts when 4 agree) |
| **turns (max harmony turns per attempt)** | 128 |
| **seed (global)** | 42 |
| **per-attempt seed** | `(seed + attempt_index) ** 2` |
| **buffer_tokens** | 512 |
| **search_tokens (streaming `\boxed{}` window)** | 32 |
| **notebook_limit** | 17,400 s (4h50m) |
| **base_problem_timeout** | 270 s |
| **high_problem_timeout** | 900 s |
| **server_timeout** | 180 s |
| **session_timeout** | 960 s |
| **jupyter_timeout** | 6 s |
| **sandbox_timeout** | 3 s |
| **per-problem budget formula** | `min(max(time_remaining − 270 × (problems_remaining − 1), 270), 900)` |

### A.1 Voting function source

```python
def _compute_weighted_entropy(self, logprobs_buffer):
    """5-component weighted entropy."""
    if not logprobs_buffer:
        return float('inf')

    entropies = []
    for top_lp in logprobs_buffer:
        if not isinstance(top_lp, dict) or not top_lp:
            continue
        h = 0.0
        for _, lp in top_lp.items():
            p = math.exp(lp)
            if p > 0:
                h -= p * math.log2(p)
        entropies.append(h)

    if not entropies:
        return float('inf')

    n = len(entropies)
    mean_ent = sum(entropies) / n

    # Variance penalty
    if n > 1:
        variance = sum((e - mean_ent) ** 2 for e in entropies) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    # Position-weighted (recent tokens dominate, decay 0.995)
    decay = 0.995
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    position_weighted = (
        sum(w * e for w, e in zip(weights, entropies)) / total_weight
        if total_weight > 0 else mean_ent
    )

    # High-entropy ratio (tokens with H > 2.0 bits)
    high_ent_count = sum(1 for e in entropies if e > 2.0)
    high_ent_ratio = high_ent_count / n

    # Low-entropy streak bonus (sustained < 1.0 bit)
    max_streak = 0
    current_streak = 0
    for e in entropies:
        if e < 1.0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    streak_bonus = -0.1 * (max_streak / n) if n > 0 else 0.0

    weighted = (
        0.3 * mean_ent
      + 0.4 * position_weighted
      + 0.2 * std_dev
      + 0.3 * high_ent_ratio * 3.0
      + streak_bonus
    )
    return max(weighted, 1e-9)


def _select_answer(self, detailed_results):
    """Inverse-entropy weighted voting."""
    answer_weights = defaultdict(float)
    answer_votes = defaultdict(int)
    for r in detailed_results:
        a = r['Answer']
        e = r['Entropy']
        if a is not None:
            answer_weights[a] += 1.0 / max(e, 1e-9)
            answer_votes[a] += 1
    if not answer_weights:
        return 0
    scored = sorted(
        [{'answer': a, 'votes': answer_votes[a], 'score': w}
         for a, w in answer_weights.items()],
        key=lambda x: x['score'], reverse=True
    )
    return scored[0]['answer']
```

### A.2 Boxed-answer scanner

```python
def _scan_for_answer(self, text):
    pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
    matches = re.findall(pattern, text)
    if matches:
        try:
            val = int(matches[-1].replace(',', ''))
            if 0 <= val <= 99999:
                return val
        except ValueError:
            pass
    pattern = r'final\s+answer\s+is\s*([0-9,]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            val = int(matches[-1].replace(',', ''))
            if 0 <= val <= 99999:
                return val
        except ValueError:
            pass
    return None
```

The streaming inspection happens inside `_process_attempt`: after each chunk of streamed text, the last `search_tokens = 32` tokens worth of text are scanned for `\boxed{...}`, and if found, the attempt short-circuits without consuming additional turns.

### A.3 Provenance and credit

`final_submission v2` was implemented from scratch by the team. It is not a fork of any public Kaggle notebook. Several ideas - particularly the entropy-weighted voting pattern and the harmony-format tool integration - were informed by discussion in the AIMO3 Kaggle forum during the competition. The specific 5-component composite formula (with the 0.30/0.40/0.20/0.30/−0.10 weight pattern, the 0.995 decay, the 2.0-bit high-entropy threshold, and the < 1.0-bit streak threshold) was tuned by hand across intermediate submissions in this competition and is not reproduced from any external source we are aware of.

---

## Appendix B: GRPO v6 configuration

| Parameter | Value |
|---|---|
| **Hardware** | RunPod B200 (1 GPU, 191.5 GB VRAM) |
| **Cost rate** | $5.03/hr |
| **Base model** | `danielhanchen/gpt-oss-120b` loaded via Unsloth with `load_in_4bit=True` |
| **Quantization during training** | BitsAndBytes NF4 (Unsloth converts MXFP4 → NF4 on load) |
| **Architecture** | Off-policy: vLLM serves base model; Unsloth trains LoRA adapter separately; vLLM restarted between cycles |
| **Learning rate** | 2e-6 with cosine decay |
| **max_grad_norm** | 0.3 |
| **Generations per problem** | 8 |
| **Temperature (rollouts)** | 1.0 |
| **top_p (rollouts)** | 0.98 |
| **max_completion_tokens** | 6,144 |
| **Advantage** | Dr.GRPO (mean subtraction, no std division) |
| **Overlong filtering** | DAPO-style (truncated completions removed from gradient) |
| **KL coefficient (β)** | 0.0 |
| **Reward function** | Binary: 1.0 if extracted boxed answer matches ground truth, 0.0 otherwise |
| **Problems per cycle** | 5 |
| **Cycles captured in telemetry** | 38 (cycles 62–99) |
| **Problems captured in telemetry** | 190 (problems 310–499) |
| **Mean cycle time** | ~1,170 s (gen ~720s + train ~440s) |
| **Cost in v6 logged segment** | $61.87 |
| **Total GRPO project cost (all phases)** | ≈$300 |
| **Merge method used (the bug)** | `merged_4bit_forced` → produces BnB NF4 |
| **Merge method that should have been used** | `merged_16bit` or `save_method="mxfp4"` |
| **Submission score** | 13/50 |

### B.1 TRL Phase A (collapsed) configuration

| Parameter | Value |
|---|---|
| **Framework** | Hugging Face TRL `GRPOTrainer` + Unsloth |
| **use_vllm** | False (TRL+vLLM crashed at startup) |
| **Learning rate** | 5e-6 |
| **max_grad_norm** | 1.0 |
| **β (KL coefficient)** | 0.0 |
| **Generations per problem** | 8 |
| **Steps completed** | 16 (model destroyed at step 16) |
| **Time per step** | ≈76 minutes |
| **Cost** | $101.73 |
| **Loss at step 1** | 0.0513 |
| **KL at steps 8–9** | 87 |
| **Gradient norm at steps 8–9** | 133 |
| **Reward at step 13** | 0.5625 (peak) |
| **Loss at step 16** | 19,751,736 |

### B.2 SFT (uploaded but not submitted)

| Parameter | Value |
|---|---|
| **Method** | QLoRA, rank 128 |
| **Training data** | 63,959 tool-integrated reasoning (TIR) traces |
| **Steps** | 1,000 |
| **Final loss** | 0.69 (plateau) |
| **Uploaded model** | `flamingice2801/gpt-oss-120b-aimo3-tir-sft` (Kaggle Models) |
| **Submitted to AIMO3?** | No - abandoned after GRPO deployment failure exposed the merge format issue |

---

## Appendix C: Full Kaggle submission history

Twenty-five successful submissions plus three early debug parquet uploads (score 0) and one notebook exception. Listed in chronological order (oldest first).

| # | Days ago at writeup | Notebook | Version | Score | Phase |
|---|---|---|---|---|---|
| 1 | 24 | `aimo3_final` | v3 | 39 | Iteration |
| 2 | 23 | `aimo3_final` | v4_tweaked_temp&model | 11 | Debugging |
| 3 | 22 | `aimo3_final` | v5_tweaked_temp | 0 | Debugging |
| 4 | 21 | `aimo3_final` | v7 | 36 | Iteration |
| 5 | 20 | `aimo3_final` | v9 | 37 | Iteration |
| 6 | 19 | `aimo3_final` | v22 | 36 | Iteration |
| 7 | 18 | `base_submission` | v2 | 33 | Failure (V11 timeout) |
| 8 | 17 | `[15/15] AIME 2026 I 120b in 20mins` | v4_grpo | **13** | **Failure (GRPO)** |
| 9 | 16 | `[15/15] AIME 2026 I 120b in 20mins` | v3 | 38 | Iteration |
| 10 | 15 | `aimo3_final` | v3 (re-run) | 34 | Iteration |
| 11 | 14 | `AIMO3_Adaptive_Router_Confidence_Vote_v1` | v1 | **28** | **Failure (router)** |
| 12 | 13 | `[15/15] AIME 2026 I 120b in 20mins` | v3 (re-run) | 35 | Iteration |
| 13 | 12 | `aimo3_final` | v9 (re-run) | 37 | Iteration |
| 14 | 11 | `aimo3_final` | v9 (re-run) | 37 | Iteration |
| 15 | 10 | `aimo3_adaptive` | _v4 | 34 | Failure |
| 16 | 9 | **`final_submission`** | **v2** | **39** | **Final (run 1)** |
| 17 | 8 | `aimo_tester` | v2 | 34 | Iteration |
| 18 | 7 | **`final_submission`** | **v2** | **40** | **Final (run 2)** |
| 19 | 6 | **`final_submission`** | **v2** | **43** | **Final (run 3)** |
| 20 | 4 | **`final_submission`** | **v2** | **40** | **Final (run 4)** |
| 21 | 3 | **`final_submission`** | **v2** | **42** | **Final (run 5)** |
| 22 | 3 | **`final_submission`** | **v2** | **40** | **Final (run 6)** |
| 23 | 2 | **`final_submission`** | **v2** | **41** | **Final (run 7)** |
| 24 | 1 | **`final_submission`** | **v2** | **41** | **Final (run 8)** |
| 25 | 0 | **`final_submission`** | **v2** | **41** | **Final (run 9)** |

`final_submission v2` distribution: {39, 40, 40, 40, 41, 41, 41, 42, 43}, n = 9, mean 40.778, median 41, σ = 1.202, 95% CI [39.85, 41.70], best 43.

---

## Appendix D: GRPO v6 telemetry summary

The full per-problem telemetry CSV (190 rows × 14 columns) is provided as a supplementary artifact. Summary statistics:

| Statistic | Value |
|---|---|
| Total rows | 190 |
| Cycle range | 62 – 99 (38 cycles) |
| Problem range | 310 – 499 |
| Generations per problem | 8 (constant) |
| Mean generation time per cycle | 720 s |
| Mean training time per cycle | 440 s |
| Mean cycle time | 1,170 s |
| Cost at cycle 62 | $2.34 |
| Cost at cycle 99 | $61.87 |
| Mean reward (all 190 problems) | +0.033 |
| Problems with reward < 0 | 90 (47.4%) |
| Problems with reward = 0 (filtered) | 34 (17.9%) |
| Problems with 0 < reward < 1 | 62 (32.6%) |
| Problems with reward = 1.0 (all 8 correct) | 4 (2.1%) |
| Problem IDs achieving reward = 1.0 | 329, 333, 379, 468 |
| Loss range | [0.0, ~0.7] |
| Loss collapses | 0 |
| Mean completion length | ~3,500 tokens |
| Max completion length observed | ~6,400 tokens (binding against 6,144 cap) |

The full CSV columns are: `cycle, problem_idx, gen_time_sec, train_time_sec, total_sec, reward_mean, reward_std, num_correct, num_total, loss, completion_len_mean, clipped_ratio, gpu_mem, cost`.

---

## Appendix E: Variance experiment statistical details

The 9-run variance experiment for `final_submission v2` produced scores {39, 40, 40, 40, 41, 41, 41, 42, 43}. Descriptive statistics:

| Statistic | Value |
|---|---|
| n | 9 |
| Sum | 367 |
| Mean | 40.778 |
| Median | 41 |
| Mode | 40 and 41 (each appear 3 times) |
| Min | 39 |
| Max | 43 |
| Range | 4 |
| Sample variance s² | 1.444 |
| Sample standard deviation s | 1.202 |
| Standard error of the mean | 0.401 |
| t-critical (df=8, 95%) | 2.306 |
| 95% CI for mean | [39.854, 41.702] |
| Coefficient of variation | 2.95% |

Under the assumption that each problem is solved with effective probability *p_eff* after voting, the score X ~ Binomial(50, *p_eff*). From mean = 40.778, *p_eff* ≈ 0.816, predicting a theoretical Binomial standard deviation of √(50 · 0.816 · 0.184) ≈ 2.74. **Our observed σ = 1.20 is meaningfully lower than the Binomial prediction**, suggesting positive correlation between problems within a single run (i.e., when the model is "having a good run," it tends to solve more problems than independent sampling would predict). Plausible mechanisms include vLLM batching state, KV-cache hit patterns, and GPU thermal/scheduling effects that affect attempts within a single notebook run more than across runs.

For two independent runs from the same configuration with σ = 1.20:
- Expected difference E[|X₁ − X₂|] ≈ 1.35
- P(|X₁ − X₂| ≥ 3) ≈ 0.07
- P(|X₁ − X₂| ≥ 4) ≈ 0.02

A 3-point single-shot difference between two notebooks therefore has approximately a 1-in-14 chance of being pure noise. A 4-point difference reaches conventional statistical significance (p < 0.05) only with this much variance assumed. **Differences smaller than 3 points should not be acted on without replication.**

---

## Appendix F: Why the public 42-notebook is not a clean baseline in our results

The public Kaggle notebook `AIMO 3 | GPT-OSS-120B (with tools)` ("the 42-notebook" in community discussion) is reported to score 42/50 by its authors with default settings. We never submitted the unmodified 42-notebook. Our `base_submission v2` is a **fork** of it with three modifications:

1. `attempts` raised from 8 to 12
2. `early_stop` raised from 4 to 5
3. `"Be concise. Stop immediately after \boxed{}"` added to the preference prompt

`base_submission v2` scored 33 - six points below baseline - primarily due to the wall-clock overrun caused by modification #1. Because we changed three variables at once, **we cannot attribute the −6 to "more attempts" alone**, and we cannot use this submission as a measurement of the unmodified 42-notebook's performance in our environment. The 42-notebook's score in our hands is, strictly speaking, unknown. We list this as a methodological limitation rather than a result.

---

## Appendix G: What we would do differently

If we were starting AIMO3 again with what we know now, in priority order:

1. **Day 1, before any architectural work:** Submit the same baseline notebook 9 times. Plot the distribution. Compute σ. *Then* decide what counts as a meaningful improvement.
2. **Day 2:** Read every Unsloth and vLLM doc page about merging fine-tuned LoRA adapters for `gpt-oss-120b`. Specifically search for "MXFP4" and "merged_4bit". This would have saved the ≈$300 GRPO sunk cost.
3. **Day 3 onward, if time allows:** Investigate inference-acceleration paths (EAGLE-3 speculative decoding, smaller-N with smarter voting, parallel sandbox pool tuning) before investigating training-time interventions. The base model is strong; the binding constraint is throughput.
4. **Throughout:** Treat the voting function as a first-class variable. A more refined confidence estimator was the only thing that produced a defensible improvement above noise in our experiments.
5. **Never:** Add a multi-stage router with verifier / GenSelect / adversarial passes without measuring its time cost on a wall-clock simulation first.

---

## Appendix H: Offline research branch catalog

Fifteen offline research branches were authored during the competition to explore architectures that we were not willing to spend Kaggle submission slots on. **None of the 15 were submitted to the AIMO3 leaderboard.** Their target benchmark was the community-sourced `hard_50_math_problems_set_v6.csv` dataset (Kaggle dataset `zfturbo/hard-math-problems-for-aimo-3`). This appendix documents each branch with its design intent (taken verbatim from the top-of-notebook docstring or markdown cell where available), its execution status, and what the preserved notebook outputs actually verify.

### H.1 Execution status summary

| Branch file                                         | Execution status                               | Verifiable score        | Mean time/problem |
|------------------------------------------------------|------------------------------------------------|-------------------------|-------------------|
| `aimo3_branch_a_contested_resolver`                  | Authored, never executed to completion         | -                       | -                 |
| `aimo3_branch_a_contested_resolver_rebuilt`          | Authored, never executed to completion         | -                       | -                 |
| `aimo3-contested-resolver`                           | Ran 50 problems; **no correctness logging**    | Cannot verify           | 216 s             |
| `aimo3_contested_resolver_conservative` (A2)         | Ran 50 problems; **no correctness logging**    | Cannot verify           | 216 s             |
| `aimo3_A3_1_clean_multicell`                         | Authored, never executed to completion         | -                       | -                 |
| `aimo3_A3_2_candidategen_multicell`                  | Authored, never executed to completion         | -                       | -                 |
| `aimo3_A3_2_patch_multicell`                         | Authored, never executed to completion         | -                       | -                 |
| **`aimo3_branch_B_multicell`**                       | **Complete 50-problem run with inline grading**| **27/50 (54.0%)**       | **233 s**         |
| `aimo3_branch_B_fixed`                               | Shares stale truncated run with C_final        | 19/29 (ambiguous)       | 218 s             |
| `aimo3_branch_B_updated`                             | Shares stale truncated run with C_final        | 19/29 (ambiguous)       | 218 s             |
| `aimo3_branch_C_final`                               | Shares stale truncated run with B_fixed/updated| 19/29 (ambiguous)       | 218 s             |
| `aimo3_branch_C_fixed2`                              | Authored, never executed to completion         | -                       | -                 |
| `aimo3_branch_C_c2`                                  | Authored, never executed to completion         | -                       | -                 |
| `aimo3_branch_C_trust_patch`                         | Authored, never executed to completion         | -                       | -                 |
| `aimo3_branch_C_plus_final` (C+)                     | Authored, never executed to completion         | -                       | -                 |

**Critical caveat on the 19/29 row.** Three notebooks (`B_fixed`, `B_updated`, `C_final`) have different code but identical saved outputs - same per-problem trace, same cumulative markers, same trace zip timestamp `20260409_092715`. The outputs come from a single Colab run that was saved into three different notebook files during iteration. We cannot attribute 19/29 to any specific architecture and therefore do not cite it as a score for any named branch in the main paper. The run terminated at problem 30 - we do not know whether termination was voluntary (user stopped) or involuntary (Colab session died).

**The only offline number that survives full provenance audit is Branch B multicell: 27/50 on hard50.** All per-problem `[EVAL] id=<X> pred=<Y> actual=<Z> correct=<bool> cumulative=<N>/<M>` lines are preserved inline in the notebook output cells.

### H.2 Branch A - Trunk + Contested Resolver

**Files:** `aimo3_branch_a_contested_resolver`, `aimo3_branch_a_contested_resolver_rebuilt`, `aimo3-contested-resolver`

**Design intent (verbatim from the notebook docstring):**

> "This notebook starts from the stable `final_submission` baseline rather than the experimental branches. The baseline kept the proven ingredients that were strongest in prior AIMO3 experiments: simple prompts, T=1.0 generation, 8 attempts, weighted entropy, 65k context, and no large bundles of interacting heuristics."
>
> Motivation: hard50 analysis showed two failure classes - *rerankable* (correct answer appears in the candidate pool but a plurality of wrong attempts shipped it) and *exploration* (correct answer never generated). Branch A targets only the first class.

**Key changes from `final_submission`:**
1. Quick-unanimous exit (3/3 agree) removed as over-confident on genuinely hard problems.
2. Early stop means "stop further generation only" - still build full candidate table; never auto-ship solely because generation stopped early.
3. Candidate aggregation table with support, inverse-entropy weight, mean entropy, mean Python calls/errors, Python error rate.
4. Conservative stage-1 winner ordering: `support DESC → inverse-entropy DESC → mean entropy ASC → Python error rate ASC`.
5. Contest detector fires when any of: top support ≤ 3; runner-up support ≥ 2; top/runner weight ratio < 1.35; stage-1 winner not "safe to ship"; a structured nearby rival (off-by-one / off-by-two / double-half) exists.
6. Rerank score used only inside contested handling: `rerank = log1p(avg_inv_entropy) − 0.15 × py_err_rate − 0.03 × log1p(mean_py_calls)`. Constants drawn from hard50 failure taxonomies (66 vs 69, 570 vs 285, 24 vs 25, 2000 vs 1999, 105 vs 106) and explicitly labeled "heuristic and intentionally mild; not claimed to be optimal."

**Execution status:** The two `aimo3_branch_a_*` variants have zero outputs (never ran end-to-end). `aimo3-contested-resolver` ran 50 problems on hard50 but only logged `answer=X` per problem, never `pred=X actual=Y correct=bool`, so correctness is not inline-verifiable. Mean time 215.9 s/problem, max 710 s.

### H.3 Branch A2 - Trunk + Conservative Challenger + Pairwise Resolver

**File:** `aimo3_contested_resolver_conservative`

**Design intent (verbatim):**

> "The first Branch A run clarified four things: (1) Candidate tables are useful. They make it obvious when the correct answer is already in the pool. (2) The resolver is NOT reliable enough to hard-override the pipeline. In several cases the rerank or stage-1 winner was better than the resolver. (3) Single-cluster early agreement is still dangerous. Some hard problems produced 4 quick agreements on one wrong answer with no serious alternative exploration. (4) The reranker is better used to nominate a challenger than to become the final answer."

**Key changes from Branch A:**
1. Single-cluster challenge mode: if generation stops early with only one candidate, run up to two extra challenge attempts (one diversity-oriented, one exact-check-oriented), each with a hard cap.
2. Stricter `safe_to_ship`: requires single-cluster challenge to have passed when applicable.
3. Suspicion-based contested detector: does not trigger on one weak signal alone; accumulates a suspicion score from multiple evidence sources.
4. Rerank only nominates a challenger (never auto-ships).
5. Pairwise resolver: compares only incumbent vs challenger; forbidden to invent a third answer; may flip only if it explicitly chooses the challenger; if undecided or chooses incumbent, keeps incumbent.
6. `safe_weight_share = 0.70` - explicitly labeled "heuristic, not hidden-set tuned, not paper-derived optima."

**Execution status:** Same as `aimo3-contested-resolver` - ran 50 problems on hard50 but without per-problem correctness logging. The outputs are byte-identical to the earlier contested-resolver run (same output hash), suggesting this notebook saved the same run's outputs without re-executing.

### H.4 Branch A3 - Trunk + Closure-Aware Challenge + Dual-Check Flip Gate

**Files:** `aimo3_A3_1_clean_multicell`, `aimo3_A3_2_candidategen_multicell`, `aimo3_A3_2_patch_multicell`

**Design intent (verbatim from markdown cell):**

> "Preserve the stable baseline generator while making post-generation decisions much more conservative and closure-aware. **The design is intended for AIMO3 itself, not for hard50-only optimization.**"

**Key changes from Branch A2:**
1. Do NOT stop an attempt at the first `\boxed{}` - collect all boxed integers seen in an attempt and use the LAST one; record box instability signals (multiple boxed values, box_changed flag, last-box position). Targets premature-boxing / double-counting / off-by-one failures.
2. De-anchored single-cluster challenge: if only one answer cluster exists, run two fresh-from-scratch extra attempts - (a) diverse solve with no incumbent mentioned, (b) closure-audit solve with no incumbent mentioned. Both prompts explicitly remind the model to check symmetry, ordered-vs-unordered counting, endpoint inclusion, and final-step closure before boxing.
3. Strict safe-to-ship: safe only if exactly one candidate remains after challenge, support is strong, and the cluster is box-stable.
4. Challengers must have support ≥ 2 (unless stage-1 support ≤ 2) - prevents noisy singleton challengers from overturning healthy incumbents.
5. Flips require TWO independent checks: pairwise exact check (incumbent vs challenger only, cannot invent a third answer) AND an independent fresh audit (solve from scratch without seeing the candidates). Flip only if both checks agree on the challenger.
6. Time discipline preserved: hard problems still get up to 900 s; extra auditing capped and skipped when time is tight.

**Constants explicitly labeled:** "not hard50-fit coefficients. They are conservative priors chosen from observed failure classes."

**Execution status:** All three A3 notebooks have zero outputs. Never executed end-to-end.

### H.5 Branch B - Trunk + Pairwise-Authorized Arbitration + Bounded Checks

**Files:** `aimo3_branch_B_fixed`, `aimo3_branch_B_updated`, **`aimo3_branch_B_multicell`**

**Design intent (verbatim from markdown cell):**

> "Preserve the stable AIMO3 generator while making post-generation arbitration stricter and more faithful to the candidate pool. **This branch is intended for AIMO3-level problems, not hard50-only tuning.**"

**Key changes:**
1. **Pairwise becomes the only component that can authorize a challenger flip.**
2. Pressure becomes a challenger stress-test / veto, not an independent flip engine.
3. Audit remains candidate-bounded and can only confirm or veto a pairwise-authorized challenger.
4. Narrow strong-challenger override: allowed only when pairwise already picked the challenger AND the incumbent does not dominate support.
5. Challenge-created rivals still force escalation, but adjacency / double-half only trigger extra checking, not easier flips.
6. Trace writing remains non-fatal; final cell zips the trace folder for download.

**Execution status - this is the one verified data point:**
- `aimo3_branch_B_multicell`: **Completed a full 50-problem hard50 run.** 27 correct, 23 wrong, **27/50 = 54.0%**. Total wall-clock: 194.4 min. Mean time per problem: 233.3 s, median 188 s, max 659 s, min 27 s. 10 problems took ≥ 400 s. **Zero problems hit the 900 s cap** - the 27/50 is not a timeout artifact, it reflects actual answer quality on the hard50 distribution. All per-problem `[EVAL]` lines with `pred`, `actual`, `correct`, and `cumulative` are preserved inline in the notebook outputs.
- `aimo3_branch_B_fixed`, `aimo3_branch_B_updated`: Share stale identical outputs with `aimo3_branch_C_final` from a Colab run with trace timestamp `20260409_092715`, truncated at problem 30, cumulative 19/29 (65.5%). Different code bodies, same saved outputs - we cannot attribute the 19/29 to any specific architecture.

**Figure 9 in the main paper is based on the Branch B multicell run.**

### H.6 Branch C - Controller + Code Rescue + Provenance-Aware Arbitration

**Files:** `aimo3_branch_C_final`, `aimo3_branch_C_c2`, `aimo3_branch_C_trust_patch`, `aimo3_branch_C_fixed2`

**Design intent (verbatim from `CFG` docstring):**

> "Branch C keeps the stable Branch B generator and pairwise-centered arbitration, but adds four portable ideas motivated by the AIMO2 analysis and the recent Branch B traces..."

**Key changes from Branch B:**
1. **Convergence / divergence / broken-state controller** - after the first few attempts, detect whether the candidate pool has already converged, fragmented, or broken (invalid-heavy / no useful generation). Stop spending more budget on repeated plain-text attempts when the pool is obviously converged or broken.
2. **Targeted code rescue** - when the pool is broken, fragmented, or definition-ambiguous, switch to an exact tool-backed rescue path rather than more of the same text sampling.
3. **One-step code repair** - if the rescue path reaches Python but fails to finish cleanly, one repair attempt that fixes the prior code / extraction instead of restarting.
4. **Provenance-aware weighting** - candidates supported by multiple evidence modes (main pool, challenge, code rescue, repair) are stronger than plain text-only singleton rivals. Verifier-only candidates are kept weak unless also supported elsewhere.
5. Pairwise remains the main discriminator; audit is weakened when pairwise + pressure already align.
6. Larger reserve and lighter verification spending to protect tail problems.

**Execution status:** `C_final` shares stale outputs with `B_fixed/B_updated` (see H.5 caveat). `C_c2`, `C_trust_patch`, and `C_fixed2` have zero outputs - never executed end-to-end. No new verifiable score.

### H.7 Branch C+ - Trusted Main-Pool Arbitration + Routed Micro-Prompts + Typed Verifiers

**File:** `aimo3_branch_C_plus_final`

**Design intent (verbatim from `CFG` docstring):**

> "Design constraints: **This branch is intended for AIMO3-level problems, not hard50-only tuning.** Pairwise remains the main discriminator in binary contests. Audit is weakened relative to Branch B when pairwise + pressure already align. Tail problems are protected by a larger reserve and by lighter verification spending on already-settled problems."

**Key changes:** Inherits the full Branch C controller + code rescue + repair + provenance weighting, then adds:
- Trusted main-pool arbitration: the main pool is trusted more strongly; other stages become confirmatory.
- Routed micro-prompts: short targeted prompts routed by inferred failure class.
- Typed verifiers: verifiers specialized for specific failure classes observed in earlier traces.

This is the most elaborate architecture in the 15-branch catalog.

**Execution status:** Zero outputs. Authored but never executed to completion.

### H.8 What the catalog tells us about our workflow

Of 15 offline research branches authored over the course of the competition:

- **1 produced a verified inline-graded hard50 score** (Branch B multicell: 27/50, 54%)
- **2 ran 50 problems without correctness logging** (contested resolver A / A2 - no verifiable score)
- **3 share stale outputs from one truncated Colab session** (B_fixed, B_updated, C_final - 19/29 ambiguous attribution)
- **9 were authored but never executed to completion** (the two `branch_a` variants, all three A3 notebooks, C_c2, C_fixed2, C_trust_patch, C_plus_final)

In other words, **60% of our offline research architecture never produced a single hard50 data point**. The ones that did were either non-graded (40% of the remaining six) or attribution-ambiguous (50% of the remaining six). We learned a lot from reading the traces and tuning heuristics, but we did not run a clean offline benchmark loop, and the result was that the only number we can defend in this paper is the one Branch B multicell saved inline. The methodological finding in Section 4 stands on that single data point plus the nine "intended for AIMO3, not hard50" design-intent statements preserved verbatim above.

The full verbatim docstrings and markdown cells quoted in this appendix are reproduced from the actual `.ipynb` files and can be verified byte-for-byte against the notebook sources supplied with this writeup.

---
---

# Streamlined branch naming guide

For consistency in this writeup, the team's notebook history has been mapped to canonical names. The mapping is as follows:

| Canonical name (used in writeup) | Kaggle notebook name & version | Submitted? | Score(s) |
|---|---|---|---|
| **Final submission** | `final_submission v2` | Yes (×9) | 39, 40, 40, 40, 41, 41, 41, 42, 43 |
| **42-notebook fork (V11)** | `base_submission v2` | Yes (×1) | 33 |
| **GRPO submission** | `[15/15] AIME 2026 I 120b in 20mins v4_grpo` | Yes (×1) | 13 |
| **Adaptive router** | `AIMO3_Adaptive_Router_Confidence_Vote_v1` | Yes (×1) | 28 |
| **5-stage AIME scaffold** | `[15/15] AIME 2026 I 120b in 20mins v3` | Yes (×2) | 35, 38 |
| **Strategy-prompt family** | `aimo3_final v7 / v9 / v22` | Yes (×5) | 36, 37, 37, 37, 36 |
| **UNDERSTAND/EXPLORE adaptive** | `aimo3_adaptive _v4` | Yes (×1) | 34 |
| **Multi-temperature tester** | `aimo_tester v2` | Yes (×1) | 34 |
| **Early iteration** | `aimo3_final v3` | Yes (×2) | 39, 34 |
| **Debug / temp tweaks** | `aimo3_final v4_tweaked_temp&model`, `v5_tweaked_temp` | Yes (×2) | 11, 0 |

The name `final_submission v2` is preserved as-is per team preference. Submissions earlier than 25 days before writeup (3 × `submission.parquet` uploads at score 0) and one notebook exception (`aimo_check v4`) are excluded from analysis.

---

*End of writeup. All scores are from the public AIMO3 leaderboard. Configurations are reproduced from the verbatim source code of each submitted notebook. The GRPO telemetry CSV (190 rows) is provided as a supplementary artifact alongside this writeup.*
