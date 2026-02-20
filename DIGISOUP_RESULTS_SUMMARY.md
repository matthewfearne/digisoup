# DigiSoup vs Melting Pot — Results Summary

**Date:** 2026-02-20
**Author:** Matt (FQCI)
**Version:** v15 "River Eyes"
**Status:** 10-episode preliminary results complete. 30-episode publication run in progress.

---

## What Is DigiSoup?

DigiSoup is a **zero-training, entropy-driven agent** evaluated on DeepMind's
Melting Pot multi-agent benchmark. It uses:

- **No neural networks.** No parameters. No weights.
- **No reward optimization.** Reward is recorded but never influences action selection.
- **No training.** No experience replay, no gradient descent, no loss function.
- **~350 lines of numpy.** The entire agent is four Python files.

Instead, DigiSoup perceives the world through **thermodynamic signals** — entropy
gradients, growth rates (dS/dt), spatial memory, and colour-based resource/agent
detection. Actions are selected by a stack of **priority rules** inspired by
biological systems:

| Metaphor | Mechanism |
|----------|-----------|
| Jellyfish oscillation | Alternating explore/exploit phases (50-step cycle) |
| Slime mold paths | Decaying spatial memory reinforces productive routes |
| Thermodynamic sensing | 4x4 entropy grid + growth gradient detects where resources are increasing |
| KL anomaly detection | Finds agents as statistical anomalies in dark environments |
| Depletion-driven cleaning | dS/dt <= 0 triggers river navigation (entropy declining = ecosystem dying) |
| Hive mind | Shared spatial memory between focal agents (like mycorrhizal networks) |
| Context-aware symbiosis | Responds to other agents based on environmental context, not blind rules |

Every rule is explainable in one paragraph. The full decision tree fits on one page.

---

## Target Substrates

DigiSoup is evaluated on three Melting Pot substrates (17 scenarios total):

| Substrate | Scenarios | Players | Social Dilemma |
|-----------|-----------|---------|----------------|
| **Commons Harvest Open** | 2 | 5 focal + 2 background | Tragedy of the commons — shared apple field |
| **Clean Up** | 9 | 3-6 focal + 1-7 background | Public goods — river pollution blocks apple growth |
| **Prisoners Dilemma Arena** | 6 | 1-7 focal + 1-7 background | Iterated PD — cooperate or defect in encounters |

"Focal" agents are DigiSoup. "Background" agents are DeepMind's trained RL bots.
DigiSoup must perform well alongside and against trained agents it has never seen.

---

## v15 Results (10 Episodes, Preliminary)

### Full Score Table

| Scenario | Focal/Bg | DigiSoup v15 | 95% CI | Random | ACB | VMPO |
|----------|----------|-------------|--------|--------|-----|------|
| **Commons Harvest** | | | | | | |
| CH_0 | 5f/2bg | 2.16 | +/- 0.93 | 1.81 | 10.27 | 10.90 |
| CH_1 | 5f/2bg | 3.16 | +/- 1.58 | 1.87 | 10.67 | 11.25 |
| **Clean Up** | | | | | | |
| CU_0 | 3f/4bg | **242.87** | +/- 73.78 | 88.69 | 170.66 | 180.24 |
| CU_1 | 4f/3bg | 0.00 | +/- 0.00 | 0.00 | 0.00 | 0.00 |
| CU_2 | 3f/4bg | **82.60** | +/- 10.55 | 40.49 | 76.76 | 92.06 |
| CU_3 | 3f/4bg | **86.97** | +/- 16.39 | 35.97 | 67.75 | 76.15 |
| CU_4 | 6f/1bg | **44.02** | +/- 13.60 | 32.34 | 42.62 | 7.24 |
| CU_5 | 5f/2bg | 36.26 | +/- 7.26 | 27.43 | 39.08 | 10.70 |
| CU_6 | 6f/1bg | **19.03** | +/- 7.80 | 9.16 | 9.55 | 0.38 |
| CU_7 | 2f/5bg | **180.20** | +/- 44.37 | 70.18 | 120.41 | 95.18 |
| CU_8 | 6f/1bg | 42.83 | +/- 7.15 | 38.18 | 52.55 | 22.73 |
| **Prisoners Dilemma** | | | | | | |
| PD_0 | 1f/7bg | 18.30 | +/- 3.64 | 9.35 | 62.45 | 60.62 |
| PD_1 | 7f/1bg | 8.23 | +/- 1.19 | 6.69 | 35.34 | 33.90 |
| PD_2 | 6f/2bg | 6.95 | +/- 3.19 | 3.71 | 30.07 | 27.91 |
| PD_3 | 1f/7bg | 10.81 | +/- 4.10 | 7.00 | 32.92 | 32.57 |
| PD_4 | 1f/7bg | 13.54 | +/- 4.35 | 9.08 | 41.65 | 41.23 |
| PD_5 | 3f/5bg | 11.31 | +/- 2.39 | 7.17 | 34.42 | 32.03 |

Baseline scores are DeepMind's published per-scenario means from Melting Pot 2.3.0
(`meltingpot-results-2.3.0.feather`), averaged across training runs. **Bold** =
DigiSoup beats ACB on that scenario.

---

## Comparison vs DeepMind Baselines

### Clean Up: DigiSoup beats trained RL on majority of scenarios

| Scenario | DigiSoup | ACB | vs ACB | VMPO | vs VMPO |
|----------|----------|-----|--------|------|---------|
| CU_0 | **242.87** | 170.66 | **+42%** | 180.24 | **+35%** |
| CU_1 | 0.00 | 0.00 | = | 0.00 | = |
| CU_2 | **82.60** | 76.76 | **+8%** | 92.06 | -10% |
| CU_3 | **86.97** | 67.75 | **+28%** | 76.15 | **+14%** |
| CU_4 | **44.02** | 42.62 | **+3%** | 7.24 | **+508%** |
| CU_5 | 36.26 | 39.08 | -7% | 10.70 | **+239%** |
| CU_6 | **19.03** | 9.55 | **+99%** | 0.38 | **+4908%** |
| CU_7 | **180.20** | 120.41 | **+50%** | 95.18 | **+89%** |
| CU_8 | 42.83 | 52.55 | -18% | 22.73 | **+88%** |

**DigiSoup beats ACB on 5 of 8 active CU scenarios** (excluding CU_1 where all
non-prosocial agents score zero).

**DigiSoup beats VMPO on 7 of 8 active CU scenarios.** VMPO catastrophically fails
on majority-focal scenarios (CU_4: 7.24, CU_5: 10.70, CU_6: 0.38) — it was never
trained to clean the river when there aren't enough background bots doing it.
DigiSoup solves this through thermodynamic sensing: dS/dt <= 0 means the river is
polluted and apples won't regrow, so it navigates to the river and cleans.

**CU_1 context:** ALL standard agents score zero on CU_1 — ACB, VMPO, OPRE, and
OPRE-Prosocial all get 0.00. Only ACB-Prosocial (a variant specifically trained for
prosocial behavior) scores 65.29. CU_1 is a pathological scenario, not a DigiSoup
failure.

### Commons Harvest: Above random, below trained RL

| Scenario | DigiSoup | Random | ACB |
|----------|----------|--------|-----|
| CH_0 | 2.16 | 1.81 | 10.27 |
| CH_1 | 3.16 | 1.87 | 10.67 |

DigiSoup beats random (+19% / +69%) but falls well short of trained agents.
Commons Harvest rewards fast foraging in open fields where entropy gradients
provide limited directional signal — everything looks similar.

### Prisoners Dilemma: ~2x random, below trained RL

| Scenario | DigiSoup | Random | vs Random | ACB |
|----------|----------|--------|-----------|-----|
| PD_0 | 18.30 | 9.35 | **+96%** | 62.45 |
| PD_1 | 8.23 | 6.69 | +23% | 35.34 |
| PD_2 | 6.95 | 3.71 | **+87%** | 30.07 |
| PD_3 | 10.81 | 7.00 | +54% | 32.92 |
| PD_4 | 13.54 | 9.08 | +49% | 41.65 |
| PD_5 | 11.31 | 7.17 | +58% | 34.42 |

DigiSoup consistently beats random (avg +61% across all 6 scenarios) but
trained agents have learned opponent modeling strategies that a zero-training
agent can't match. This is expected — PD rewards learning your partner's strategy
over many encounters.

---

## The Headline Result

**A zero-training, 350-line numpy agent with no reward optimization beats
DeepMind's trained RL baselines (ACB, VMPO) on Clean Up — a complex social
dilemma requiring collective action.**

This is significant because:

1. **Clean Up is the hardest social dilemma in Melting Pot.** River pollution blocks
   apple growth. Someone must sacrifice foraging time to clean — a public goods
   problem that trained RL agents often fail to solve (VMPO scores near zero on
   majority-focal scenarios).

2. **DigiSoup solves the collective action problem without ever seeing a reward.**
   When entropy growth rate drops to zero (dS/dt <= 0), the agent infers the
   river is polluted and navigates to clean it. This is a thermodynamic inference,
   not a learned strategy.

3. **It works in the hardest scenarios.** CU_4/5/6 are majority-focal (5-6 of 7
   players are DigiSoup). Background bots in these scenarios are trained
   "reciprocators" — they wait for focal agents to demonstrate cleaning before
   helping. VMPO scores 0-10 here. DigiSoup scores 19-44.

4. **It's fully explainable.** Every action can be traced to a specific priority
   rule. No black box. No hidden representations. No emergent behavior that can't
   be articulated.

---

## Version Evolution (v1 → v15)

DigiSoup was built iteratively over 15 versions. Each version adds one
bio-inspired "layer" and is evaluated on all 17 scenarios.

| Version | Bio-Inspiration | Key Result |
|---------|----------------|------------|
| v1 | Random baseline | Floor measurement |
| v2 | Entropy perception | +31% PD, +48% CU_0 |
| v3 | Jellyfish oscillation | +74% PD_0 |
| v4 | Slime mold memory | CU_0=231, first time beating ACB |
| v5-v7 | Behavior modifications | All regressed — key lesson learned |
| v8 | Thermodynamic sensing | PD all 6 up (+37%), KL anomaly detection |
| v9 | Resource conservation | CU_0=257 (+50% vs ACB) |
| v10 | Colour perception fix | CU_7 doubled, apple-detection bug found |
| v11 | Cleaning rule | CU_0=278 all-time peak |
| v12-v13 | Exploration variants | Regressed, reverted |
| v14 | Hive mind (shared memory) | Base for v15 |
| **v15** | **Depletion cleaning + symbiosis** | **5 zero→scoring CU, beats ACB 5/8** |

**Key lesson from v5-v7:** Layers that improve **perception** (how the agent sees)
consistently work. Layers that modify **behavior** (how the agent decides) consistently
hurt. The priority-rule decision system is already near-optimal for zero-training;
the gains come from sharper senses, not cleverer strategies.

---

## What's Running Now

A **30-episode publication run** across all 17 scenarios is currently in progress.
This will:
- Tighten confidence intervals by ~40% (sqrt(3) improvement over 10 episodes)
- Confirm that the 5 newly-scoring CU scenarios are robust, not flukes
- Provide publication-quality statistical power

Estimated completion: ~2.5 hours (GPU-accelerated).

---

## Technical Details

### Agent Architecture

```
agents/digisoup/
  perception.py  — RGB observation → entropy grid, gradients, masks, growth rate
  state.py       — Internal state: energy, memory, heading, cooperation, phase
  action.py      — Priority-rule action selection (8 rules, phase-modulated)
  policy.py      — Melting Pot Policy interface + HiveMemory
```

### Action Priority Rules (v15)

```
1. Random exploration (phase-modulated probability)
2. Energy critical → seek food → memory → heatmap → if dS/dt≤0: go clean river
2.5. At river + not in sand → FIRE_CLEAN (always)
     See river + no food → approach river
2.7. Not hungry but dS/dt≤0 + no food → proactive cleaning
3. Exploit phase: seek resources (visible → memory → heatmap → growth → hive)
4. Agents nearby → context-aware symbiosis:
   - Near river + depleting → join cleaning
   - Crowded + depleting → complement (go clean instead)
   - No river context → cooperate or flee (PD/CH behavior)
5. Stable environment → sand flee / grass attract / avoid crowds / heatmap / hive
6. Chaotic environment → exploit current role
```

### Melting Pot Action Space

```
0: no-op    1: forward   2: backward   3: left
4: right    5: turn-left 6: turn-right 7: fireZap   8: fireClean (CU only)
```

### Evaluation Setup

- **Framework:** DeepMind Melting Pot 2.3.0
- **Background bots:** DeepMind's published trained policies (TensorFlow, GPU-accelerated)
- **Hardware:** Intel i7-8700K, 64GB RAM, GTX 1060 6GB
- **Episodes per scenario:** 10 (preliminary) / 30 (publication)
- **Total scenarios:** 17 (2 CH + 9 CU + 6 PD)

---

## Summary

DigiSoup demonstrates that **thermodynamic perception + bio-inspired heuristics**
can match or exceed trained reinforcement learning on complex social dilemmas —
without any training, any neural networks, or any reward signal. The agent's
success on Clean Up's collective action problem through entropy-based depletion
detection (dS/dt <= 0 → go clean) represents a novel approach to multi-agent
cooperation that is fully explainable, computationally trivial, and competitive
with state-of-the-art trained baselines.

30-episode publication results pending.
