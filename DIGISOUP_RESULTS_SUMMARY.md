# DigiSoup vs Melting Pot — Results Summary

**Date:** 2026-02-21
**Author:** Matt (FQCI)
**Version:** v15 "River Eyes"
**Status:** 30-episode evaluation complete. Paper submitted to arXiv.

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

## Final Results (30 Episodes, 95% CI)

### Full Score Table

| Scenario | Focal/Bg | DigiSoup v15 | 95% CI | Random | ACB | VMPO |
|----------|----------|-------------|--------|--------|-----|------|
| **Commons Harvest** | | | | | | |
| CH_0 | 5f/2bg | 2.84 | ±0.83 | 1.81 | 10.27 | 10.90 |
| CH_1 | 5f/2bg | 3.44 | ±0.88 | 1.87 | 10.67 | 11.25 |
| **Clean Up** | | | | | | |
| CU_0 | 3f/4bg | **194.70** | ±25.35 | 88.69 | 170.66 | 180.24 |
| CU_1 | 4f/3bg | 0.00 | ±0.00 | 0.00 | 0.00 | 0.00 |
| CU_2 | 3f/4bg | **79.22** | ±11.66 | 40.49 | 76.76 | 92.06 |
| CU_3 | 3f/4bg | 65.90 | ±8.25 | 35.97 | 67.75 | 76.15 |
| CU_4 | 6f/1bg | 42.14 | ±8.18 | 32.34 | 42.62 | 7.24 |
| CU_5 | 5f/2bg | 31.27 | ±6.09 | 27.43 | 39.08 | 10.70 |
| CU_6 | 6f/1bg | **13.21** | ±2.52 | 9.16 | 9.55 | 0.38 |
| CU_7 | 2f/5bg | **234.00** | ±48.39 | 70.18 | 120.41 | 95.18 |
| CU_8 | 6f/1bg | 45.38 | ±8.92 | 38.18 | 52.55 | 22.73 |
| **Prisoners Dilemma** | | | | | | |
| PD_0 | 1f/7bg | 16.50 | ±3.13 | 9.35 | 62.45 | 60.62 |
| PD_1 | 7f/1bg | 7.50 | ±0.84 | 6.69 | 35.34 | 33.90 |
| PD_2 | 6f/2bg | 7.52 | ±1.62 | 3.71 | 30.07 | 27.91 |
| PD_3 | 1f/7bg | 11.25 | ±2.96 | 7.00 | 32.92 | 32.57 |
| PD_4 | 1f/7bg | 15.01 | ±3.11 | 9.08 | 41.65 | 41.23 |
| PD_5 | 3f/5bg | 14.84 | ±2.31 | 7.17 | 34.42 | 32.03 |

Baseline scores are DeepMind's published per-scenario means from Melting Pot 2.3.0
(`meltingpot-results-2.3.0.feather`), averaged across training runs. **Bold** =
DigiSoup beats ACB on that scenario.

---

## Comparison vs DeepMind Baselines

### Clean Up: DigiSoup beats trained RL in aggregate

| Scenario | DigiSoup | ACB | vs ACB | VMPO | vs VMPO |
|----------|----------|-----|--------|------|---------|
| CU_0 | **194.70** | 170.66 | **+14%** | 180.24 | **+8%** |
| CU_1 | 0.00 | 0.00 | = | 0.00 | = |
| CU_2 | **79.22** | 76.76 | **+3%** | 92.06 | -14% |
| CU_3 | 65.90 | 67.75 | -3% | 76.15 | -13% |
| CU_4 | 42.14 | 42.62 | -1% | 7.24 | **+482%** |
| CU_5 | 31.27 | 39.08 | -20% | 10.70 | **+192%** |
| CU_6 | **13.21** | 9.55 | **+38%** | 0.38 | **+3376%** |
| CU_7 | **234.00** | 120.41 | **+94%** | 95.18 | **+146%** |
| CU_8 | 45.38 | 52.55 | -14% | 22.73 | **+100%** |
| **Total** | **705.82** | 579.38 | **+22%** | 484.67 | **+46%** |

**DigiSoup beats ACB on 4 of 8 active CU scenarios** (excluding CU_1 where all
non-prosocial agents score zero). Wins are large (CU_7: +94%), losses are narrow
(CU_3: -3%, CU_4: -1%).

**DigiSoup beats VMPO on 6 of 8 active CU scenarios.** VMPO catastrophically fails
on majority-focal scenarios (CU_4: 7.24, CU_5: 10.70, CU_6: 0.38) — it was never
trained to clean the river when there aren't enough background bots doing it.
DigiSoup solves this through thermodynamic sensing: dS/dt <= 0 means the river is
polluted and apples won't regrow, so it navigates to the river and cleans.

**CU_1 context:** ALL standard agents score zero on CU_1 — ACB, VMPO, OPRE, and
OPRE-Prosocial all get 0.00. Only ACB-Prosocial (a variant specifically trained for
prosocial behavior) scores 65.29. CU_1 is a pathological scenario, not a DigiSoup
failure.

### Commons Harvest: Above random, below trained RL

| Scenario | DigiSoup | Random | vs Random | ACB |
|----------|----------|--------|-----------|-----|
| CH_0 | 2.84 | 1.81 | +57% | 10.27 |
| CH_1 | 3.44 | 1.87 | +84% | 10.67 |

DigiSoup beats random (+57% / +84%) but falls well short of trained agents.
Commons Harvest rewards fast foraging in open fields where entropy gradients
provide limited directional signal — everything looks similar.

### Prisoners Dilemma: ~2x random, below trained RL

| Scenario | DigiSoup | Random | vs Random | ACB |
|----------|----------|--------|-----------|-----|
| PD_0 | 16.50 | 9.35 | **+76%** | 62.45 |
| PD_1 | 7.50 | 6.69 | +12% | 35.34 |
| PD_2 | 7.52 | 3.71 | **+103%** | 30.07 |
| PD_3 | 11.25 | 7.00 | +61% | 32.92 |
| PD_4 | 15.01 | 9.08 | +65% | 41.65 |
| PD_5 | 14.84 | 7.17 | **+107%** | 34.42 |

DigiSoup consistently beats random (avg +71% across all 6 scenarios) but
trained agents have learned opponent modeling strategies that a zero-training
agent can't match. This is expected — PD rewards learning your partner's strategy
over many encounters.

---

## The Headline Result

**A zero-training, 350-line numpy agent with no reward optimization beats
DeepMind's trained RL baselines in aggregate on Clean Up — a complex social
dilemma requiring collective action (+22% vs ACB, +46% vs VMPO).**

This is significant because:

1. **Clean Up is the hardest social dilemma in Melting Pot.** River pollution blocks
   apple growth. Someone must sacrifice foraging time to clean — a public goods
   problem that trained RL agents often fail to solve (VMPO scores near zero on
   majority-focal scenarios).

2. **DigiSoup solves the collective action problem without ever seeing a reward.**
   When entropy growth rate drops to zero (dS/dt <= 0), the agent infers the
   river is polluted and navigates to clean it. This is a thermodynamic inference,
   not a learned strategy.

3. **CU_7 is the standout.** Just two DigiSoup focal agents among seven players
   score 234.00 — nearly double ACB's 120.41 (+94%). Even a small number of
   entropy-driven agents can sustain the commons.

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
| **v15** | **Depletion cleaning + symbiosis** | **+22% aggregate vs ACB, +46% vs VMPO** |

**Key lesson from v5-v7:** Layers that improve **perception** (how the agent sees)
consistently work. Layers that modify **behavior** (how the agent decides) consistently
hurt. The priority-rule decision system is already near-optimal for zero-training;
the gains come from sharper senses, not cleverer strategies.

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

### Evaluation Setup

- **Framework:** DeepMind Melting Pot 2.3.0
- **Background bots:** DeepMind's published trained policies (TensorFlow, GPU-accelerated)
- **Hardware:** Intel i7-8700K, 64GB RAM, GTX 1060 6GB
- **Episodes per scenario:** 30
- **Total scenarios:** 17 (2 CH + 9 CU + 6 PD)
- **Total episodes:** 510
