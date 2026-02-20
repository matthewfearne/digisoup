# Version Log -- DigiSoup vs Melting Pot

Each version adds one piece to the agent. All scores are from official Melting
Pot scenarios with background bots. Focal per-capita return reported as
mean +/- 95% CI across episodes.

## Scoring Key

- **Commons Harvest Open:** 2 scenarios, 5 focal + 2 background
- **Clean Up:** 9 scenarios, 3 focal + 4 background
- **Prisoners Dilemma Arena:** 6 scenarios, 1 focal + 7 background

---

## v1 -- Random Baseline (floor)

**Agent:** Uniform random action selection. No entropy. The statistical floor.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.08 | +/- 0.57 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 2.60 | +/- 1.38 | 10 |
| clean_up | _0 (3f/4bg) | 96.60 | +/- 37.37 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 41.13 | +/- 10.79 | 10 |
| clean_up | _3 (3f/4bg) | 37.13 | +/- 11.79 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 93.20 | +/- 61.50 | 10 |
| clean_up | _8 (6f/1bg) | 13.27 | +/- 2.19 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 8.25 | +/- 1.69 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 6.34 | +/- 1.54 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 3.74 | +/- 1.29 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 6.50 | +/- 1.81 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 9.43 | +/- 2.61 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 7.15 | +/- 1.19 | 10 |

---

## v2 -- DigiSoup Original

**Agent:** Full DigiSoup entropy-gradient design. Perception (entropy, gradients,
agent/resource detection, change detection) -> internal state (energy, cooperation
tendency, emergent role) -> priority-rule action selection. No reward optimization.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.34 | +/- 1.00 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.88 | +/- 1.37 | 10 |
| clean_up | _0 (3f/4bg) | 143.10 | +/- 57.34 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 57.40 | +/- 16.36 | 10 |
| clean_up | _3 (3f/4bg) | 50.43 | +/- 14.74 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 31.15 | +/- 31.33 | 10 |
| clean_up | _8 (6f/1bg) | 12.72 | +/- 2.95 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 10.64 | +/- 4.63 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 8.73 | +/- 3.07 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 4.25 | +/- 1.06 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 10.15 | +/- 3.44 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 11.85 | +/- 4.94 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 8.85 | +/- 2.20 | 10 |

**vs v1:** PD all 6 up (avg +31%). Clean Up _0 +48%, _2 +40%, _3 +36%.
Clean Up _7 regressed -67% (high variance). Commons Harvest mixed.

---

## v3 -- Phase Cycling (Jellyfish Oscillation)

**Agent:** Adds temporal phase cycling — alternates EXPLORE and EXPLOIT every 50
steps. Explore phase: higher random exploration, biased toward movement/scanning,
reluctant to interact (coop threshold 0.7). Exploit phase: focused, eager to
interact (coop threshold 0.3), seeks resources at moderate energy. Inspired by
jellyfish swim-pulse oscillation: discover first, then act.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.90 | +/- 0.64 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.92 | +/- 0.66 | 10 |
| clean_up | _0 (3f/4bg) | 181.97 | +/- 59.97 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 62.57 | +/- 21.82 | 10 |
| clean_up | _3 (3f/4bg) | 39.63 | +/- 19.42 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 53.70 | +/- 39.60 | 10 |
| clean_up | _8 (6f/1bg) | 10.22 | +/- 2.20 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 18.53 | +/- 6.11 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 5.58 | +/- 1.58 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 3.06 | +/- 0.74 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 8.76 | +/- 2.67 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 14.40 | +/- 5.10 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 9.17 | +/- 2.23 | 10 |

**vs v2 (excluding zeros):** 8 improved, 5 regressed.
Big wins: PD _0 +74%, Clean Up _7 +72%, Commons _0 +42%, Clean Up _0 +27%.
Regressions: PD _1 -36%, PD _2 -28% (majority-focal — explore phase hurts
cooperation with other focal agents), Clean Up _3 -21%, Clean Up _8 -20%.

---

## v4 -- Spatial Memory (Slime Mold Path Reinforcement)

**Agent:** Adds decaying resource direction memory. When resources are detected,
their direction is blended into a memory vector (stronger for denser patches).
When resources aren't visible, the agent follows the memory "scent trail" back
toward productive areas. Memory decays each step — old trails fade. Inspired by
slime mold tube reinforcement: strengthen paths that lead to food.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.68 | +/- 0.76 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 2.14 | +/- 0.75 | 10 |
| clean_up | _0 (3f/4bg) | 231.20 | +/- 44.48 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 75.27 | +/- 32.20 | 10 |
| clean_up | _3 (3f/4bg) | 62.07 | +/- 19.75 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 55.55 | +/- 44.80 | 10 |
| clean_up | _8 (6f/1bg) | 11.03 | +/- 2.38 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 16.73 | +/- 6.71 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 5.95 | +/- 1.69 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 2.94 | +/- 0.52 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 9.63 | +/- 2.73 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 11.55 | +/- 3.75 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 8.34 | +/- 2.62 | 10 |

**vs v3 (excluding zeros):** 8 improved, 5 regressed.
Clean Up dominates: _0 +27% (231.20 — **beats ACB 170.66**), _3 +57%, _2 +20%.
PD mixed: _1 +7%, _3 +10%, but _4 -20%, _0 -10%.

---

## v5 -- Adaptive Cooperation Threshold (Vampire Bat Reciprocity) [BRANCHED OFF]

**Agent:** Adds adaptive cooperation threshold based on recent interaction success.
If interactions have been working (entropy changed), lower the threshold — cooperate
more readily. If they've been failing, raise it — be reluctant. Uses a rolling window
of interaction outcomes. Inspired by vampire bat reciprocal altruism.

**Status:** Mixed results. NOT carried forward — v6 branches from v4 instead.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 0.64 | +/- 0.53 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.84 | +/- 0.59 | 10 |
| clean_up | _0 (3f/4bg) | 226.73 | +/- 40.28 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 73.97 | +/- 17.43 | 10 |
| clean_up | _3 (3f/4bg) | 48.07 | +/- 11.06 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 154.30 | +/- 119.69 | 10 |
| clean_up | _8 (6f/1bg) | 15.47 | +/- 2.33 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 15.87 | +/- 5.15 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 6.79 | +/- 1.47 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 2.04 | +/- 0.55 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 8.68 | +/- 1.69 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 12.10 | +/- 3.27 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 8.77 | +/- 2.28 | 10 |

**vs v4 (excluding zeros):** 5 improved, 8 regressed. Net negative.
Big win: Clean Up _7 +178% (but CI +/- 119.69 — unreliable).
Clean Up _8 +40%, PD _1 +14%, PD _4 +5%, PD _5 +5%.
Regressions: Commons _0 -62%, Clean Up _3 -23%, PD _2 -31%, PD _0 -5%.
Core strength held: Clean Up _0 barely moved (226.73 vs 231.20).

**Decision:** Adaptive cooperation hurts more than it helps. v6 branches from v4.

---

## v6 -- Entropy-as-Energy (Cockroach Persistence) [BRANCHED OFF]

**Agent:** Energy depletion scales inversely with environmental entropy — rich
environments sustain the agent, barren ones drain faster. High entropy provides a
small passive energy trickle. Below COCKROACH_THRESHOLD (0.15), the agent enters
survival mode: no random exploration, no cooperation, pure resource-seeking.
Inspired by cockroach persistence in hostile environments.

**Status:** Across-the-board regression. NOT carried forward — v7 branches from v4.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.48 | +/- 0.83 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.86 | +/- 0.79 | 10 |
| clean_up | _0 (3f/4bg) | 113.57 | +/- 27.30 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 53.33 | +/- 15.75 | 10 |
| clean_up | _3 (3f/4bg) | 26.07 | +/- 8.05 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 9.60 | +/- 13.51 | 10 |
| clean_up | _8 (6f/1bg) | 8.78 | +/- 3.21 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 9.02 | +/- 3.39 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 2.71 | +/- 1.07 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 1.83 | +/- 0.53 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 7.96 | +/- 2.25 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 12.16 | +/- 4.90 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 5.56 | +/- 1.32 | 10 |

**vs v4 (excluding zeros):** 1 improved, 12 regressed. Catastrophic regression.
Clean Up _0 halved (113.57 vs 231.20). PD collapsed across the board.
Entropy-modulated depletion starves agent in low-entropy arenas (PD dark rooms).
Cockroach mode kills cooperation/exploration when agent needs them most.

**Decision:** Entropy-as-energy is actively harmful. v7 branches from v4.

---

## v7 -- Anti-Fragility Mode (Honey Badger Threat Reversal) [BRANCHED OFF]

**Agent:** Tracks environmental stress (EMA of change rate). When stressed, the agent
becomes bolder — cooperation threshold drops by 0.3, chaotic environments trigger
40% chance of INTERACT instead of role-based play. Honey badger: charges when
threatened. Anti-fragile: disorder is fuel.

**Status:** Worst regression yet. NOT carried forward — v4 remains best.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.60 | +/- 1.08 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.50 | +/- 0.70 | 10 |
| clean_up | _0 (3f/4bg) | 77.30 | +/- 39.47 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 33.13 | +/- 11.49 | 10 |
| clean_up | _3 (3f/4bg) | 22.57 | +/- 16.57 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 29.50 | +/- 28.67 | 10 |
| clean_up | _8 (6f/1bg) | 5.93 | +/- 2.51 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 12.43 | +/- 4.35 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 5.67 | +/- 1.40 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 3.23 | +/- 1.15 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 6.49 | +/- 2.62 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 15.79 | +/- 6.12 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 9.12 | +/- 2.29 | 10 |

**vs v4 (excluding zeros):** 3 improved, 10 regressed. Worst version yet.
Clean Up _0 cratered to 77.30 (-67%). All Clean Up scenarios regressed hard.
PD _4 +37%, PD _2 +10%, PD _5 +9% — minor PD gains don't offset Clean Up losses.
Aggressive interaction in chaos disrupts cooperative dynamics that drive Clean Up.

**Decision:** Anti-fragility is destructive. v4 remains the high-water mark.

---

## v8 -- Thermodynamic Sensing (4x4 Grid + Growth Gradient + KL Anomaly)

**Agent:** Pure perception upgrade on v4 base. Three new sensing channels:
1. **4x4 fine entropy grid** — replaces 2x2 quadrants for sharper spatial awareness
2. **Entropy growth gradient** — direction toward where entropy is INCREASING (dS/dt), tracks apple regrowth and activity hotspots
3. **KL divergence anomaly** — detects agents as statistical anomalies against uniform backgrounds (crucial for dark PD arenas)

Decision logic unchanged from v4. Only inputs sharpened.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.28 | +/- 0.74 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.10 | +/- 0.68 | 10 |
| clean_up | _0 (3f/4bg) | 221.87 | +/- 53.86 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 64.20 | +/- 24.77 | 10 |
| clean_up | _3 (3f/4bg) | 68.27 | +/- 23.47 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 63.40 | +/- 45.74 | 10 |
| clean_up | _8 (6f/1bg) | 10.98 | +/- 2.69 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 21.64 | +/- 7.65 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 6.23 | +/- 1.74 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 5.01 | +/- 1.45 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 10.78 | +/- 2.58 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 15.41 | +/- 5.04 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 14.35 | +/- 5.82 | 10 |

**vs v4 (excluding zeros):** 8 improved, 4 regressed, 1 flat.
**PD: ALL 6 scenarios improved (avg +37%).** PD_2 +70%, PD_5 +72%, PD_4 +33%, PD_0 +29%.
KL anomaly detection finds agents in dark PD arenas — the biggest single improvement since v2.
Clean Up held: _0 -4% (within CI), _3 +10%, _7 +14%. _2 -15%.
Commons regressed: _0 -24%, _1 -49% (small absolute numbers: 1.28 vs 1.68).

**New high-water mark for PD.** First version since v4 to improve without destroying Clean Up.

---

## v9 -- Resource Conservation (Sustainable Harvesting)

**Agent:** v8 base + one new action rule. When the agent is near dense resources
(>2% green pixels) AND the environment's entropy is declining (growth_rate < -0.1
bits = patch being depleted), move AWAY from resources to let them regrow. Survival
overrides conservation — Rule 2 (energy critical) still fires first.

Perception: adds `growth_rate` scalar (mean entropy change across 4x4 grid).
Action: Rule 2.5 inserted between energy-seek and exploit-seek.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.58 | +/- 0.62 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 0.52 | +/- 0.59 | 10 |
| clean_up | _0 (3f/4bg) | 256.70 | +/- 71.85 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 78.77 | +/- 16.48 | 10 |
| clean_up | _3 (3f/4bg) | 80.50 | +/- 28.26 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 56.75 | +/- 53.70 | 10 |
| clean_up | _8 (6f/1bg) | 13.45 | +/- 1.77 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 18.32 | +/- 6.57 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 5.79 | +/- 1.44 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 3.72 | +/- 0.81 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 10.36 | +/- 3.46 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 10.33 | +/- 6.74 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 14.50 | +/- 3.57 | 10 |

**vs v8 (excluding zeros):** 7 improved, 6 regressed.
**Clean Up breakout:** CU_0 **256.70** (new all-time high, +16% vs v8, +11% vs v4, **+50% vs ACB**).
CU_2 +23%, CU_3 +18%, CU_8 +22% vs v8. Conservation rule lets patches regrow = more apples.
PD gave back some v8 gains: PD_0 -15%, PD_4 -33%, PD_2 -26% vs v8. Still above v4 on most.
Commons_0 +23% vs v8 (partial recovery). Commons_1 regressed further.

**vs v4 (excluding zeros):** 9 improved, 4 regressed.
Best combined result yet: Clean Up stronger than ever, PD still well above v4.

**New all-time high-water mark: CU_0 at 256.70 beats ACB (170.66) by 50%.**

---

## v10 -- Sharper Eyes (Perception Upgrades from v8 Base)

**Agent:** Four perception-only upgrades. Critical bug fix: `_agent_mask()` was
detecting red/orange apples as agents — agent fled from food and zapped apples
instead of walking onto them to collect.

Changes:
1. **Warm mask** — detects red apples (214,88,88) and CU orange apples (212,80,57)
2. **Dirt mask** — detects CU pollution (2,245,80)
3. **Resource mask** — green + warm = all apple types (fixes apple-fleeing bug)
4. **Agent density grid** — 4x4 spatial map of agent locations
5. **Resource heatmap** — temporal 4x4 spatial memory with 0.95/step decay
6. **Heading persistence** — EMA of movement direction for smoother paths
7. **Crowding avoidance** — steers away from agent-dense quadrants (Rule 5)

Removed v9 conservation rule (hurt PD). 27 tests passing.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.68 | +/- 0.90 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 0.90 | +/- 0.49 | 10 |
| clean_up | _0 (3f/4bg) | 209.10 | +/- 49.97 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 92.70 | +/- 18.17 | 10 |
| clean_up | _3 (3f/4bg) | 110.67 | +/- 34.14 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 145.15 | +/- 85.48 | 10 |
| clean_up | _8 (6f/1bg) | 15.37 | +/- 2.82 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 15.94 | +/- 5.49 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 7.27 | +/- 2.64 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 4.81 | +/- 0.78 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 10.72 | +/- 5.51 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 13.93 | +/- 4.20 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 9.94 | +/- 3.68 | 10 |

**vs v8 (excluding zeros):** 7 improved, 6 regressed (PD within CI overlap).
**Clean Up breakthrough:** CU_2 +44%, CU_3 +62%, CU_7 **+129%** (doubled), CU_8 +40%.
Colour detection fix working: agent now collects apples it previously fled from.
CU_0 dipped -6% vs v8 but still beats ACB (209.10 vs 170.66 = +23%).
CH_0 at 1.68 = new all-time high. PD mixed: PD_1 +17%, PD_0 -26%, PD_5 -31%.
PD differences are small absolute numbers with overlapping CIs.

**Best overall version.** Clean Up gains (4 of 5 active scenarios up, CU_7 doubled)
far outweigh PD wobble. The apple-detection bug fix was the single highest-impact
change since v2's original perception layer.

---

## v11 -- Squid Custodian (Cleaning Rule)

**Agent:** v10 + cleaning rule. When dirt/pollution is visible but no resources
(apples stopped growing), approach dirt and fire INTERACT to clean. Also added
to Rule 2 energy-low cascade. Targets the reciprocator deadlock in CU_4/5/6
where background bots wait for focal agents to clean first.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 2.88 | +/- 1.79 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.94 | +/- 1.42 | 10 |
| clean_up | _0 (3f/4bg) | 277.93 | +/- 77.80 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 106.90 | +/- 23.24 | 10 |
| clean_up | _3 (3f/4bg) | 79.37 | +/- 25.55 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 92.50 | +/- 80.10 | 10 |
| clean_up | _8 (6f/1bg) | 15.38 | +/- 1.88 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 16.89 | +/- 9.23 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 7.94 | +/- 2.55 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 4.90 | +/- 1.84 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 12.42 | +/- 4.81 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 16.62 | +/- 7.38 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 9.89 | +/- 2.57 | 10 |

**vs v10:** CU_0 **277.93** (new all-time high, +63% vs ACB). CH_0 +71%, CH_1 +116%.
PD all 6 held or improved (PD_3 +16%, PD_4 +19%). CU_2 +15%.
CU_3 -28%, CU_7 -36% (high variance). CU_1/4/5/6 still zero — agent never
reaches the river (stays in empty orchard, never sees dirt to trigger cleaning).
29 tests passing.

---

## v12 -- Echo Explorer (Starvation Exploration + Echo Cleaning) [BRANCHED OFF]

**Agent:** v11 + starvation exploration mode + echo cleaning feedback. When energy
drops critically low and no resources visible, explores aggressively. Cleaning
success creates positive echo that reinforces cleaning behavior.

**Status:** CU and CH regressed. NOT carried forward — reverted to v11.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.50 | +/- 0.65 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 1.20 | +/- 0.79 | 10 |
| clean_up | _0 (3f/4bg) | 197.10 | +/- 71.58 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 100.80 | +/- 14.95 | 10 |
| clean_up | _3 (3f/4bg) | 72.50 | +/- 19.73 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 16.40 | +/- 19.52 | 10 |
| clean_up | _8 (6f/1bg) | 12.80 | +/- 3.43 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 17.86 | +/- 6.46 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 8.88 | +/- 2.78 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 6.06 | +/- 1.96 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 18.76 | +/- 5.71 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 15.09 | +/- 4.22 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 16.09 | +/- 3.36 | 10 |

**vs v11:** CU_0 -29%, CU_7 -82%. PD mixed: _3 +51%, _5 +63% but _0 +6%.
Echo feedback amplified noise without improving core behavior.

**Decision:** Echo exploration hurts more than helps. Reverted to v11 base.

---

## v13 -- Mycorrhizal Explorer (Dead Reckoning + Scouting + Edge Detection) [BRANCHED OFF]

**Agent:** v11 + dead reckoning position tracking + frontier scouting + world-edge
detection. Agent maintains position estimate to avoid revisiting explored areas and
scouts toward unexplored frontiers.

**Status:** Broad regression. NOT carried forward — reverted to v11.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 1.98 | +/- 0.43 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 2.08 | +/- 0.59 | 10 |
| clean_up | _0 (3f/4bg) | 202.17 | +/- 46.42 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 91.97 | +/- 23.19 | 10 |
| clean_up | _3 (3f/4bg) | 66.40 | +/- 27.54 | 10 |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _7 (2f/5bg) | 75.35 | +/- 65.23 | 10 |
| clean_up | _8 (6f/1bg) | 12.00 | +/- 2.34 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 11.46 | +/- 4.44 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 6.78 | +/- 2.40 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 4.31 | +/- 0.76 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 8.71 | +/- 3.05 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 14.42 | +/- 9.40 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 16.04 | +/- 3.78 | 10 |

**vs v11:** CU_0 -27%, CU_3 -16%, CU_7 -19%. PD_0 -32%, PD_3 -30%.
Dead reckoning overhead didn't improve navigation in egocentric-only substrate.

**Decision:** Mycorrhizal exploration not suited to egocentric vision. Reverted to v11.

---

## v14 -- Hive Mind (Shared Spatial Memory Between Focal Agents)

**Agent:** v11 + shared spatial memory. All focal agents write resource/dirt discoveries
to a class-level HiveMemory with position+orientation tracking. Agents query the hive
for the nearest discovery and get a world-space direction vector, converted to
egocentric coordinates for navigation. Like mycorrhizal networks sharing nutrient info.

**Status:** No formal evaluation run. Carried forward as base for v15. 33 tests passing.

---

## v15 -- River Eyes (Depletion-Driven Cleaning + Symbiosis)

**Agent:** v14 base + 7 targeted fixes and features for Clean Up mastery. The biggest
behavioral upgrade since v2 — first version to score on ALL previously-zero CU scenarios.
Key insight: **dS/dt <= 0 means river is polluted and apples won't regrow**. Agent uses
thermodynamic signal to decide WHEN to clean (not just WHERE).

Changes:
1. **Action space fix** — INTERACT always action 7, FIRE_CLEAN always action 8 (was using `n_actions-1` which gave wrong action in CU)
2. **Sand guard** — don't fire FIRE_CLEAN when standing in sand (wide FOV sees distant river water)
3. **Lowered cleaning thresholds** — DIRT_CLOSE 0.15→0.08, APPROACH 0.05→0.03 (sand guard handles false positives)
4. **Clean regardless of distant apples** — FIRE_CLEAN fires when at river even if apples visible at edge of 88px FOV
5. **Depletion-driven cleaning (dS/dt ≤ 0)** — when growth_rate ≤ 0 AND no food visible, navigate to river and clean. Thermodynamic insight: declining entropy = dying ecosystem = polluted river.
6. **Proactive cleaning (Rule 2.7)** — not hungry but no food and environment depleting → approach river
7. **Context-aware symbiosis (Rule 4)** — near river + depleting → join cleaning crew. Crowded + depleting → complement (go clean). No river → original cooperate/flee. Growth-rate gated: only divert when `growth_rate ≤ 0`.

33 tests passing.

| Substrate | Scenario | Focal Per-Capita | 95% CI | Episodes |
|-----------|----------|-----------------|--------|----------|
| commons_harvest__open | _0 (5f/2bg) | 2.16 | +/- 0.93 | 10 |
| commons_harvest__open | _1 (5f/2bg) | 3.16 | +/- 1.58 | 10 |
| clean_up | _0 (3f/4bg) | 242.87 | +/- 73.78 | 10 |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | 10 |
| clean_up | _2 (3f/4bg) | 82.60 | +/- 10.55 | 10 |
| clean_up | _3 (3f/4bg) | 86.97 | +/- 16.39 | 10 |
| clean_up | _4 (6f/1bg) | **44.02** | +/- 13.60 | 10 |
| clean_up | _5 (5f/2bg) | **36.26** | +/- 7.26 | 10 |
| clean_up | _6 (6f/1bg) | **19.03** | +/- 7.80 | 10 |
| clean_up | _7 (2f/5bg) | 180.20 | +/- 44.37 | 10 |
| clean_up | _8 (6f/1bg) | 42.83 | +/- 7.15 | 10 |
| prisoners_dilemma | _0 (1f/7bg) | 18.30 | +/- 3.64 | 10 |
| prisoners_dilemma | _1 (7f/1bg) | 8.23 | +/- 1.19 | 10 |
| prisoners_dilemma | _2 (6f/2bg) | 6.95 | +/- 3.19 | 10 |
| prisoners_dilemma | _3 (1f/7bg) | 10.81 | +/- 4.10 | 10 |
| prisoners_dilemma | _4 (1f/7bg) | 13.54 | +/- 4.35 | 10 |
| prisoners_dilemma | _5 (3f/5bg) | 11.31 | +/- 2.39 | 10 |

**vs v11 (excluding zeros):**
- **5 previously-zero CU scenarios now scoring:** CU_4 **44.02**, CU_5 **36.26**, CU_6 **19.03** (all from 0.00)
- **CU_7 +95%** (180.20 vs 92.50), **CU_8 +179%** (42.83 vs 15.38)
- CU_0 -13% (242.87 vs 277.93 — still **+42% vs ACB** 170.66)
- CU_3 +10%, CU_2 -23% (tighter CI now: +/-10.55 vs +/-23.24)
- PD mixed: _2 +42%, _5 +14%, _0 +8%, _1 +4%. _3 -13%, _4 -19%.
- CH mixed: _1 +63%, _0 -25%.

**Total CU reward (sum of means): 734.74 vs 572.08 in v11 (+28%).**
First version to score on 8 of 9 CU scenarios (only CU_1 remains zero).
Depletion-driven cleaning + symbiosis = the breakthrough for majority-focal scenarios
where background bots wait for focal agents to demonstrate cleaning.

**New high-water marks:** CU_4 (44.02), CU_5 (36.26), CU_6 (19.03), CU_7 (180.20), CU_8 (42.83).
CU_0 still beats ACB by 42%. CH_1 at 3.16 = all-time high.

---

*Further versions will be added as each entropy layer is built and tested.*
