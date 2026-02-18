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

*Further versions will be added as each entropy layer is built and tested.*
