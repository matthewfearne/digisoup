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

## v2 -- Original DigiSoup (perception + state + action rules)

**Agent:** Original DigiSoup design from spec. Perception (entropy, gradients,
agent/resource detection, change detection), internal state (energy, cooperation
tendency, role emergence), 6-rule entropy-gradient action selection.

**vs v1:** PD +31% avg (all 6 up). Clean Up _0/2/3 +36-48%. Clean Up _7 -67%.

| Substrate | Scenario | Focal Per-Capita | 95% CI | vs v1 |
|-----------|----------|-----------------|--------|-------|
| commons_harvest__open | _0 (5f/2bg) | 1.34 | +/- 1.00 | +24% |
| commons_harvest__open | _1 (5f/2bg) | 1.88 | +/- 1.37 | -28% |
| clean_up | _0 (3f/4bg) | 143.10 | +/- 57.34 | +48% |
| clean_up | _1 (4f/3bg) | 0.00 | +/- 0.00 | -- |
| clean_up | _2 (3f/4bg) | 57.40 | +/- 16.36 | +40% |
| clean_up | _3 (3f/4bg) | 50.43 | +/- 14.74 | +36% |
| clean_up | _4 (6f/1bg) | 0.00 | +/- 0.00 | -- |
| clean_up | _5 (5f/2bg) | 0.00 | +/- 0.00 | -- |
| clean_up | _6 (6f/1bg) | 0.00 | +/- 0.00 | -- |
| clean_up | _7 (2f/5bg) | 31.15 | +/- 31.33 | -67% |
| clean_up | _8 (6f/1bg) | 12.72 | +/- 2.95 | -4% |
| prisoners_dilemma | _0 (1f/7bg) | 10.64 | +/- 4.63 | +29% |
| prisoners_dilemma | _1 (7f/1bg) | 8.73 | +/- 3.07 | +38% |
| prisoners_dilemma | _2 (6f/2bg) | 4.25 | +/- 1.06 | +14% |
| prisoners_dilemma | _3 (1f/7bg) | 10.15 | +/- 3.44 | +56% |
| prisoners_dilemma | _4 (1f/7bg) | 11.85 | +/- 4.94 | +26% |
| prisoners_dilemma | _5 (3f/5bg) | 8.85 | +/- 2.20 | +24% |

---

*Further versions will be added as each entropy layer is built and tested.*
