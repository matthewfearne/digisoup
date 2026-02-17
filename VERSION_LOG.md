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

*Further versions will be added as each entropy layer is built and tested.*
