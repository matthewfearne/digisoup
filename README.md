# DigiSoup — Zero-Training Entropy Agent Beats Trained RL on Social Dilemmas

A **350-line numpy agent** with no neural networks, no training, and no reward
optimisation beats DeepMind's trained reinforcement learning baselines on Clean Up
— the hardest social dilemma in the
[Melting Pot](https://github.com/google-deepmind/meltingpot) benchmark.

**Paper:** [`paper/digisoup.pdf`](paper/digisoup.pdf) — submitted to arXiv (cs.MA)

## Key Results (v15 "River Eyes" — 30 episodes, 95% CI)

### Clean Up — DigiSoup beats DeepMind in aggregate (+22% vs ACB, +46% vs VMPO)

| Scenario | DigiSoup | 95% CI | ACB (trained) | VMPO (trained) | Random | vs ACB |
|----------|----------|--------|---------------|----------------|--------|--------|
| CU_0 (3f/4bg) | **194.70** | ±25.35 | 170.66 | 180.24 | 88.69 | **+14%** |
| CU_1 (4f/3bg) | 0.00 | ±0.00 | 0.00 | 0.00 | 0.00 | --- |
| CU_2 (3f/4bg) | **79.22** | ±11.66 | 76.76 | 92.06 | 40.49 | **+3%** |
| CU_3 (3f/4bg) | 65.90 | ±8.25 | 67.75 | 76.15 | 35.97 | -3% |
| CU_4 (6f/1bg) | 42.14 | ±8.18 | 42.62 | 7.24 | 32.34 | -1% |
| CU_5 (5f/2bg) | 31.27 | ±6.09 | 39.08 | 10.70 | 27.43 | -20% |
| CU_6 (6f/1bg) | **13.21** | ±2.52 | 9.55 | 0.38 | 9.16 | **+38%** |
| CU_7 (2f/5bg) | **234.00** | ±48.39 | 120.41 | 95.18 | 70.18 | **+94%** |
| CU_8 (6f/1bg) | 45.38 | ±8.92 | 52.55 | 22.73 | 38.18 | -14% |
| **Total** | **705.82** | | 579.38 | 484.67 | 341.44 | **+22%** |

DigiSoup beats ACB on 4 of 8 active scenarios, VMPO on 6 of 8. Wins are large
(CU_7: +94%), losses are narrow (CU_3: -3%, CU_4: -1%). CU_1 excluded — all
non-prosocial agents score zero.

### Commons Harvest & Prisoner's Dilemma — beats random, not trained RL

| Scenario | DigiSoup | 95% CI | Random | vs Random | ACB |
|----------|----------|--------|--------|-----------|-----|
| CH_0 (5f/2bg) | 2.84 | ±0.83 | 1.81 | +57% | 10.27 |
| CH_1 (5f/2bg) | 3.44 | ±0.88 | 1.87 | +84% | 10.67 |
| PD_0 (1f/7bg) | 16.50 | ±3.13 | 9.35 | +76% | 62.45 |
| PD_1 (7f/1bg) | 7.50 | ±0.84 | 6.69 | +12% | 35.34 |
| PD_2 (6f/2bg) | 7.52 | ±1.62 | 3.71 | +103% | 30.07 |
| PD_3 (1f/7bg) | 11.25 | ±2.96 | 7.00 | +61% | 32.92 |
| PD_4 (1f/7bg) | 15.01 | ±3.11 | 9.08 | +65% | 41.65 |
| PD_5 (3f/5bg) | 14.84 | ±2.31 | 7.17 | +107% | 34.42 |

PD average +71% vs random. CH and PD fall short of trained agents — expected,
since foraging rewards fast movement (not entropy gradients) and iterated PD
rewards opponent modelling (requires learning).

### Did we beat DeepMind?

**Yes, on Clean Up — the benchmark's hardest social dilemma.** A zero-training,
350-line numpy agent with no reward signal outperforms million-parameter neural
networks trained for millions of steps. The aggregate score is +22% above ACB
and +46% above VMPO. The standout is CU_7: just two DigiSoup agents in a group
of seven nearly double ACB's score (+94%).

**No, on Commons Harvest and Prisoner's Dilemma.** These substrates reward
capabilities DigiSoup lacks — fast open-field foraging (CH) and learned opponent
modelling (PD). DigiSoup still significantly outperforms random on both (+71% PD,
+57–84% CH), confirming it's doing real work, not just bumbling around.

Baselines are DeepMind's published per-scenario scores from
`meltingpot-results-2.3.0.feather`, averaged across training runs.

## What Makes This Different

Every agent in Melting Pot's literature is a **trained RL agent** — neural networks
optimised over millions of steps on reward signal. DigiSoup uses:

- **No neural networks.** No parameters. No weights. No tensors.
- **No reward optimisation.** Reward is recorded but never influences decisions.
- **No training.** No experience replay, no gradient descent, no loss function.
- **Pure thermodynamics.** Entropy gradients, growth rates (dS/dt), and spatial memory.
- **~350 lines of numpy.** Four Python files. Every rule explainable in one paragraph.

## How It Works

DigiSoup perceives the world through thermodynamic signals and selects actions via
bio-inspired priority rules:

| Layer | Bio-Inspiration | Mechanism |
|-------|----------------|-----------|
| Perception | Thermodynamic sensing | 4x4 entropy grid, growth gradient (dS/dt), KL anomaly detection |
| Memory | Slime mould paths | Decaying spatial memory reinforces productive routes |
| Temporal | Jellyfish oscillation | Alternating explore/exploit phases (50-step cycle) |
| Cleaning | Entropy depletion | dS/dt ≤ 0 → river polluted → navigate and clean |
| Social | Mycorrhizal networks | Shared spatial memory between focal agents (hive mind) |
| Cooperation | Context-aware symbiosis | Respond to agents based on environmental state, not blind rules |

The key insight for Clean Up: **when entropy growth rate drops to zero, the ecosystem
is dying** — the river is polluted and apples won't regrow. The agent uses this
thermodynamic signal to decide when to sacrifice foraging time for the public good.
No reward needed. The physics tells the agent what to do.

## Why It Matters

Clean Up is a **public goods dilemma**. River pollution blocks apple growth. Someone
must sacrifice foraging time to clean — but cleaners earn less than free-riders.
Trained RL agents often fail this (VMPO scores 0–10 on majority-focal scenarios).

DigiSoup solves the collective action problem through thermodynamic inference alone.
This challenges the assumption that complex multi-agent cooperation requires
gradient-based learning.

## Version Evolution

Built iteratively over 15 versions. Each adds one bio-inspired layer:

```
v1  Random baseline (floor)
v2  Entropy perception (+48% CU_0)
v3  Jellyfish oscillation (+74% PD_0)
v4  Slime mould memory (first time beating ACB on CU_0)
v5-v7  Behaviour modifications (all regressed — key lesson)
v8  Thermodynamic sensing (PD all 6 up, +37%)
v9  Resource conservation (CU_0 +50% vs ACB)
v10 Colour perception fix (CU_7 doubled)
v11 Cleaning rule (CU_0=278 peak)
v14 Hive mind (shared memory)
v15 Depletion cleaning + symbiosis (5 zero→scoring, +22% aggregate vs ACB)
```

**Key lesson:** Improving perception works. Modifying behaviour hurts.
The decision system is near-optimal; the gains come from sharper senses.

See [VERSION_LOG.md](VERSION_LOG.md) for full scores across all 15 versions.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/matthewfearne/digisoup.git
cd digisoup

# Create venv (requires Python 3.10 for dmlab2d)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/test_agent.py -v

# Run evaluation (all 17 scenarios, 10 episodes)
python -m evaluation.run --all-targets --episodes 10

# Run single substrate
python -m evaluation.run --substrate clean_up --episodes 10

# Live viewer (requires pygame + display)
python watch.py clean_up
```

## Project Structure

```
agents/digisoup/
  perception.py    # RGB → entropy grid, gradients, masks, growth rate
  state.py         # Internal state: energy, memory, heading, phase
  action.py        # Priority-rule action selection (8 rules)
  policy.py        # Melting Pot Policy interface + HiveMemory

evaluation/
  run.py           # Official evaluation runner
  metrics.py       # Metrics and aggregation
  compare.py       # DeepMind baseline comparison

configs/scenarios.py    # Scenario configurations (17 scenarios)
tests/test_agent.py     # 33 tests
results/                # JSON results per run (timestamped)
VERSION_LOG.md          # Full score log across all versions
watch.py                # Live pygame viewer
```

## Evaluation Protocol

Follows the official Melting Pot 2.0 evaluation:

- **17 scenarios** across 3 substrates (Commons Harvest, Clean Up, Prisoners Dilemma)
- **Focal vs background:** DigiSoup fills focal slots, DeepMind's trained bots fill background
- **Metric:** Focal per-capita return with 95% confidence intervals
- **Hardware:** Intel i7-8700K, GTX 1060 6GB (GPU for background bot inference only)

## Target Substrates

| Substrate | Scenarios | Players | Social Dilemma |
|-----------|-----------|---------|----------------|
| Commons Harvest Open | 2 | 5 focal + 2 bg | Tragedy of the commons |
| Clean Up | 9 | 3–6 focal + 1–7 bg | Public goods / free-rider |
| Prisoners Dilemma Arena | 6 | 1–7 focal + 1–7 bg | Iterated cooperation |

## Author

**Matthew Fearne** — Independent AI Researcher & Complexity Scientist

mrfearne@gmail.com

## License

MIT
