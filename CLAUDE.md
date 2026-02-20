# CLAUDE.md -- DigiSoup vs Melting Pot

## Project

**Location:** `/home/fqci/digisoup-meltingpot/`
**What:** Zero-training entropy-driven agents evaluated on DeepMind's Melting Pot benchmark.
**Python:** 3.10 (required by dmlab2d). Venv at `.venv/`.
**Author:** Matt (FQCI)
**GPU:** GTX 1060 6GB with CUDA 12.0 + cuDNN 9.19 (TensorFlow GPU acceleration for background bots)

## Critical Rules

1. NO neural networks. NO reward optimization. NO training. Pure entropy.
2. Reward is RECORDED but NEVER used to select actions.
3. Agent must implement the official Melting Pot `Policy` interface.
4. Evaluation uses real Melting Pot scenarios with background bots.
5. Every version is git tagged with scores in the commit message.
6. Report honest results including losses.
7. Always save/tag current version before building next. Never overwrite versions.
8. If a version regresses, revert to best base (currently v4) and branch from there.

## How To Run

```bash
source .venv/bin/activate
python -m pytest tests/test_agent.py -v           # Run tests (33 passing)
python -m evaluation.run --all-targets --episodes 10
python -m evaluation.run --substrate clean_up --episodes 10
python -m evaluation.run --scenario prisoners_dilemma_in_the_matrix__arena_0 --episodes 10
```

Evaluation runs ~85 min (CPU) or ~55 min (with GPU) for all 17 scenarios x 10 episodes.
For publication-quality results, use `--episodes 30`.

## GPU / CUDA Setup

NVIDIA driver 535.288.01 + CUDA toolkit 12.0 + cuDNN 9.19 installed.
TensorFlow 2.20.0 uses GPU for background bot inference (the trained RL policies).
Our DigiSoup agent is pure numpy — no GPU needed for the agent itself.

Installed via:
```bash
sudo apt install -y nvidia-cuda-toolkit
# cuDNN from NVIDIA repo:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcudnn9-cuda-12
```

## Version History

All versions git tagged. See VERSION_LOG.md for full scores.

| Version | Layer | Result | Tag |
|---------|-------|--------|-----|
| v1 | Random baseline (floor) | Baseline | `v1-random-baseline` |
| v2 | Entropy perception + state machine | +31% PD, +48% CU_0 | `v2-digisoup-original` |
| v3 | Phase cycling (jellyfish oscillation) | +74% PD_0, +27% CU_0 | `v3-phase-cycling` |
| v4 | Spatial memory (slime mold paths) | CU_0=231 beats ACB=171 | `v4-spatial-memory` |
| v5 | Adaptive coop (vampire bat) | Mixed, net negative | `v5-adaptive-coop` |
| v6 | Entropy-as-energy (cockroach) | Catastrophic regression | `v6-cockroach-persistence` |
| v7 | Anti-fragility (honey badger) | Worst regression | `v7-honey-badger` |
| v8 | Thermodynamic sensing (4x4 grid + growth + KL) | PD all 6 up avg +37%, CU held | `v8-thermodynamic-sensing` |
| v9 | Resource conservation (sustainable harvesting) | CU_0=257 (+50% vs ACB), CU breakout | `v9-resource-conservation` |
| v10 | Sharper Eyes (colour fix + heatmap + heading) | CU_7 doubled, CU_2/3/8 up 40-62% | `v10-sharper-eyes` |
| v11 | Squid Custodian (cleaning rule) | CU_0=278 (+63% ACB), CH +71-116% | `v11-squid-custodian` |
| v12 | Echo Explorer (starvation + echo feedback) | CU/CH regressed | `v12-echo-explorer` |
| v13 | Mycorrhizal Explorer (dead reckoning + scouting) | Broad regression | `v13-mycorrhizal-explorer` |
| v14 | Hive Mind (shared spatial memory) | Base for v15, no formal eval | `v14-hive-mind` |
| v15 | River Eyes (depletion cleaning + symbiosis) | **5 zero→scoring CU, CU total +28%** | `v15-river-eyes` |

**High-water mark: v15** — 5 previously-zero CU scenarios now scoring (CU_4=44, CU_5=36, CU_6=19).
CU_7=180 (+95% vs v11), CU_8=43 (+179%). CU_0=243 still beats ACB=171 by +42%.
Total CU reward 735 vs 572 in v11 (+28%). CH_1=3.16 new all-time high.
v5-v7 modified behavior (cooperation/energy/aggression) and regressed.
v8-v15: perception + targeted rules — all improved over v4.

### Key Insight

Layers that improve **perception and navigation** (v2-v4) work.
Layers that modify **cooperation, energy, or aggression** (v5-v7) hurt.
The agent's decision system is already near-optimal for zero-training; make it see better, not act differently.

## Agent Architecture

Four modules, cleanly separated:

```
agents/digisoup/
  perception.py  # RGB -> entropy, gradients, growth, anomaly, agents, resources, change
  state.py       # DigiSoupState NamedTuple, update_state(), phase/role helpers
  action.py      # Priority-rule action selection (6 rules, phase-modulated)
  policy.py      # Melting Pot Policy interface (wires perception -> state -> action)
```

### Current Agent (v15 = v14 + river eyes):
- **Perception:** 4x4 entropy grid, growth gradient (dS/dt), KL anomaly, warm mask (red/orange apples), dirt/water mask (CU pollution), grass mask (orchard floor), sand mask (dead zone), resource mask (green+warm), agent density grid (4x4), change detection
- **State:** Energy, cooperation tendency, emergent role, entropy EMA, spatial memory, resource heatmap (4x4 temporal), heading (movement EMA), orientation + position (dead reckoning), prev entropy grid
- **Action:** 8 priority rules: random explore → energy-seek (depletion→river) → **river cleaning (sand-guarded)** → proactive cleaning (dS/dt≤0) → exploit-seek → **context-aware symbiosis** → stable-navigate (sand flee + grass attract + crowding avoidance + heatmap) → chaotic-exploit
- **Phase:** Jellyfish oscillation — 50 steps explore, 50 steps exploit
- **Memory:** Slime mold path reinforcement + temporal resource heatmap + heading persistence + **hive mind** (shared focal agent spatial memory)

## Target Scenarios

- `commons_harvest__open`: 2 scenarios (5 focal, 2 background)
- `clean_up`: 9 scenarios (3-6 focal, 1-4 background)
- `prisoners_dilemma_in_the_matrix__arena`: 6 scenarios (1-7 focal, 1-7 background)

## Key Paths

```
agents/digisoup/policy.py      # Policy interface implementation
agents/digisoup/perception.py  # Thermodynamic perception (v8: 4x4 grid + growth + KL)
agents/digisoup/state.py       # Internal state machine
agents/digisoup/action.py      # Priority-rule action selection
tests/test_agent.py            # 33 tests
evaluation/run.py              # Main evaluation runner
evaluation/metrics.py          # Metrics and aggregation
configs/scenarios.py           # Scenario configurations
results/                       # JSON results per run (timestamped)
ref/                           # v11 reference code (parts bin for entropy layers)
VERSION_LOG.md                 # Full score log per version with comparisons
```

## DeepMind Baselines

Raw baseline scores from `https://storage.googleapis.com/dm-meltingpot/meltingpot-results-2.3.0.feather`.
Key comparison: Clean Up _0 — ACB: 170.66, VMPO: 180.24, DigiSoup v15: **242.87** (+42% vs ACB).
Best version: v15 — 8 of 9 CU scenarios scoring (5 unlocked from zero). Total CU reward +28% vs v11.
