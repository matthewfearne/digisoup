# CLAUDE.md -- DigiSoup vs Melting Pot

## Project

**Location:** `/home/fqci/digisoup-meltingpot/`
**What:** Zero-training entropy-driven agents evaluated on DeepMind's Melting Pot benchmark.
**Python:** 3.10 (required by dmlab2d). Venv at `.venv/`.
**Author:** Matt (FQCI)

## Critical Rules

1. NO neural networks. NO reward optimization. NO training. Pure entropy.
2. Reward is RECORDED but NEVER used to select actions.
3. Agent must implement the official Melting Pot `Policy` interface.
4. Evaluation uses real Melting Pot scenarios with background bots.
5. Every version is git tagged with scores in the commit message.
6. Report honest results including losses.

## How To Run

```bash
source .venv/bin/activate
python -m evaluation.run --all-targets --episodes 10
python -m evaluation.run --substrate clean_up --episodes 10
python -m evaluation.run --scenario prisoners_dilemma_in_the_matrix__arena_0 --episodes 10
```

## Agent Architecture

The agent is built incrementally. Each layer adds one entropy computation.
See VERSION_LOG.md for what each version adds and its scores.

Current layers are tracked in `agents/digisoup/entropy.py`.

## Target Scenarios

- `commons_harvest__open`: 2 scenarios (5 focal, 2 background)
- `clean_up`: 9 scenarios (3 focal, 4 background)
- `prisoners_dilemma_in_the_matrix__arena`: 6 scenarios (1 focal, 7 background)

## Key Paths

```
agents/digisoup/policy.py    # Policy interface implementation
agents/digisoup/entropy.py   # Entropy stack (built incrementally)
agents/digisoup/action.py    # Action selection
evaluation/run.py            # Main evaluation runner
evaluation/metrics.py        # Metrics and aggregation
configs/scenarios.py         # Scenario configurations
results/                     # JSON results per run
ref/                         # v11 reference code (parts bin)
VERSION_LOG.md               # Score log per version
```
