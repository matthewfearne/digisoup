# DigiSoup vs Melting Pot: Zero-Training Entropy Agents on DeepMind's Benchmark

## The Thesis

DeepMind built Melting Pot -- the standard benchmark for multi-agent cooperation.
50+ substrates, 262 test scenarios, pre-trained background bots. Their approach:
reinforcement learning, neural networks, reward optimization, GPU clusters,
billions of training steps.

**Their own finding:** Trained cooperation doesn't generalize. Agents overfit to
self-play. Prosocial agents underperform selfish ones on average. Only on Clean Up
did prosocial architectures significantly beat random.

DigiSoup proved that cooperation emerges from pure entropy -- no training, no
reward signals, no neural networks. 434% cooperation increase. 12 emergent
behavioural types. On a laptop.

**This project:** Run DigiSoup's entropy-driven agents on DeepMind's actual
benchmark. Same scenarios, same background bots, same scoring. Zero training
vs billion-dollar RL. The agents implement Melting Pot's official Policy
interface and are evaluated using the official protocol.

## Rules

1. **NO NEURAL NETWORKS.** Perception is histogram-based pixel statistics.
2. **NO REWARD OPTIMIZATION.** Reward is recorded but never used for decisions.
3. **NO TRAINING.** No pre-training, no fine-tuning, no cross-episode learning.
4. **HONEST RESULTS.** Report losses as well as wins.
5. **SIMPLE CODE.** Every component explainable in one paragraph.
6. **STATISTICAL RIGOUR.** 100 episodes per scenario. Report confidence intervals.
7. **OFFICIAL PROTOCOL.** Real Melting Pot scenarios with background bots.

## Evaluation Protocol

Follows the Melting Pot 2.0 evaluation exactly:

- **Scenarios, not just substrates.** Each scenario pairs a substrate with
  pre-trained background bots the focal agents have never seen.
- **Focal vs background.** DigiSoup agents fill the focal slots. DeepMind's
  pre-trained bots fill the background slots. Only focal rewards count.
- **Resident and visitor modes.** Some scenarios have DigiSoup as majority
  (resident), others as minority (visitor).
- **Primary metric:** Focal per-capita return.
- **Normalization:** (raw - random) / (exploiter - random), where random is
  a uniform-random agent and exploiter is a scenario-specific trained agent.
- **100 episodes per scenario.** Results include mean, std, 95% CI.

## Target Substrates

| Substrate | Scenarios | Social Dynamic |
|-----------|-----------|----------------|
| `commons_harvest__open` | 2 | Tragedy of the commons |
| `clean_up` | 9 | Public goods / free-rider dilemma |
| `prisoners_dilemma_in_the_matrix__arena` | 6 | Cooperation vs defection |

Total: 17 scenarios across 3 substrates.

## Incremental Build

The agent is built up one layer at a time. Each version is tagged, scored on
all scenarios, and committed with results. See VERSION_LOG.md for the full
progression.

## How To Run

```bash
# Activate the environment
source .venv/bin/activate

# Run evaluation on all scenarios for a substrate
python -m evaluation.run --substrate commons_harvest__open --episodes 10

# Run all 3 target substrates
python -m evaluation.run --all-targets --episodes 10

# Run a specific scenario
python -m evaluation.run --scenario clean_up_0 --episodes 10
```

## Project Structure

```
digisoup-meltingpot/
  agents/digisoup/          # The agent (built incrementally)
    policy.py               # Official Melting Pot Policy implementation
    entropy.py              # Entropy computations (added layer by layer)
    action.py               # Action selection
  evaluation/               # Evaluation pipeline
    run.py                  # Main runner (official Melting Pot scenarios)
    metrics.py              # Metrics collection and aggregation
    compare.py              # Baseline comparison tables
  analysis/                 # Post-hoc analysis
    visualize.py            # Publication-ready figures
  configs/                  # Substrate and scenario configuration
    scenarios.py            # Target scenarios and metadata
  results/                  # All evaluation results (JSON)
  ref/                      # Reference material from v11 agent
    agent/                  # Original entropy.py, action.py, core.py
    docs/                   # Original BATTLE_PLAN.md, CHANGELOG.md
  tests/                    # Smoke tests
  VERSION_LOG.md            # Score tracking per version
  CLAUDE.md                 # Session context
  README.md                 # This file
```

## What Success Looks Like

**Minimum:** Above random on all 3 substrates (17 scenarios).
**Strong:** Above one DeepMind baseline on 2+ substrates.
**Paradigm:** Above ALL baselines on Clean Up.

The result is a paper regardless of numbers. The question has never been asked:
can zero-training entropy agents achieve cooperation on the industry standard
multi-agent benchmark?

## Author

Matt (FQCI) -- Independent AI Researcher & Complexity Scientist
