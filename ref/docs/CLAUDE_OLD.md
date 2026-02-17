# CLAUDE.md — DigiSoup vs Melting Pot: Entropy Meets DeepMind's Benchmark

## Project Identity

**Project:** DigiSoup agents on DeepMind's Melting Pot benchmark
**Author:** Matt (FQCI) — Independent AI Researcher & Complexity Scientist
**Language:** Python 3.10
**Philosophy:** Zero training. Zero reward optimization. Entropy drives cooperation. Prove it on their benchmark.

---

## What This Is

A direct challenge to the assumption that multi-agent cooperation requires reinforcement learning.

DeepMind built Melting Pot — the standard benchmark for multi-agent RL. 50+ substrates, 256+ test scenarios. They test TRAINED agents on cooperation, competition, trust.

**Their finding:** Trained cooperation doesn't generalize. Agents overfit. Prosocial agents underperform selfish ones on average. Only on Clean Up did prosocial agents significantly beat random.

Matt built DigiSoup — cooperation emerges from pure entropy. 434% cooperation increase. 12 emergent behavioural types. Zero training. On a laptop.

**This project:** Port DigiSoup's entropy-driven agents to Melting Pot's API. Run them on DeepMind's benchmark. Same test suite, same evaluation, same scoring. Zero training vs billion-dollar RL.

---

## Critical Rules

1. **NO NEURAL NETWORKS.** Perception is simple pixel statistics. This is the whole point.
2. **NO REWARD OPTIMIZATION.** Reward is RECORDED but NEVER used to select actions.
3. **NO TRAINING.** No pre-training, no fine-tuning, no cross-episode optimization.
4. **HONEST RESULTS.** Report losses as well as wins.
5. **SIMPLE CODE.** Every component explainable in one paragraph.
6. **STATISTICAL RIGOUR.** Minimum 10 episodes, 30 preferred. Report confidence intervals.

---

## Agent Architecture

### Observation Processing (perception.py)
88x88x3 RGB observations processed with SIMPLE pixel statistics:
- Shannon entropy of pixel distribution
- Gradient across observation quadrants
- Agent detection by colour signature
- Resource detection by pixel patterns
- Change detection via observation differencing
NOT a convolutional neural network. Histogram-based features only.

### Internal State Machine (state.py)
Ported from DigiSoup:
- Energy: accumulated from successful interactions, depletes over time
- Cooperation tendency: [0,1] float, shifts based on interaction outcomes
- Role: emerges from action pattern history
- Interaction history: rolling window
- Entropy state: running estimate of local environmental entropy

### Action Selection (action.py)
Entropy-gradient-based:
1. Compute observation entropy and gradient direction
2. Energy low → move toward resources
3. Agents nearby → cooperation tendency determines interact vs avoid
4. Environment stable → explore (move toward higher entropy)
5. Environment chaotic → exploit current strategy
6. Periodic random exploration proportional to internal entropy

---

## Target Substrates

### Priority 1 (Strongest DigiSoup parallels):
1. `commons_harvest__open` — Shared resources, tragedy of the commons
2. `clean_up` — Public goods, requires altruistic cleaning
3. `prisoners_dilemma_in_the_matrix__arena` — Classic cooperation/defection

### Priority 2:
4. `allelopathic_harvest__open` — Coordination without communication
5. `stag_hunt_in_the_matrix__arena` — Trust dynamics
6. `collaborative_cooking__ring` — Division of labor

---

## Evaluation Protocol

1. **Random baselines:** 30 episodes per substrate. Floor measurement.
2. **DigiSoup agents:** 30 episodes per substrate. Same metrics.
3. **Background populations:** Melting Pot official eval with pre-trained bots. Tests generalization.
4. **Comparison:** Against DeepMind baselines from Melting Pot 2.0 Tech Report. Normalised (worst=0, best=1).
5. **Emergence analysis:** Cooperation growth, role specialization, spatial patterns, 98/2 ratio.

---

## Melting Pot API

```python
# PettingZoo (recommended)
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="commons_harvest__open")
observations = env.reset()
while env.agents:
    actions = {agent: your_agent.act(observations[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

# Agent interface:
# observation['RGB']: np.array(88, 88, 3) — 11x11 sprites, 8px each
# action: int 0-7 — move, turn, interact
# reward: float per step
```

---

## Tech Stack

- dm-meltingpot, dmlab2d, shimmy — DeepMind's benchmark suite
- NumPy, SciPy — Entropy computation, statistics (NOT for neural networks)
- Rich — Terminal output
- Matplotlib, Pandas — Visualization and analysis

**NO TensorFlow. NO PyTorch.** DigiSoup runs on NumPy. That's the point.

System: Linux x86_64, Python 3.10. No GPU required.

---

## Commands

```bash
pip install -r requirements.txt
python -m src.evaluation.runner --substrate commons_harvest__open --agent random --episodes 30
python -m src.evaluation.runner --substrate commons_harvest__open --agent digisoup --episodes 30
python -m src.evaluation.runner --all-targets --agent digisoup --episodes 30
python -m src.evaluation.compare --results results/latest/
python -m src.analysis.emergence --results results/latest/
```

---

## What Success Looks Like

**Minimum:** Above random on 3+ cooperation substrates.
**Strong:** Above one DeepMind baseline on 2+ substrates.
**Paradigm:** Above ALL baselines on Clean Up.

The result is a paper regardless of numbers. The question has never been asked.
