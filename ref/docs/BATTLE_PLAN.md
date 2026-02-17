# DigiSoup vs Melting Pot — The Entropy Challenge

## The Thesis

DeepMind built Melting Pot — 50+ multi-agent substrates, 256 test scenarios, the industry standard benchmark for multi-agent cooperation. Their approach: train RL agents with reward signals, gradient descent, millions of episodes, GPU clusters.

**Their own results:** Agents overfit. High self-play scores, near-random on test scenarios. Prosocial agents underperformed selfish ones on average. Only on Clean Up did prosocial architectures significantly beat random.

**DigiSoup's result:** 434% cooperation increase. 12 emergent behavioural types. Zero training. Zero reward signals. Zero fitness functions. Cooperation emerges from pure entropy.

**The challenge:** Port DigiSoup to Melting Pot's API. Same test suite, same evaluation, same scoring. Show that zero-training entropy agents achieve cooperation that DeepMind's trained agents struggle to produce.

---

## What Is Melting Pot?

Python library built on DeepMind Lab2D:
- **50+ substrates** — 2D grid-world games with different social dynamics
- **256+ test scenarios** — Substrate + background population of pre-trained bots
- **Standard API** — `env.reset()` → observations, `env.step(actions)` → rewards
- **Evaluation protocol** — Per-capita reward normalised against baselines (worst=0, best=1)

**Agent interface:**
- Observations: 11x11 RGB pixel window (88x88 pixels) — partial observability
- Actions: Discrete 0-7 — move, turn, interact
- Rewards: Float per step, substrate-dependent

**Key:** They explicitly say "anything goes" for agent design. Including zero training.

---

## Target Substrates

### Tier 1 — Direct DigiSoup Analogues

**1. Commons Harvest (Open)** — Shared resources, tragedy of the commons. DigiSoup agents learned sustainable harvesting from entropy. DeepMind: prosocial agents outperform here, but they're TRAINED to be prosocial.

**2. Clean Up** — Public goods dilemma. Cleaning benefits everyone but costs the cleaner. DeepMind: ONLY prosocial architectures beat random. Their hardest cooperation substrate.

**3. Prisoners Dilemma in the Matrix** — Classic game theory. DigiSoup proved cooperation becomes dominant under entropy.

### Tier 2 — Strong Candidates

**4. Allelopathic Harvest** — Coordination without communication. DigiSoup developed implicit signalling.

**5. Stag Hunt** — Trust dynamics. DigiSoup built trust from entropy.

**6. Collaborative Cooking** — Division of labor. DigiSoup showed role specialization.

---

## The DigiSoup → Melting Pot Agent

### Core Design:

```python
class DigiSoupMeltingPotAgent:
    def __init__(self):
        self.energy = initial_energy
        self.cooperation_tendency = 0.5  # Neutral start, evolves
        self.role = None                 # Emerges from dynamics
        self.interaction_history = []
        
    def act(self, observation, reward):
        # 1. Compute observation entropy (how complex is my view?)
        # 2. Update internal state (energy, cooperation tendency)
        # 3. Select action via entropy gradient (NOT reward optimization)
        # Reward is RECORDED but NEVER drives decisions
        return action
```

### Observation Processing (NO neural network):
- Shannon entropy of pixel distribution
- Gradient direction across quadrants
- Agent detection by colour signature
- Resource detection by pixel patterns
- All histogram-based. NumPy only.

### Action Selection (entropy-gradient):
- Move toward areas of higher entropy (where things happen)
- Near agents: cooperation tendency determines interact/avoid
- Near resources: energy level determines harvest/conserve
- Periodic random exploration based on internal entropy state

### What You Do NOT Do:
- No neural networks, no gradient descent, no reward optimization
- No policy search, no value functions, no experience replay
- No pre-training, no fine-tuning
- Record rewards for ANALYSIS, never for DECISIONS

---

## Implementation Phases

### Phase 1: Setup (2-3 hours)
```bash
conda create -n meltingpot python=3.10
conda activate meltingpot
pip install dmlab2d dm-meltingpot shimmy[meltingpot]
```
Verify API. Run random agents. Record baseline scores.

### Phase 2: Port DigiSoup Agent (3-4 hours)
Build DigiSoupMeltingPotAgent:
- entropy.py — Shannon entropy from pixels
- perception.py — Agent/resource detection
- state.py — Energy, cooperation tendency, role emergence
- action.py — Entropy-gradient action selection
Each component: simple, explainable in one paragraph.

### Phase 3: Run on Substrates (2-3 hours)
For each target substrate:
1. 30 episodes random agents → baseline
2. 30 episodes DigiSoup agents → experimental
3. Official Melting Pot evaluation with background bots → generalization
Record everything.

### Phase 4: Analysis (2-3 hours)
Compare against DeepMind baselines from Melting Pot 2.0 Tech Report.
Document emergent behaviours. Check for 98/2 ratio.

### Phase 5: Report (1-2 hours)
Comparison tables. Emergence documentation. Honest wins AND losses.

---

## DeepMind Baselines to Compare Against

From Melting Pot 2.0 Tech Report:
- A3C (standard RL)
- A3C Prosocial (group reward)
- OPRE (their fancy architecture)
- OPRE Prosocial (group reward)
- Random

| Substrate | Random | A3C | A3C-Pro | OPRE | OPRE-Pro | **DigiSoup** |
|---|---|---|---|---|---|---|
| Commons Harvest | 0.0 | X | X | X | X | **?** |
| Clean Up | 0.0 | X | X | X | X | **?** |
| Prisoners Dilemma | 0.0 | X | X | X | X | **?** |

---

## Why This Is Devastating

1. **Same benchmark, opposite philosophy.** Trained vs untrained on identical tests.
2. **Their results support your thesis.** Trained cooperation overfits. Entropy-driven has nothing to overfit to.
3. **Compute disparity.** GPU clusters vs laptop. Comparable results = revolution.
4. **Connects everything.** DigiSoup → Melting Pot proves cooperation principle. CHAOS → MycoNet proves networking. InfoSoup → thermodynamics provides theory.

---

## Honest Caveats

1. Melting Pot's visual observations are a handicap for DigiSoup (designed for direct state access)
2. Discrete grid actions vs DigiSoup's continuous interactions
3. Competitive substrates are out-of-scope
4. Background bots may exploit non-adversarial agents
5. Need sufficient episodes for statistical significance

Report losses honestly. That's what makes wins credible.

---

## Victory Conditions

**Minimum:** Above random on 3+ cooperation substrates. Proves entropy-driven cooperation transfers.
**Strong:** Above one DeepMind baseline on 2+ substrates. Zero-training competes with RL.
**Paradigm:** Above ALL baselines on Clean Up. Entropy > engineering.

Either way: "We ran zero-training entropy agents on DeepMind's benchmark. Here's what happened." That's a paper.
