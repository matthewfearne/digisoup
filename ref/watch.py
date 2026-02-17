"""Watch DigiSoup agents play on actual Melting Pot substrates.

Opens a pygame window showing the substrate in real-time.
Prints live stats to the terminal: step, reward, entropy, temperature, actions.

Usage:
    .venv310/bin/python3.10 watch.py                    # Default: PD
    .venv310/bin/python3.10 watch.py prisoners_dilemma
    .venv310/bin/python3.10 watch.py commons_harvest
    .venv310/bin/python3.10 watch.py clean_up
"""
import sys
import time
import numpy as np
import pygame
from collections import Counter
from shimmy import MeltingPotCompatibilityV0
from src.agent.core import DigiSoupAgent
from src.agent.entropy import information_temperature, entropy_rate

ALIASES = {
    "pd": "prisoners_dilemma_in_the_matrix__arena",
    "prisoners_dilemma": "prisoners_dilemma_in_the_matrix__arena",
    "commons": "commons_harvest__open",
    "commons_harvest": "commons_harvest__open",
    "clean_up": "clean_up",
    "cleanup": "clean_up",
}

ACTION_NAMES = ["noop", "fwd", "back", "left", "right", "trnL", "trnR", "zap", "CLEAN"]

# Substrate-specific action counts (Clean Up has 9: action 8 = fireClean)
SUBSTRATE_N_ACTIONS = {"clean_up": 9}

substrate = sys.argv[1] if len(sys.argv) > 1 else "prisoners_dilemma"
substrate = ALIASES.get(substrate, substrate)
short_name = substrate.split("__")[0].replace("_in_the_matrix", "")
n_actions = SUBSTRATE_N_ACTIONS.get(substrate, 8)

print(f"\n  DigiSoup vs Melting Pot — THERMODYNAMIC ENTROPY EDITION")
print(f"  Substrate: {short_name}")
print(f"  Agent: pure information thermodynamics (zero training, zero reward, zero rules)")
print(f"  Entropy stack: observation + fine gradient + spatial change + dS/dt + temperature + Boltzmann")
print(f"  Close window or Ctrl+C to stop.\n")

env = MeltingPotCompatibilityV0(substrate_name=substrate, render_mode="human")
obs, info = env.reset()

# Thermodynamic personality profiles: 4 cleaners + 3 harvesters
# Same entropy physics, different metabolic constants.
PERSONALITY_PROFILES = [
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},  # cleaner
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},  # cleaner
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},  # cleaner
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},  # cleaner
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},  # harvester
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},  # harvester
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},  # harvester
]
ROLE_NAMES = ["cleaner", "cleaner", "cleaner", "cleaner", "harvester", "harvester", "harvester"]

agents = {}
for i, a in enumerate(env.possible_agents):
    personality = PERSONALITY_PROFILES[i % len(PERSONALITY_PROFILES)]
    agents[a] = DigiSoupAgent(
        agent_id=i, seed=42 + i, n_actions=n_actions, personality=personality
    )
    role = ROLE_NAMES[i % len(ROLE_NAMES)]
    print(f"  Agent {i} ({role}): interact={personality['interact_bias']:.2f} "
          f"move={personality['move_bias']:.2f} "
          f"scan={personality['scan_bias']:.2f}")
print()

n_agents = len(env.possible_agents)
total_reward = 0.0
action_counts = Counter()
step = 0
t0 = time.time()

try:
    while env.agents:
        actions = {}
        step_entropy = 0.0
        step_temp = 0.0
        for agent_id in env.agents:
            a = agents[agent_id].act(obs[agent_id])
            actions[agent_id] = a
            action_counts[a] += 1
            step_entropy += agents[agent_id]._last_entropy
            step_temp += information_temperature(agents[agent_id]._entropy_history)

        obs, rew, term, trunc, info = env.step(actions)
        env.render()

        # Process pygame events so the OS doesn't think we're frozen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        step_r = sum(float(v) for v in rew.values())
        total_reward += step_r
        step += 1
        avg_ent = step_entropy / n_agents
        avg_temp = step_temp / n_agents

        # Print live stats every 50 steps
        if step % 50 == 0:
            elapsed = time.time() - t0
            fps = step / elapsed if elapsed > 0 else 0
            act_str = " ".join(
                f"{ACTION_NAMES[k] if k < len(ACTION_NAMES) else f'a{k}'}={action_counts[k]}"
                for k in sorted(action_counts.keys())
            )
            # Sample one agent's dS/dt for display
            sample_agent = list(agents.values())[0]
            ds_dt = entropy_rate(sample_agent._entropy_history)
            print(
                f"  Step {step:5d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Per-cap: {total_reward/n_agents:7.2f} | "
                f"H: {avg_ent:.3f} | "
                f"T: {avg_temp:.3f} | "
                f"dS/dt: {ds_dt:+.3f} | "
                f"FPS: {fps:.0f} | "
                f"{act_str}"
            )

        if any(term.values()) or any(trunc.values()):
            break

except KeyboardInterrupt:
    print("\n  Interrupted.")

elapsed = time.time() - t0
print(f"\n  === FINAL ===")
print(f"  Steps: {step}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Per-capita: {total_reward / n_agents:.2f}")
print(f"  Time: {elapsed:.1f}s ({step/elapsed:.0f} FPS)")
print()

env.close()
