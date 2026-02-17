#!/usr/bin/env python3
"""Run DeepMind's pretrained bot agents on Melting Pot substrates.

Gets raw per-capita reward numbers from DeepMind's own trained agents
for direct comparison against DigiSoup results.

All agents are DeepMind's pretrained bots (not just background population).
Uses the native meltingpot dm_env interface, not shimmy.

Usage:
    .venv310/bin/python3.10 run_deepmind_baseline.py
    .venv310/bin/python3.10 run_deepmind_baseline.py --episodes 30
    .venv310/bin/python3.10 run_deepmind_baseline.py --substrate commons_harvest__open --episodes 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Any

import numpy as np

# Suppress noisy warnings from TF/absl during bot loading.
# The PermissiveModel warns about WORLD.RGB and first-step reward/discount
# being "unexpected" -- these are harmless (it drops them gracefully).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Substrate configurations
# ---------------------------------------------------------------------------

SUBSTRATE_CONFIGS: dict[str, dict[str, Any]] = {
    "commons_harvest__open": {
        "n_agents": 7,
        "bots": [
            "commons_harvest__open__free_0",
            "commons_harvest__open__free_1",
            "commons_harvest__open__pacifist_0",
            "commons_harvest__open__pacifist_1",
        ],
    },
    "clean_up": {
        "n_agents": 7,
        "bots": [
            "clean_up__cleaner_0",
            "clean_up__cleaner_1",
            "clean_up__consumer_0",
            "clean_up__consumer_1",
            "clean_up__puppet_alternator_first_cleans_0",
        ],
    },
    "prisoners_dilemma_in_the_matrix__arena": {
        "n_agents": 8,
        "bots": [
            "prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_0",
            "prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_margin_0",
            "prisoners_dilemma_in_the_matrix__arena__puppet_defector_0",
            "prisoners_dilemma_in_the_matrix__arena__puppet_defector_margin_0",
        ],
    },
}

DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 1000


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_substrate(
    substrate_name: str,
    n_episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run pretrained DeepMind bots on a substrate.

    Returns dict with per-episode and aggregated results.
    """
    from meltingpot import substrate as mp_substrate, bot as mp_bot
    import dm_env

    config = SUBSTRATE_CONFIGS[substrate_name]
    n_agents = config["n_agents"]
    bot_names = config["bots"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Substrate: {substrate_name}")
        print(f"Agents: {n_agents}, Episodes: {n_episodes}, Max steps: {max_steps}")
        print(f"Bot pool: {bot_names}")
        print(f"{'='*70}")

    # Build bot policies -- one per agent, cycling through available bots
    if verbose:
        print("Loading bot policies...", end=" ", flush=True)
    policies = []
    policy_names = []
    for i in range(n_agents):
        bot_name = bot_names[i % len(bot_names)]
        policy = mp_bot.build(bot_name)
        policies.append(policy)
        policy_names.append(bot_name)
    if verbose:
        print("done.")
        for i, name in enumerate(policy_names):
            print(f"  Agent {i}: {name}")

    # Run episodes
    all_episode_rewards = []       # list of lists: [episode][agent] = total_reward
    all_episode_per_capita = []    # list of per-capita rewards

    for ep in range(n_episodes):
        t0 = time.time()

        # Build fresh environment each episode
        env = mp_substrate.build(
            substrate_name,
            roles=["default"] * n_agents,
        )

        # Initialize bot states
        states = [p.initial_state() for p in policies]

        # Reset environment
        timestep = env.reset()

        # Track per-agent total reward
        agent_rewards = [0.0] * n_agents

        for step_idx in range(max_steps):
            # Create per-player timesteps and get actions
            actions = []
            new_states = []
            for i in range(n_agents):
                player_ts = dm_env.TimeStep(
                    step_type=timestep.step_type,
                    reward=timestep.reward[i],
                    discount=timestep.discount,
                    observation=timestep.observation[i],
                )
                action, new_state = policies[i].step(player_ts, states[i])
                actions.append(action)
                new_states.append(new_state)
            states = new_states

            # Step environment
            timestep = env.step(actions)

            # Accumulate rewards
            for i in range(n_agents):
                agent_rewards[i] += float(timestep.reward[i])

            # Check if episode ended
            if timestep.step_type == dm_env.StepType.LAST:
                break

        env.close()

        total_reward = sum(agent_rewards)
        per_capita = total_reward / n_agents
        all_episode_rewards.append(agent_rewards)
        all_episode_per_capita.append(per_capita)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"  Episode {ep+1:3d}/{n_episodes}: "
                f"per_capita={per_capita:8.2f}  "
                f"total={total_reward:10.2f}  "
                f"steps={step_idx+1:5d}  "
                f"time={elapsed:5.1f}s"
            )

    # Clean up policies
    for p in policies:
        try:
            p.close()
        except Exception:
            pass

    # Aggregate
    pc_array = np.array(all_episode_per_capita)
    result = {
        "substrate": substrate_name,
        "n_agents": n_agents,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "bot_names": policy_names,
        "per_capita_reward": {
            "mean": float(np.mean(pc_array)),
            "std": float(np.std(pc_array, ddof=1)) if len(pc_array) > 1 else 0.0,
            "min": float(np.min(pc_array)),
            "max": float(np.max(pc_array)),
            "ci95": float(2.0 * np.std(pc_array, ddof=1) / np.sqrt(len(pc_array)))
            if len(pc_array) > 1
            else 0.0,
            "all_episodes": pc_array.tolist(),
        },
        "per_agent_rewards": [
            {
                "agent": i,
                "bot": policy_names[i],
                "mean": float(
                    np.mean([ep[i] for ep in all_episode_rewards])
                ),
            }
            for i in range(n_agents)
        ],
    }

    if verbose:
        print(f"\n  --- {substrate_name} Summary ---")
        print(
            f"  Per-capita reward: "
            f"{result['per_capita_reward']['mean']:.3f} "
            f"+/- {result['per_capita_reward']['ci95']:.3f} "
            f"(std={result['per_capita_reward']['std']:.3f})"
        )
        print(
            f"  Range: [{result['per_capita_reward']['min']:.3f}, "
            f"{result['per_capita_reward']['max']:.3f}]"
        )

    return result


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print results in a format ready for comparison with DigiSoup."""
    print("\n" + "=" * 78)
    print("DEEPMIND PRETRAINED BOT BASELINES -- Per-Capita Reward")
    print("=" * 78)
    print(
        f"{'Substrate':<45s}  {'Mean':>8s}  {'Std':>8s}  "
        f"{'CI95':>10s}  {'N_ep':>4s}"
    )
    print("-" * 78)
    for r in results:
        pc = r["per_capita_reward"]
        print(
            f"{r['substrate']:<45s}  "
            f"{pc['mean']:8.3f}  "
            f"{pc['std']:8.3f}  "
            f"+/- {pc['ci95']:6.3f}  "
            f"{r['n_episodes']:4d}"
        )
    print("-" * 78)
    print()

    # Per-agent breakdown
    for r in results:
        print(f"\n  {r['substrate']} -- per-agent mean reward:")
        for a in r["per_agent_rewards"]:
            short_name = a["bot"].split("__")[-1]
            print(f"    Agent {a['agent']}: {a['mean']:8.2f}  ({short_name})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepMind pretrained bots on Melting Pot substrates."
    )
    parser.add_argument(
        "--substrate",
        type=str,
        default=None,
        help="Single substrate to run (default: all three).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episodes per substrate (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode output.",
    )
    args = parser.parse_args()

    if args.substrate:
        if args.substrate not in SUBSTRATE_CONFIGS:
            print(f"ERROR: Unknown substrate '{args.substrate}'")
            print(f"Available: {list(SUBSTRATE_CONFIGS.keys())}")
            sys.exit(1)
        substrates = [args.substrate]
    else:
        substrates = list(SUBSTRATE_CONFIGS.keys())

    verbose = not args.quiet

    all_results = []
    for sub_name in substrates:
        result = run_substrate(
            sub_name,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=verbose,
        )
        all_results.append(result)

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    if args.save:
        save_path = args.save
    else:
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join(
            "results",
            f"deepmind_baseline_{int(time.time())}.json",
        )

    with open(save_path, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "description": "DeepMind pretrained bot baselines",
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
