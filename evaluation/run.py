"""Evaluation runner for DigiSoup on Melting Pot scenarios.

Runs the official Melting Pot evaluation protocol:
- Builds scenarios (substrate + background bots)
- DigiSoup fills focal slots, DeepMind bots fill background slots
- Collects focal per-capita return (primary metric)
- Reports results with confidence intervals

Usage:
    python -m evaluation.run --substrate commons_harvest__open --episodes 10
    python -m evaluation.run --all-targets --episodes 10
    python -m evaluation.run --scenario clean_up_0 --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import logging
from typing import Any

# Suppress TF/CUDA/absl noise and disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import dm_env

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from meltingpot import scenario as mp_scenario

from agents.digisoup.policy import DigiSoupPolicy
from agents.digisoup.state import DigiSoupState
from configs.scenarios import (
    TARGET_SUBSTRATES,
    get_scenarios_for_substrate,
    get_scenario_info,
)
from evaluation.metrics import EpisodeMetrics, aggregate_episode_metrics
from evaluation.compare import print_results_table

console = Console()

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)


def run_episode(
    scenario_name: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one episode of a Melting Pot scenario with DigiSoup focal agents.

    The scenario handles background bots internally — we only control focal
    players. The env exposes only focal observations, rewards, and actions.
    """
    # Build the scenario (background bots are built-in)
    env = mp_scenario.build(scenario_name)

    # Number of focal players = length of observation tuple
    timestep = env.reset()
    n_focal = len(timestep.observation)

    # Detect action space size (Clean Up has 9: action 8 = FIRE_CLEAN)
    action_spec = env.action_spec()
    n_actions = action_spec[0].num_values if action_spec else 8

    # Create one DigiSoup policy per focal slot
    policies: list[DigiSoupPolicy] = []
    states: list[DigiSoupState] = []
    for i in range(n_focal):
        policy_seed = (seed + i) if seed is not None else (42 + i)
        p = DigiSoupPolicy(seed=policy_seed, n_actions=n_actions)
        policies.append(p)
        states.append(p.initial_state())

    # Run the episode
    metrics = EpisodeMetrics(n_focal=n_focal)

    while timestep.step_type != dm_env.StepType.LAST:
        actions = []
        new_states = []

        for i in range(n_focal):
            # Build per-player timestep from the tuple elements
            reward = float(timestep.reward[i]) if timestep.reward is not None else 0.0
            discount = timestep.discount if not isinstance(timestep.discount, tuple) else float(timestep.discount[i])
            player_ts = dm_env.TimeStep(
                step_type=timestep.step_type,
                reward=reward,
                discount=discount,
                observation=timestep.observation[i],
            )

            action, new_state = policies[i].step(player_ts, states[i])
            actions.append(action)
            new_states.append(new_state)

        states = new_states

        # Step the environment with focal actions only
        timestep = env.step(actions)

        # Record focal rewards
        focal_rewards = []
        for i in range(n_focal):
            r = float(timestep.reward[i]) if timestep.reward is not None else 0.0
            focal_rewards.append(r)
        metrics.record_step(focal_rewards)

    # Clean up
    env.close()
    for p in policies:
        p.close()

    return metrics.compute()


def run_scenario(
    scenario_name: str,
    n_episodes: int = 10,
    base_seed: int | None = None,
) -> dict[str, Any]:
    """Run multiple episodes on a scenario and aggregate results."""
    info = get_scenario_info(scenario_name)

    console.print(
        f"  [cyan]{scenario_name}[/cyan] "
        f"({info['n_focal']} focal, {info['n_background']} background, "
        f"{n_episodes} episodes)"
    )

    episode_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"    {scenario_name}", total=n_episodes)
        for ep in range(n_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            result = run_episode(scenario_name, seed=seed)
            episode_results.append(result)
            progress.update(task, advance=1)

    aggregated = aggregate_episode_metrics(episode_results)
    aggregated["scenario"] = scenario_name
    aggregated["substrate"] = info["substrate"]
    return aggregated


def run_substrate(
    substrate_name: str,
    n_episodes: int = 10,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run all scenarios for a substrate."""
    scenarios = get_scenarios_for_substrate(substrate_name)
    if not scenarios:
        console.print(f"[red]No scenarios found for {substrate_name}[/red]")
        return []

    console.print()
    console.print(
        f"[bold cyan]Substrate: {substrate_name}[/bold cyan] "
        f"({len(scenarios)} scenarios)"
    )

    results = []
    for sc in scenarios:
        result = run_scenario(sc, n_episodes, base_seed)
        results.append(result)

    return results


def save_results(
    results: list[dict[str, Any]],
    label: str = "digisoup",
) -> str:
    """Save results to JSON. Returns the file path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = int(time.time())
    filename = f"{label}_{ts}.json"
    path = os.path.join(RESULTS_DIR, filename)

    # Make results JSON-serializable
    def _clean(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    payload = {
        "timestamp": time.time(),
        "label": label,
        "results": json.loads(json.dumps(results, default=_clean)),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    console.print(f"[green]Results saved to {path}[/green]")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DigiSoup evaluation on Melting Pot scenarios.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--substrate", type=str, help="Run all scenarios for a substrate.")
    group.add_argument("--scenario", type=str, help="Run a single scenario.")
    group.add_argument("--all-targets", action="store_true", help="Run all target substrates.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per scenario (default: 10).")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--label", type=str, default="digisoup", help="Label for results file.")
    args = parser.parse_args(argv)

    console.print()
    console.print("[bold]DigiSoup vs Melting Pot -- Official Evaluation[/bold]")
    console.print(f"Agent: zero-training entropy-driven (reward never used)")
    console.print()

    all_results: list[dict[str, Any]] = []

    if args.scenario:
        result = run_scenario(args.scenario, args.episodes, args.seed)
        all_results.append(result)

    elif args.substrate:
        results = run_substrate(args.substrate, args.episodes, args.seed)
        all_results.extend(results)

    elif args.all_targets:
        for sub in TARGET_SUBSTRATES:
            results = run_substrate(sub, args.episodes, args.seed)
            all_results.extend(results)

    # Print results
    if all_results:
        print_results_table(all_results)
        save_results(all_results, label=args.label)

    console.print("[bold green]Evaluation complete.[/bold green]")


if __name__ == "__main__":
    main()
