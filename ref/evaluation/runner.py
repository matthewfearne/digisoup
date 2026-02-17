# runner.py -- Run agents on Melting Pot substrates.
# Handles substrate loading, agent instantiation, episode loop, metrics.
"""
Evaluation runner for DigiSoup vs Melting Pot.

Usage:
    python3 -m src.evaluation.runner --substrate commons_harvest --agent digisoup --episodes 30
    python3 -m src.evaluation.runner --substrate commons_harvest --agent random --episodes 30
    python3 -m src.evaluation.runner --all-targets --agent digisoup --episodes 30

Supports both "digisoup" and "random" agent types.  Random agent uniformly
samples from the action space and serves as the statistical floor.

Results saved to results/<substrate>_<agent>_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

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
from rich.table import Table

from src.evaluation.metrics import MetricsCollector, aggregate_metrics
from src.evaluation.recorder import EpisodeRecorder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical substrate names used throughout the project.
SUBSTRATE_ALIASES: dict[str, str] = {
    "commons_harvest": "commons_harvest__open",
    "commons_harvest__open": "commons_harvest__open",
    "clean_up": "clean_up",
    "prisoners_dilemma": "prisoners_dilemma_in_the_matrix__arena",
    "prisoners_dilemma_in_the_matrix__arena": (
        "prisoners_dilemma_in_the_matrix__arena"
    ),
    "allelopathic_harvest": "allelopathic_harvest__open",
    "allelopathic_harvest__open": "allelopathic_harvest__open",
    "stag_hunt": "stag_hunt_in_the_matrix__arena",
    "stag_hunt_in_the_matrix__arena": "stag_hunt_in_the_matrix__arena",
    "collaborative_cooking": "collaborative_cooking__ring",
    "collaborative_cooking__ring": "collaborative_cooking__ring",
}

PRIORITY_1_SUBSTRATES = [
    "commons_harvest__open",
    "clean_up",
    "prisoners_dilemma_in_the_matrix__arena",
]

ALL_TARGET_SUBSTRATES = [
    "commons_harvest__open",
    "clean_up",
    "prisoners_dilemma_in_the_matrix__arena",
    "allelopathic_harvest__open",
    "stag_hunt_in_the_matrix__arena",
    "collaborative_cooking__ring",
]

DEFAULT_N_AGENTS = 7
DEFAULT_MAX_CYCLES = 1000
DEFAULT_EPISODES = 30
N_ACTIONS = 8  # Default Melting Pot discrete action space: 0-7

# Substrate-specific action counts (Clean Up has 9: action 8 = fireClean)
SUBSTRATE_N_ACTIONS: dict[str, int] = {
    "clean_up": 9,
}


def _get_n_actions(substrate_name: str) -> int:
    """Get the number of discrete actions for a substrate."""
    return SUBSTRATE_N_ACTIONS.get(substrate_name, N_ACTIONS)


# Thermodynamic personality profiles — 4 cleaners + 3 harvesters.
# Same entropy physics, different metabolic constants. Roles EMERGE from the
# interaction between these constants and the environment, not from assignment.
PERSONALITY_PROFILES = [
    # 4 cleaners: interact-biased (prefer to engage with what's nearby)
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},
    {"interact_bias": 1.15, "move_bias": 0.95, "scan_bias": 0.95},
    # 3 harvesters: move/scan-biased (prefer to explore and relocate)
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},
    {"interact_bias": 0.85, "move_bias": 1.05, "scan_bias": 1.05},
]

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)

console = Console()


# ---------------------------------------------------------------------------
# Substrate loader
# ---------------------------------------------------------------------------

def _load_substrate_env(substrate_name: str, n_agents: int, max_cycles: int):
    """Load a substrate environment.

    Tries in order:
    1. shimmy/meltingpot (actual Melting Pot via dmlab2d) → ParallelEnv
    2. Built-in replications → AEC env
    3. Mock environment → AEC-like
    """
    # 1. Try actual Melting Pot via shimmy (preferred)
    try:
        from shimmy import MeltingPotCompatibilityV0  # type: ignore
        env = MeltingPotCompatibilityV0(
            substrate_name=substrate_name, render_mode=None,
        )
        console.print(
            f"  [green]Using actual Melting Pot (dmlab2d)[/green]"
        )
        return env
    except (ImportError, Exception):
        pass

    # 2. Built-in replications
    _BUILTIN = {}
    try:
        from src.substrates.commons_harvest import CommonsHarvestEnv
        _BUILTIN["commons_harvest__open"] = CommonsHarvestEnv
    except ImportError:
        pass
    try:
        from src.substrates.clean_up import CleanUpEnv
        _BUILTIN["clean_up"] = CleanUpEnv
    except ImportError:
        pass
    try:
        from src.substrates.prisoners_dilemma import PrisonersDilemmaEnv
        _BUILTIN["prisoners_dilemma_in_the_matrix__arena"] = PrisonersDilemmaEnv
    except ImportError:
        pass

    if substrate_name in _BUILTIN:
        EnvClass = _BUILTIN[substrate_name]
        console.print(
            f"  [yellow]Using built-in replication[/yellow]"
        )
        return EnvClass(n_agents=n_agents, max_cycles=max_cycles)

    # 3. Fall back to mock environment
    console.print(
        f"[yellow]WARNING: No implementation for '{substrate_name}'. "
        f"Using mock environment.[/yellow]"
    )
    return _MockAECEnv(substrate_name, n_agents, max_cycles)


class _MockAECEnv:
    """Minimal PettingZoo AEC-like mock for testing without Melting Pot.

    Generates random 88x88x3 observations and small random rewards.
    Terminates after max_cycles steps.
    """

    def __init__(
        self,
        substrate_name: str,
        n_agents: int = 7,
        max_cycles: int = 1000,
    ):
        self.substrate_name = substrate_name
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.possible_agents = [f"player_{i}" for i in range(n_agents)]
        self.agents: list[str] = []
        self._agent_iter_list: list[str] = []
        self._cycle = 0
        self._rng = np.random.default_rng()
        self._last: dict[str, tuple] = {}
        self._current_agent_idx = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._cycle = 0
        self._current_agent_idx = 0
        for agent in self.agents:
            obs = self._rng.integers(0, 256, (88, 88, 3), dtype=np.uint8)
            self._last[agent] = (obs, 0.0, False, False, {})

    def agent_iter(self):
        """Yield agents one at a time, AEC-style."""
        while self.agents and self._cycle < self.max_cycles:
            for agent in list(self.agents):
                if not self.agents:
                    return
                yield agent

    def last(self):
        """Return (observation, reward, termination, truncation, info)."""
        agent = self.agents[self._current_agent_idx % len(self.agents)]
        return self._last.get(
            agent,
            (
                self._rng.integers(0, 256, (88, 88, 3), dtype=np.uint8),
                0.0,
                False,
                False,
                {},
            ),
        )

    def step(self, action: int) -> None:
        """Advance one step for the current agent."""
        if not self.agents:
            return
        agent = self.agents[self._current_agent_idx % len(self.agents)]
        self._current_agent_idx += 1

        # Generate next observation and reward
        obs = self._rng.integers(0, 256, (88, 88, 3), dtype=np.uint8)
        reward = float(self._rng.normal(0.0, 0.5))

        # Count cycles (one full round of agents = one cycle)
        if self._current_agent_idx % len(self.possible_agents) == 0:
            self._cycle += 1

        terminated = self._cycle >= self.max_cycles
        truncated = False

        self._last[agent] = (obs, reward, terminated, truncated, {})

        if terminated:
            self.agents = []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

class _RandomAgent:
    """Uniformly random agent -- the statistical floor."""

    def __init__(self, n_actions: int = N_ACTIONS, seed: int | None = None):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Any, reward: float = 0.0) -> int:
        return int(self._rng.integers(0, self.n_actions))

    def reset(self) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {"type": "random"}


class _DigiSoupAgentWrapper:
    """Wraps the DigiSoup agent, handling import and instantiation."""

    def __init__(
        self,
        seed: int | None = None,
        n_actions: int = N_ACTIONS,
        personality: dict | None = None,
    ):
        self._agent = None
        self._seed = seed
        self._n_actions = n_actions
        self._personality = personality
        self._step_count = 0
        try:
            from src.agent.core import DigiSoupAgent
            self._agent = DigiSoupAgent(
                n_actions=n_actions, personality=personality
            )
        except (ImportError, Exception):
            try:
                from src.agent.core import DigiSoupMeltingPotAgent
                self._agent = DigiSoupMeltingPotAgent()
            except (ImportError, Exception):
                # Agent core is still a stub; use fallback
                self._agent = None

    @property
    def available(self) -> bool:
        return self._agent is not None

    def act(self, observation: Any, reward: float = 0.0) -> int:
        if self._agent is not None:
            try:
                return int(self._agent.act(observation, reward))
            except Exception:
                pass
        # Fallback: entropy-biased random if agent is unavailable
        return self._entropy_fallback(observation)

    def _entropy_fallback(self, observation: Any) -> int:
        """Simple entropy-gradient action when full agent is not built yet.

        Uses the entropy module directly to pick a direction.
        """
        try:
            from src.agent.entropy import fine_entropy_gradient
            grad = fine_entropy_gradient(np.asarray(observation))
            # Map gradient to action: 0=noop, 1=up, 2=down, 3=left, 4=right,
            # 5=turn_left, 6=turn_right, 7=interact
            dy, dx = float(grad[0]), float(grad[1])
            if abs(dy) < 0.1 and abs(dx) < 0.1:
                # Low gradient -- explore or interact
                self._step_count += 1
                if self._step_count % 5 == 0:
                    return 7  # interact periodically
                return int(np.random.randint(0, N_ACTIONS))
            if abs(dy) > abs(dx):
                return 2 if dy > 0 else 1  # down / up
            else:
                return 4 if dx > 0 else 3  # right / left
        except Exception:
            return int(np.random.randint(0, N_ACTIONS))

    def reset(self) -> None:
        self._step_count = 0
        if self._agent is not None:
            try:
                self._agent.reset()
            except Exception:
                pass

    def get_state(self) -> dict[str, Any]:
        if self._agent is not None:
            for attr in ("summary", "get_state", "state"):
                fn_or_val = getattr(self._agent, attr, None)
                if callable(fn_or_val):
                    try:
                        return fn_or_val()
                    except Exception:
                        pass
                elif fn_or_val is not None and isinstance(fn_or_val, dict):
                    return fn_or_val
        return {"type": "digisoup", "available": self.available}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """Runs a single episode of agents on a substrate."""

    def __init__(
        self,
        substrate_name: str,
        agent_type: str = "digisoup",
        n_agents: int = DEFAULT_N_AGENTS,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        record: bool = True,
    ):
        self.substrate_name = SUBSTRATE_ALIASES.get(
            substrate_name, substrate_name
        )
        self.agent_type = agent_type
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.record = record

    def run_episode(
        self,
        seed: int | None = None,
        episode_num: int = 0,
    ) -> dict[str, Any]:
        """Run one episode, return per-agent metrics and optional recording.

        Supports both ParallelEnv (actual Melting Pot via shimmy) and
        AEC environments (our built-in substrates).

        Returns:
            dict with keys: metrics (dict), recording_path (str or None),
            seed (int or None), episode (int).
        """
        env = _load_substrate_env(
            self.substrate_name, self.n_agents, self.max_cycles
        )

        # Detect environment type: ParallelEnv vs AEC
        _is_parallel = not hasattr(env, "agent_selection")

        # Metrics and recording
        n_agents_actual = len(env.possible_agents)
        collector = MetricsCollector(n_agents_actual)
        recorder = (
            EpisodeRecorder(self.substrate_name, self.agent_type, episode_num)
            if self.record
            else None
        )

        # Create agents for each player in the environment
        n_act = _get_n_actions(self.substrate_name)
        agents: dict[str, Any] = {}
        for i, agent_id in enumerate(env.possible_agents):
            agent_seed = seed + i if seed is not None else None
            if self.agent_type == "random":
                agents[agent_id] = _RandomAgent(
                    n_actions=n_act, seed=agent_seed
                )
            elif self.agent_type == "digisoup":
                # Thermodynamic personality: same physics, different constants
                personality = PERSONALITY_PROFILES[i % len(PERSONALITY_PROFILES)]
                agents[agent_id] = _DigiSoupAgentWrapper(
                    seed=agent_seed, n_actions=n_act, personality=personality
                )
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type!r}")

        # Reset agents
        for ag in agents.values():
            ag.reset()

        step_count = 0

        try:
            if _is_parallel:
                # === ParallelEnv API (actual Melting Pot via shimmy) ===
                observations, infos = env.reset(seed=seed)
                # Track rewards from previous step (0 on first step)
                prev_rewards = {a: 0.0 for a in env.agents}

                while env.agents:
                    # All agents act simultaneously
                    actions = {}
                    for agent_id in env.agents:
                        obs = observations.get(agent_id)
                        reward = float(prev_rewards.get(agent_id, 0.0))
                        agent = agents.get(agent_id)
                        if agent is None:
                            actions[agent_id] = int(np.random.randint(0, n_act))
                        else:
                            actions[agent_id] = agent.act(obs, reward)

                    # Step all agents at once
                    observations, rewards, terminations, truncations, infos = env.step(actions)

                    # Record metrics for each agent
                    for agent_id in actions:
                        r = float(rewards.get(agent_id, 0.0))
                        collector.record_step(agent_id, r, actions[agent_id], {})

                        if recorder is not None:
                            obs_ent = 0.0
                            try:
                                from src.agent.entropy import observation_entropy
                                raw_obs = observations.get(agent_id, {})
                                if isinstance(raw_obs, dict):
                                    raw_obs = raw_obs.get("RGB", np.zeros((88, 88, 3), dtype=np.uint8))
                                obs_ent = observation_entropy(np.asarray(raw_obs))
                            except Exception:
                                pass
                            agent_state = (
                                agents[agent_id].get_state()
                                if agent_id in agents
                                else {}
                            )
                            recorder.record(
                                step_count, agent_id, obs_ent, r,
                                actions[agent_id], agent_state,
                            )

                    prev_rewards = {a: float(rewards.get(a, 0.0)) for a in env.agents}
                    step_count += 1

            else:
                # === AEC API (our built-in substrates) ===
                env.reset(seed=seed)
                _has_observe = hasattr(env, "observe")
                _has_agent_iter = hasattr(env, "agent_iter")

                if _has_observe:
                    while env.agents:
                        agent_id = env.agent_selection
                        if env.terminations.get(agent_id) or env.truncations.get(agent_id):
                            env.step(None)
                            continue

                        obs = env.observe(agent_id)
                        reward = env.rewards.get(agent_id, 0.0)

                        agent = agents.get(agent_id)
                        if agent is None:
                            action = int(np.random.randint(0, n_act))
                        else:
                            action = agent.act(obs, reward)

                        env.step(action)
                        collector.record_step(agent_id, reward, action, {})

                        if recorder is not None:
                            obs_ent = 0.0
                            try:
                                from src.agent.entropy import observation_entropy
                                obs_ent = observation_entropy(np.asarray(obs))
                            except Exception:
                                pass
                            agent_state = (
                                agent.get_state() if agent is not None else {}
                            )
                            recorder.record(
                                step_count, agent_id, obs_ent, reward, action,
                                agent_state,
                            )
                        step_count += 1

                elif _has_agent_iter:
                    for agent_id in env.agent_iter():
                        if not env.agents:
                            break
                        obs, reward, terminated, truncated, info = env.last()
                        if terminated or truncated:
                            env.step(None)
                            continue
                        agent = agents.get(agent_id)
                        if agent is None:
                            action = int(np.random.randint(0, n_act))
                        else:
                            action = agent.act(obs, reward)
                        env.step(action)
                        collector.record_step(agent_id, reward, action, info)
                        step_count += 1

        except Exception as exc:
            console.print(f"[red]Episode error at step {step_count}: {exc}[/red]")

        # Clean up
        try:
            env.close()
        except Exception:
            pass

        # Compute metrics
        metrics = collector.compute()

        # Save recording
        recording_path = None
        if recorder is not None:
            rec_dir = os.path.join(RESULTS_DIR, "recordings")
            try:
                recording_path = recorder.save(rec_dir)
            except Exception as exc:
                console.print(
                    f"[yellow]Warning: could not save recording: {exc}[/yellow]"
                )

        return {
            "metrics": metrics,
            "recording_path": recording_path,
            "seed": seed,
            "episode": episode_num,
        }


# ---------------------------------------------------------------------------
# Multi-episode evaluation runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """Runs multiple episodes and aggregates results."""

    def __init__(
        self,
        substrate_name: str,
        agent_type: str,
        episodes: int = DEFAULT_EPISODES,
        n_agents: int = DEFAULT_N_AGENTS,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        record: bool = True,
        base_seed: int | None = None,
    ):
        self.substrate_name = SUBSTRATE_ALIASES.get(
            substrate_name, substrate_name
        )
        self.agent_type = agent_type
        self.episodes = episodes
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.record = record
        self.base_seed = base_seed

        self.episode_results: list[dict[str, Any]] = []
        self.aggregated: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Run all episodes with a Rich progress bar.

        Returns the aggregated metrics dict.
        """
        runner = EpisodeRunner(
            self.substrate_name,
            self.agent_type,
            self.n_agents,
            self.max_cycles,
            self.record,
        )

        console.print()
        console.print(
            f"[bold cyan]Evaluating[/bold cyan] "
            f"[bold]{self.agent_type}[/bold] on "
            f"[bold]{self.substrate_name}[/bold] "
            f"({self.episodes} episodes, {self.n_agents} agents, "
            f"max {self.max_cycles} cycles)"
        )
        console.print()

        self.episode_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Running episodes...", total=self.episodes
            )

            for ep in range(self.episodes):
                seed = (
                    self.base_seed + ep
                    if self.base_seed is not None
                    else None
                )
                result = runner.run_episode(seed=seed, episode_num=ep)
                self.episode_results.append(result)
                progress.update(task, advance=1)

        # Aggregate
        all_metrics = [r["metrics"] for r in self.episode_results]
        self.aggregated = aggregate_metrics(all_metrics)
        self.aggregated["substrate"] = self.substrate_name
        self.aggregated["agent_type"] = self.agent_type

        # Print summary table
        self._print_summary()

        return self.aggregated

    def _print_summary(self) -> None:
        """Print a Rich summary table of aggregated results."""
        table = Table(
            title=(
                f"Results: {self.agent_type} on {self.substrate_name} "
                f"({self.aggregated.get('n_episodes', '?')} episodes)"
            ),
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("CI 95%", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        display_keys = [
            ("per_capita_reward", "Per-Capita Reward"),
            ("total_reward", "Total Reward"),
            ("cooperation_ratio", "Cooperation Ratio"),
            ("cooperation_events", "Cooperation Events"),
            ("resource_efficiency", "Resource Efficiency"),
            ("gini_coefficient", "Gini Coefficient"),
            ("sustainability", "Sustainability"),
            ("steps", "Steps"),
        ]

        for key, label in display_keys:
            info = self.aggregated.get(key)
            if info is None or not isinstance(info, dict):
                continue
            table.add_row(
                label,
                f"{info['mean']:.4f}",
                f"{info['std']:.4f}",
                f"+/- {info['ci95']:.4f}",
                f"{info['min']:.4f}",
                f"{info['max']:.4f}",
            )

        console.print()
        console.print(table)
        console.print()

    def save_results(self, path: str | None = None) -> str:
        """Save aggregated results and per-episode metrics to JSON.

        Returns the path to the saved file.
        """
        if path is None:
            ts = int(time.time())
            filename = (
                f"{self.substrate_name}_{self.agent_type}_{ts}.json"
            )
            path = os.path.join(RESULTS_DIR, filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "substrate": self.substrate_name,
            "agent_type": self.agent_type,
            "n_episodes": self.episodes,
            "n_agents": self.n_agents,
            "max_cycles": self.max_cycles,
            "timestamp": time.time(),
            "aggregated": self.aggregated,
            "episodes": [
                {
                    "episode": r["episode"],
                    "seed": r["seed"],
                    "metrics": r["metrics"],
                    "recording_path": r["recording_path"],
                }
                for r in self.episode_results
            ],
        }

        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        console.print(f"[green]Results saved to {path}[/green]")
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DigiSoup vs Melting Pot evaluation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m src.evaluation.runner "
            "--substrate commons_harvest --agent random --episodes 10\n"
            "  python3 -m src.evaluation.runner "
            "--substrate commons_harvest --agent digisoup --episodes 10\n"
            "  python3 -m src.evaluation.runner "
            "--all-targets --agent digisoup --episodes 10\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--substrate",
        type=str,
        help="Substrate name (e.g., commons_harvest, clean_up).",
    )
    group.add_argument(
        "--all-targets",
        action="store_true",
        help="Run on all priority-1 target substrates.",
    )
    group.add_argument(
        "--all-substrates",
        action="store_true",
        help="Run on all 6 target substrates.",
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["digisoup", "random"],
        help="Agent type.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of episodes per substrate (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=DEFAULT_N_AGENTS,
        help=f"Number of agents per episode (default: {DEFAULT_N_AGENTS}).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_MAX_CYCLES,
        help=f"Max cycles per episode (default: {DEFAULT_MAX_CYCLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable per-step recording (faster, less disk usage).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    record = not args.no_record

    # Determine substrates to run
    if args.all_targets:
        substrates = PRIORITY_1_SUBSTRATES
    elif args.all_substrates:
        substrates = ALL_TARGET_SUBSTRATES
    else:
        canonical = SUBSTRATE_ALIASES.get(args.substrate, args.substrate)
        substrates = [canonical]

    all_aggregated: list[dict[str, Any]] = []

    for substrate in substrates:
        runner = EvaluationRunner(
            substrate_name=substrate,
            agent_type=args.agent,
            episodes=args.episodes,
            n_agents=args.n_agents,
            max_cycles=args.max_cycles,
            record=record,
            base_seed=args.seed,
        )
        aggregated = runner.run()
        all_aggregated.append(aggregated)

        # Save per-substrate results
        runner.save_results(args.output)

    # If multiple substrates, show comparison
    if len(all_aggregated) > 1:
        _print_multi_substrate_summary(all_aggregated)

    console.print("[bold green]Evaluation complete.[/bold green]")


def _print_multi_substrate_summary(
    all_aggregated: list[dict[str, Any]],
) -> None:
    """Print a cross-substrate summary table."""
    table = Table(
        title="Cross-Substrate Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Substrate", style="bold")
    table.add_column("Agent", justify="center")
    table.add_column("Per-Capita Reward", justify="right")
    table.add_column("Cooperation", justify="right")
    table.add_column("Sustainability", justify="right")
    table.add_column("Gini", justify="right")

    for agg in all_aggregated:
        pcr = agg.get("per_capita_reward", {})
        coop = agg.get("cooperation_ratio", {})
        sust = agg.get("sustainability", {})
        gini = agg.get("gini_coefficient", {})
        table.add_row(
            agg.get("substrate", "?"),
            agg.get("agent_type", "?"),
            f"{pcr.get('mean', 0.0):.4f} +/- {pcr.get('ci95', 0.0):.4f}",
            f"{coop.get('mean', 0.0):.3f}",
            f"{sust.get('mean', 0.0):.3f}",
            f"{gini.get('mean', 0.0):.3f}",
        )

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
