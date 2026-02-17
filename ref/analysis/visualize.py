# visualize.py -- Matplotlib publication-ready figures. Comparison charts.
"""
Visualization module for DigiSoup vs Melting Pot results.

Generates:
    - Comparison bar chart (DigiSoup vs DeepMind baselines)
    - Cooperation timeline plots
    - Role distribution pie/bar charts
    - Reward distribution box plots
    - Multi-panel summary figure

All figures are saved as PNG files suitable for publication.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.evaluation.compare import DEEPMIND_BASELINES


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
}

# Colour palette
_COLOURS = {
    "digisoup": "#2ecc71",      # green
    "random": "#95a5a6",        # grey
    "A3C": "#e74c3c",           # red
    "A3C_prosocial": "#e67e22", # orange
    "OPRE": "#3498db",          # blue
    "OPRE_prosocial": "#9b59b6",# purple
}

_ROLE_COLOURS = [
    "#2ecc71", "#e74c3c", "#3498db", "#e67e22", "#9b59b6",
    "#1abc9c", "#f39c12", "#2c3e50", "#d35400", "#16a085",
    "#8e44ad", "#c0392b",
]


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Comparison bar chart: DigiSoup vs DeepMind baselines
# ---------------------------------------------------------------------------

def plot_comparison_table(
    results: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]] | None = None,
    output_path: str = "results/comparison.png",
) -> str:
    """Bar chart comparing DigiSoup normalised scores against DeepMind baselines.

    *results* is a list of dicts from BaselineComparator.compare().
    Returns the output path.
    """
    _apply_style()
    if baselines is None:
        baselines = DEEPMIND_BASELINES

    substrates = [r["substrate"] for r in results]
    n_substrates = len(substrates)
    if n_substrates == 0:
        return output_path

    # Agents to display
    agent_keys = ["A3C", "A3C_prosocial", "OPRE", "OPRE_prosocial"]
    all_keys = agent_keys + ["DigiSoup"]
    n_bars = len(all_keys)
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n_substrates * 2.5), 6))

    x = np.arange(n_substrates)

    for i, key in enumerate(agent_keys):
        vals = []
        for r in results:
            sub_baselines = baselines.get(r["substrate"], {})
            vals.append(sub_baselines.get(key, 0.0))
        ax.bar(
            x + i * bar_width,
            vals,
            bar_width,
            label=key.replace("_", " "),
            color=_COLOURS.get(key, "#cccccc"),
            alpha=0.85,
        )

    # DigiSoup bar
    ds_vals = [r.get("digisoup_normalised", 0.0) for r in results]
    ax.bar(
        x + len(agent_keys) * bar_width,
        ds_vals,
        bar_width,
        label="DigiSoup",
        color=_COLOURS["digisoup"],
        edgecolor="black",
        linewidth=1.2,
    )

    # Labels
    short_names = [s.split("__")[0].replace("_", " ").title() for s in substrates]
    ax.set_xticks(x + bar_width * n_bars / 2)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_ylabel("Normalised Score")
    ax.set_title("DigiSoup vs DeepMind Baselines (Melting Pot)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylim(bottom=min(-0.1, min(ds_vals) - 0.05))

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Cooperation timeline
# ---------------------------------------------------------------------------

def plot_cooperation_timeline(
    data: list[dict[str, Any]],
    output_path: str = "results/cooperation_timeline.png",
    title: str = "Cooperation Ratio Over Time",
    label: str = "Cooperation",
) -> str:
    """Line plot: cooperation ratio over time.

    *data* is a list of window dicts containing ``cooperation_ratio_mean``
    (aggregated) or ``cooperation_ratio`` (single episode), plus ``window``.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    windows = [d["window"] for d in data]
    mean_key = (
        "cooperation_ratio_mean"
        if "cooperation_ratio_mean" in data[0]
        else "cooperation_ratio"
    )
    means = [d[mean_key] for d in data]

    ax.plot(windows, means, color=_COLOURS["digisoup"], linewidth=2, label=label)

    # Add std band if available
    if "cooperation_ratio_std" in data[0]:
        stds = [d["cooperation_ratio_std"] for d in data]
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(
            windows, lower, upper,
            color=_COLOURS["digisoup"], alpha=0.2,
        )

    ax.set_xlabel("Time Window")
    ax.set_ylabel("Cooperation Ratio")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Role distribution
# ---------------------------------------------------------------------------

def plot_role_distribution(
    data: dict[str, Any],
    output_path: str = "results/role_distribution.png",
    chart_type: str = "bar",
) -> str:
    """Bar or pie chart showing behavioural type distribution.

    *data* is a dict as returned by ``EmergenceAnalyzer.role_distribution()``,
    with a ``fractions`` key.
    """
    _apply_style()

    fracs = data.get("fractions", {})
    if not fracs:
        # Create empty placeholder figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No role data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _ensure_dir(output_path)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    roles = sorted(fracs.keys(), key=lambda r: fracs[r], reverse=True)
    values = [fracs[r] for r in roles]
    colours = [_ROLE_COLOURS[i % len(_ROLE_COLOURS)] for i in range(len(roles))]

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "pie":
        wedges, texts, autotexts = ax.pie(
            values,
            labels=roles,
            colors=colours,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Behavioural Type Distribution")
    else:
        bars = ax.barh(roles, values, color=colours)
        ax.set_xlabel("Fraction")
        ax.set_title("Behavioural Type Distribution")
        ax.set_xlim(0, max(values) * 1.15)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                fontsize=9,
            )

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 4. Reward distribution box plots
# ---------------------------------------------------------------------------

def plot_reward_distribution(
    data: dict[str, list[float]],
    output_path: str = "results/reward_distribution.png",
) -> str:
    """Box plots of per-capita reward distribution.

    *data* is a dict mapping label (e.g., "DigiSoup", "Random") to a list
    of per-episode per-capita rewards.
    """
    _apply_style()

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No reward data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _ensure_dir(output_path)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    labels = list(data.keys())
    values = [data[k] for k in labels]
    colours = [_COLOURS.get(k.lower(), "#cccccc") for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6},
    )

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Capita Reward")
    ax.set_title("Reward Distribution Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. Reward trajectory over time
# ---------------------------------------------------------------------------

def plot_reward_trajectory(
    data: list[dict[str, Any]],
    output_path: str = "results/reward_trajectory.png",
    title: str = "Reward Over Time",
) -> str:
    """Line plot of mean reward over time windows.

    *data* is aggregated reward trajectory from TemporalAnalyzer.
    """
    _apply_style()

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _ensure_dir(output_path)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    windows = [d["window"] for d in data]
    means = [d.get("mean_reward", 0.0) for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(windows, means, color=_COLOURS["digisoup"], linewidth=2)

    if "std_reward" in data[0]:
        stds = [d["std_reward"] for d in data]
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(windows, lower, upper, color=_COLOURS["digisoup"], alpha=0.2)

    ax.set_xlabel("Time Window")
    ax.set_ylabel("Mean Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 6. Role timeline (stacked area)
# ---------------------------------------------------------------------------

def plot_role_timeline(
    data: list[dict[str, Any]],
    output_path: str = "results/role_timeline.png",
) -> str:
    """Stacked area chart of role distribution over time.

    *data* is from EmergenceAnalyzer.role_timeline().
    """
    _apply_style()

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No role timeline data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _ensure_dir(output_path)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Collect all roles across all windows
    all_roles: set[str] = set()
    for d in data:
        all_roles.update(d.get("role_distribution", {}).keys())
    all_roles_sorted = sorted(all_roles)

    windows = [d["window"] for d in data]
    stacks = {
        role: [d.get("role_distribution", {}).get(role, 0.0) for d in data]
        for role in all_roles_sorted
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = [_ROLE_COLOURS[i % len(_ROLE_COLOURS)] for i in range(len(all_roles_sorted))]

    ax.stackplot(
        windows,
        *stacks.values(),
        labels=all_roles_sorted,
        colors=colours,
        alpha=0.8,
    )
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Fraction")
    ax.set_title("Role Distribution Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 7. Multi-panel summary
# ---------------------------------------------------------------------------

def plot_summary(
    comparison_results: list[dict[str, Any]] | None = None,
    cooperation_timeline: list[dict[str, Any]] | None = None,
    role_distribution: dict[str, Any] | None = None,
    reward_data: dict[str, list[float]] | None = None,
    output_path: str = "results/summary.png",
) -> str:
    """Multi-panel summary figure (2x2 grid).

    Combines: comparison bar chart, cooperation timeline,
    role distribution, reward box plots.
    """
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Comparison (top-left)
    ax1 = axes[0, 0]
    if comparison_results:
        substrates = [r["substrate"].split("__")[0].replace("_", " ").title()
                      for r in comparison_results]
        ds_scores = [r.get("digisoup_normalised", 0.0) for r in comparison_results]
        best_baselines = []
        for r in comparison_results:
            bl = DEEPMIND_BASELINES.get(r["substrate"], {})
            best_baselines.append(max(bl.values()) if bl else 0.0)

        x = np.arange(len(substrates))
        w = 0.35
        ax1.bar(x - w / 2, best_baselines, w, label="Best Baseline",
                color=_COLOURS["OPRE_prosocial"], alpha=0.7)
        ax1.bar(x + w / 2, ds_scores, w, label="DigiSoup",
                color=_COLOURS["digisoup"], edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels(substrates, rotation=15, ha="right", fontsize=9)
        ax1.set_ylabel("Normalised Score")
        ax1.set_title("DigiSoup vs Best Baseline")
        ax1.legend(fontsize=9)
        ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    else:
        ax1.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_title("DigiSoup vs Best Baseline")

    # Panel 2: Cooperation timeline (top-right)
    ax2 = axes[0, 1]
    if cooperation_timeline:
        ws = [d["window"] for d in cooperation_timeline]
        key = ("cooperation_ratio_mean"
               if "cooperation_ratio_mean" in cooperation_timeline[0]
               else "cooperation_ratio")
        vals = [d[key] for d in cooperation_timeline]
        ax2.plot(ws, vals, color=_COLOURS["digisoup"], linewidth=2)
        if "cooperation_ratio_std" in cooperation_timeline[0]:
            stds = [d["cooperation_ratio_std"] for d in cooperation_timeline]
            ax2.fill_between(ws, [v - s for v, s in zip(vals, stds)],
                             [v + s for v, s in zip(vals, stds)],
                             color=_COLOURS["digisoup"], alpha=0.2)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("Window")
        ax2.set_ylabel("Cooperation Ratio")
    else:
        ax2.text(0.5, 0.5, "No timeline data", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Cooperation Over Time")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Role distribution (bottom-left)
    ax3 = axes[1, 0]
    if role_distribution and role_distribution.get("fractions"):
        fracs = role_distribution["fractions"]
        roles = sorted(fracs.keys(), key=lambda r: fracs[r], reverse=True)
        values = [fracs[r] for r in roles]
        colours = [_ROLE_COLOURS[i % len(_ROLE_COLOURS)] for i in range(len(roles))]
        ax3.barh(roles, values, color=colours)
        ax3.set_xlabel("Fraction")
    else:
        ax3.text(0.5, 0.5, "No role data", ha="center", va="center",
                 transform=ax3.transAxes)
    ax3.set_title("Role Distribution")

    # Panel 4: Reward distribution (bottom-right)
    ax4 = axes[1, 1]
    if reward_data:
        labels = list(reward_data.keys())
        vals = [reward_data[k] for k in labels]
        colours = [_COLOURS.get(k.lower(), "#cccccc") for k in labels]
        bp = ax4.boxplot(vals, labels=labels, patch_artist=True, widths=0.5,
                         showmeans=True)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax4.set_ylabel("Per-Capita Reward")
    else:
        ax4.text(0.5, 0.5, "No reward data", ha="center", va="center",
                 transform=ax4.transAxes)
    ax4.set_title("Reward Distribution")
    ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DigiSoup vs Melting Pot -- Summary", fontsize=16, y=1.02)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
