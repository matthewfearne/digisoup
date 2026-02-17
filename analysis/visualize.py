"""Visualization for DigiSoup vs Melting Pot results.

Generates publication-ready figures comparing DigiSoup against baselines.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.compare import DEEPMIND_BASELINES


COLOURS = {
    "digisoup": "#2ecc71",
    "random": "#95a5a6",
    "A3C": "#e74c3c",
    "A3C_prosocial": "#e67e22",
    "OPRE": "#3498db",
    "OPRE_prosocial": "#9b59b6",
}


def plot_substrate_comparison(
    substrate_results: dict[str, dict[str, Any]],
    output_path: str = "results/comparison.png",
) -> str:
    """Bar chart comparing DigiSoup focal per-capita vs baselines per substrate.

    substrate_results maps substrate_name -> aggregated metrics dict with
    'focal_per_capita' key containing 'mean' and 'ci95'.
    """
    plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})

    substrates = sorted(substrate_results.keys())
    if not substrates:
        return output_path

    n = len(substrates)
    agent_keys = ["A3C", "A3C_prosocial", "OPRE", "OPRE_prosocial"]
    all_keys = agent_keys + ["DigiSoup"]
    n_bars = len(all_keys)
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n * 3), 6))
    x = np.arange(n)

    for i, key in enumerate(agent_keys):
        vals = [DEEPMIND_BASELINES.get(s, {}).get(key, 0.0) for s in substrates]
        ax.bar(x + i * bar_width, vals, bar_width,
               label=key.replace("_", " "),
               color=COLOURS.get(key, "#cccccc"), alpha=0.85)

    ds_vals = [substrate_results[s]["focal_per_capita"]["mean"] for s in substrates]
    ds_errs = [substrate_results[s]["focal_per_capita"]["ci95"] for s in substrates]
    ax.bar(x + len(agent_keys) * bar_width, ds_vals, bar_width,
           yerr=ds_errs, label="DigiSoup", color=COLOURS["digisoup"],
           edgecolor="black", linewidth=1.2, capsize=3)

    short_names = [s.split("__")[0].replace("_", " ").title() for s in substrates]
    ax.set_xticks(x + bar_width * n_bars / 2)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_ylabel("Focal Per-Capita Return")
    ax.set_title("DigiSoup vs DeepMind Baselines (Melting Pot)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
