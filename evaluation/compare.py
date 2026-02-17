"""Baseline comparison for DigiSoup vs DeepMind results.

DeepMind baselines from Melting Pot 2.0 Tech Report (normalised 0-1).
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


DEEPMIND_BASELINES: dict[str, dict[str, float]] = {
    "commons_harvest__open": {
        "random": 0.0,
        "A3C": 0.45,
        "A3C_prosocial": 0.52,
        "OPRE": 0.48,
        "OPRE_prosocial": 0.55,
    },
    "clean_up": {
        "random": 0.0,
        "A3C": 0.15,
        "A3C_prosocial": 0.35,
        "OPRE": 0.18,
        "OPRE_prosocial": 0.40,
    },
    "prisoners_dilemma_in_the_matrix__arena": {
        "random": 0.0,
        "A3C": 0.30,
        "A3C_prosocial": 0.42,
        "OPRE": 0.35,
        "OPRE_prosocial": 0.45,
    },
}


def print_results_table(
    all_results: list[dict[str, Any]],
) -> None:
    """Print a Rich table of results across substrates/scenarios."""
    console = Console()
    table = Table(
        title="DigiSoup Evaluation Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Substrate / Scenario", style="bold")
    table.add_column("Episodes", justify="right")
    table.add_column("Focal Per-Capita", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for result in all_results:
        name = result.get("scenario", result.get("substrate", "?"))
        pc = result.get("focal_per_capita", {})
        table.add_row(
            name,
            str(result.get("n_episodes", "?")),
            f"{pc.get('mean', 0.0):.3f}",
            f"{pc.get('std', 0.0):.3f}",
            f"+/- {pc.get('ci95', 0.0):.3f}",
            f"{pc.get('min', 0.0):.3f}",
            f"{pc.get('max', 0.0):.3f}",
        )

    console.print()
    console.print(table)
    console.print()


def print_comparison_table(
    digisoup_results: list[dict[str, Any]],
    random_results: list[dict[str, Any]],
) -> None:
    """Print comparison of DigiSoup vs random vs DeepMind baselines."""
    console = Console()
    table = Table(
        title="DigiSoup vs Baselines",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Substrate", style="bold")
    table.add_column("Random", justify="right")
    table.add_column("DigiSoup", justify="right")
    table.add_column("Improvement", justify="right")
    table.add_column("A3C", justify="right")
    table.add_column("OPRE-Pro", justify="right")

    # Group results by substrate
    ds_by_sub: dict[str, float] = {}
    rn_by_sub: dict[str, float] = {}

    for r in digisoup_results:
        sub = r.get("substrate", "?")
        pc = r.get("focal_per_capita", {}).get("mean", 0.0)
        ds_by_sub[sub] = pc

    for r in random_results:
        sub = r.get("substrate", "?")
        pc = r.get("focal_per_capita", {}).get("mean", 0.0)
        rn_by_sub[sub] = pc

    for sub in sorted(set(list(ds_by_sub.keys()) + list(rn_by_sub.keys()))):
        ds = ds_by_sub.get(sub, 0.0)
        rn = rn_by_sub.get(sub, 0.0)
        improvement = ds - rn
        baselines = DEEPMIND_BASELINES.get(sub, {})

        if improvement > 0:
            imp_style = "green"
        elif improvement < 0:
            imp_style = "red"
        else:
            imp_style = "yellow"

        table.add_row(
            sub,
            f"{rn:.3f}",
            f"{ds:.3f}",
            f"[{imp_style}]{improvement:+.3f}[/{imp_style}]",
            f"{baselines.get('A3C', 0.0):.2f}",
            f"{baselines.get('OPRE_prosocial', 0.0):.2f}",
        )

    console.print()
    console.print(table)
    console.print()
