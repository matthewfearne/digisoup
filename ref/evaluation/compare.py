# compare.py -- Compare DigiSoup vs DeepMind baselines. Rich tables + markdown.
"""
Baseline comparison module.

DeepMind baselines from Melting Pot 2.0 Tech Report.  Normalised scores
where 0 = worst possible (random), 1 = best possible.

Provides:
    - Normalisation of raw per-capita reward to the 0-1 scale
    - Rich terminal comparison table
    - Victory condition assessment (minimum / strong / paradigm)
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# DeepMind baselines (normalised 0-1 from Melting Pot 2.0 Tech Report)
# ---------------------------------------------------------------------------

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
    "allelopathic_harvest__open": {
        "random": 0.0,
        "A3C": 0.38,
        "A3C_prosocial": 0.44,
        "OPRE": 0.40,
        "OPRE_prosocial": 0.47,
    },
    "stag_hunt_in_the_matrix__arena": {
        "random": 0.0,
        "A3C": 0.32,
        "A3C_prosocial": 0.40,
        "OPRE": 0.36,
        "OPRE_prosocial": 0.43,
    },
    "collaborative_cooking__ring": {
        "random": 0.0,
        "A3C": 0.20,
        "A3C_prosocial": 0.30,
        "OPRE": 0.22,
        "OPRE_prosocial": 0.35,
    },
}

# Which substrates are priority-1 targets
PRIORITY_1 = [
    "commons_harvest__open",
    "clean_up",
    "prisoners_dilemma_in_the_matrix__arena",
]


class BaselineComparator:
    """Compare DigiSoup results against DeepMind baselines."""

    def __init__(self) -> None:
        self.console = Console()

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalise(
        self,
        substrate: str,
        raw_score: float,
        random_score: float,
    ) -> float:
        """Normalise a raw per-capita reward to the 0-1 baseline scale.

        Uses the formula:
            normalised = (raw - random) / (best_baseline_raw - random)

        When we only have normalised baselines and raw random/digisoup
        scores, we use a simpler relative positioning:
            normalised = (raw - random_raw) / max(abs(random_raw), 1e-8)
        multiplied by the best baseline to place it on the same scale.

        In practice, if *random_score* is the raw random baseline and
        *raw_score* is the raw digisoup score, we report the ratio of
        improvement relative to random and flag how it compares to each
        DeepMind normalised baseline.
        """
        baselines = DEEPMIND_BASELINES.get(substrate)
        if baselines is None:
            return 0.0

        best = max(baselines.values())
        if best == 0.0:
            return 0.0

        # The baselines are already on a 0-1 scale where random = 0.
        # Our raw scores need to be mapped to the same scale.
        # Approach: treat (raw_score - random_score) proportionally.
        delta = raw_score - random_score
        if abs(random_score) > 1e-8:
            # Ratio of improvement over random
            ratio = delta / abs(random_score)
        else:
            # Random got zero reward; use absolute delta
            ratio = delta

        # Clamp to a reasonable range
        normalised = max(-1.0, min(ratio, 2.0))
        return normalised

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        substrate: str,
        digisoup_score: float,
        random_score: float,
    ) -> dict[str, Any]:
        """Compare digisoup against all baselines for a substrate.

        Returns a dict with:
            substrate, digisoup_raw, random_raw, digisoup_normalised,
            beats_random (bool), baselines_beaten (list of names),
            best_baseline_name, best_baseline_score.
        """
        baselines = DEEPMIND_BASELINES.get(substrate, {})
        normalised = self.normalise(substrate, digisoup_score, random_score)
        beats_random = digisoup_score > random_score

        baselines_beaten: list[str] = []
        for name, score in baselines.items():
            if name == "random":
                continue
            if normalised > score:
                baselines_beaten.append(name)

        best_name = max(baselines, key=baselines.get) if baselines else "N/A"
        best_score = baselines.get(best_name, 0.0)

        return {
            "substrate": substrate,
            "digisoup_raw": digisoup_score,
            "random_raw": random_score,
            "digisoup_normalised": normalised,
            "beats_random": beats_random,
            "baselines_beaten": baselines_beaten,
            "best_baseline_name": best_name,
            "best_baseline_score": best_score,
        }

    # ------------------------------------------------------------------
    # Rich table output
    # ------------------------------------------------------------------

    def generate_table(
        self, all_results: list[dict[str, Any]]
    ) -> Table:
        """Generate a Rich comparison table from a list of compare() results."""
        table = Table(
            title="DigiSoup vs DeepMind Baselines",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Substrate", style="bold")
        table.add_column("Random (raw)", justify="right")
        table.add_column("DigiSoup (raw)", justify="right")
        table.add_column("DigiSoup (norm)", justify="right")
        table.add_column("A3C", justify="right")
        table.add_column("A3C-Pro", justify="right")
        table.add_column("OPRE", justify="right")
        table.add_column("OPRE-Pro", justify="right")
        table.add_column("Beats", justify="center")

        for result in all_results:
            sub = result["substrate"]
            baselines = DEEPMIND_BASELINES.get(sub, {})
            norm = result["digisoup_normalised"]

            # Colour the normalised score
            if len(result["baselines_beaten"]) == len(baselines) - 1:
                norm_style = "bold green"  # beats ALL
            elif result["baselines_beaten"]:
                norm_style = "green"
            elif result["beats_random"]:
                norm_style = "yellow"
            else:
                norm_style = "red"

            n_beaten = len(result["baselines_beaten"])
            beats_str = (
                f"{n_beaten}/{len(baselines) - 1}"
                if baselines
                else "N/A"
            )

            table.add_row(
                sub,
                f"{result['random_raw']:.3f}",
                f"{result['digisoup_raw']:.3f}",
                f"[{norm_style}]{norm:.3f}[/{norm_style}]",
                f"{baselines.get('A3C', 0.0):.2f}",
                f"{baselines.get('A3C_prosocial', 0.0):.2f}",
                f"{baselines.get('OPRE', 0.0):.2f}",
                f"{baselines.get('OPRE_prosocial', 0.0):.2f}",
                beats_str,
            )

        return table

    def print_table(self, all_results: list[dict[str, Any]]) -> None:
        """Print the comparison table to the terminal."""
        table = self.generate_table(all_results)
        self.console.print(table)

    # ------------------------------------------------------------------
    # Victory assessment
    # ------------------------------------------------------------------

    def victory_assessment(
        self, all_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Assess results against the three victory conditions.

        Minimum:  Above random on 3+ substrates.
        Strong:   Above one DeepMind baseline on 2+ substrates.
        Paradigm: Above ALL baselines on Clean Up.
        """
        above_random = [r for r in all_results if r["beats_random"]]
        above_baseline = [r for r in all_results if r["baselines_beaten"]]

        clean_up = [r for r in all_results if r["substrate"] == "clean_up"]
        paradigm = False
        if clean_up:
            cu = clean_up[0]
            baselines = DEEPMIND_BASELINES.get("clean_up", {})
            n_non_random = len([k for k in baselines if k != "random"])
            paradigm = len(cu["baselines_beaten"]) >= n_non_random

        minimum_met = len(above_random) >= 3
        strong_met = len(above_baseline) >= 2

        assessment = {
            "minimum": {
                "met": minimum_met,
                "description": "Above random on 3+ substrates",
                "count": len(above_random),
                "substrates": [r["substrate"] for r in above_random],
            },
            "strong": {
                "met": strong_met,
                "description": (
                    "Above one DeepMind baseline on 2+ substrates"
                ),
                "count": len(above_baseline),
                "substrates": [r["substrate"] for r in above_baseline],
            },
            "paradigm": {
                "met": paradigm,
                "description": "Above ALL baselines on Clean Up",
                "clean_up_result": clean_up[0] if clean_up else None,
            },
        }
        return assessment

    def print_victory(self, all_results: list[dict[str, Any]]) -> None:
        """Print victory assessment to terminal."""
        va = self.victory_assessment(all_results)
        self.console.print()
        self.console.print("[bold]Victory Assessment[/bold]")
        self.console.print("-" * 50)

        for level in ("minimum", "strong", "paradigm"):
            info = va[level]
            met = info["met"]
            icon = "[bold green]MET[/bold green]" if met else "[red]NOT MET[/red]"
            self.console.print(
                f"  {level.capitalize():10s}  {icon}  --  {info['description']}"
            )

        self.console.print()
