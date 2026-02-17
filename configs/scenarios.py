"""Target substrates and scenarios for DigiSoup evaluation.

All scenario metadata comes directly from Melting Pot's configs.
"""
from __future__ import annotations

from meltingpot import scenario as mp_scenario


# The 3 priority substrates and their official scenarios.
TARGET_SUBSTRATES = [
    "commons_harvest__open",
    "clean_up",
    "prisoners_dilemma_in_the_matrix__arena",
]


def get_scenarios_for_substrate(substrate_name: str) -> list[str]:
    """Return the list of official scenario names for a substrate."""
    return sorted(mp_scenario.SCENARIOS_BY_SUBSTRATE.get(substrate_name, []))


def get_all_target_scenarios() -> list[str]:
    """Return all scenario names across all target substrates."""
    scenarios = []
    for sub in TARGET_SUBSTRATES:
        scenarios.extend(get_scenarios_for_substrate(sub))
    return sorted(scenarios)


def get_scenario_info(scenario_name: str) -> dict:
    """Return metadata for a scenario: substrate, roles, focal/background counts."""
    cfg = mp_scenario.get_config(scenario_name)
    n_focal = sum(cfg.is_focal)
    n_background = len(cfg.is_focal) - n_focal
    return {
        "scenario": scenario_name,
        "substrate": cfg.substrate,
        "n_players": len(cfg.is_focal),
        "n_focal": n_focal,
        "n_background": n_background,
        "is_focal": cfg.is_focal,
        "roles": cfg.roles,
    }


def print_target_summary() -> None:
    """Print a summary of all target substrates and scenarios."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Target Scenarios", show_header=True,
                  header_style="bold cyan")
    table.add_column("Scenario", style="bold")
    table.add_column("Substrate")
    table.add_column("Players", justify="right")
    table.add_column("Focal", justify="right")
    table.add_column("Background", justify="right")

    for sc_name in get_all_target_scenarios():
        info = get_scenario_info(sc_name)
        table.add_row(
            sc_name,
            info["substrate"],
            str(info["n_players"]),
            str(info["n_focal"]),
            str(info["n_background"]),
        )

    console.print(table)


if __name__ == "__main__":
    print_target_summary()
