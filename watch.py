"""Watch DigiSoup v14 agents play Melting Pot scenarios live.

Renders WORLD.RGB (full map) alongside focal agent POV tiles in a pygame window.
Uses actual scenarios with DeepMind background bots — same as evaluation.

Usage:
    source .venv/bin/activate
    python watch.py                         # Default: clean_up_0
    python watch.py clean_up_0              # Specific scenario
    python watch.py clean_up_2 --slow       # Half speed
    python watch.py clean_up_0 --fast       # No delay

Controls:
    SPACE  — pause/unpause
    +/-    — speed up / slow down
    Q/ESC  — quit
"""
from __future__ import annotations

import os
import sys
import time

# Suppress TF/CUDA/absl noise, disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import dm_env
import pygame

from meltingpot import scenario as mp_scenario

from agents.digisoup.policy import DigiSoupPolicy
from agents.digisoup.state import DigiSoupState


# ---------------------------------------------------------------------------
# Scenario aliases for convenience
# ---------------------------------------------------------------------------

ALIASES = {
    "cu0": "clean_up_0", "cu1": "clean_up_1", "cu2": "clean_up_2",
    "cu3": "clean_up_3", "cu4": "clean_up_4", "cu5": "clean_up_5",
    "cu6": "clean_up_6", "cu7": "clean_up_7", "cu8": "clean_up_8",
    "ch0": "commons_harvest__open_0", "ch1": "commons_harvest__open_1",
    "pd0": "prisoners_dilemma_in_the_matrix__arena_0",
    "pd1": "prisoners_dilemma_in_the_matrix__arena_1",
    "pd2": "prisoners_dilemma_in_the_matrix__arena_2",
    "pd3": "prisoners_dilemma_in_the_matrix__arena_3",
    "pd4": "prisoners_dilemma_in_the_matrix__arena_4",
    "pd5": "prisoners_dilemma_in_the_matrix__arena_5",
}

# Substrate -> all scenario names
SUBSTRATE_SCENARIOS = {
    "clean_up": [f"clean_up_{i}" for i in range(9)],
    "commons_harvest": [f"commons_harvest__open_{i}" for i in range(2)],
    "prisoners_dilemma": [f"prisoners_dilemma_in_the_matrix__arena_{i}" for i in range(6)],
}

ACTION_NAMES = ["noop", "fwd", "back", "left", "right", "trnL", "trnR", "zap"]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)
STAT_COLOR = (180, 220, 180)
HEADER_COLOR = (100, 200, 255)
BORDER_COLOR = (80, 80, 80)


def run_scenario_visual(
    scenario_name: str,
    delay_ms: int,
    screen: pygame.Surface | None = None,
    scenario_idx: int = 0,
    total_scenarios: int = 1,
) -> tuple[pygame.Surface | None, bool]:
    """Run one scenario visually. Returns (screen, user_quit)."""

    print(f"\n  [{scenario_idx+1}/{total_scenarios}] Loading {scenario_name}...", end=" ", flush=True)

    env = mp_scenario.build(scenario_name)
    timestep = env.reset()
    n_focal = len(timestep.observation)

    # Access underlying substrate for WORLD.RGB
    substrate = env._substrate
    sub_obs = substrate.observation()
    world_rgb = sub_obs[0]["WORLD.RGB"]
    world_h, world_w = world_rgb.shape[:2]

    # Detect action space (Clean Up has 9: action 8 = FIRE_CLEAN)
    action_spec = env.action_spec()
    n_actions = action_spec[0].num_values if action_spec else 8
    print(f"OK ({n_focal} focal, {n_actions} actions, world {world_w}x{world_h})")

    # Create DigiSoup policies
    policies: list[DigiSoupPolicy] = []
    states: list[DigiSoupState] = []
    for i in range(n_focal):
        p = DigiSoupPolicy(seed=42 + i, n_actions=n_actions)
        policies.append(p)
        states.append(p.initial_state())

    # Layout: world view on left, focal POVs stacked on right, stats below
    SCALE = max(1, min(4, 600 // world_h))
    world_display_w = world_w * SCALE
    world_display_h = world_h * SCALE

    pov_size = 88
    pov_scale = max(1, min(3, world_display_h // (max(n_focal, 1) * pov_size + max(n_focal, 1) * 4)))
    pov_display = pov_size * pov_scale

    right_panel_w = pov_display + 20
    stats_h = 140
    win_w = world_display_w + right_panel_w + 20
    win_h = world_display_h + stats_h + 20

    # Init or resize pygame window
    if screen is None:
        pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"DigiSoup v14 — {scenario_name} [{scenario_idx+1}/{total_scenarios}]"
    )
    font = pygame.font.SysFont("monospace", 14)
    font_big = pygame.font.SysFont("monospace", 16, bold=True)

    # State
    user_quit = False
    paused = False
    step = 0
    total_focal_reward = 0.0
    per_player_rewards = [0.0] * n_focal
    t0 = time.time()
    action_counts = [0] * 8

    try:
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    user_quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        user_quit = True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        delay_ms = max(0, delay_ms - 10)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        delay_ms = min(200, delay_ms + 10)
                    elif event.key == pygame.K_n:
                        # Skip to next scenario
                        break

            if user_quit:
                break

            if paused:
                screen.fill(BG_COLOR)
                pause_surf = font_big.render("PAUSED (SPACE to resume)", True, (255, 255, 100))
                screen.blit(pause_surf, (win_w // 2 - pause_surf.get_width() // 2, win_h // 2))
                pygame.display.flip()
                pygame.time.delay(100)
                continue

            if timestep.step_type == dm_env.StepType.LAST:
                break

            # --- Agent step ---
            actions = []
            new_states = []

            for i in range(n_focal):
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
                if 0 <= action < 8:
                    action_counts[action] += 1

            states = new_states
            timestep = env.step(actions)

            # Record rewards
            for i in range(n_focal):
                r = float(timestep.reward[i]) if timestep.reward is not None else 0.0
                per_player_rewards[i] += r
                total_focal_reward += r

            step += 1

            # --- Render ---
            screen.fill(BG_COLOR)

            # Get WORLD.RGB from substrate
            sub_obs = substrate.observation()
            world_rgb = sub_obs[0]["WORLD.RGB"]

            # Draw world view (left)
            world_surf = pygame.surfarray.make_surface(
                np.transpose(world_rgb, (1, 0, 2))
            )
            world_surf = pygame.transform.scale(world_surf, (world_display_w, world_display_h))
            screen.blit(world_surf, (10, 10))

            # Draw focal POVs (right panel)
            pov_x = world_display_w + 20
            for i in range(n_focal):
                obs_dict = timestep.observation[i]
                if hasattr(obs_dict, "get"):
                    rgb = obs_dict.get("RGB", np.zeros((88, 88, 3), dtype=np.uint8))
                else:
                    rgb = np.asarray(obs_dict, dtype=np.uint8)
                rgb = np.asarray(rgb, dtype=np.uint8)

                pov_y = 10 + i * (pov_display + 24)

                # Label
                label = font.render(f"Focal {i}", True, HEADER_COLOR)
                screen.blit(label, (pov_x, pov_y))

                # POV image
                pov_surf = pygame.surfarray.make_surface(
                    np.transpose(rgb, (1, 0, 2))
                )
                pov_surf = pygame.transform.scale(pov_surf, (pov_display, pov_display))
                screen.blit(pov_surf, (pov_x, pov_y + 16))

                # Per-player reward
                r_text = font.render(f"R:{per_player_rewards[i]:.1f}", True, STAT_COLOR)
                screen.blit(r_text, (pov_x, pov_y + 16 + pov_display + 2))

            # --- Stats panel (bottom) ---
            stats_y = world_display_h + 20
            elapsed = time.time() - t0
            fps = step / elapsed if elapsed > 0 else 0
            per_cap = total_focal_reward / n_focal if n_focal > 0 else 0

            lines = [
                f"Scenario: {scenario_name}  [{scenario_idx+1}/{total_scenarios}]",
                f"Step: {step:5d}  |  Total R: {total_focal_reward:8.1f}  |  "
                f"Per-cap: {per_cap:7.1f}  |  FPS: {fps:.0f}  |  Delay: {delay_ms}ms",
                f"Actions: " + "  ".join(
                    f"{ACTION_NAMES[a]}={action_counts[a]}"
                    for a in range(8)
                ),
            ]

            # Per-player line
            player_strs = []
            for i in range(n_focal):
                phase = "explore" if (states[i].step_count % 100) < 50 else "exploit"
                player_strs.append(
                    f"F{i}: E={states[i].energy:.2f} C={states[i].cooperation_tendency:.2f} "
                    f"[{phase}]"
                )
            lines.append("  ".join(player_strs))

            for j, line in enumerate(lines):
                surf = font.render(line, True, TEXT_COLOR)
                screen.blit(surf, (10, stats_y + j * 18))

            pygame.display.flip()

            if delay_ms > 0:
                pygame.time.delay(delay_ms)

    except KeyboardInterrupt:
        user_quit = True

    # Final stats
    elapsed = time.time() - t0
    per_cap = total_focal_reward / n_focal if n_focal > 0 else 0
    print(f"  {scenario_name}: {step} steps, per-cap={per_cap:.1f}, "
          f"{elapsed:.0f}s")
    for i in range(n_focal):
        print(f"    Focal {i}: {per_player_rewards[i]:.1f}")

    env.close()
    for p in policies:
        p.close()

    return screen, user_quit


def main() -> None:
    # Parse args
    scenario_arg = "clean_up"
    delay_ms = 30  # default ~33 FPS

    for arg in sys.argv[1:]:
        if arg == "--slow":
            delay_ms = 60
        elif arg == "--fast":
            delay_ms = 0
        elif arg.startswith("--"):
            pass
        else:
            scenario_arg = arg

    # Resolve to scenario list
    resolved = ALIASES.get(scenario_arg, scenario_arg)
    if resolved in SUBSTRATE_SCENARIOS:
        scenarios = SUBSTRATE_SCENARIOS[resolved]
    elif scenario_arg in SUBSTRATE_SCENARIOS:
        scenarios = SUBSTRATE_SCENARIOS[scenario_arg]
    else:
        scenarios = [resolved]

    print(f"\n  DigiSoup v14 'Hive Mind' — Live Viewer")
    print(f"  Scenarios: {len(scenarios)} ({scenarios[0]} ... {scenarios[-1]})" if len(scenarios) > 1 else f"  Scenario: {scenarios[0]}")
    print(f"  Controls: SPACE=pause  +/-=speed  N=skip  Q/ESC=quit")

    screen = None
    for idx, sc in enumerate(scenarios):
        screen, user_quit = run_scenario_visual(
            sc, delay_ms, screen,
            scenario_idx=idx, total_scenarios=len(scenarios),
        )
        if user_quit:
            break

    print(f"\n  All done.")
    pygame.quit()


if __name__ == "__main__":
    main()
