# DigiSoup Changelog

## Baseline (v11) — the version that scored Commons 36.20, PD 11.83, Clean Up 5.04
- 9-layer entropy stack (obs, gradient, change, dS/dt, temp, d2S/dt2, growth, thermal, KL)
- Gradient cap 0.5 in all gradient functions (entropy.py)
- Boltzmann action selection: INTERACT, MOVE, SCAN
- Personality profiles: 4 cleaners (interact=1.15) + 3 harvesters (move/scan=1.05)
- Move energy: (grad_strength + ch_strength) / 2.0 — equal weight
- Movement direction: ch_grad if change detected, else grad — no blending
- Echo chase: toward reactive zones only (resonance > 1.0)
- Echo trend: scan boost only when declining
- Cooling pressure: move boost when dS/dt negative
- Thermal saturation: anti-pheromone flip + move/scan boost
- Langevin noise: 0.3 + ramp to 0.5 over 600 steps

## Gradient cap fix (this session, attempt 2)
- Restored gradient cap from 0.5 back to 1.0 in ALL gradient functions in entropy.py
- Affects: fine_entropy_gradient, spatial_change_gradient, entropy_growth_gradient,
  thermal_gradient, peripheral_entropy_gradient, kl_anomaly_gradient
- The 0.5 cap was halving gradient signal strength, making agents indecisive

## v12 FAILED — reverted (this session)
Attempted: comfort zones, habituation (80/20 change/static), cold_mix, redundancy
pressure, echo chase flip, echo trend move boost, novelty starvation, peripheral
sensing, 5+2 personality split.
Results: ALL substrates regressed. Commons 36.20→11.40, PD 11.83→6.45, Clean Up 5.04→3.23.
Cause: Too many interacting mechanisms destabilized the core entropy-gradient system.
Reverted to baseline.

## Files and what they do
- `src/agent/action.py` — Action selection (Boltzmann energies, gradient→movement)
- `src/agent/entropy.py` — All entropy computations (9 layers)
- `src/agent/core.py` — DigiSoupAgent class (state management, echo tracking)
- `src/evaluation/runner.py` — Formal evaluation (episodes, metrics, profiles)
- `watch.py` — Visual testing with pygame
