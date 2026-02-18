"""Tests for the DigiSoup agent."""
import dm_env
import numpy as np

from agents.digisoup.policy import DigiSoupPolicy
from agents.digisoup.perception import perceive, MAX_ENTROPY
from agents.digisoup.state import get_role


def test_policy_interface():
    """Test that the policy implements the required interface."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    obs = {"RGB": np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)}
    timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=0.0,
        discount=1.0,
        observation=obs,
    )

    action, new_state = policy.step(timestep, state)

    assert isinstance(action, int)
    assert 0 <= action < 8
    assert new_state.step_count == 1

    policy.close()


def test_deterministic_with_same_seed():
    """Same seed should produce same actions."""
    actions_a = []
    actions_b = []

    for actions_list, seed in [(actions_a, 99), (actions_b, 99)]:
        policy = DigiSoupPolicy(seed=seed)
        state = policy.initial_state()
        obs = {"RGB": np.zeros((88, 88, 3), dtype=np.uint8)}

        for _ in range(20):
            timestep = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=1.0,
                observation=obs,
            )
            action, state = policy.step(timestep, state)
            actions_list.append(action)
        policy.close()

    assert actions_a == actions_b


def test_multiple_steps():
    """Run 100 steps without error."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    for step in range(100):
        obs = {"RGB": np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)}
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=float(np.random.normal()),
            discount=1.0,
            observation=obs,
        )
        action, state = policy.step(timestep, state)
        assert isinstance(action, int)
        assert 0 <= action < 8

    assert state.step_count == 100
    policy.close()


def test_energy_depletes():
    """Energy should decrease over steps without successful interaction."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()
    initial_energy = state.energy

    # Run steps with blank observation (no agents, no resources, no change)
    obs = {"RGB": np.zeros((88, 88, 3), dtype=np.uint8)}
    for _ in range(50):
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )
        _, state = policy.step(timestep, state)

    assert state.energy < initial_energy
    policy.close()


def test_cooperation_tendency_bounded():
    """Cooperation tendency should stay in [0, 1]."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    for _ in range(200):
        obs = {"RGB": np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)}
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )
        _, state = policy.step(timestep, state)

    assert 0.0 <= state.cooperation_tendency <= 1.0
    policy.close()


def test_perception_entropy():
    """Perception should compute entropy from RGB observations."""
    # Uniform noise has high entropy
    noisy = np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)
    p_noisy = perceive(noisy)
    assert p_noisy.entropy > 0.0

    # Solid black has zero entropy
    black = np.zeros((88, 88, 3), dtype=np.uint8)
    p_black = perceive(black)
    assert p_black.entropy == 0.0

    # Noisy should have higher entropy than black
    assert p_noisy.entropy > p_black.entropy


def test_perception_resource_detection():
    """Green pixels should be detected as resources."""
    obs = np.zeros((88, 88, 3), dtype=np.uint8)
    # Paint a green patch in the top-right
    obs[10:30, 60:80, 1] = 200  # Green channel high
    p = perceive(obs)
    assert p.resources_nearby
    assert p.resource_density > 0.0


def test_perception_change_detection():
    """Change between frames should be detected."""
    obs1 = np.zeros((88, 88, 3), dtype=np.uint8)
    obs2 = np.zeros((88, 88, 3), dtype=np.uint8)
    obs2[20:60, 20:60] = 200  # Big bright area appears

    p = perceive(obs2, prev_obs=obs1)
    assert p.change > 0.0


def test_role_emergence():
    """Role should emerge from action history."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    # Run enough steps to accumulate action history
    for _ in range(50):
        obs = {"RGB": np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)}
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )
        _, state = policy.step(timestep, state)

    role = get_role(state)
    assert role in ("cooperator", "explorer", "scanner", "generalist")
    policy.close()
