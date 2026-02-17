"""Smoke tests for the DigiSoup agent."""
import dm_env
import numpy as np

from agents.digisoup.policy import DigiSoupPolicy


def test_policy_interface():
    """Test that the policy implements the required interface."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    # Simulate a timestep with a random observation
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
