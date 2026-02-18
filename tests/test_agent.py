"""Tests for the DigiSoup agent."""
import dm_env
import numpy as np

from agents.digisoup.policy import DigiSoupPolicy
from agents.digisoup.perception import perceive, MAX_ENTROPY
from agents.digisoup.state import (
    get_role, get_phase, initial_state, update_state, PHASE_LENGTH,
    MEMORY_REINFORCE, RECENCY_DECAY, get_interaction_success_rate,
)
from agents.digisoup.action import _adaptive_coop_threshold


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


def test_phase_cycling():
    """Phase should alternate between explore and exploit."""
    state = initial_state()

    # Step 0 = explore
    assert get_phase(state) == "explore"

    # Step PHASE_LENGTH-1 = still explore
    state = state._replace(step_count=PHASE_LENGTH - 1)
    assert get_phase(state) == "explore"

    # Step PHASE_LENGTH = exploit
    state = state._replace(step_count=PHASE_LENGTH)
    assert get_phase(state) == "exploit"

    # Step 2*PHASE_LENGTH - 1 = still exploit
    state = state._replace(step_count=2 * PHASE_LENGTH - 1)
    assert get_phase(state) == "exploit"

    # Step 2*PHASE_LENGTH = back to explore (new cycle)
    state = state._replace(step_count=2 * PHASE_LENGTH)
    assert get_phase(state) == "explore"


def test_phase_covers_full_episode():
    """Agent should cycle through multiple phases during a full run."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()
    phases_seen = set()

    for _ in range(200):
        phases_seen.add(get_phase(state))
        obs = {"RGB": np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)}
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )
        _, state = policy.step(timestep, state)

    assert "explore" in phases_seen
    assert "exploit" in phases_seen
    policy.close()


def test_resource_memory_reinforcement():
    """Seeing resources should build spatial memory."""
    state = initial_state()
    obs = np.zeros((88, 88, 3), dtype=np.uint8)
    resource_dir = np.array([0.0, 1.0])  # resources to the right

    # Update with resources visible
    state = update_state(
        state, obs, action=1,
        perception_entropy=1.0, perception_change=0.0,
        resources_nearby=True, resource_direction=resource_dir,
        resource_density=0.05,
    )

    # Memory should point toward resources
    assert state.resource_recency == 1.0
    assert state.resource_memory[1] > 0.0  # dx > 0 = rightward


def test_resource_memory_decays():
    """Memory should decay when no resources are seen."""
    state = initial_state()
    obs = np.zeros((88, 88, 3), dtype=np.uint8)

    # First: see resources to build memory
    state = update_state(
        state, obs, action=1,
        perception_entropy=1.0, perception_change=0.0,
        resources_nearby=True, resource_direction=np.array([0.0, 1.0]),
        resource_density=0.05,
    )
    initial_mem_strength = np.linalg.norm(state.resource_memory)
    initial_recency = state.resource_recency

    # Then: 30 steps with no resources
    for _ in range(30):
        state = update_state(
            state, obs, action=1,
            perception_entropy=1.0, perception_change=0.0,
            resources_nearby=False,
        )

    # Memory and recency should have decayed
    assert np.linalg.norm(state.resource_memory) < initial_mem_strength
    assert state.resource_recency < initial_recency


def test_memory_persists_through_policy():
    """Memory should be maintained through the policy step interface."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    # Feed observations with green resources for 10 steps
    obs_with_green = np.zeros((88, 88, 3), dtype=np.uint8)
    obs_with_green[10:30, 60:80, 1] = 200  # Green patch top-right

    for _ in range(10):
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation={"RGB": obs_with_green},
        )
        _, state = policy.step(timestep, state)

    # Memory should be non-zero (resources were detected)
    assert state.resource_recency > 0.0
    assert np.linalg.norm(state.resource_memory) > 0.0

    # Now feed blank observations — memory should decay but not vanish instantly
    blank_obs = {"RGB": np.zeros((88, 88, 3), dtype=np.uint8)}
    for _ in range(5):
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=blank_obs,
        )
        _, state = policy.step(timestep, state)

    # Should still have some memory left after only 5 blank steps
    assert np.linalg.norm(state.resource_memory) > 0.0
    policy.close()


def test_interaction_success_rate_insufficient_data():
    """Success rate should be None with too few interactions."""
    state = initial_state()
    assert get_interaction_success_rate(state) is None

    # Add 2 interactions (below threshold of 3)
    state = state._replace(interaction_outcomes=(0.5, 0.0))
    assert get_interaction_success_rate(state) is None


def test_interaction_success_rate_computes():
    """Success rate should reflect recent interaction outcomes."""
    state = initial_state()

    # 3 successes, 2 failures = 60% success
    state = state._replace(interaction_outcomes=(0.5, 0.4, 0.0, 0.6, 0.0))
    rate = get_interaction_success_rate(state)
    assert rate is not None
    assert abs(rate - 0.6) < 0.01

    # All successes
    state = state._replace(interaction_outcomes=(0.5, 0.4, 0.3))
    assert get_interaction_success_rate(state) == 1.0

    # All failures
    state = state._replace(interaction_outcomes=(0.0, 0.0, 0.0))
    assert get_interaction_success_rate(state) == 0.0


def test_adaptive_threshold_lowers_on_success():
    """High success rate should lower cooperation threshold."""
    state = initial_state()

    # No data: base threshold
    base_explore = _adaptive_coop_threshold(state, "explore")
    base_exploit = _adaptive_coop_threshold(state, "exploit")
    assert abs(base_explore - 0.7) < 0.01
    assert abs(base_exploit - 0.3) < 0.01

    # All successes: threshold should drop (more willing to cooperate)
    state = state._replace(interaction_outcomes=(0.5, 0.4, 0.6, 0.5, 0.3))
    adapted_explore = _adaptive_coop_threshold(state, "explore")
    adapted_exploit = _adaptive_coop_threshold(state, "exploit")
    assert adapted_explore < base_explore
    assert adapted_exploit < base_exploit


def test_adaptive_threshold_raises_on_failure():
    """Low success rate should raise cooperation threshold."""
    state = initial_state()
    base_explore = _adaptive_coop_threshold(state, "explore")

    # All failures: threshold should rise (more reluctant)
    state = state._replace(interaction_outcomes=(0.0, 0.0, 0.0, 0.0, 0.0))
    adapted = _adaptive_coop_threshold(state, "explore")
    assert adapted > base_explore
