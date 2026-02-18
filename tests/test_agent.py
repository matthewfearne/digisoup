"""Tests for the DigiSoup agent."""
import dm_env
import numpy as np

from agents.digisoup.policy import DigiSoupPolicy
from agents.digisoup.perception import perceive, MAX_ENTROPY
from agents.digisoup.action import (
    select_action, CONSERVATION_DENSITY, CONSERVATION_DEPLETION,
)
from agents.digisoup.state import (
    get_role, get_phase, initial_state, update_state, PHASE_LENGTH,
    MEMORY_REINFORCE, RECENCY_DECAY,
)


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


def test_fine_entropy_grid():
    """Perception should return a 4x4 entropy grid."""
    obs = np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)
    p = perceive(obs)
    assert p.entropy_grid.shape == (4, 4)
    assert p.entropy_grid.max() > 0.0  # noisy obs has nonzero entropy


def test_entropy_growth_gradient():
    """Growth gradient should point toward where entropy increased."""
    # Frame 1: blank
    obs1 = np.zeros((88, 88, 3), dtype=np.uint8)
    p1 = perceive(obs1)

    # Frame 2: noisy patch appears bottom-right
    obs2 = np.zeros((88, 88, 3), dtype=np.uint8)
    obs2[60:88, 60:88] = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
    p2 = perceive(obs2, prev_obs=obs1, prev_entropy_grid=p1.entropy_grid)

    # Growth gradient should point toward bottom-right (positive dy, positive dx)
    assert p2.growth_gradient[0] > 0.0 or p2.growth_gradient[1] > 0.0


def test_kl_anomaly_detection():
    """KL anomaly should detect unusual patches in uniform backgrounds."""
    # Mostly black image with a bright colored patch (simulating agent sprite)
    obs = np.zeros((88, 88, 3), dtype=np.uint8)
    obs[20:35, 20:35, 0] = 200  # bright red patch
    obs[20:35, 20:35, 2] = 150  # with some blue

    p = perceive(obs)
    assert p.anomaly_strength > 0.0
    # Anomaly direction should point toward the patch (top-left area)
    assert np.linalg.norm(p.anomaly_direction) > 0.0


def test_perception_grid_flows_through_policy():
    """Entropy grid should persist through policy for growth gradient."""
    policy = DigiSoupPolicy(seed=42)
    state = policy.initial_state()

    # First step: establish baseline grid
    obs1 = {"RGB": np.zeros((88, 88, 3), dtype=np.uint8)}
    timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID, reward=0.0, discount=1.0,
        observation=obs1,
    )
    _, state = policy.step(timestep, state)

    # State should now have a prev_entropy_grid
    assert state.prev_entropy_grid.shape == (4, 4)

    # Second step: different observation should produce growth signal
    obs2_arr = np.zeros((88, 88, 3), dtype=np.uint8)
    obs2_arr[44:88, 44:88] = np.random.randint(50, 200, (44, 44, 3), dtype=np.uint8)
    obs2 = {"RGB": obs2_arr}
    timestep2 = dm_env.TimeStep(
        step_type=dm_env.StepType.MID, reward=0.0, discount=1.0,
        observation=obs2,
    )
    _, state = policy.step(timestep2, state)

    # Grid should have updated
    assert state.prev_entropy_grid.max() > 0.0
    policy.close()


def test_growth_rate_signal():
    """Growth rate should be negative when entropy declines between frames."""
    # Frame 1: noisy (high entropy)
    obs1 = np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)
    p1 = perceive(obs1)

    # Frame 2: mostly black (low entropy) — environment has been depleted
    obs2 = np.zeros((88, 88, 3), dtype=np.uint8)
    obs2[0:10, 0:10] = 50  # small dim patch to avoid perfect zero
    p2 = perceive(obs2, prev_obs=obs1, prev_entropy_grid=p1.entropy_grid)

    assert p2.growth_rate < 0.0  # entropy declined = depletion

    # Frame 3: noisy again (entropy grew)
    obs3 = np.random.randint(0, 256, (88, 88, 3), dtype=np.uint8)
    p3 = perceive(obs3, prev_obs=obs2, prev_entropy_grid=p2.entropy_grid)

    assert p3.growth_rate > 0.0  # entropy grew = recovery


def test_conservation_backs_off_depleting_patch():
    """Agent should move AWAY from resources when patch is actively depleting."""
    from agents.digisoup.perception import Perception

    # Craft a perception with dense resources and negative growth rate
    perception = Perception(
        entropy=2.0,
        gradient=np.array([0.0, 1.0]),
        entropy_grid=np.ones((4, 4)),
        growth_gradient=np.zeros(2),
        anomaly_direction=np.zeros(2),
        anomaly_strength=0.0,
        agents_nearby=False,
        agent_direction=np.zeros(2),
        agent_density=0.0,
        resources_nearby=True,
        resource_direction=np.array([0.0, 1.0]),  # resources to the right
        resource_density=0.05,  # above CONSERVATION_DENSITY (0.02)
        growth_rate=-0.3,  # below CONSERVATION_DEPLETION (-0.1) = depleting
        change=0.1,
        change_direction=np.zeros(2),
    )

    state = initial_state()
    # Set energy above LOW_ENERGY_THRESHOLD so Rule 2 doesn't fire
    state = state._replace(energy=0.8)

    rng = np.random.default_rng(42)
    actions = [select_action(perception, state, 8, rng) for _ in range(20)]

    # Agent should NOT move toward resources (action RIGHT=4 would mean
    # chasing resources to the right). Most actions should be LEFT=3 or
    # BACKWARD=2 (moving away from resources at [0, 1]).
    from agents.digisoup.action import RIGHT
    right_count = sum(1 for a in actions if a == RIGHT)
    assert right_count < 5, f"Agent moved right {right_count}/20 times — should back off"
