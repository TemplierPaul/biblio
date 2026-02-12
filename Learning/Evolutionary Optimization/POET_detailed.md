# POET: Detailed Implementation Guide

## Algorithm Overview

POET (Paired Open-Ended Trailblazer) consists of three interleaved processes:
1. **Environment Mutation**: Generate new challenges via novelty-driven reproduction
2. **Agent Optimization**: Improve agents within their paired environments using ES
3. **Transfer Mechanism**: Attempt agent transfers between environments

All operations are highly parallelizable across multiple CPU cores.

---

## Main Algorithm

### POET Main Loop

```python
def POET(E_init, θ_init, α, σ, T, N_mutate, N_transfer):
    """
    Main POET algorithm loop.

    Args:
        E_init: Initial environment (flat ground)
        θ_init: Initial agent parameters
        α: Learning rate for ES (0.01 → 0.001 with decay)
        σ: Noise standard deviation for ES (0.1 → 0.01 with decay)
        T: Total iterations (25,200 in experiments)
        N_mutate: Mutation interval
        N_transfer: Transfer interval

    Returns:
        EA_list: Final list of (environment, agent) pairs
    """
    EA_list = [(E_init, θ_init)]

    for t in range(T):
        # Environment mutation phase
        if t > 0 and t % N_mutate == 0:
            EA_list = mutate_envs(EA_list, α, σ)

        # Agent optimization phase
        M = len(EA_list)
        for m in range(M):
            E_m, θ_m = EA_list[m]
            θ_update = ES_step(θ_m, E_m, α, σ)
            θ_m_new = θ_m + θ_update
            EA_list[m] = (E_m, θ_m_new)

        # Transfer phase
        for m in range(M):
            if M > 1 and t % N_transfer == 0:
                E_m, θ_m_new = EA_list[m]
                # Gather all other agents
                other_agents = [EA_list[i][1] for i in range(M) if i != m]
                # Attempt transfer
                θ_best = evaluate_candidates(other_agents, E_m, α, σ)
                # Accept if better
                if E_m(θ_best) > E_m(θ_m_new):
                    EA_list[m] = (E_m, θ_best)

        # Decay learning rate and noise
        α *= 0.9999
        σ *= 0.999

    return EA_list
```

**Key Parameters**:
- **T = 25,200**: Total iterations (~10 days with 256 cores)
- **N_mutate**: Environment mutation interval (check supplemental for exact value)
- **N_transfer**: Transfer attempt interval (periodic throughout run)
- **Population size**: 20 active environments maximum
- **ES samples**: 512 perturbations per ES step

---

## Environment Mutation

### Mutation Process

```python
def mutate_envs(EA_list, α, σ, max_children=512, max_admitted=20, capacity=20):
    """
    Reproduce environments and admit novelty-filtered children.

    Args:
        EA_list: Current list of (environment, agent) pairs
        α, σ: ES parameters
        max_children: Maximum children per reproduction cycle
        max_admitted: Maximum new environments to admit
        capacity: Maximum active population size

    Returns:
        Updated EA_list with new environments
    """
    # Step 1: Identify eligible parents (reward >= 200)
    parent_list = []
    for E, θ in EA_list:
        if E(θ) >= 200:  # Reproduction threshold
            parent_list.append((E, θ))

    # Step 2: Generate child environments via mutation
    child_list = []
    for E_parent, θ_parent in parent_list:
        for _ in range(max_children // len(parent_list)):
            E_child = mutate_environment(E_parent)
            θ_child = θ_parent.copy()  # Child inherits parent's agent
            child_list.append((E_child, θ_child))

    # Step 3: Filter by Minimal Criterion (50 <= reward <= 300)
    child_list = [
        (E, θ) for E, θ in child_list
        if 50 <= E(θ) <= 300
    ]

    # Step 4: Rank by novelty
    child_list = rank_by_novelty(child_list, EA_list)

    # Step 5: Attempt transfer and admit best candidates
    admitted = 0
    for E_child, θ_child in child_list:
        # Gather all current agents
        all_agents = [θ for _, θ in EA_list]
        # Try transfer from existing agents to child
        θ_best = evaluate_candidates(all_agents, E_child, α, σ)

        # Check Minimal Criterion with best agent
        if 50 <= E_child(θ_best) <= 300:
            EA_list.append((E_child, θ_best))
            admitted += 1
            if admitted >= max_admitted:
                break

    # Step 6: Enforce capacity (remove oldest if needed)
    if len(EA_list) > capacity:
        EA_list = EA_list[-capacity:]  # Keep most recent

    return EA_list
```

### Environment Encoding and Mutation

**Environment Parameters** (2D Bipedal Walker):

```python
class BipedalWalkerEnvironment:
    """
    Environment encoding for 2D bipedal walker obstacle course.
    """
    def __init__(self, seed=None):
        self.seed = seed or random.randint(0, 2**31-1)

        # Obstacle parameters (interval-based)
        self.stump_height = (0.0, 0.4)      # Min, max range
        self.gap_width = (0.0, 0.8)
        self.step_height = (0.0, 0.4)
        self.step_number = 1
        self.roughness = 0.3  # uniform(0, 0.6)

    def mutate(self):
        """
        Create mutated child environment.
        """
        child = copy.deepcopy(self)
        child.seed = random.randint(0, 2**31-1)

        # Mutate each parameter with probability
        if random.random() < 0.5:
            # Add or subtract mutation step
            delta = random.choice([-0.2, 0.2])
            child.stump_height = (
                clip(self.stump_height[0] + delta, 0.0, 5.0),
                clip(self.stump_height[1] + delta, 0.0, 5.0)
            )

        if random.random() < 0.5:
            delta = random.choice([-0.4, 0.4])
            child.gap_width = (
                clip(self.gap_width[0] + delta, 0.0, 10.0),
                clip(self.gap_width[1] + delta, 0.0, 10.0)
            )

        if random.random() < 0.5:
            delta = random.choice([-0.2, 0.2])
            child.step_height = (
                clip(self.step_height[0] + delta, 0.0, 5.0),
                clip(self.step_height[1] + delta, 0.0, 5.0)
            )

        if random.random() < 0.5:
            delta = random.choice([-1, 1])
            child.step_number = clip(self.step_number + delta, 1, 9)

        if random.random() < 0.5:
            child.roughness = random.uniform(0, 10.0)

        return child

    def encode(self):
        """
        Return encoding vector for novelty calculation.
        """
        return np.array([
            self.stump_height[0], self.stump_height[1],
            self.gap_width[0], self.gap_width[1],
            self.step_height[0], self.step_height[1],
            self.step_number,
            self.roughness
        ])
```

**Mutation Rules**:
- Each parameter mutates independently with 50% probability
- Mutation step sizes calibrated to parameter ranges
- Parameters clipped to valid ranges [initial, maximum]
- New seed ensures different environment instances

---

## Novelty Calculation

```python
def rank_by_novelty(child_list, EA_list, k=5):
    """
    Rank child environments by novelty score (descending).

    Args:
        child_list: List of (E_child, θ_child) pairs
        EA_list: Current population + archive (all historical environments)
        k: Number of nearest neighbors (5 in experiments)

    Returns:
        child_list sorted by novelty (most novel first)
    """
    # Get encodings of all current + archived environments
    archive_encodings = [E.encode() for E, _ in EA_list]

    # Calculate novelty for each child
    novelties = []
    for E_child, θ_child in child_list:
        e_child = E_child.encode()

        # Find k-nearest neighbors
        distances = [np.linalg.norm(e_child - e_archive)
                     for e_archive in archive_encodings]
        distances.sort()
        knn_distances = distances[:k]

        # Novelty = average distance to k-NN
        novelty = np.mean(knn_distances)
        novelties.append(novelty)

    # Sort by novelty (descending)
    sorted_indices = np.argsort(novelties)[::-1]
    child_list = [child_list[i] for i in sorted_indices]

    return child_list
```

**Key Details**:
- **Distance metric**: L2 norm (Euclidean distance)
- **k = 5**: Number of nearest neighbors
- **Archive**: Includes all current environments AND historical environments (not pruned)
- **Purpose**: Encourages divergence into different problem types

---

## Agent Optimization (Evolution Strategies)

### ES Step Implementation

```python
def ES_step(θ, environment, α, σ, population_size=512):
    """
    Single Evolution Strategies optimization step.

    Args:
        θ: Current agent parameters (neural network weights)
        environment: Function E(·) mapping parameters to reward
        α: Learning rate (decayed from 0.01 to 0.001)
        σ: Noise standard deviation (decayed from 0.1 to 0.01)
        population_size: Number of perturbations to sample (512)

    Returns:
        Δθ: Parameter update
    """
    # Initialize gradient accumulator
    gradients = np.zeros_like(θ)

    # Sample perturbations and evaluate
    for i in range(population_size):
        # Sample Gaussian noise
        ε = np.random.normal(0, σ, size=θ.shape)

        # Evaluate perturbed parameters
        reward_plus = environment(θ + ε)
        reward_minus = environment(θ - ε)

        # Accumulate gradient estimate
        gradients += (reward_plus - reward_minus) * ε

    # Average over population
    gradients /= population_size

    # Apply Adam optimizer
    Δθ = adam_update(gradients, α)

    return Δθ
```

**Neural Network Controller**:

```python
class AgentNetwork:
    """
    3-layer fully connected network for bipedal walker.
    """
    def __init__(self):
        # Architecture: 24 → 40 → 40 → 4
        self.W1 = np.random.randn(24, 40) * 0.1
        self.b1 = np.zeros(40)
        self.W2 = np.random.randn(40, 40) * 0.1
        self.b2 = np.zeros(40)
        self.W3 = np.random.randn(40, 4) * 0.1
        self.b3 = np.zeros(4)

    def forward(self, x):
        """
        Forward pass with tanh activations.

        Args:
            x: Input state (24-dim: 10 LIDAR + 14 internal state)

        Returns:
            actions: 4-dim output (2 hips + 2 knees)
        """
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        actions = np.tanh(h2 @ self.W3 + self.b3)  # Bounded [-1, 1]
        return actions

    def get_parameters(self):
        """Flatten all parameters into single vector."""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_parameters(self, θ):
        """Set parameters from flattened vector."""
        idx = 0
        self.W1 = θ[idx:idx+24*40].reshape(24, 40); idx += 24*40
        self.b1 = θ[idx:idx+40]; idx += 40
        self.W2 = θ[idx:idx+40*40].reshape(40, 40); idx += 40*40
        self.b2 = θ[idx:idx+40]; idx += 40
        self.W3 = θ[idx:idx+40*4].reshape(40, 4); idx += 40*4
        self.b3 = θ[idx:idx+4]
```

**Key Details**:
- **Input**: 24 dimensions (10 LIDAR + 14 internal state)
- **Hidden layers**: 40 neurons each with tanh activation
- **Output**: 4 dimensions (hip and knee torques) bounded [-1, 1]
- **Total parameters**: ~4,000 weights and biases

---

## Transfer Mechanism

### Candidate Evaluation

```python
def evaluate_candidates(candidate_agents, target_env, α, σ):
    """
    Evaluate all candidate agents in target environment.
    Returns best agent.

    Args:
        candidate_agents: List of agent parameters from other environments
        target_env: Target environment E(·)
        α, σ: ES parameters for proposal transfers

    Returns:
        θ_best: Best agent for target environment
    """
    candidates = []

    # Direct transfers: add original agents
    for θ in candidate_agents:
        candidates.append(θ)

    # Proposal transfers: add one-step ES improvements
    for θ in candidate_agents:
        Δθ = ES_step(θ, target_env, α, σ)
        candidates.append(θ + Δθ)

    # Evaluate all and return best
    rewards = [target_env(θ) for θ in candidates]
    best_idx = np.argmax(rewards)
    return candidates[best_idx]
```

**Transfer Types**:

1. **Direct Transfer**:
   - Agent θ from environment A transferred directly to environment B
   - No additional optimization
   - Fast, works when skills generalize immediately

2. **Proposal Transfer**:
   - Agent θ takes one ES step in target environment before evaluation
   - Allows quick adaptation to new environment
   - More robust but computationally more expensive

**Transfer Statistics** (from experiments):
- 18,000-19,000 transfer attempts per 25,200 iteration run
- ~50% success rate (transferred agent outperforms current champion)
- Critical for solving extremely challenging environments

---

## Reward Function

```python
def bipedal_walker_reward(state_trajectory, action_trajectory):
    """
    Reward function for 2D bipedal walker.

    Args:
        state_trajectory: Sequence of states over episode
        action_trajectory: Sequence of actions over episode

    Returns:
        total_reward: Scalar reward for episode
    """
    total_reward = 0

    for state, action in zip(state_trajectory, action_trajectory):
        x_pos, hull_angle = state['x'], state['hull_angle']
        torques = action

        # Encourage forward progress
        total_reward += 130 * (x_pos - prev_x_pos)

        # Penalize tilting
        total_reward -= 5 * abs(hull_angle - prev_hull_angle)

        # Penalize high torques (energy efficiency)
        total_reward -= 0.00035 * np.sum(torques**2)

        # Penalty for falling
        if is_fallen(state):
            total_reward -= 100
            break

    return total_reward
```

**Success Threshold**: Reward ≥ 230 (ensures efficient forward locomotion)

**Episode Limit**: 2,000 time steps maximum

---

## Hyperparameters Summary

### Main Loop
| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 25,200 | Total iterations |
| N_mutate | Variable | Mutation interval |
| N_transfer | Variable | Transfer interval |
| Population size | 20 | Max active environments |

### Evolution Strategies
| Parameter | Initial | Final | Decay |
|-----------|---------|-------|-------|
| Learning rate α | 0.01 | 0.001 | 0.9999 per step |
| Noise σ | 0.1 | 0.01 | 0.999 per step |
| Population size | 512 | 512 | N/A |

### Environment Mutation
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_children | 512 | Children per reproduction cycle |
| max_admitted | 20 | Max new environments per cycle |
| Reproduction threshold | 200 | Min reward to reproduce |
| Minimal Criterion | [50, 300] | Admitted reward range |
| Novelty k | 5 | Nearest neighbors for novelty |

### Environment Parameters
| Obstacle | Initial | Step | Maximum |
|----------|---------|------|---------|
| Stump height | (0.0, 0.4) | 0.2 | (5.0, 5.0) |
| Gap width | (0.0, 0.8) | 0.4 | (10.0, 10.0) |
| Step height | (0.0, 0.4) | 0.2 | (5.0, 5.0) |
| Step number | 1 | 1 | 9 |
| Roughness | uniform(0, 0.6) | resample | 10.0 |

### Computational Resources
- **CPU cores**: 256 (highly parallelizable)
- **Runtime**: ~10 days per 25,200 iteration run
- **Parallelization**: Ipyparallel framework

---

## Implementation Tips

### Parallelization Strategy

```python
from ipyparallel import Client

# Initialize parallel workers
rc = Client()
dview = rc[:]

# Parallelize ES sampling
def parallel_ES_step(θ, environment, α, σ, population_size=512):
    """
    Parallelized Evolution Strategies step.
    """
    # Distribute perturbation evaluations across workers
    tasks = []
    for i in range(population_size):
        ε = np.random.normal(0, σ, size=θ.shape)
        tasks.append((θ, ε, environment))

    # Evaluate in parallel
    results = dview.map_sync(evaluate_perturbation, tasks)

    # Aggregate gradients
    gradients = np.zeros_like(θ)
    for reward_plus, reward_minus, ε in results:
        gradients += (reward_plus - reward_minus) * ε

    gradients /= population_size
    return adam_update(gradients, α)

def evaluate_perturbation(task):
    """Worker function for evaluating single perturbation."""
    θ, ε, environment = task
    reward_plus = environment(θ + ε)
    reward_minus = environment(θ - ε)
    return reward_plus, reward_minus, ε
```

### Pluggable Optimizer Design

```python
class POETOptimizer:
    """
    Abstract optimizer interface for POET.
    """
    def step(self, θ, environment):
        """
        Single optimization step.

        Args:
            θ: Current parameters
            environment: Evaluation function

        Returns:
            Δθ: Parameter update
        """
        raise NotImplementedError

class ESOptimizer(POETOptimizer):
    """Evolution Strategies optimizer."""
    def step(self, θ, environment):
        return ES_step(θ, environment, self.α, self.σ)

class PPOOptimizer(POETOptimizer):
    """Proximal Policy Optimization."""
    def step(self, θ, environment):
        # Collect trajectories, compute advantages, update policy
        return ppo_update(θ, environment, self.hyperparams)

# Use in POET
optimizer = ESOptimizer(α=0.01, σ=0.1)
# Or: optimizer = PPOOptimizer(...)
```

**Advantage**: Any RL algorithm can replace ES (TRPO, PPO, genetic algorithms)

---

## Experimental Insights

### Why Direct Optimization Fails

**Problem**: ES on POET-generated environments converges to degenerate behaviors.

**Example**: Wide gap environment
- ES learns to freeze immediately (score: 17.9)
- Reasoning: Moving risks -100 fall penalty, freezing gives 0
- POET agents: Learn jumping/running skills (score: 230+)

**Root Cause**: Without stepping stones, agents cannot discover complex skills through random exploration.

### Why Single-Path Curriculum Fails

**Control Experiment**: Progressive curriculum from flat → POET target

**Results**:
- **Challenging** (distance ≈0.081): Control sometimes matches POET
- **Very challenging** (distance ≈0.214): Control significantly worse
- **Extremely challenging** (distance ≈0.375): Control completely fails

**Conclusion**: Multi-path exploration essential. Optimal stepping stones cannot be predicted; parallel paths increase discovery probability.

### Why Transfer Matters

**Ablation**: POET without transfer mechanism

**Results**:
- Zero extremely challenging environments solved
- Significantly worse environment coverage (p < 2.2e-16)
- Agents trapped in local optima within single environment

**With Transfer**:
- ~50% transfer success rate
- Child solutions rescue parents from local optima
- Enables previously impossible problem solving

---

## Extensions and Future Work

### Enhanced POET

**Improvements**:
1. **PATA-EC**: Performance of All Transferred Agents Environment Characterization
   - More robust environment difficulty estimation
   - Uses aggregate transfer performance instead of single paired agent
2. **CPPN Environments**: Compositional Pattern-Producing Networks
   - Unbounded, arbitrarily complex environment generation
   - Indirect encoding enables geometric patterns and symmetries

**See [[Enhanced_POET_detailed]] for details.**

### Domain Extensions

**Potential Applications**:
1. **Autonomous Driving**: Generate edge case scenarios automatically
2. **Robotic Manipulation**: Coevolve object geometries and manipulation strategies
3. **Game AI**: Generate game levels and AI players simultaneously
4. **Protein Folding**: Coevolve targets and folding strategies
5. **Engineering Design**: Generate design challenges and solutions

### Algorithmic Extensions

**Possible Improvements**:
1. **Reward Coevolution**: Evolve reward functions per environment
2. **Morphology Coevolution**: Evolve agent body structure alongside controller
3. **Meta-Learning Integration**: Use POET-generated task distributions for meta-learning
4. **Hierarchical POET**: Nested coevolution at multiple abstraction levels

---

## Common Pitfalls

### 1. Minimal Criterion Too Strict

**Problem**: MC range [50, 300] calibrated for bipedal walker. Other domains need adjustment.

**Solution**: Set MC based on:
- Lower bound: Enough reward signal for learning (not random)
- Upper bound: Not trivial (requires optimization)

### 2. Mutation Step Sizes

**Problem**: Too large → discontinuous environment spaces; too small → slow exploration

**Solution**: Calibrate mutation steps to:
- ~10-20% of parameter range
- Sufficient for diversity but smooth transitions

### 3. Transfer Frequency

**Problem**: Too frequent → computational waste; too rare → missed opportunities

**Solution**: Balance based on:
- Optimization speed (how fast agents improve)
- Population size (more environments → more transfer chances)
- Typically every 50-100 iterations

### 4. Novelty vs. Quality

**Problem**: Pure novelty → trivial but diverse environments; pure quality → convergence to single type

**Solution**: POET's approach:
- Novelty for **admission** (which children to add)
- Quality for **reproduction eligibility** (which parents can create children)
- Combined, these ensure diverse AND high-quality environments

---

## JAX Implementation Sketch

```python
import jax
import jax.numpy as jnp
from jax import random, vmap, jit

@jit
def es_step_jax(key, θ, env_params, α, σ, population_size=512):
    """
    JAX-accelerated Evolution Strategies step.
    """
    keys = random.split(key, population_size)

    def evaluate_pair(key):
        ε = random.normal(key, shape=θ.shape) * σ
        reward_plus = evaluate_agent(θ + ε, env_params)
        reward_minus = evaluate_agent(θ - ε, env_params)
        return (reward_plus - reward_minus) * ε

    # Vectorized evaluation
    gradients = vmap(evaluate_pair)(keys)
    gradients = jnp.mean(gradients, axis=0)

    # Adam update (simplified)
    Δθ = α * gradients
    return Δθ

@jit
def evaluate_agent(θ, env_params):
    """
    Evaluate agent in environment (should be JIT-compiled).
    """
    # Run episode, accumulate reward
    state = env_reset(env_params)
    total_reward = 0.0

    for t in range(2000):  # Max episode length
        action = network_forward(θ, state)
        state, reward, done = env_step(state, action, env_params)
        total_reward += reward
        if done:
            break

    return total_reward
```

**Advantages of JAX**:
- GPU/TPU acceleration for agent evaluation
- Vectorized ES sampling
- JIT compilation for fast environment stepping
- Automatic differentiation (if switching from ES to gradient-based methods)

---

## Summary

POET's implementation combines:
1. **Novelty-driven environment mutation** with Minimal Criterion filtering
2. **Evolution Strategies optimization** for agent improvement (pluggable)
3. **Bidirectional transfer mechanism** for stepping stone discovery

**Key Insight**: The magic happens at the **intersection** of these three components. Remove any one, and performance degrades significantly.

**Computational Profile**:
- Highly parallelizable (256+ cores)
- ~10 days per full run
- Bottleneck: Agent evaluation (can be GPU-accelerated)

**Practical Recommendation**: Start with small-scale experiments (fewer iterations, smaller population) to validate implementation, then scale to full POET runs.

---

**See [[POET]] for high-level overview and motivation.**
