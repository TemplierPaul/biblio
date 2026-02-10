# PSRO — Detailed Implementation Notes

> **Quick overview**: [[PSRO]]

## Paper

**Title**: A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning

**Authors**: Lanctot et al. (DeepMind)

**Year**: 2017

**ArXiv**: [1711.00832](https://arxiv.org/abs/1711.00832)

## Algorithm

### Core Framework

PSRO iteratively builds a population of policies and computes a meta-strategy (mixture) over them:

```
Initialize: Population Π = {π₀}, Payoff table M = []
for iteration t = 1 to convergence:
    1. Evaluate: Fill payoff table M[i,j] = u(πᵢ, πⱼ)
    2. Meta-solve: σ = MetaSolver(M)  (e.g., Nash, PRD, α-Rank)
    3. Best-response: πₜ = TrainBR(σ, Π)  via RL
    4. Expand: Π = Π ∪ {πₜ}
```

### Pseudocode (Full)

```python
class PSRO:
    def __init__(self, game, meta_solver='nash'):
        self.population = [random_policy()]
        self.payoff_matrix = np.zeros((1, 1))
        self.meta_solver = meta_solver

    def run(self, num_iterations):
        for t in range(num_iterations):
            # Evaluate all policy matchups
            n = len(self.population)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    M[i,j] = evaluate(self.population[i], self.population[j])
            self.payoff_matrix = M

            # Compute meta-strategy
            sigma = self.solve_meta_game(M)

            # Train best-response via RL
            new_policy = self.train_best_response(sigma)

            # Add to population
            self.population.append(new_policy)

    def train_best_response(self, sigma):
        """Train policy to maximize return against σ-weighted opponents"""
        for episode in range(num_episodes):
            opp_idx = np.random.choice(len(sigma), p=sigma)
            opponent = self.population[opp_idx]
            trajectory = play_game(learner, opponent)
            update_policy(learner, trajectory)
        return learner
```

## Network Architectures

### Standard Policy Networks

```
For board/card games:
    Input: Game state encoding
    → Conv2D layers (if spatial) or Linear layers
    → ReLU activations
    → Policy head: Softmax over actions
    → Value head: Single scalar

For continuous control:
    Input: Observation vector
    → Linear(obs_dim, 256) → ReLU
    → Linear(256, 256) → ReLU
    → Policy head: Gaussian (μ, σ) for actions
    → Value head: Linear → scalar
```

### Recurrent Policies (Partial Observation)

```
Input: (observation, hidden_state)
→ Linear encoder
→ LSTM/GRU cell
→ Policy head + Value head
```

## Best Response Objectives

The BR oracle maximizes return against the meta-strategy:

$$\pi^* = \arg\max_\pi \mathbb{E}_{j \sim \sigma} \left[ u(\pi, \pi_j) \right]$$

**Common RL Algorithms for BR Training**:
- PPO (discrete actions, on-policy, stable)
- SAC (continuous actions, off-policy, sample-efficient)
- MPO (KL-constrained, very stable)
- CrossQ (no target networks, lightweight)

## Meta-Strategy Solvers

| Solver | Type | Speed | Best For |
|--------|------|-------|----------|
| **Nash (LP)** | Exact | Slow | Small games, guarantees |
| **PRD** | Approximate | Fast | Large games, exploration |
| **α-Rank** | Approximate | Medium | Multi-player |
| **Rectified Nash** | Modified | Varies | Symmetric games only |
| **Uniform** | None | Instant | Baseline |

See [[PRD_Rectified_Nash]] for details on meta-solvers.

## Variants

### Anytime PSRO
- No fixed BR budget; continuously refine best-responses
- More practical for complex environments

### Simulation-Free PSRO
- Avoids expensive game simulations through model-based evaluation

### Pipeline PSRO (P2SRO)
- Parallel workers in hierarchical pipeline
- Near-linear speedup while preserving convergence
- See [[Pipeline_PSRO]]

### Rectified PSRO
- Modified support selection
- ⚠️ Breaks convergence guarantees in asymmetric games

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| BR training episodes | 10k–1M steps |
| Meta-game evaluation games | 100–10k per matchup |
| Population growth per iter | 1 policy |
| Meta-solver | Nash or PRD (recommended) |
| RL algorithm for BR | PPO, SAC, MPO, CrossQ |

## Evaluation Metrics

1. **Exploitability**: $\varepsilon = \max_{\pi'} u(\pi', \sigma) - v^*$ — how far from Nash
2. **NashConv**: Sum of exploitabilities across players
3. **Win Rate**: Against fixed baselines or historical policies
4. **Elo Rating**: Relative skill tracking over iterations
5. **Population Diversity**: Behavioral variance across policies

## Computational Efficiency

**Bottleneck**: BR training is the dominant cost.

**Optimizations**:
- Parallelized BR training (see [[Pipeline_PSRO]])
- Warm-starting from previous BR
- Reducing evaluation games via payoff estimation
- Single-network population (see [[NeuPL]])

## Benchmark Tasks

| Game | Info | Players | Typical Result |
|------|------|---------|----------------|
| Kuhn Poker | Imperfect | 2 | Exact convergence |
| Leduc Poker | Imperfect | 2 | Near-optimal |
| MuJoCo games | Perfect | 2 | Competitive play |
| Barrage Stratego | Imperfect | 2 | State-of-the-art (with P2SRO) |

## Code Resources

- [OpenSpiel PSRO](https://github.com/google-deepmind/open_spiel) (Google DeepMind)
- [Pipeline PSRO](https://github.com/JBLanier/pipeline-psro)
- [PSRO Survey (2024)](https://arxiv.org/abs/2403.02227)

## References

- [A Unified Game-Theoretic Approach to MARL (Lanctot et al., 2017)](https://arxiv.org/abs/1711.00832)
- [Policy Space Response Oracles: A Survey (2024)](https://arxiv.org/abs/2403.02227)
- [Pipeline PSRO (McAleer et al., NeurIPS 2020)](https://arxiv.org/abs/2006.08555)

## Related

- [[PSRO]] — Quick overview
- [[NeuPL]] / [[NeuPL_detailed]] — Single-network alternative
- [[Pipeline_PSRO]] — Parallel PSRO
- [[JPSRO]] / [[JPSRO_detailed]] — General-sum extension
- [[PRD_Rectified_Nash]] — Meta-strategy solvers
- [[RL_Methods_for_PSRO]] — BR oracle algorithms
