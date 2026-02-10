# Pipeline PSRO — Detailed Implementation Notes

> **Quick overview**: [[Pipeline_PSRO]]

## Paper

**Title**: Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games

**Authors**: Stephen McAleer, John Lanier, Kevin Wang, Pierre Baldi, Roy Fox (UC Irvine)

**Published**: NeurIPS 2020

**ArXiv**: [2006.08555](https://arxiv.org/abs/2006.08555)

**Code**: [GitHub](https://github.com/JBLanier/pipeline-psro)

## Algorithm

### Sequential PSRO (Bottleneck)

```
Initialize: Population Π = {π₀}
for iteration t = 1 to convergence:
    σ = NashSolver(payoffs)
    πₜ = TrainBR(σ)        ← BLOCKS until complete
    Π = Π ∪ {πₜ}
```

### Pipeline PSRO (Parallel)

```
Initialize: Hierarchical pipeline with L levels
Level 0: Start Worker 0 training π₀

for level l = 1 to L:
    Wait until Worker (l-1) has partially trained
    Start Worker l training against current π_{l-1}

Concurrent operation:
    All workers train simultaneously
    Higher-level workers periodically update opponent to latest lower-level policy
    Meta-game updated asynchronously with converged policies
```

### Worker Implementation

```python
class RLWorker:
    def __init__(self, level, rl_algorithm):
        self.level = level
        self.rl_algorithm = rl_algorithm
        self.steps_trained = 0

    def train_step(self, num_steps=1000):
        for _ in range(num_steps):
            episode = self.play_game(self.current_policy, self.opponent)
            self.rl_algorithm.update(episode)
            self.steps_trained += 1

    def should_update_opponent(self):
        return self.steps_trained % self.opponent_update_freq == 0

    def update_opponent(self, new_opponent):
        self.opponent = new_opponent

    def converged(self):
        return (self.steps_trained >= self.max_steps or
                self.performance_plateaued())
```

### Meta-Game Manager

```python
class MetaGameManager:
    def add_policy(self, policy):
        self.population.append(policy)
        new_payoffs = self.evaluate_policy(policy)
        self.update_payoff_matrix(new_payoffs)
        self.meta_strategy = self.compute_nash(self.payoff_matrix)
```

## Convergence Guarantees

| Method | Convergence | Why |
|--------|-------------|-----|
| **PSRO** | ✓ | Sequential BR, proven |
| **DCH** | ✗ | Simultaneous BR violates assumptions |
| **Rectified PSRO** | ✗ | Modification breaks proof |
| **Pipeline PSRO** | ✓ | Hierarchical structure preserves iterative BR |

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| Number of levels L | 3–10 |
| Workers per level | 1–5 |
| Warmup steps | 10k–100k |
| RL algorithm | DQN, PPO, etc. |
| Training steps per BR | 100k–1M |
| Opponent update freq | 1k–10k steps |

## Comparison

| Method | Parallelization | Convergence | Scalability |
|--------|----------------|-------------|-------------|
| **PSRO** | None | ✓ Guaranteed | Low |
| **DCH** | Full simultaneous | ✗ | High (broken) |
| **Rectified PSRO** | Partial | ✗ | Medium (broken) |
| **Pipeline PSRO** | Hierarchical | ✓ Guaranteed | High |
| **NeuPL** | Single network | ✓ Approximate | High |

## References

- [Pipeline PSRO (McAleer et al., NeurIPS 2020)](https://arxiv.org/abs/2006.08555)
- [GitHub Implementation](https://github.com/JBLanier/pipeline-psro)

## Related

- [[Pipeline_PSRO]] — Quick overview
- [[PSRO]] / [[PSRO_detailed]] — Base algorithm
- [[NeuPL]] — Alternative scalable approach
