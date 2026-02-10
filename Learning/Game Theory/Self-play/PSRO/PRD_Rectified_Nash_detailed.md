# PRD & Rectified Nash — Detailed Implementation Notes

> **Quick overview**: [[PRD_Rectified_Nash]]

## Projected Replicator Dynamics (PRD)

### Continuous-Time Formulation

```
d/dt σᵢ(t) = σᵢ(t) · [u(eᵢ, σ(t)) - u(σ(t), σ(t))]

subject to:
    Σ σᵢ = 1
    σᵢ ≥ η > 0  (minimum probability)
```

### Discrete-Time Update

```python
class PRDMetaSolver:
    def __init__(self, min_prob=0.01, step_size=0.1):
        self.eta = min_prob
        self.alpha = step_size

    def update(self, payoff_matrix, current_sigma):
        n = len(current_sigma)
        avg_payoff = np.dot(current_sigma, np.dot(payoff_matrix, current_sigma))

        growth = np.zeros(n)
        for i in range(n):
            pure_payoff = np.dot(payoff_matrix[i], current_sigma)
            growth[i] = current_sigma[i] * (pure_payoff - avg_payoff)

        new_sigma = current_sigma + self.alpha * growth
        new_sigma = np.maximum(new_sigma, self.eta)
        new_sigma = new_sigma / new_sigma.sum()
        return new_sigma

    def solve(self, payoff_matrix, num_iterations=1000):
        n = payoff_matrix.shape[0]
        sigma = np.ones(n) / n  # Uniform init
        for _ in range(num_iterations):
            sigma = self.update(payoff_matrix, sigma)
        return sigma
```

### Projection Operation

```python
def project_to_simplex(sigma, min_prob=0.01):
    sigma_proj = np.maximum(sigma, min_prob)
    sigma_proj = sigma_proj / sigma_proj.sum()
    return sigma_proj
```

## Rectified Nash

### Key Modifications
1. **Restricted support**: Only considers subset of strategies
2. **Regularization**: Added to Nash computation
3. **Symmetry exploitation**: Leverages game symmetry when applicable

### Empirical Issues (ICLR 2020)

On Kuhn Poker convergence to zero NashConv:
1. α-Rank, Nash, PRD: ~Same rate ✓
2. Uniform: Slower rate
3. Rectified PRD: Very slow
4. **Rectified Nash: Constant (no progress) ✗**

**Recommendation**: Use PRD or α-Rank for general cases; avoid Rectified Nash in asymmetric games.

## PSRO Integration

```python
class PSROwithPRD:
    def __init__(self, game, min_prob=0.01):
        self.game = game
        self.population = [self.init_policy()]
        self.prd_solver = PRDMetaSolver(min_prob=min_prob)

    def run_iteration(self):
        payoff_matrix = self.evaluate_population()
        self.meta_strategy = self.prd_solver.solve(payoff_matrix)
        new_policy = self.train_best_response(self.meta_strategy)
        self.population.append(new_policy)
```

## Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| η (min_prob) | 0.01–0.1 | More exploration ↔ closer to Nash |
| α (step_size) | 0.01–0.5 | Faster ↔ more stable |
| Iterations | 100–10000 | Until convergence |
| Convergence threshold | 1e-6 to 1e-3 | Change in σ for stopping |

## When to Use Each

| **Use PRD** | **Use Nash** | **Use α-Rank** | **Avoid Rectified Nash** |
|---|---|---|---|
| Large games | Small games | >2 players | Asymmetric games |
| Need exploration | Need guarantees | Need exploration | Need convergence |
| Empirical perf | Convergence proof | Multi-player | Production systems |

## References

- [A Generalized Training Approach for Multi-Agent Learning (ICLR 2020)](https://openreview.net/pdf/7f3a5843704098cc456ab961def2f56902eaf1dd.pdf)
- [PSRO Survey (2024)](https://arxiv.org/abs/2403.02227)

## Related

- [[PRD_Rectified_Nash]] — Quick overview
- [[PSRO]] / [[PSRO_detailed]] — Framework using these solvers
- [[JPSRO]] — Uses CE/CCE solvers instead
