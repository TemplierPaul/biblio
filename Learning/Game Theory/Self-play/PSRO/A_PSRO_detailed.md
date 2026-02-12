# A-PSRO — Detailed Implementation Notes

> **Quick overview**: [[A_PSRO]]

## Paper
**Title**: A-PSRO: A Unified Strategy Learning Method with Advantage Function for Normal-form Games
**Venue**: ICML 2024
**Authors**: Hu et al. (UCAS)

## Core Concept: The Advantage Function

The central metric is the **Advantage Function** $V(\pi)$, quantifying how well a strategy performs against its best response:

$$V_i(\pi_i) = U_i(\pi_i, BR(\pi_i))$$

**Properties**:
1. **Nash Characterization**: $V(\pi) \le 0$, and $V(\pi) = 0 \iff$ Nash Equilibrium.
2. **Convexity**: $-V(\pi)$ is convex (in symmetric zero-sum), aiding optimization.
3. **Lipschitz Continuity**: Ensures stable gradients.

## Algorithm: LookAhead Update

Instead of just adding a Best Response (BR) to the population, A-PSRO intelligently mixes the current meta-strategy $\theta_i$ with a search direction $\Delta \pi$.

```python
def a_psro_update(P_i, theta_i, theta_j, lr=0.1):
    # 1. LookAhead Search
    # Find direction Delta_pi that maximizes advantage of mixed strategy
    l_r_sampled = random.uniform(0, lr)
    
    Delta_pi = argmax_A [ 
        V( (1 - l_r_sampled) * theta_i + l_r_sampled * candidate ) 
    ]
    
    # 2. Candidate Strategy
    pi_star = (1 - l_r_sampled) * theta_i + l_r_sampled * Delta_pi
    
    # 3. Acceptance Check (sufficient improvement?)
    if improvement(pi_star) >= threshold:
        return pi_star  # Update current policy in-place
    else:
        return P_i + {new_random_policy} # Add new dimension
```

## Algorithm: General-Sum & Multi-Player

For non-zero-sum games, A-PSRO runs **multiple oracle trials** to find the equilibrium with the **highest advantage value**.

**Theorem**: Local maxima of the advantage function in general-sum games correspond to Nash Equilibria. The global maximum often corresponds to the Pareto-optimal Nash Equilibrium (highest joint reward).

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate ($\lambda$)** | 0.05 - 0.2 | Max step size for LookAhead mixing. |
| **Diversity Weight ($\lambda_d$)** | 0.3 - 0.7 | Probability of using Diversity vs LookAhead. |
| **Oracle Repeats ($k$)** | 10 | Runs to find best equilibrium (general-sum). |
| **Improvement ($c_m$)** | 0.01 | Threshold to accept in-place update. |

## Comparison to PSRO Variants

| Method | Zero-Sum Guarantee | General-Sum Selection | Compute Cost |
|--------|--------------------|-----------------------|--------------|
| **PSRO** | No (cycles possible) | Random NE | Low |
| **DPP-PSRO** | Better (diversity) | Random NE | High ($O(N^3)$) |
| **A-PSRO** | **Deterministic** | **Pareto-Optimal** | Medium ($O(|A|N^2)$) |

## Related

- [[PSRO]] / [[PSRO_detailed]] — Baseline algorithm
- [[Pipeline_PSRO]] — for parallelization
- [[SP_PSRO]] — using mixed strategies
