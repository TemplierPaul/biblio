# NeuPL-JPSRO — Detailed Implementation Notes

> **Quick overview**: [[NeuPL_JPSRO]]

## Paper

**Title**: Neural Population Learning beyond Symmetric Zero-sum Games

**Authors**: Siqi Liu, Luke Marris, Marc Lanctot, Nicolas Heess (DeepMind)

**Year**: 2024

**ArXiv**: [2401.05133](https://arxiv.org/abs/2401.05133)

## Core Algorithm (Algorithm 2)

```
function NeuPL-JPSRO(game, ε):
    # Initialization
    for each player p:
        Initialize embedding νₚ⁰ ∈ ℝᵈ
        𝒱ₚ = {νₚ⁰}
    Initialize conditional policy Πθ(·|s,ν)
    Initialize payoff estimator ψw(ν₁,...,νₙ)
    σ⁰ = CCE_Solver(evaluate_payoffs(𝒱, Πθ))

    for iteration t = 1 to convergence:
        θ̂ = θ       # Freeze reference parameters
        𝒱̂ = 𝒱       # Freeze embeddings

        for each player p:
            # Step 1: Best Response (against frozen co-players)
            πₚᵗ = BR(player=p, σ₋ₚᵗ⁻¹, Πθ̂, 𝒱̂)

            # Step 2: Distill BR into population
            νₚᵗ = new_embedding()
            min KL(πₚᵗ || Πθ(·|·,νₚᵗ))

            # Step 3: Regularize existing strategies
            for ν ∈ 𝒱ₚ:
                min KL(Πθ̂(·|·,ν̂) || Πθ(·|·,ν))

            𝒱ₚ = 𝒱ₚ ∪ {νₚᵗ}

        # Step 4: Update meta-game
        Gᵗ = evaluate_payoffs(𝒱, Πθ, ψw)
        σᵗ = CCE_Solver(Gᵗ)

        if max_p CCE_Gap_p(Gᵗ, σᵗ) < ε: break

    return Πθ, 𝒱, σᵗ
```

## Network Architecture

### Policy Network (Πθ)

```
Input: (observation s, strategy embedding ν)
    ↓
Observation Encoder (shared)
    Conv/MLP → encoded_obs
    ↓
Recurrent Memory (shared, LSTM 128-256 units)
    → memory_state
    ↓
Conditioning (FiLM, concat, or attention)
    [encoded_obs, memory_state, ν] → features
    ↓
Policy Head → π(a|s,ν) = Softmax(logits)
```

### Best-Response Head (Πϕ)

```
Input: (observation s, co-player mixed-strategy encoding)
    ↓
Reuse: Encoder + Memory from Πθ (frozen or trainable)
    ↓
Mixed-Strategy Encoding (top-k=96 joint strategies)
    g(𝒱, σ₋ₚ) = Σ σ₋ₚ(a₋ₚ) · f(ν embeddings)
    ↓
BR Policy Head (separate) → πϕ(a|s,σ₋ₚ)
```

### Payoff Estimator (ψw)

```
Input: Joint embeddings (ν₁,...,νₙ)
    → Concatenation or symmetric encoding
    → MLP
    → Output: [payoff_p1, ..., payoff_pn] ∈ ℝⁿ
```

## Key Design Decisions

### Iterative vs Concurrent Training

| NeuPL-JPSRO (Iterative) | Original NeuPL (Concurrent) |
|---|---|
| Freeze θ̂, train BR against stationary co-players | Continuously train all policies |
| Ensures JPSRO convergence guarantees | Co-players are moving targets |
| More expensive per iteration | More sample-efficient |

### Reference Parameter Freezing

```python
θ̂ = θ   # Ensures co-player stationarity
𝒱̂ = 𝒱   # Prevents "moving targets" problem
```

### Distillation + Regularization

```python
# Distill BR into population
loss_distill = KL(πₚᵗ(·|s) || Πθ(·|s,νₚᵗ))

# Prevent catastrophic forgetting
loss_reg = KL(Πθ̂(·|s,ν̂) || Πθ(·|s,ν))  # for each existing ν
```

## CCE Solver (Linear Program)

```python
import cvxpy as cp

def solve_CCE(payoff_tensor, num_players, actions_per_player):
    joint_actions = list(itertools.product(
        *[range(a) for a in actions_per_player]
    ))
    sigma = cp.Variable(len(joint_actions))
    constraints = [sigma >= 0, cp.sum(sigma) == 1]

    # Incentive constraints for each player
    for player in range(num_players):
        for ap in range(actions_per_player[player]):
            for ap_dev in range(actions_per_player[player]):
                if ap == ap_dev: continue
                payoff_comply = sum(...)  # expected payoff following σ
                payoff_deviate = sum(...)  # expected payoff deviating
                constraints.append(payoff_comply >= payoff_deviate)

    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve()
    return sigma.value
```

## Convergence

**Theorem 3.2**: Under exact distillation and regularization, NeuPL-JPSRO converges to a normal-form CCE.

**In practice**: Distillation/regularization are approximate → bounded error, but empirically still converges to near-CCE.

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| Strategy embedding dim | 64–256 |
| Encoder hidden units | 128–512 |
| LSTM units | 128–256 |
| LR (policy) | 1e-4 to 1e-3 |
| LR (payoff estimator) | 1e-4 to 1e-3 |
| Entropy regularization α | 0.001–0.01 |
| KL distillation weight | 1.0–10.0 |
| KL regularization weight | 1.0–10.0 |
| Top-k for mixed-strategy | k=96 |
| CCE solver | LP (CVXPY / OSQP) |

## Evaluation Metrics

1. **CCE Gap**: δ(σ) = Σₚ max(0, max_{a'ₚ}[E_{a₋ₚ~σ₋ₚ} Gₚ(a'ₚ,a₋ₚ) - E_{a~σ} Gₚ(a)])
2. **Exploitability**: Via independent RL exploiters
3. **Payoff Estimator Accuracy**: MSE vs actual rollouts
4. **Policy Diversity**: KL divergence between population members

## Benchmark Tasks

| Task | Players | Type | Key Challenge |
|------|---------|------|---------------|
| OpenSpiel games (6) | 2–3 | Various | Analytical CCE verification |
| MuJoCo Cheetah-Run | 2 | Cooperative | Coordinated motor control |
| Capture-the-Flag | 4 (2v2) | Mixed | Partial obs, sparse rewards, teams |

## References

- [NeuPL beyond Symmetric Zero-sum (Liu et al., 2024)](https://arxiv.org/abs/2401.05133)
- [JPSRO (Marris et al., ICML 2021)](https://arxiv.org/abs/2106.09435)
- [NeuPL (Liu et al., ICLR 2022)](https://arxiv.org/abs/2202.07415)

## Related

- [[NeuPL_JPSRO]] — Quick overview
- [[NeuPL]] / [[NeuPL_detailed]] — Single-network foundation
- [[JPSRO]] / [[JPSRO_detailed]] — CCE convergence backbone
- [[Simplex_NeuPL]] — Mixture-optimal (symmetric ZS only)
