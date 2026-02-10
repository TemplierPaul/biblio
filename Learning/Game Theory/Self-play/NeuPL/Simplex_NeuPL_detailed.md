# Simplex-NeuPL — Detailed Implementation Notes

> **Quick overview**: [[Simplex_NeuPL]]

## Paper

**Title**: Simplex Neural Population Learning: Any-Mixture Bayes-Optimality in Symmetric Zero-sum Games

**Authors**: Siqi Liu, Marc Lanctot, Luke Marris, Nicolas Heess (DeepMind)

**Published**: ICML 2022

**ArXiv**: [2205.15879](https://arxiv.org/abs/2205.15879)

## Two-Level Conditioning

### Network Architecture

```
Input: (observation, condition)
    condition: basis index i ∈ {1,...,k}  OR  mixture μ ∈ Δₖ
    ↓
Observation Encoder → obs_features
    ↓
Condition Encoder
    Basis: one_hot(i) → Linear → embedding
    Mixture: μ → Linear → embedding (or direct use)
    ↓
Fusion (FiLM or concatenation)
    ↓
Shared Trunk → Policy Head → π(a|o,c)
```

## Training Loop

```
Initialize network θ, basis population Π = {π₁}

for iteration t:
    # Phase 1: Train basis policies (like NeuPL)
    for basis_index i in Π:
        Sample opponent j ~ meta_strategy
        Train π_θ(·|·,i) vs π_θ(·|·,j)

    # Phase 2: Train mixture best-responses (key innovation)
    for step in mixture_training_steps:
        μ ~ Dirichlet(α)           # Random mixture from simplex
        opp_idx ~ μ                # Opponent sampled from mixture
        Train π_θ(·|·,μ) vs π_θ(·|·,opp_idx)

    # Phase 3: Add new basis policy
    new_basis = BR to meta_strategy
    Π = Π ∪ {new_basis}
    Update meta_strategy via Nash solver
```

## Simplex Sampling

```python
# Dirichlet sampling (uniform on simplex)
μ = np.random.dirichlet(alpha * np.ones(k))

# Stratified for better coverage:
# 1. Corners (pure): μ = eᵢ
# 2. Edges (pairwise): μ = λeᵢ + (1-λ)eⱼ
# 3. Interior (general): μ ~ Dirichlet(α)
```

## Theoretical Properties

**Any-Mixture Best-Response**: For any μ ∈ Δₖ:
$$u(\pi_\theta(\cdot|\cdot,\mu),\; \text{opp}\sim\mu) \approx \max_\pi u(\pi,\; \text{opp}\sim\mu)$$

**Key Insight**: Mixture-BR training is an effective *auxiliary task* that promotes strategic exploration, better generalization, and faster convergence of basis policies.

## Comparison: NeuPL vs Simplex-NeuPL

| Aspect | NeuPL | Simplex-NeuPL |
|--------|-------|---------------|
| Conditioning | Discrete only | Discrete + Continuous |
| Optimality | BR to pure strategies | BR to any mixture |
| Training | Basis only | Basis + mixture responses |
| Robustness | Good | Better (Bayes-optimal) |
| Applications | Known opponents | Unknown/mixed opponents |

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| Basis population k | 8–32 |
| Dirichlet α | 0.5–2.0 |
| Mixture training steps | 100s–1000s per iter |
| Condition embedding dim | 32–128 |
| Learning rate | 1e-4 to 1e-3 |

## Limitations

1. **Symmetric games only**: Designed for symmetric zero-sum
2. **Computational cost**: Mixture training adds overhead
3. **Network capacity**: Must handle both discrete and continuous conditioning

## References

- [Simplex NeuPL (Liu et al., ICML 2022)](https://proceedings.mlr.press/v162/liu22h.html)
- [ArXiv 2205.15879](https://arxiv.org/abs/2205.15879)

## Related

- [[Simplex_NeuPL]] — Quick overview
- [[NeuPL]] / [[NeuPL_detailed]] — Base framework
- [[NeuPL_JPSRO]] — General-sum extension
