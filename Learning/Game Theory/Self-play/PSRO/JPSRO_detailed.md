# JPSRO — Detailed Implementation Notes

> **Quick overview**: [[JPSRO]]

## Paper

**Title**: Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium Meta-Solvers

**Authors**: Luke Marris, Ian Gemp, Thomas Anthony, et al.

**Published**: ICML 2021

**ArXiv**: [2106.09435](https://arxiv.org/abs/2106.09435)

**Code**: [OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/jpsro.py)

## Algorithm

```
Input: Initial policies, equilibrium concept (CE or CCE)

1. Initialize policy sets Π₁⁰,...,Πₙ⁰
2. G⁰ ← evaluate all policy combinations
3. σ⁰ ← MetaSolver(G⁰)

4. For iteration t = 1 to convergence:

    For each player p:
        If JPSRO(CCE):
            βₚ ← BR_p(Π, σᵗ⁻¹)         # Single BR against marginal
        If JPSRO(CE):
            For each recommendation r:
                βₚ,ᵣ ← BR_p(Π, σᵗ⁻¹|r)  # One BR per recommendation

        Πₚ = Πₚ ∪ {new policies}

    Gᵗ ← evaluate all new matchups
    σᵗ ← MetaSolver(Gᵗ)

    If (C)CE_Gap < ε: break

5. Return Π, σᵗ
```

## JPSRO(CE) vs JPSRO(CCE)

| Aspect | JPSRO(CE) | JPSRO(CCE) |
|--------|-----------|------------|
| BRs per player | \|recommendations\| | 1 |
| Computational cost | Higher | Lower |
| Equilibrium strength | Stronger (CE ⊆ CCE) | Weaker |
| Coordination | Full conditional | Marginals only |
| Convergence speed | Faster (empirically) | Slower |

## Meta-Strategy Solvers

### Maximum Welfare MW(C)CE
```
σ* = argmax_{σ ∈ (C)CE} Σᵢ uᵢ(σ)
```

### Maximum Entropy MG(C)CE (Recommended)
```
maximize: -½σᵀσ  (Gini impurity)
subject to: A·σ ≤ ε, σ ≥ 0, eᵀ·σ = 1
```
Robust to payoff perturbations, encourages diversity.

### Random Vertex RV(C)CE
```
Enumerate vertices of (C)CE polytope → sample uniformly
```

## Convergence Guarantees

**Theorem 6** (CCE): Mixed joint policy converges to CCE under meta-solver distribution.

**Theorem 7** (CE): Mixed joint policy converges to CE under meta-solver distribution.

**Conditions**: Finite policies, exact BR, correct meta-solver.

## Computational Optimizations

- **IESDS**: Eliminate dominated strategies → guarantee full-support solutions
- **Sparse advantage matrices**: Faster QP solving
- **Parallelization**: BRs and evaluation parallelizable

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Meta-Solver | MG(C)CE (recommended) |
| QP Optimizer | CVXPY or OSQP |
| Approximation ε | 0.0 for strict |
| Random seeds | 5 seeds for stochastic |

## Benchmark Tasks

| Game | Type | Players | Challenge |
|------|------|---------|-----------|
| Kuhn Poker | Zero-sum | 2–3 | Convergence speed |
| Trade Comm | Common-payoff | 2 | Large coordination space |
| Sheriff | General-sum | 2 | Negotiation dynamics |

## Key Results (ICML 2021)

1. MG(C)CE consistently best meta-solver for general-sum
2. JPSRO(CE) converges faster than JPSRO(CCE)
3. CE enables significantly higher payoffs in cooperative games
4. Maximum entropy solvers more robust than greedy welfare

## Code Resources

- [OpenSpiel JPSRO](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/jpsro.py)
- CVXPY for quadratic programming
- OSQP for fast QP solving

## References

- [Multi-Agent Training beyond Zero-Sum (Marris et al., ICML 2021)](https://arxiv.org/abs/2106.09435)
- [PSRO Survey (2024)](https://arxiv.org/abs/2403.02227)

## Related

- [[JPSRO]] — Quick overview (with CE/CCE math)
- [[PSRO]] / [[PSRO_detailed]] — Two-player zero-sum parent
- [[NeuPL_JPSRO]] — Single-network version
