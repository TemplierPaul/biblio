# PRD & Rectified Nash (Meta-Strategy Solvers)

## Definition
Projected Replicator Dynamics (PRD) and Rectified Nash are **meta-strategy solvers** used within PSRO to compute how to mix over the policy population. They provide alternatives to exact Nash computation.

## PRD — Projected Replicator Dynamics

Continuous-time dynamical system combining replicator dynamics with projection to enforce exploration:

```
σ̇ᵢ = σᵢ · [u(eᵢ, σ) - u(σ, σ)]     (replicator dynamics)
σᵗ⁺¹ = Project(σᵗ + α·σ̇, min=η)     (projection to simplex)
```

**Key property**: Minimum probability η keeps all strategies viable → exploration.

## Rectified Nash

Modified Nash with restricted support and regularization. Designed for symmetric games.

⚠️ **Empirical issues**: Very slow or non-convergent in asymmetric games (Kuhn, Leduc poker).

## Comparison of Meta-Solvers

| Solver | Speed | Exploration | Best For |
|--------|-------|-------------|----------|
| **Nash (LP)** | Slow | None | Small games, guarantees |
| **PRD** | Fast | Good (via η) | Large games |
| **α-Rank** | Medium | Good | Multi-player |
| **Rectified Nash** | Varies | Varies | Symmetric only |
| **Uniform** | Instant | Maximum | Baseline |

## Interview Relevance
- **Why not exact Nash?** Slow (LP) for large meta-games; PRD is faster and encourages exploration
- **Rectified Nash issue?** Fails in asymmetric games (constant NashConv, no convergence)
- **PRD's η parameter?** Acts as entropy regularization; trades off convergence quality vs exploration

> Detailed implementation: [[PRD_Rectified_Nash_detailed]]

## Related
- [[PSRO]] / [[PSRO_detailed]] — Framework using these solvers
- [[JPSRO]] — Uses CE/CCE solvers instead
