# Pipeline PSRO (P2SRO)

## Definition
Pipeline PSRO is a scalable parallelization of PSRO that organizes RL workers in a **hierarchical pipeline**, enabling distributed best-response training while preserving convergence guarantees to approximate Nash equilibrium.

## Motivation
Standard PSRO is sequential: each BR must finish before the next begins. Previous parallelization attempts (DCH, Rectified PSRO) broke convergence guarantees. P2SRO solves this with a hierarchical structure.

## Key Idea
Workers at higher pipeline levels get a "head start" by pre-training against lower-level policies while those are still training:

```
Level 0: [Worker 0] → π₀ (trains vs random)
Level 1: [Worker 1] → π₁ (trains vs π₀)
Level 2: [Worker 2] → π₂ (trains vs π₁)
...
Level k: [Worker k] → πₖ (trains vs πₖ₋₁)
```

## Convergence
✓ Maintains PSRO convergence guarantees (unlike DCH or Rectified PSRO)

Each worker effectively performs one PSRO iteration, but iterations overlap in time.

## Key Results
- **Near-linear speedup** with number of workers
- **Barrage Stratego**: State-of-the-art performance, beats all existing bots
- **Leduc Poker**: 5-10x faster convergence with 5 workers vs sequential

## Interview Relevance
- **PSRO bottleneck?** Sequential BR training. P2SRO parallelizes via hierarchy.
- **Why not simple parallelism?** Breaks PSRO invariants. Hierarchy preserves the iterative BR property.
- **Comparison**: DCH ✗, Rectified PSRO ✗, P2SRO ✓ (convergence proven)

> Detailed implementation: [[Pipeline_PSRO_detailed]]

## Related
- [[PSRO]] / [[PSRO_detailed]] — Base algorithm
- [[NeuPL]] — Alternative scalable approach (single network)
