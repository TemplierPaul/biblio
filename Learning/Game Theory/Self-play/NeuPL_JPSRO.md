# NeuPL-JPSRO

## Definition
NeuPL-JPSRO extends Neural Population Learning to **n-player general-sum games** by leveraging the convergence guarantees of JPSRO. It combines NeuPL's single-network efficiency with JPSRO's Coarse Correlated Equilibrium (CCE) convergence.

## Motivation
- **NeuPL**: Efficient (single network) but restricted to symmetric zero-sum
- **JPSRO**: Handles general-sum n-player but uses separate networks (O(N) memory)
- **NeuPL-JPSRO**: Best of both — single network, general-sum, CCE convergence

## Key Properties
1. **General-sum games**: Not restricted to zero-sum
2. **n-player**: Any number of players
3. **CCE convergence**: Proven via reduction to JPSRO
4. **Skill transfer**: Shared representations via single network
5. **O(1) memory**: Single conditional network for all policies

## Algorithm Sketch
```
for iteration t:
    Freeze reference parameters θ̂
    for each player p:
        Train BR against co-player marginal σ₋ₚ (using frozen network)
        Distill BR into population network (KL minimization)
        Regularize existing strategies (prevent forgetting)
    Update meta-game → solve CCE
```

## Comparison

| Method | Players | Game Type | Equilibrium | Networks | Transfer |
|--------|---------|-----------|-------------|----------|----------|
| **JPSRO** | n | General-sum | CCE | O(N) | None |
| **NeuPL** | 2 | Symmetric ZS | Nash approx | 1 | Yes |
| **NeuPL-JPSRO** | n | General-sum | CCE | 1 | Yes |

## Interview Relevance
- **Why not just NeuPL?** Limited to symmetric zero-sum; can't handle coordination
- **Why not just JPSRO?** O(N) networks, no transfer learning
- **Key innovation?** Iterative (not concurrent) training + distillation ensures JPSRO convergence while using a single network

> Detailed implementation: [[NeuPL_JPSRO_detailed]]

## Related
- [[NeuPL]] / [[NeuPL_detailed]] — Base framework
- [[JPSRO]] / [[JPSRO_detailed]] — Convergence backbone
- [[Simplex_NeuPL]] — Mixture-optimal variant (symmetric ZS only)
