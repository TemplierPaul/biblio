# Simplex-NeuPL

## Definition
Simplex-NeuPL extends NeuPL to achieve **any-mixture Bayes-optimality**: it learns both diverse basis policies *and* best-responses to arbitrary mixtures over those policies, all in a single conditional network.

## Key Innovation
Two simultaneous objectives within one network π_θ(a|o,c):
1. **Diverse basis policies**: A population Π = {π₁,…,πₖ} spanning the strategy space
2. **Any-mixture BR**: For any μ ∈ Δₖ, learn BR(μ) = argmax_π E_{opp~μ}[u(π, opp)]

## Conditioning
- **Discrete** c ∈ {1,…,k}: selects a basis policy
- **Continuous** c = μ ∈ Δₖ: selects BR to mixture μ (sampled via Dirichlet)

## Why It Matters
- **Bayes-optimal**: Near-optimal play against unknown opponent mixtures
- **Auxiliary task**: Mixture-BR training actually improves basis policies
- **Robust**: Superior generalization to unseen opponent distributions

## Interview Relevance
- **NeuPL vs Simplex-NeuPL?** NeuPL trains BRs to pure strategies; Simplex trains BRs to *any mixture*
- **Why Bayes-optimal?** When opponent is uncertain, optimal play integrates over all possible opponents
- **Limitation?** Designed for symmetric zero-sum games

> Detailed implementation: [[Simplex_NeuPL_detailed]]

## Related
- [[NeuPL]] / [[NeuPL_detailed]] — Base framework
- [[NeuPL_JPSRO]] — General-sum extension
- [[PSRO]] — Population-based predecessor
