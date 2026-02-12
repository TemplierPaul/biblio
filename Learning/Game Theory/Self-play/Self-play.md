# Self-play

Self-play is a paradigm in multi-agent reinforcement learning and game theory where agents learn by interacting with versions of themselves. It creates a natural "auto-curriculum" that can drive agents toward super-human performance.

## Core Methods
- [[SP|Vanilla Self-Play (SP)]] · [[SP_detailed|detailed]]
- [[FSP|Fictitious Self-Play (FSP)]] · [[FSP_detailed|detailed]]
- [[PSRO|Policy Space Response Oracles (PSRO)]] · [[PSRO_detailed|detailed]]
- [[NeuPL|Neural Population Learning (NeuPL)]] · [[NeuPL_detailed|detailed]]

## Advanced Variants
- [[Pipeline_PSRO|Pipeline PSRO (P2SRO)]] · [[Pipeline_PSRO_detailed|detailed]] — Parallel PSRO with convergence
- [[A_PSRO|A-PSRO]] · [[A_PSRO_detailed|detailed]] — Advantage-based PSRO (deterministic convergence)
- [[SP_PSRO|SP-PSRO]] · [[SP_PSRO_detailed|detailed]] — Self-Play PSRO (mixed strategies)
- [[SF_PSRO|SF-PSRO]] · [[SF_PSRO_detailed|detailed]] — Simulation-Free PSRO (Dynamic Window)
- [[Simplex_NeuPL|Simplex-NeuPL]] · [[Simplex_NeuPL_detailed|detailed]] — Any-mixture Bayes-optimality
- [[NeuPL_JPSRO|NeuPL-JPSRO]] · [[NeuPL_JPSRO_detailed|detailed]] — General-sum n-player NeuPL
- [[PRD_Rectified_Nash|PRD & Rectified Nash]] · [[PRD_Rectified_Nash_detailed|detailed]] — Meta-strategy solvers

## Key Concepts for Interviews
- **Auto-curriculum**: How self-play automatically increases task difficulty.
- **Cycling**: The risk of getting stuck in loops in non-transitive games (e.g., Rock-Paper-Scissors).
- **Nash Equilibrium**: The target convergence point for many self-play algorithms.
- **Exploitability**: A metric for measuring how far a strategy is from being un-exploitable.
