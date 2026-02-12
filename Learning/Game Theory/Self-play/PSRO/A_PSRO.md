# A-PSRO (Advantage-based PSRO)

## Definition
A-PSRO constitutes a unified open-ended learning framework for **zero-sum**, **general-sum**, and **multi-player** games. It uses the **advantage function** (performance against best response) as a universal learning objective, enabling deterministic convergence properties that standard PSRO lacks.

## Key Innovation
Classic PSRO mixes strategies blindly or via diversity metrics. A-PSRO introduces an **Advantage-based LookAhead** module that specifically searches for update directions that maximize the advantage function $V(\pi) = U(\pi, BR(\pi))$.

## Core Components
1. **Advantage Function**: $V_i(\pi_i) = \min_{\pi_{-i}} U_i(\pi_i, \pi_{-i})$ (for zero-sum). $V(\pi)=0$ iff Nash.
2. **LookAhead Module**: Optimizes step direction using advantage gradients.
3. **Diversity Module**: Optional fallback for cyclic game landscapes.

## Why Use It?
- **Converges deterministically** in zero-sum games (unlike standard PSRO).
- **Selects better equilibria** (Pareto-optimal) in general-sum games.
- **Unified**: Works for n-player games out of the box.

> Detailed implementation: [[A_PSRO_detailed]]
