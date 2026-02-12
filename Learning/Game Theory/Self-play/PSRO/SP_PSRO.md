# SP-PSRO (Self-Play PSRO)

## Definition
SP-PSRO (Self-Play PSRO) extends Anytime PSRO by adding **mixed strategies** directly into the population, rather than just pure best responses. It trains a new policy $\nu$ via off-policy RL to approximate a Nash equilibrium limited to the current population context.

## Key Innovation
Standard PSRO adds one deterministic Best Response (BR) per iteration. In games like Rock-Paper-Scissors, this requires iterating through all pure strategies ($N$ steps) to converge. SP-PSRO adds a **time-averaged mixed strategy** $\bar{\nu}$, allowing it to approximate the Nash equilibrium in $O(1)$ iterations.

## Core Components
1. **Three Policies**:
    - $\beta_{-i}$: Opponent Best Response (trained vs restricted distribution).
    - $\nu_i$: New Strategy (trained vs $\beta_{-i}$).
    - $\pi^r$: Restricted Distribution (no-regret learner).
2. **Off-Policy Training**: $\nu_i$ learns from *all* generated data, not just its own trajectories.
3. **Time-Averaging**: Distills the average behavior of $\nu_i$ into a new network $\bar{\nu}$.

## Why Use It?
- **Speed**: Converges ~5-15x faster than APSRO in games requiring mixed strategies.
- **Efficiency**: "Anytime" property ensures exploitability doesn't degrade.
- **Lower Exploitability**: consistently beats standard PSRO baselines.

> Detailed implementation: [[SP_PSRO_detailed]]
