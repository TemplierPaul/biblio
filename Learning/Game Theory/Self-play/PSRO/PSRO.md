# Policy Space Response Oracles (PSRO)

## Definition
PSRO is a general framework that combines the **Double Oracle (DO)** algorithm with deep reinforcement learning. It scales game-theoretic solving to games with vast strategy spaces.

## The Algorithm
1.  **Initialize**: Start with a small set (population) of initial strategies for each player.
2.  **Meta-Game**: Compute the "Meta-Nash" of the game restricted to the current population (the **Empirical Game**).
3.  **Oracle**: For each player, find a **Best Response (BR)** to the Meta-Nash using RL.
4.  **Expand**: Add the new BRs to the population and repeat.

## Key Concepts
-   **Meta-Nash Convergence**: PSRO converges to a Nash Equilibrium of the full game when the Oracle can no longer find a better response that improves a player's payoff against the current Meta-Nash.
-   **Empirical Game Theory (EGT)**: Using simulations to build a payoff matrix for the current population.

## Variants
-   **Pipeline PSRO (P-PSRO)**: Parallelizes the BR computation by having multiple "oracles" training at once against the current population.
-   **Extensive-Form PSRO**: Adapted for games with sequential moves.

## Why it's powerful (Interview Point)
PSRO decouples **Strategy Exploration** (finding new BRs) from **Strategy Interaction** (evaluating how strategies perform against each other). It is the backbone of many modern MARL successes.
