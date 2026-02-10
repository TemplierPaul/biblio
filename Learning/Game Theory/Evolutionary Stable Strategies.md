# Evolutionary Stable Strategies (ESS)

An ESS is a strategy which, if adopted by a population in a given environment, is impenetrable, meaning that it cannot be invaded by any alternative (mutant) strategy.

## Context: Evolutionary Game Theory
Unlike classical game theory, players aren't necessarily rational. Instead:
-   **Strategies** are inherited traits.
-   **Payoffs** represent biological fitness (reproductive success).
-   **Dynamics**: Strategies with higher payoffs increase in frequency (Replicator Dynamics).

## The Maynard Smith Conditions
A strategy $S$ is an ESS if, for any mutant strategy $M$:
1.  **Equilibrium Condition**: $u(S, S) \geq u(M, S)$ (The mutant does no better against the population strategy than the population strategy does against itself; $S$ is a Nash Equilibrium).
2.  **Stability Condition**: If $u(S, S) = u(M, S)$, then $u(S, M) > u(M, M)$ (If the mutant does as well as the population strategy against $S$, it must do worse than $S$ against the mutant itself).

## Key Examples
-   **Hawk-Dove Game**: Analyzes aggressive vs. passive behavior.
-   **Cooperation**: How altruism can be an ESS under certain conditions (e.g., kin selection).

## Connection to RL & ML
-   **Evolutionary Optimization**: Used to evolve neural network weights or architectures.
-   **Self-play Stability**: An ESS is a very strong form of stability in self-play. If a self-play algorithm converges to an ESS, it is robust against "mutant" strategies that might try to exploit the learned policy.

## Interview Point
-   **ESS vs Nash**: Every ESS is a Nash Equilibrium, but not every Nash Equilibrium is an ESS. ESS is a **refined** equilibrium concept that requires stability against small perturbations (invasions).
