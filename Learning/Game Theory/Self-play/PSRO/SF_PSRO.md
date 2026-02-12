# SF-PSRO (Simulation-Free PSRO)

## Definition
Simulation-Free PSRO (SF-PSRO) describes a family of PSRO variants that eliminate the **Game Simulation (GS)** phase—the computationally expensive step of filling the full meta-payoff matrix. Instead, they rely on "sketchy" or implicit payoff matrices built during best-response training.

## Key Innovation
Standard PSRO requires $O(M^N)$ simulations to evaluate a meta-game with $M$ strategies and $N$ players. SF-PSRO removes this bottleneck by:
1.  **Dynamic Window**: maintaining a fixed-size window of active strategies.
2.  **Sketchy Matrix**: updating payoff estimates only from training interactions.
3.  **Nash Clustering**: intelligently selecting which strategies to evict from the window.

## Core Components
- **Dynamic Window**: Keeps the strategy population size constant $(|X^w| \approx 30)$.
- **Nash Clustering**: Identifies the "least useful" strategy to remove when the window is full (based on Relative Population Performance).
- **Filling**: Updates the sparse payoff matrix using fresh data from the Best Response Solver (BRS).

## Why Use It?
- **Speed**: Removes the primary runtime bottleneck of PSRO.
- **Scalability**: Enables scaling to games with more players or iterations where full evaluation is impossible.
- **Performance**: Dynamic Window + Nash Clustering maintains a high-quality population.

> Detailed implementation: [[SF_PSRO_detailed]]
