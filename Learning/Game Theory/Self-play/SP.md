# Vanilla Self-Play (SP)

## Definition
Vanilla Self-Play is the simplest form of self-play where a learning agent interacts with its current or recent self to improve its policy.

## Mechanism
1.  **Initialize**: Start with a random policy $\pi_0$.
2.  **Iterate**: 
    -   Generate data by playing $\pi_t$ against $\pi_t$.
    -   Update $\pi_t$ to $\pi_{t+1}$ using RL (e.g., PPO, DQN).

## Key Examples
-   **TD-Gammon (1992)**: Used temporal difference learning and self-play to reach expert level in Backgammon.
-   **AlphaGo / AlphaZero**: Combined MCTS with self-play and deep learning to solve Go, Chess, and Shogi.

## Strengths
-   No need for expert human data (in AlphaZero).
-   Scaleable: Performance scales with compute and data generation.

## Limitations
-   **Cycling**: In non-transitive games (transitivity $\neq$ monotonic improvement), SP can circle around strategies without converging (e.g., Rock $\to$ Paper $\to$ Scissors $\to$ Rock).
-   **Narrowness**: May converge to a specialized "cluster" of strategies that is easily exploitable by a different strategy outside the self-play distribution.
