# Fictitious Self-Play (FSP)

## Definition
Fictitious Self-Play is a machine learning implementation of **Fictitious Play**, where agents choose a best response to the **average strategy** of their opponents.

## Fictitious Play (Classical)
-   Agents track the empirical frequency of their opponent's actions.
-   Action $a_{t+1}$ is the best response to the opponent's historical average policy.
-   **Guarantees**: Converges to Nash Equilibrium in zero-sum games and certain other classes (e.g., potential games).

## Fictitious Self-Play (FSP)
-   Sample-based version of Fictitious Play.
-   Uses two datasets:
    -   **Reinforcement Learning**: To learn a best response to the current average strategy.
    -   **Supervised Learning**: To track the average strategy by fitting a model to the agent's historical actions.

## Neural Fictitious Self-Play (NFSP)
-   Integrates deep neural networks into FSP.
-   **RL agent**: DQN or similar to find the best response.
-   **SL agent**: A network trained to predict the actions taken by the RL agent, representing the average policy.
-   **Policy Selection**: During data generation, the agent acts according to a mix of the RL policy and the SL policy (e.g., $\epsilon$-greedy style).

## Interview Relevance
-   **Convergence**: Why is FSP more stable than Vanilla SP? (It targets the average strategy rather than just the latest, mitigating cycling).
-   **NFSP**: The first end-to-end RL method to solve large extensive-form games like Limit Texas Hold'em without hand-crafted features.
