# Neural Population Learning (NeuPL)

## Definition
NeuPL is a framework designed to learn a population of diverse policies within a **single conditional neural network**. It aims to solve the inefficiency of learning new strategies from scratch and the lack of skill transfer.

## Core Idea
-   Instead of training $N$ separate networks for $N$ strategies (as in standard population-based methods like PSRO), NeuPL uses a conditional network $\pi(a|s, i)$ where $i$ is a strategy index.
-   **Transitive Learning / Transfer**: New strategies $i+1$ can build upon the features and skills learned by strategy $i$.

## Key Mechanism: NeuPL Matrix
-   A strategy $i$ is defined by what it plays against.
-   NeuPL builds a hierarchy of responses.
-   Supports general-sum games and $n$-player scenarios.

## NeuPL-JPSRO (Joint PSRO)
-   Combines NeuPL with Joint-PSRO for general-sum games.
-   Allows convergence to **Coarse Correlated Equilibrium (CCE)**.
-   Efficiently handles large populations by leveraging transfer learning.

## Advantages
-   **Memory Efficiency**: One network for many policies.
-   **Sample Efficiency**: Exploits shared representations.
-   **Diversity**: Explicitly seeks to expand the repertoire of strategies the agent can respond to.
