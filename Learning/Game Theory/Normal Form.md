# Normal Form Games

Also known as **Strategic Form Games**, this is a model of a game where players choose their actions simultaneously.

## Components
A normal form game is defined by:
1.  **Players**: A finite set $N = \{1, 2, \dots, n\}$.
2.  **Strategies**: For each player $i$, a set of available actions $S_i$.
3.  **Payoffs**: A function $u_i: S \to \mathbb{R}$ for each player, where $S = S_1 \times S_2 \times \dots \times S_n$ is the set of all possible strategy profiles.

## Representation: Payoff Matrix
For 2-player games, this is typically represented as a matrix where:
-   Rows represent Player 1's strategies.
-   Columns represent Player 2's strategies.
-   Cells contain tuples $(u_1, u_2)$ representing the payoff for each player.

| P1 \ P2 | Left | Right |
| :--- | :---: | :---: |
| **Up** | (3, 2) | (0, 0) |
| **Down** | (0, 0) | (2, 3) |

## Key Concepts
-   **Pure Strategy**: A deterministic choice of one action.
-   **Mixed Strategy**: A probability distribution over pure strategies.
-   **Dominant Strategy**: A strategy that is better than any other strategy for a player, regardless of what the opponents do.
-   **Zero-Sum Game**: A game where $\sum u_i = 0$ for all strategy profiles (total gain = total loss).

## Interview Tips
-   Be prepared to find **Strictly Dominant Strategies** first to simplify a matrix.
-   Understand that any finite normal form game has at least one **Nash Equilibrium** in mixed strategies.
