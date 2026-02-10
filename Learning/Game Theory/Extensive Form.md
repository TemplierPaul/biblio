# Extensive Form Games

Extensive form games model situations where players move sequentially, capturing the timing of actions and the information available at each step.

## Components
1.  **Game Tree**: A directed tree where nodes represent states of the game.
2.  **Players**: Assigned to each non-terminal node.
3.  **Actions**: Edges leading from a node.
4.  **Payoffs**: Defined at each terminal (leaf) node for all players.
5.  **Information Sets**: Groups of nodes that a player cannot distinguish between when it is their turn to move.

## Information Structures
-   **Perfect Information**: Every information set contains exactly one node (players know everything that happened before).
-   **Imperfect Information**: Some information sets contain multiple nodes (e.g., card games where you don't know the opponent's hand).
-   **Incomplete Information**: Players don't know the exact payoffs or types of other players (often transformed into imperfect information using "Nature" moves).

## Solving Extensive Form Games
-   **Backward Induction**: Starting from the leaves and working up to the root to find the optimal move at each node. Applicable to finite games of perfect information.
-   **Subgame Perfect Equilibrium (SPE)**: A refinement of Nash Equilibrium that requires the strategy to be a Nash Equilibrium in every subgame of the original game.

## Connection to Reinforcement Learning
Extensive form games are the natural framework for most competitive RL environments (Chess, Poker, Go). Methods like MCTS (Monte Carlo Tree Search) and Counterfactual Regret Minimization (CFR) operate primarily on these structures.

## Interview Relevance
-   **Game Tree Complexity**: Difference between state space and game tree size.
-   **Information Sets**: Why they make games harder to solve than perfect information games.
