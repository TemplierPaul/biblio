# Monte Carlo Tree Search (MCTS)

MCTS is a best-first search algorithm used to find optimal decisions in complex domains (like Go, Chess) by building a search tree according to the results of random (or guided) simulations.

## The 4 Core Phases
MCTS iterates through these four phases until a time/memory budget is reached:

1.  **Selection**: Start from root $R$ and traverse down the tree to a leaf node $L$. At each step, select the child that maximizes a selection policy (e.g., UCT).
2.  **Expansion**: If $L$ is not a terminal state, create one or more child nodes.
3.  **Simulation (Rollout)**: Run a simulation from the new node to a terminal state to produce an outcome (e.g., Win/Loss).
    -   *Classic MCTS*: Uses random moves.
    -   *AlphaZero*: Uses a Value Network evaluation instead of a rollout.
4.  **Backpropagation**: Update the statistics (visit count $N$, total value $W$) of all nodes on the path from the new node back to the root.

## Selection Policies

### UCT (Upper Confidence Bounds for Trees)
The standard policy balances **Exploitation** (high win rate) and **Exploration** (low visit count).
$$UCT(s, a) = \frac{W(s, a)}{N(s, a)} + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$
-   $W/N$: Average value (Exploitation).
-   Square root term: Exploration boost for rarely visited nodes.
-   $c$: Exploration constant (often $\sqrt{2}$).

### PUCT (Predictor + UCT)
Used in **AlphaZero / MuZero**. It incorporates a prior probability $P(s, a)$ from a policy network.
$$PUCT(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
-   $Q(s, a)$: Mean action value.
-   $P(s, a)$: Prior probability from the neural network.
-   Allows the search to focus on moves the neural network "thinks" are good, drastically pruning the search space compared to random exploration.

## MCTS vs Minimax
-   **Minimax**: Explores the full width of the tree to a fixed depth. Good for tactical, short-horizon calculations.
-   **MCTS**: Grows an **asymmetric** tree. It explores promising paths much deeper than poor paths. Better for strategic, long-horizon games with high branching factors (like Go).

## Interview Relevance
-   **Asymmetry**: Why is MCTS better for Go than Minimax? (Branching factor too high for Minimax; MCTS focuses selectively).
-   **Rollouts vs Value Nets**: How AlphaZero removed the need for hand-crafted rollout policies (using a learned value function $v \approx z$).
