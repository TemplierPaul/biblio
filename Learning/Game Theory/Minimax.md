# Minimax Algorithm

## Definition
Minimax is a decision rule for minimizing the maximum possible loss in zero-sum, perfect-information games. It assumes the opponent plays optimally to maximize their gain (minimize your utility).

## Core Idea

### Zero-Sum Games
- Player 1's gain = Player 2's loss
- Competitive: $u_1(s) + u_2(s) = 0$
- Examples: Chess, Tic-Tac-Toe, Checkers

### Minimax Principle
**Maximizing player** (you):
$$\max_{a \in A} \min_{s' \in S(a)} v(s')$$
- Choose action that **maximizes** the **minimum** value over opponent responses

**Minimizing player** (opponent):
$$\min_{a \in A} \max_{s' \in S(a)} v(s')$$
- Choose action that **minimizes** the **maximum** value over your responses

**Optimal value**: Converges to same value at equilibrium (minimax theorem)

## Algorithm (Two-Player, Turn-Based)

### Recursive Definition
```
function minimax(state, depth, isMaximizingPlayer):
    if terminal(state) or depth == 0:
        return evaluate(state)

    if isMaximizingPlayer:
        maxEval = -∞
        for each child in children(state):
            eval = minimax(child, depth-1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = +∞
        for each child in children(state):
            eval = minimax(child, depth-1, True)
            minEval = min(minEval, eval)
        return minEval
```

### Tree Exploration
1. **Maximizing node**: Choose max of children
2. **Minimizing node**: Choose min of children
3. **Terminal node**: Return game outcome
4. **Depth limit**: Return heuristic evaluation

## Example: Tic-Tac-Toe

```
        O's turn (MIN)
       /     |      \
     /       |       \
   X       X   O     O
  (MAX)   (MAX)     (MAX)
  / | \   / | \     / | \
 -1 0 +1 ...

Backpropagate: MIN selects minimum of children's values
```

**Result**: Tic-Tac-Toe is a draw with optimal play

## Complexity

### Time Complexity
$$O(b^d)$$
- $b$: Branching factor (average number of legal moves)
- $d$: Depth of search

**Example** (Chess):
- Branching factor: ~35
- Depth 10: $35^{10} \approx 2.8 \times 10^{15}$ nodes

**Problem**: Exponential explosion - infeasible for deep search

### Space Complexity
$$O(bd)$$
- DFS-style search: Only store path from root to current node

## Optimizations

### 1. Alpha-Beta Pruning
**Idea**: Prune branches that can't affect final decision

**Maintain**:
- $\alpha$: Best value for maximizer found so far
- $\beta$: Best value for minimizer found so far

**Prune when**: $\alpha \geq \beta$

**Savings**: $O(b^{d/2})$ in best case (good move ordering)

**Example**:
```
MAX node: α = -∞
  Child 1: returns 3, α = 3
  Child 2:
    MIN node: β = +∞
      Grandchild 1: returns 2, β = 2
      Grandchild 2: prune (α=3 ≥ β=2, no need to explore)
```

### 2. Move Ordering
**Better pruning** with good move ordering:
- Try likely-best moves first (captures, checks in chess)
- Previous iteration best move (iterative deepening)
- Killer moves, history heuristic

### 3. Transposition Tables
**Cache** previously evaluated positions:
- Same position can occur via different move orders
- Store (position hash → value, depth, best move)
- Massive speedup in practice

### 4. Iterative Deepening
Search depths 1, 2, 3, ... until time limit:
- Combines DFS space efficiency with BFS completeness
- Move ordering benefits from shallow searches
- Anytime algorithm (always has best move so far)

### 5. Quiescence Search
**Problem**: Depth limit during tactical exchange gives bad evaluation

**Solution**: Continue search for "quiet" position
- Extend search for captures, checks
- Stop when position is stable

## Evaluation Functions

### Terminal States
- Win: $+\infty$ (or +1)
- Loss: $-\infty$ (or -1)
- Draw: 0

### Non-Terminal (Heuristic)
**Chess examples**:
- Material: $+9$ (Queen), $+5$ (Rook), $+3$ (Bishop/Knight), $+1$ (Pawn)
- Position: Center control, king safety, pawn structure
- Mobility: Number of legal moves

**Requirements**:
- Fast to compute (evaluated millions of times)
- Accurate estimate of win probability

## Minimax Theorem (von Neumann)

**Statement**: In zero-sum games, optimal strategies satisfy:
$$\max_{s_1} \min_{s_2} u(s_1, s_2) = \min_{s_2} \max_{s_1} u(s_1, s_2) = v^*$$

**Implications**:
- **Value of the game** $v^*$ exists
- **Saddle point**: Optimal strategies form Nash equilibrium
- Player 1 can guarantee $\geq v^*$, Player 2 can guarantee $\leq v^*$

## Minimax vs MCTS

| Aspect | Minimax | MCTS |
|--------|---------|------|
| **Exploration** | Full-width to depth | Asymmetric, focused |
| **Branching** | Best for low $b$ | Handles high $b$ |
| **Evaluation** | Heuristic function | Simulation/network |
| **Anytime** | No (needs depth) | Yes (improves with time) |
| **Best for** | Chess, Checkers | Go, large $b$ games |

**Why MCTS for Go?**
- Branching factor ~250 (vs ~35 for chess)
- Minimax can't search deep enough
- MCTS focuses on promising variations

## Limitations

### 1. Exponential Complexity
- Can't search to game end for complex games
- Relies on heuristic evaluation (imperfect)

### 2. Perfect Information Required
- Needs full game tree
- Doesn't handle uncertainty (poker, etc.)

### 3. Horizon Effect
- Depth limit causes tactical blindness
- Opponent can push bad consequences beyond horizon

### 4. Two-Player Zero-Sum Only
- Doesn't generalize to multi-player
- Doesn't handle general-sum payoffs

## Interview Relevance

**Common Questions**:
1. **How does minimax work?** Maximize worst-case outcome by assuming optimal opponent
2. **Complexity?** $O(b^d)$ time, $O(bd)$ space
3. **Alpha-beta pruning?** Prune branches that can't affect decision; $O(b^{d/2})$ best case
4. **Why not use for Go?** Branching factor too large (~250 vs ~35 for chess)
5. **Minimax theorem?** Max-min = min-max at equilibrium
6. **Evaluation function?** Heuristic to estimate position value (material, position, etc.)
7. **Minimax vs MCTS?** Minimax: full-width, low branching; MCTS: selective, high branching
8. **Transposition table?** Cache evaluated positions (same position via different moves)
9. **Iterative deepening?** Search depths 1, 2, 3... until time limit (anytime, better move ordering)

**Key Concepts**:
- **Minimax principle**: Maximize minimum payoff
- **Alpha-beta**: Prune irrelevant branches
- **Evaluation**: Heuristic for non-terminal states
- **Zero-sum**: One player's gain = other's loss

**Key Insight**: Minimax is optimal for small perfect-information zero-sum games, but exponential complexity limits applicability - modern games use alpha-beta pruning, transposition tables, and (for high branching) MCTS instead.
