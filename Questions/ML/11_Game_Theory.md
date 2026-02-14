# Game Theory & Algorithms - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: Game Theory Foundations]]
  - [[#What's a normal form game? Components?]]
  - [[#What's a payoff matrix?]]
  - [[#Pure vs mixed strategy Nash equilibrium?]]
  - [[#Does every finite game have a Nash equilibrium?]]
  - [[#What's an extensive form game?]]
  - [[#Difference between perfect and imperfect information?]]
  - [[#What are information sets?]]
  - [[#How to solve perfect information games (backward induction)?]]
  - [[#What's a subgame perfect equilibrium?]]
  - [[#Nash equilibrium vs social optimum - are they the same?]]
- [[#Part 2: Social Dilemmas]]
  - [[#What's the Nash equilibrium of prisoner's dilemma?]]
  - [[#Is Nash equilibrium Pareto optimal in PD?]]
  - [[#What's a dominant strategy?]]
  - [[#What makes it a "dilemma"?]]
  - [[#How does iterated PD change the game?]]
  - [[#What's tit-for-tat strategy?]]
  - [[#Real-world examples of prisoner's dilemmas?]]
  - [[#How to achieve cooperation in PD?]]
  - [[#What's the n-player version (public goods game)?]]
- [[#Part 3: Classical Game Algorithms]]
  - [[#What's the minimax principle?]]
  - [[#How does minimax algorithm work (recursive)?]]
  - [[#What's alpha-beta pruning and how much does it save?]]
  - [[#What's the complexity of minimax?]]
  - [[#When is minimax optimal?]]
  - [[#Explain the four phases of MCTS (Selection, Expansion, Simulation, Backup)]]
  - [[#What's UCT formula?]]
  - [[#What's PUCT and how does it use neural networks?]]
  - [[#Minimax vs MCTS - when to use each?]]
  - [[#Why is MCTS better for Go than minimax?]]
- [[#Part 4: AlphaZero]]
  - [[#What are the two heads of AlphaZero's network?]]
  - [[#How does PUCT use the policy network?]]
  - [[#Why doesn't AlphaZero use rollouts (unlike classic MCTS)?]]
  - [[#What's the training loop (self-play → train → repeat)?]]
  - [[#What's the loss function (three components)?]]
  - [[#How does value network replace rollouts?]]
  - [[#What's the difference between AlphaGo and AlphaZero?]]
  - [[#Why is AlphaZero considered "tabula rasa"?]]
  - [[#What are limitations (perfect info, two-player, etc.)?]]
  - [[#How does temperature affect move selection?]]
- [[#Part 5: PSRO & Population-Based Methods]]
  - [[#What's the PSRO algorithm (4 steps)?]]
  - [[#What's an empirical game?]]
  - [[#What's a best response oracle?]]
  - [[#How does PSRO converge to Nash equilibrium?]]
  - [[#Difference between PSRO and double oracle?]]
  - [[#What's JPSRO and how does it differ from PSRO?]]
  - [[#What's Correlated Equilibrium (CE)?]]
  - [[#What's Coarse Correlated Equilibrium (CCE)?]]
  - [[#Hierarchy: Nash ⊆ CE ⊆ CCE - explain]]
  - [[#Why use CCE for general-sum games?]]
  - [[#How many BRs needed per iteration (PSRO vs JPSRO)?]]
- [[#Part 6: Neural Population Learning]]
  - [[#What's the key innovation of NeuPL vs PSRO?]]
  - [[#How does conditional network work: π(a|s,i)?]]
  - [[#What's the memory advantage (O(1) vs O(N))?]]
  - [[#What's transfer learning in NeuPL?]]
  - [[#NeuPL vs NeuPL-JPSRO differences?]]
  - [[#What equilibrium does NeuPL converge to?]]
  - [[#When to use NeuPL vs standard PSRO?]]
- [[#Part 7: Fictitious Self-Play]]
  - [[#What's fictitious play (classical)?]]
  - [[#How does FSP use RL and SL?]]
  - [[#What's NFSP (Neural FSP)?]]
  - [[#What are the two networks and two buffers?]]
  - [[#Why does FSP converge to Nash in some games?]]
  - [[#FSP vs vanilla self-play - why more stable?]]
  - [[#When to use FSP vs PSRO?]]
- [[#Part 8: Counterfactual Regret Minimization]]
  - [[#What games is CFR designed for (imperfect info, extensive form)?]]
  - [[#What's an information set?]]
  - [[#What's counterfactual value?]]
  - [[#Explain regret matching strategy update]]
  - [[#Why use average strategy (not current strategy)?]]
  - [[#What's CFR+ and how does it improve vanilla CFR?]]
  - [[#What's Monte Carlo CFR (sampling)?]]
  - [[#What's Deep CFR?]]
  - [[#Complexity of CFR per iteration?]]
  - [[#Why does CFR converge to Nash in two-player zero-sum?]]

---

## Part 1: Game Theory Foundations

### What's a normal form game? Components?

**Normal form** (strategic form): Model where players choose actions **simultaneously**.

**Components**:
1. **Players**: Finite set $N = \{1, 2, \ldots, n\}$
2. **Strategies**: Action sets $S_i$ for each player $i$
3. **Payoffs**: Utility functions $u_i: S \to \mathbb{R}$ where $S = S_1 \times \cdots \times S_n$

**Representation** (2-player): Payoff matrix
- Rows: Player 1's actions
- Columns: Player 2's actions
- Cells: $(u_1, u_2)$ payoffs

**Example** (Coordination):
```
         Left    Right
Up       (2,2)   (0,0)
Down     (0,0)   (1,1)
```

### What's a payoff matrix?

**Payoff matrix**: Tabular representation of normal form game.

**Format** (2-player):
- **Rows**: Player 1 (row player) actions
- **Columns**: Player 2 (column player) actions
- **Entries**: Tuple $(u_1, u_2)$ - payoffs for each player

**Example** (Prisoner's Dilemma):
```
         Cooperate  Defect
Coop     (-1,-1)    (-3,0)
Defect   (0,-3)     (-2,-2)
```

**Reading**: If P1 plays Cooperate, P2 plays Defect → P1 gets -3, P2 gets 0.

### Pure vs mixed strategy Nash equilibrium?

**Pure strategy NE**: Each player plays single action with probability 1
- Example: (Defect, Defect) in Prisoner's Dilemma
- Deterministic

**Mixed strategy NE**: Players randomize over actions
- Example: Rock-Paper-Scissors → (1/3, 1/3, 1/3) for each player
- Probability distribution over pure strategies

**Finding mixed NE**: Make opponent **indifferent** between their actions
- Set expected payoffs equal, solve for probabilities

### Does every finite game have a Nash equilibrium?

**Yes** - **Nash's Theorem** (1950):

Every finite game (finite players, finite actions) has **at least one Nash equilibrium in mixed strategies**.

**Key points**:
- May be only **mixed**, no pure NE (e.g., matching pennies)
- May have **multiple** NE (coordination games)
- Guarantees existence, not uniqueness

**Proof**: Uses Brouwer fixed-point theorem (beyond scope).

### What's an extensive form game?

**Extensive form**: Game tree representing **sequential** decision-making.

**Components**:
1. **Game tree**: Directed tree, nodes = game states
2. **Players**: Assigned to decision nodes
3. **Actions**: Edges from nodes
4. **Payoffs**: At terminal (leaf) nodes
5. **Information sets**: Group nodes player can't distinguish

**Example**: Chess, poker, tic-tac-toe

**Contrast normal form**: Sequential vs simultaneous moves.

### Difference between perfect and imperfect information?

**Perfect information**: Every player knows **complete history** when making decision
- Every information set is singleton (one node)
- Examples: Chess, Go, Tic-Tac-Toe
- Can solve via backward induction

**Imperfect information**: Some information is **hidden**
- Information sets contain multiple nodes (indistinguishable)
- Examples: Poker (hidden cards), Battleship
- Harder to solve (can't use backward induction directly)

**Incomplete information**: Players don't know payoffs/types (Bayesian games, not same as imperfect info).

### What are information sets?

**Information set**: Group of game tree nodes that player **cannot distinguish** when making a decision.

**Formal**: Set of nodes $I$ where:
- Same player acts at all nodes in $I$
- Player doesn't know which node in $I$ they're at
- Must choose same action at all nodes in $I$

**Example** (Poker):
- Two nodes: opponent has Ace, opponent has King
- You can't tell which → nodes in same information set
- Must choose same action (fold/call/raise) for both

**Perfect info**: Every information set has 1 node.

### How to solve perfect information games (backward induction)?

**Backward induction**: Work backwards from terminal nodes.

**Algorithm**:
1. Start at **terminal nodes** (leaves)
2. Assign payoffs from game definition
3. Move to **parent nodes**:
   - If parent is Player i's turn, choose action maximizing $u_i$
   - Assign that payoff to parent
4. Repeat until **root**

**Result**: Optimal strategy (subgame perfect equilibrium).

**Example** (simple game):
```
      P1
     /  \
    L    R
   /      \
  2,1    P2
        /  \
       l    r
      /      \
    3,0      0,2
```
- P2's turn: choose r (payoff 2 > 0)
- P1's turn: choose L (payoff 2 > 0)
- Solution: (L, r) with payoff (2,1)

**Limitation**: Only works for perfect information (no information sets with >1 node).

### What's a subgame perfect equilibrium?

**Subgame Perfect Equilibrium (SPE)**: Refinement of Nash equilibrium for extensive form games.

**Definition**: Strategy profile that is Nash equilibrium in **every subgame** (including the full game).

**Key property**: No player wants to deviate at **any** point in the game (not just at start).

**Why needed**: Rules out non-credible threats

**Example** (Entry game):
- Entrant: Enter or Stay Out
- Incumbent: Fight or Accommodate
- Payoffs: Fight (-1,-1), Accommodate (1,1), Stay Out (0,2)

**Nash**: (Enter, Fight) - but Fight is non-credible threat
**SPE**: (Enter, Accommodate) - only credible strategies

**Finding SPE**: Backward induction (in perfect info games).

### Nash equilibrium vs social optimum - are they the same?

**No** - Nash equilibrium ≠ Social optimum in general.

**Nash equilibrium**: No player wants to deviate unilaterally (individual rationality)

**Social optimum**: Maximizes sum of utilities (collective rationality)

**Counterexample** (Prisoner's Dilemma):
- Nash: (Defect, Defect) with payoff (-2, -2), sum = -4
- Social optimum: (Cooperate, Cooperate) with payoff (-1, -1), sum = -2

**Key insight**: Individual incentives don't always align with collective welfare.

**When they match**: Some games (e.g., coordination with unique optimum).

---

## Part 2: Social Dilemmas

### What's the Nash equilibrium of prisoner's dilemma?

**(Defect, Defect)** - both players defect.

**Why**:
- If opponent cooperates: Defect gives 0 > -1 (Cooperate)
- If opponent defects: Defect gives -2 > -3 (Cooperate)
- **Dominant strategy**: Defect regardless of opponent

**Verification**: Neither player can improve by deviating:
- P1: Can't improve from (D,D) by switching to C (-3 < -2)
- P2: Can't improve from (D,D) by switching to C (-3 < -2)

### Is Nash equilibrium Pareto optimal in PD?

**No** - Nash equilibrium is **Pareto dominated**.

**Pareto optimal**: Can't make any player better off without making another worse off.

**PD outcomes**:
- (D, D): (-2, -2) - **Nash equilibrium**
- (C, C): (-1, -1) - **Pareto optimal** (both better than Nash!)

**Pareto dominance**: (C, C) is better for **both** players than (D, D)

**The dilemma**: Individually rational to defect, but collectively better to cooperate.

### What's a dominant strategy?

**Dominant strategy**: Strategy that is **best response to all opponent strategies**.

**Formal**: Strategy $s_i^*$ is dominant for player $i$ if:
$$u_i(s_i^*, s_{-i}) \geq u_i(s_i, s_{-i}) \quad \forall s_i, s_{-i}$$

**Strictly dominant**: Strict inequality (always better).

**In Prisoner's Dilemma**: Defect is strictly dominant for both players
- Better than Cooperate whether opponent Cooperates or Defects

**Note**: Not all games have dominant strategies (e.g., coordination games).

### What makes it a "dilemma"?

**Dilemma**: **Individual rationality** (Nash) conflicts with **collective rationality** (Pareto optimum).

**Components**:
1. **Dominant strategy**: Defect (individual incentive)
2. **Nash equilibrium**: (D, D) with payoff (-2, -2)
3. **Pareto optimum**: (C, C) with payoff (-1, -1)
4. **Gap**: Nash is worse for everyone than cooperation

**Insight**: What's rational individually leads to collectively bad outcome.

**Broader implications**: Many real-world problems have this structure (climate change, arms race, etc.).

### How does iterated PD change the game?

**Iterated PD**: Play prisoner's dilemma **repeatedly** (finite or infinite rounds).

**Changes**:
1. **Future matters**: Today's actions affect tomorrow's outcomes (reputation)
2. **Cooperation emerges**: Conditional strategies like tit-for-tat can sustain cooperation
3. **Folk theorem**: Many outcomes (including cooperation) sustainable as equilibria

**Key difference**:
- **One-shot**: Defect dominant
- **Repeated**: Cooperate if you value future (discount factor δ high enough)

**Intuition**: "I'll cooperate if you do, otherwise I'll punish you" becomes credible threat.

### What's tit-for-tat strategy?

**Tit-for-tat**:
1. **Round 1**: Cooperate
2. **Round t**: Do whatever opponent did in round t-1 (copy opponent's last move)

**Properties**:
- **Nice**: Never defects first
- **Retaliatory**: Punishes defection immediately
- **Forgiving**: Returns to cooperation if opponent does
- **Simple**: Easy to understand and implement

**Success**: Won Axelrod's tournament (1980s) - very effective in iterated PD

**Vulnerability**: Noise can cause defection spirals (improved: generous tit-for-tat).

### Real-world examples of prisoner's dilemmas?

1. **Climate change**: Reduce emissions (C) vs pollute (D)
   - Individual: Better to pollute (save costs)
   - Collective: Everyone reducing emissions is best

2. **Arms race**: Disarm (C) vs arm (D)
   - Nash: Both arm (expensive, dangerous)
   - Optimal: Both disarm (peaceful, cheap)

3. **Overfishing**: Limit catch (C) vs overfish (D)
   - Individual: Catch more
   - Collective: Sustainable fishing

4. **Free riding on public goods**: Contribute (C) vs don't (D)
   - Individual: Don't contribute, enjoy benefits
   - Collective: Everyone contributes

5. **Code of silence** (original story): Stay silent (C) vs betray (D)

### How to achieve cooperation in PD?

1. **Repeated interaction**: Iterated PD with reputation
   - Conditional cooperation (tit-for-tat)
   - Requires high enough discount factor

2. **Communication**: Pre-play discussion
   - Limited without enforcement
   - But can establish norms

3. **Binding contracts**: Enforceable agreements
   - Changes payoff structure
   - Requires third-party enforcement

4. **Punishment mechanisms**: Sanctions for defectors
   - Social punishment, fines

5. **Changing preferences**: Altruism, social norms
   - Internalize externalities
   - Care about opponent's payoff

6. **Incomplete information**: Uncertainty about opponent's type
   - Some players might be "always cooperate"
   - Pooling equilibria possible

### What's the n-player version (public goods game)?

**Public Goods Game**: n-player extension of PD.

**Setup**:
- Each player $i$ contributes $c_i \in [0, C]$
- Total public good: $G = \sum_i c_i$
- Payoff: $u_i = \alpha G - c_i$
- Parameters: $\alpha < 1$ (individual return) but $n\alpha > 1$ (social return)

**Dilemma**:
- **Individual incentive**: Contribute 0 (free ride)
- **Collective optimum**: Everyone contributes C

**Nash equilibrium**: Everyone contributes 0 (tragedy of the commons)

**Examples**: Donations, volunteer work, team projects, environmental protection.

---

## Part 3: Classical Game Algorithms

### What's the minimax principle?

**Minimax**: Choose action that **maximizes** your **minimum** payoff (worst-case optimization).

**For maximizing player**:
$$\max_{a} \min_{opponent} value(a, opponent)$$

**For minimizing player**:
$$\min_{a} \max_{opponent} value(a, opponent)$$

**Assumption**: Opponent plays **optimally** to hurt you.

**Zero-sum games**: Your gain = opponent's loss, so:
$$\max \min = \min \max = v^*$$ (game value)

**Intuition**: Secure the best possible worst-case outcome.

### How does minimax algorithm work (recursive)?

**Recursive definition**:

```
function minimax(state, depth, isMaximizing):
    if terminal(state) or depth == 0:
        return evaluate(state)

    if isMaximizing:
        maxEval = -∞
        for child in children(state):
            eval = minimax(child, depth-1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = +∞
        for child in children(state):
            eval = minimax(child, depth-1, True)
            minEval = min(minEval, eval)
        return minEval
```

**Process**:
1. **Leaf nodes**: Return evaluation
2. **MAX nodes**: Choose max of children
3. **MIN nodes**: Choose min of children
4. **Propagate** values up tree

**Returns**: Best achievable value assuming optimal opponent.

### What's alpha-beta pruning and how much does it save?

**Alpha-beta pruning**: Optimization that prunes branches that can't affect final decision.

**Maintain**:
- $\alpha$: Best value for MAX found so far
- $\beta$: Best value for MIN found so far

**Prune when**: $\alpha \geq \beta$
- MAX won't choose this branch (MIN can force worse)

**Savings**:
- **Worst case**: No pruning, still $O(b^d)$
- **Best case**: $O(b^{d/2})$ with perfect move ordering
- **Practical**: Often 2-10x speedup

**Key**: Explore best moves first (move ordering critical).

### What's the complexity of minimax?

**Time**: $O(b^d)$
- $b$: Branching factor (average legal moves)
- $d$: Depth of search

**Space**: $O(bd)$
- DFS: Store path from root to leaf only

**Example** (Chess to depth 10):
- $b \approx 35$, $d = 10$
- $35^{10} \approx 2.8 \times 10^{15}$ nodes

**Problem**: Exponential explosion - can't search to game end for complex games.

**Solutions**: Alpha-beta, transposition tables, iterative deepening, evaluation functions.

### When is minimax optimal?

**Optimal when**:
1. **Two-player zero-sum game**
2. **Perfect information** (know complete state)
3. **Deterministic** (no randomness)
4. **Can search to terminal states** or have accurate evaluation

**Guarantees**: Finds game-theoretic optimal play (minimax value).

**Limitations**:
- Not optimal if can't search deep enough (relies on heuristic eval)
- Doesn't handle uncertainty (probabilistic games)
- Only two-player zero-sum

**Used successfully**: Chess (with alpha-beta), Checkers (solved!), Othello.

### Explain the four phases of MCTS (Selection, Expansion, Simulation, Backup)

**1. Selection**: Start at root, traverse tree using **UCT policy**
$$UCT(s,a) = \frac{W(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$$
- Choose child with highest UCT
- Balance exploitation (high $W/N$) and exploration (low $N$)
- Stop at leaf node

**2. Expansion**: If leaf is non-terminal, **add child node** to tree
- Expand one or more children
- Add to tree for future selection

**3. Simulation (Rollout)**: From new node, **play out** to terminal state
- Classic MCTS: Random moves
- AlphaZero: Use value network (no rollout)

**4. Backup**: **Propagate** result back to root
- Update visit counts: $N(s,a) \leftarrow N(s,a) + 1$
- Update values: $W(s,a) \leftarrow W(s,a) + result$
- Update all nodes on path

**Repeat** until budget exhausted → select most visited action.

### What's UCT formula?

**UCT** (Upper Confidence Bound for Trees):

$$UCT(s, a) = \underbrace{\frac{W(s,a)}{N(s,a)}}_{\text{exploitation}} + \underbrace{c \sqrt{\frac{\ln N(s)}{N(s,a)}}}_{\text{exploration}}$$

**Components**:
- $W(s,a)$: Total value (sum of rewards from action $a$)
- $N(s,a)$: Visit count for action $a$ from state $s$
- $N(s)$: Visit count for state $s$
- $c$: Exploration constant (typically $\sqrt{2}$)

**Intuition**:
- High $W/N$: Exploitation (action has high average value)
- High exploration term: Exploration (action rarely visited)
- Logarithm: Grows slowly (diminishing exploration bonus)

**Convergence**: UCT → optimal policy as visits → ∞ (proven).

### What's PUCT and how does it use neural networks?

**PUCT** (Predictor + UCT): UCT enhanced with neural network prior.

$$PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

**Components**:
- $Q(s,a)$: Mean action value (like $W/N$ in UCT)
- $P(s,a)$: **Prior probability** from policy network
- $c_{puct}$: Exploration constant

**Key difference from UCT**: Uses learned $P(s,a)$ to guide exploration

**Effect**:
- Network thinks move is good → explore more (high $P$)
- Network thinks move is bad → explore less (low $P$)
- **Dramatically reduces search space** vs random exploration

**Used in**: AlphaZero, MuZero

### Minimax vs MCTS - when to use each?

**Minimax**:
- ✅ Low branching factor ($b \approx 10$-50)
- ✅ Good evaluation function available
- ✅ Tactical games (Chess, Checkers)
- ❌ Struggles with $b > 100$

**MCTS**:
- ✅ High branching factor ($b > 100$)
- ✅ Hard to evaluate positions
- ✅ Strategic games (Go: $b \approx 250$)
- ✅ Anytime algorithm (improves with time)

**Key difference**:
- **Minimax**: Uniform exploration (full width to depth)
- **MCTS**: Asymmetric tree (focuses on promising variations)

**Modern**: AlphaZero uses MCTS + neural networks (best of both).

### Why is MCTS better for Go than minimax?

**Go characteristics**:
- **Huge branching factor**: ~250 legal moves (vs ~35 for chess)
- **Deep game**: ~150 moves
- **Hard to evaluate**: Positional evaluation very difficult

**Minimax problems**:
- Can't search deep: $250^{10}$ is astronomical
- Needs evaluation function: Go positions hard to evaluate

**MCTS advantages**:
1. **Selective search**: Focuses on promising moves (not all 250)
2. **Asymmetric tree**: Explores best variations deeply, ignores bad ones
3. **No evaluation needed**: Playouts to end (or neural network)
4. **Scales with time**: Always has best move, improves with more iterations

**Result**: MCTS enabled AlphaGo to beat world champion (minimax couldn't).

---

## Part 4: AlphaZero

### What are the two heads of AlphaZero's network?

**1. Policy head** ($\mathbf{p}$):
$$p(a|s) = \text{softmax}(\text{PolicyNet}(s))$$
- Outputs: Probability distribution over legal moves
- Used as: Prior in MCTS ($P(s,a)$)

**2. Value head** ($v$):
$$v(s) \in [-1, +1]$$
- Outputs: Scalar game outcome prediction
- -1 = loss, 0 = draw, +1 = win
- Used as: Evaluation in MCTS (replaces rollout)

**Shared**: Single ResNet backbone, two heads branch at end.

### How does PUCT use the policy network?

**PUCT selection** formula:
$$PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

**Policy network provides** $P(s,a)$:
- Network's prior belief about move quality
- High $P(s,a)$ → larger exploration bonus
- Guides MCTS toward promising moves

**Effect**:
- **Without prior**: MCTS explores randomly (slow)
- **With prior**: MCTS focuses on likely-good moves (fast)

**Intuition**: Network says "this move looks good" → MCTS explores it more.

### Why doesn't AlphaZero use rollouts (unlike classic MCTS)?

**Classic MCTS**: Random rollout to terminal state
- Fast but noisy
- Quality depends on rollout policy

**AlphaZero**: Value network evaluation $v(s)$
- **Learned** from self-play (accurate)
- **Direct** prediction (no randomness)
- **Fast** (single forward pass)

**Advantages**:
1. More accurate than random rollouts
2. Faster than smart rollouts
3. Single network for policy + value (efficient)

**Key insight**: Learned value function better than simulation.

### What's the training loop (self-play → train → repeat)?

```
Initialize network θ randomly
Initialize replay buffer D = ∅

repeat:
    // Self-play: Generate training data
    for N games in parallel:
        Play game using MCTS with π_θ
        Store (s, π, z) tuples in D
            s: board state
            π: MCTS visit distribution
            z: game outcome (+1/-1/0)

    // Training: Update network
    for M mini-batches:
        Sample batch from D
        Compute loss: L = (z - v)² - π^T log p + c||θ||²
        Update θ via SGD

until convergence
```

**Key**: Self-play provides training data, training improves network, better network → better self-play (curriculum).

### What's the loss function (three components)?

$$\mathcal{L} = \underbrace{(z - v)^2}_{\text{value loss}} - \underbrace{\pi^T \log \mathbf{p}}_{\text{policy loss}} + \underbrace{c \|\theta\|^2}_{\text{regularization}}$$

**1. Value loss** $(z - v)^2$:
- MSE between predicted value and game outcome
- Trains network to predict winner

**2. Policy loss** $-\pi^T \log \mathbf{p}$:
- Cross-entropy between MCTS policy and network policy
- Distills MCTS search into network

**3. Regularization** $c \|\theta\|^2$:
- L2 weight decay (prevents overfitting)
- $c = 10^{-4}$ typically

**Insight**: Network learns to match MCTS (which is stronger due to search).

### How does value network replace rollouts?

**Rollout** (classic MCTS):
1. From leaf node, play random moves
2. Reach terminal state
3. Get outcome (+1/-1/0)
4. Slow, noisy

**Value network** (AlphaZero):
1. From leaf node, evaluate $v_\theta(s)$
2. Single forward pass
3. Predict outcome directly
4. Fast, learned

**Why better**:
- **Learned**: Trained on millions of positions (better than random)
- **Fast**: No simulation needed
- **Accurate**: Approximates true value function

**Trade-off**: Requires good network (pre-training via self-play).

### What's the difference between AlphaGo and AlphaZero?

**AlphaGo** (2016):
- **Supervised pre-training**: Learn from human games
- **Separate networks**: Policy and value
- **Rollouts**: Fast rollout policy + value network
- **Handcrafted features**: Domain knowledge
- **Go-specific**

**AlphaZero** (2017):
- **Tabula rasa**: No human data, self-play from random
- **Single network**: Shared policy + value
- **No rollouts**: Value network only
- **Raw board**: No features, just position
- **General**: Chess, Go, Shogi (same algorithm)

**Performance**: AlphaZero beat AlphaGo 100-0 after 40 days of training.

### Why is AlphaZero considered "tabula rasa"?

**Tabula rasa** (blank slate):
- **No human knowledge**: Doesn't use human games, opening books, or expert features
- **Random initialization**: Starts with random weights
- **Learns from scratch**: Pure self-play discovers strategies
- **Only game rules**: Just needs legal moves definition

**Contrast**:
- Traditional engines: Heavily handcrafted (evaluation, openings)
- AlphaGo: Pre-trained on human games

**Significance**: Shows general learning can match/exceed human knowledge.

### What are limitations (perfect info, two-player, etc.)?

1. **Perfect information only**: Requires known game state (no hidden info)
   - Works: Chess, Go
   - Doesn't work: Poker

2. **Two-player zero-sum**: Designed for competitive games
   - Not multi-player
   - Not general-sum (cooperation)

3. **Discrete actions**: Finite move set
   - Not continuous control

4. **Massive compute**: 5000 TPUs for Go (44 million games)
   - Not sample-efficient

5. **Deterministic**: No randomness in game
   - Doesn't handle stochasticity well

6. **No transfer**: Trained separately for each game

### How does temperature affect move selection?

**After MCTS**, select action based on visit counts $N(s,a)$:

$$\pi(a|s) \propto N(s,a)^{1/\tau}$$

**Temperature $\tau$**:

- **$\tau = 1$**: Proportional to visits (stochastic, exploration)
  - Used during early game

- **$\tau \to 0$**: Greedy (always pick most visited)
  - $\pi(a^*) = 1$ where $a^* = \arg\max_a N(s,a)$
  - Used during later game

**Schedule** (AlphaZero):
- First 30 moves: $\tau = 1$ (exploration)
- After 30 moves: $\tau \to 0$ (exploitation)

**Why**: Exploration early for diversity, exploitation later for best play.

---

## Part 5: PSRO & Population-Based Methods

### What's the PSRO algorithm (4 steps)?

**PSRO loop**:

**1. Initialize**: Start with small population $\Pi_i^{(0)}$ for each player
   - Example: Random policies

**2. Meta-game**: Compute empirical payoff matrix
   - Evaluate all policy pairs via simulation
   - Build payoff table

**3. Meta-strategy**: Solve for Nash equilibrium $\sigma$ on empirical game
   - Linear programming or iterative solver

**4. Best Response**: Train BR policy to $\sigma$ using RL oracle
   - Each player trains $\pi_i^{new} = BR(\sigma_{-i})$

**5. Expand**: Add new BRs to population
   - $\Pi_i^{(k+1)} = \Pi_i^{(k)} \cup \{\pi_i^{new}\}$

**Repeat** until convergence (no profitable BR).

### What's an empirical game?

**Empirical game**: Finite game constructed from **simulations** of continuous game.

**Construction**:
- **Players**: Same as original
- **Actions**: Finite set of policies from population
- **Payoffs**: Average returns from playing policies against each other

**Example** (2-player, 3 policies each):
```
         π₂¹   π₂²   π₂³
π₁¹    (2,1) (1,3) (0,0)
π₁²    (3,2) (2,2) (1,1)
π₁³    (1,0) (2,1) (3,3)
```

**Purpose**: Reduce infinite strategy space to finite meta-game that can be solved.

### What's a best response oracle?

**Best Response (BR) oracle**: Algorithm that finds optimal policy against given opponent strategy.

**Formal**: For opponent strategy $\sigma_{-i}$, find:
$$\pi_i^{BR} = \arg\max_{\pi_i} \mathbb{E}_{\sigma_{-i}}[u_i(\pi_i, \sigma_{-i})]$$

**In PSRO**: Typically RL algorithm (PPO, DQN, SAC)
- Train against opponents sampled from $\sigma_{-i}$

**Exactness**:
- **Exact BR**: Truly optimal (hard to achieve)
- **Approximate BR**: Good enough in practice

**Convergence**: PSRO converges to Nash if BRs are exact (approximate BRs → approximate Nash).

### How does PSRO converge to Nash equilibrium?

**Convergence condition**: When no player can find profitable best response.

**Algorithm**:
1. Current meta-Nash: $\sigma^*$
2. Compute BR for each player: $\pi_i^{BR}$
3. If $u_i(\pi_i^{BR}, \sigma_{-i}^*) \leq u_i(\sigma_i^*, \sigma_{-i}^*)$ for all $i$:
   - **Converged**: $\sigma^*$ is approximate Nash
4. Else: Add $\pi_i^{BR}$ to population, repeat

**Guarantee**: Exploitability decreases monotonically (anytime PSRO variant)

**In practice**: Stop when exploitability below threshold.

**Theoretical**: Exact BRs → exact Nash. Approximate BRs → approximate Nash.

### Difference between PSRO and double oracle?

**Double Oracle** (classical):
- Game theory algorithm for matrix games
- Exact BR computation
- Tabular (small state/action spaces)

**PSRO** (modern):
- Extends double oracle to **deep RL**
- Approximate BR via RL (PPO, DQN)
- Handles **large/continuous** state/action spaces
- **Empirical game**: Estimate payoffs via simulation

**Relationship**: PSRO is double oracle + deep RL oracle.

**Key innovation**: Scalability to complex games (Starcraft, Dota).

### What's JPSRO and how does it differ from PSRO?

**JPSRO** (Joint PSRO):
- Handles **general-sum n-player** games
- Uses **joint strategy distributions** (not independent mixing)
- Computes **Correlated Equilibrium (CE)** or **CCE** (not Nash)
- Train BR against **joint marginal** $\nu_{-i}$

**PSRO**:
- Two-player **zero-sum** focus
- **Independent** strategy mixing (Nash)
- Train BR against **independent marginal** $\sigma_{-i}$

**Key difference**: Joint distribution captures **coordination** opportunities.

**Example** (Traffic Light):
- Nash: Mix independently → 2.5 utility
- CE: Coordinate (both go same direction) → 5 utility

### What's Correlated Equilibrium (CE)?

**Correlated Equilibrium**: Joint strategy distribution where no player wants to deviate **after** seeing their recommended action.

**Setup**: Mediator samples $(a_1, \ldots, a_n) \sim \mu$, recommends $a_i$ to each player privately.

**Condition**: No incentive to deviate:
$$\mathbb{E}_{a_{-i} | a_i}[u_i(a_i, a_{-i})] \geq \mathbb{E}_{a_{-i} | a_i}[u_i(a'_i, a_{-i})]$$
for all players $i$, actions $a_i$, deviations $a'_i$.

**Example** (Traffic):
```
         Left    Right
Up       (5,5)   (0,0)
Down     (0,0)   (5,5)
```
CE: Flip coin → both Up-Left or both Down-Right (EU = 5)
Nash: Mix 50-50 independently (EU = 2.5)

**Why better**: Coordination via mediator.

### What's Coarse Correlated Equilibrium (CCE)?

**CCE**: Relaxation of CE - players commit to follow or deviate **before** seeing recommendation.

**Condition**: No incentive to deviate from marginal:
$$\mathbb{E}_{\mu}[u_i(a)] \geq \mathbb{E}_{\mu}[u_i(a'_i, a_{-i})]$$
for all players $i$, deviations $a'_i$.

**Difference from CE**:
- **CE**: React after seeing recommendation
- **CCE**: Commit before seeing recommendation

**Computational**: CCE easier to compute (fewer constraints in LP)

**Used in**: JPSRO (tractable, general-sum)

### Hierarchy: Nash ⊆ CE ⊆ CCE - explain

**Inclusion**:
$$\text{Nash Equilibrium} \subseteq \text{Correlated Equilibrium} \subseteq \text{Coarse Correlated Equilibrium}$$

**Nash → CE**: Independent mixing is special case of joint distribution
- Nash: $\mu(a) = \prod_i \sigma_i(a_i)$ (product distribution)
- CE allows correlations

**CE → CCE**: Stronger incentive constraints
- CE: No profitable deviation **after** seeing recommendation
- CCE: No profitable deviation **before** seeing recommendation

**Why hierarchy matters**:
- Easier to compute: Nash (hardest) < CE < CCE (easiest)
- CCE largest set (most solutions)
- Nash smallest (most restrictive)

### Why use CCE for general-sum games?

1. **Enables coordination**: Beyond independent mixing (Nash)
   - Captures correlated strategies
   - Example: Traffic light coordination

2. **Efficient computation**: Linear program (polynomial time)
   - Nash can be PPAD-complete

3. **Existence**: Always exists (like Nash in mixed strategies)

4. **n-player**: Naturally extends to many players

5. **One BR per player**: CCE needs 1 BR/player (vs $|A_i|$ BRs for CE)

6. **Proven convergence**: JPSRO converges to CCE

**Trade-off**: Weaker than Nash (larger solution set), but more practical.

### How many BRs needed per iteration (PSRO vs JPSRO)?

**PSRO** (Nash):
- **1 BR per player** per iteration
- Total: $n$ BRs (for $n$ players)

**JPSRO with CCE**:
- **1 BR per player** per iteration
- Total: $n$ BRs
- Same as PSRO!

**JPSRO with CE** (full correlated equilibrium):
- **$|A_i|$ BRs per player** (one per possible recommendation)
- Much more expensive
- Rarely used in practice

**Conclusion**: CCE has same BR cost as Nash, but captures coordination.

---

## Part 6: Neural Population Learning

### What's the key innovation of NeuPL vs PSRO?

**PSRO**: $N$ separate policy networks for $N$ strategies
- Memory: $O(N)$ networks
- No transfer between strategies

**NeuPL**: **Single conditional network** for all strategies
$$\pi_\theta(a | s, i)$$
where $i$ is policy index.

**Memory**: $O(1)$ network (constant, regardless of $N$)

**Key innovation**:
- All policies share representation
- **Transfer learning** across strategies
- 10,000x parameter reduction

### How does conditional network work: π(a|s,i)?

**Conditioning on policy index $i$**:

**Methods**:
1. **Embedding**: $i \to e_i$ (learned embedding)
2. **Concatenation**: $[s, e_i] \to$ network
3. **FiLM**: Use $e_i$ to modulate features (affine transformation)
4. **Hypernetwork**: $e_i \to$ subset of network weights

**Example** (embedding + concat):
```
s: state (observation)
i: policy index (0, 1, 2, ...)
e_i = Embed(i)  # learnable embedding
input = concat(s, e_i)
output = Network(input)  # action distribution
```

**Effect**: Different $i$ → different policies, but shared network.

### What's the memory advantage (O(1) vs O(N))?

**PSRO**:
- Population size: $N$ policies
- Storage: $N \times$ (network parameters)
- Example: 100 policies × 10M params = 1B parameters

**NeuPL**:
- Population size: $N$ policies
- Storage: 1 network + $N$ embeddings
- Example: 10M params + 100 × 128 (embeddings) ≈ 10M parameters

**Reduction**: ~100x for population size 100

**Enables**: Large populations (1000+) that would be infeasible with separate networks.

### What's transfer learning in NeuPL?

**Transfer**: New strategies benefit from features learned by previous strategies.

**Mechanism**:
1. Train $\pi_\theta(a|s, 0)$ (first policy)
2. Network learns useful representations of states
3. Train $\pi_\theta(a|s, 1)$ (second policy)
4. Reuses learned features, only adapts via index 1
5. Faster learning, better performance

**Analogy**: Like transfer learning in vision (ImageNet → specific tasks), but for game strategies.

**Evidence**: NeuPL converges faster than PSRO (fewer samples needed).

### NeuPL vs NeuPL-JPSRO differences?

**NeuPL** (original):
- **Symmetric zero-sum** games only
- Nash equilibrium (implicit)
- Two-player focus

**NeuPL-JPSRO**:
- **General-sum n-player** games
- CCE equilibrium (explicit)
- Combines NeuPL efficiency + JPSRO generality
- Additional components: payoff estimator network

**Relationship**: NeuPL-JPSRO extends NeuPL to broader game class.

### What equilibrium does NeuPL converge to?

**Original NeuPL** (symmetric zero-sum):
- Empirically converges to **approximate Nash equilibrium**
- No formal convergence guarantee in original paper

**NeuPL-JPSRO** (general-sum):
- Proven convergence to **CCE** (under exact distillation assumption)
- Formal guarantee

**In practice**: Both converge to reasonable equilibria, NeuPL-JPSRO has theoretical backing.

### When to use NeuPL vs standard PSRO?

**Use NeuPL when**:
- ✅ Memory constrained (can't store many networks)
- ✅ Want large populations (100+ strategies)
- ✅ Transfer learning beneficial (similar strategies)
- ✅ Symmetric game (original NeuPL)

**Use PSRO when**:
- ✅ Small populations (< 20 strategies sufficient)
- ✅ Very heterogeneous strategies (little transfer)
- ✅ Simplicity preferred (separate networks easier to debug)

**Modern trend**: NeuPL-JPSRO increasingly default (efficiency + generality).

---

## Part 7: Fictitious Self-Play

### What's fictitious play (classical)?

**Fictitious Play**: Iterative algorithm where players best-respond to opponent's **empirical frequency**.

**Algorithm**:
1. Initialize: Play arbitrary action
2. Track: Opponent's action history
3. Compute: Empirical frequency $\hat{\sigma}_{-i}(t) = \frac{1}{t}\sum_{\tau=1}^t a_{-i}^\tau$
4. Best respond: $a_i^{t+1} = BR(\hat{\sigma}_{-i}(t))$

**Convergence**: Converges to Nash in:
- Two-player zero-sum games (guaranteed)
- Potential games
- Some other classes

**Does NOT converge**: Shapley's example (cycling).

### How does FSP use RL and SL?

**Fictitious Self-Play** (sample-based version):

**Two components**:
1. **RL**: Learn best response to current average strategy
   - Buffer: $M_{RL}$ (recent trajectories)
   - Trains: Best-response policy

2. **SL**: Learn average strategy
   - Buffer: $M_{SL}$ (all historical actions)
   - Trains: Average policy (supervised on own actions)

**Data flow**:
- Play using mixture of RL and SL policies
- RL data → $M_{RL}$ (for best response learning)
- Actions → $M_{SL}$ (for average strategy learning)

### What's NFSP (Neural FSP)?

**Neural Fictitious Self-Play**: Deep NN version of FSP.

**Two networks**:
1. **Q-network**: Best-response (DQN-style)
   - Trained from $M_{RL}$ (RL buffer)
   - $\epsilon$-greedy behavior

2. **$\Pi$-network**: Average strategy
   - Trained from $M_{SL}$ (SL buffer)
   - Supervised learning (predict own actions)

**Behavior policy**:
- With probability $\eta$: Use Q-network (explore)
- With probability $1-\eta$: Use $\Pi$-network (exploit average)

**Convergence**: To approximate Nash in large imperfect-info games.

### What are the two networks and two buffers?

**Networks**:
1. **Q-network** ($Q_\theta$): Best response
   - Input: State
   - Output: Q-values for actions
   - Training: RL (Q-learning / DQN)

2. **Average policy network** ($\Pi_\phi$):
   - Input: State
   - Output: Action probabilities
   - Training: SL (classification on historical actions)

**Buffers**:
1. **$M_{RL}$**: Reservoir for RL (recent data)
   - Stores: Transitions $(s, a, r, s')$
   - For: Training Q-network

2. **$M_{SL}$**: Reservoir for SL (all historical actions)
   - Stores: State-action pairs $(s, a)$
   - For: Training $\Pi$-network

**Why two buffers**: Different data requirements (RL needs recent, SL needs all history).

### Why does FSP converge to Nash in some games?

**Proof sketch** (two-player zero-sum):

1. **Average strategy**: $\bar{\sigma}_i^t = \frac{1}{t}\sum_{\tau=1}^t \sigma_i^\tau$

2. **Best response**: Each iteration plays BR to opponent's average

3. **Regret minimization**: Average regret goes to 0
   $$\frac{1}{T}\sum_{t=1}^T [u_i(BR_i^t, \bar{\sigma}_{-i}^t) - u_i(\bar{\sigma}_i^t, \bar{\sigma}_{-i}^t)] \to 0$$

4. **Folk theorem**: No-regret in two-player zero-sum → Nash

**Intuition**: Best-responding to average prevents cycling (vs best-responding to latest in vanilla self-play).

### FSP vs vanilla self-play - why more stable?

**Vanilla self-play**:
- Agent plays against latest version of itself
- Can cycle: Rock → Paper → Scissors → Rock → ...
- No convergence guarantee

**FSP**:
- Agent best-responds to **average** of all past opponents
- Average smooths out oscillations
- Provably converges (in some games)

**Example** (Rock-Paper-Scissors):
- Vanilla SP: Cycles through pure strategies
- FSP: Converges to (1/3, 1/3, 1/3) mixed Nash

**Why**: Averaging acts as regularization, prevents exploitation of single strategy.

### When to use FSP vs PSRO?

**FSP / NFSP**:
- ✅ Two-player zero-sum
- ✅ Imperfect information (poker)
- ✅ When Nash convergence critical
- ✅ Continuous learning (no population)

**PSRO**:
- ✅ Multi-player or general-sum
- ✅ Want diverse population
- ✅ Can afford sequential training
- ✅ Empirical game analysis important

**Modern**: PSRO/JPSRO more general, FSP for specific settings (poker-like).

---

## Part 8: Counterfactual Regret Minimization

### What games is CFR designed for (imperfect info, extensive form)?

**CFR** is designed for:
- **Extensive-form games** (game trees)
- **Imperfect information** (hidden information, information sets)
- **Two-player** (mainly, extensions exist for multi-player)
- **Zero-sum** (convergence guarantee)

**Examples**: Poker (Kuhn, Leduc, Texas Hold'em), Bridge, Liar's Dice

**Not for**: Perfect info (use minimax/MCTS instead), large continuous action spaces.

### What's an information set?

**Information set** $I$: Collection of game tree nodes that player **cannot distinguish**.

**Properties**:
- Same player acts at all nodes in $I$
- Player doesn't know which node they're at
- Must choose same action at all nodes in $I$

**Example** (Poker):
- Player sees own cards but not opponent's
- Nodes with same visible info → same information set
- E.g., "I have King, opponent has ???" is one infoset

**Perfect info**: Every infoset has 1 node (can distinguish all states).

### What's counterfactual value?

**Counterfactual value** $v_i(\sigma, I)$: Expected utility for player $i$ if they:
1. **Reach** information set $I$ (counterfactually, regardless of their own actions)
2. Then **follow** strategy $\sigma$ from $I$ onward

**Formula**:
$$v_i(\sigma, I) = \sum_{h \in I} \sum_{z \in Z} \pi_{-i}^\sigma(h) \pi^\sigma(h, z) u_i(z)$$

Where:
- $h$: History (node in $I$)
- $z$: Terminal node
- $\pi_{-i}^\sigma(h)$: Opponent's reach probability
- $u_i(z)$: Utility at terminal

**"Counterfactual"**: Assumes we reached $I$, even if our strategy wouldn't have.

### Explain regret matching strategy update

**Regret** for action $a$ at infoset $I$:
$$R^T(I, a) = \sum_{t=1}^T [v_i(\sigma^t_{I \to a}, I) - v_i(\sigma^t, I)]$$

Cumulative difference between playing $a$ and current strategy.

**Regret matching**: Convert regrets to strategy
$$\sigma^{T+1}(I, a) = \frac{R^{T,+}(I, a)}{\sum_{a'} R^{T,+}(I, a')}$$

Where $R^{+} = \max(R, 0)$ (positive part).

**If all regrets ≤ 0**: Play uniform random.

**Intuition**: Play actions proportional to how much we regret not playing them.

### Why use average strategy (not current strategy)?

**Current strategy** $\sigma^T$: Can oscillate (like vanilla self-play)

**Average strategy** $\bar{\sigma}^T$:
$$\bar{\sigma}^T(I, a) = \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I) \sigma^t(I, a)}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$

Weighted by reach probability.

**Convergence**: $\bar{\sigma}^T \to$ Nash equilibrium

**Why**: Regret matching guarantees **average** regret → 0, not instantaneous regret.

**Analogy**: Like FSP - averaging prevents cycling.

### What's CFR+ and how does it improve vanilla CFR?

**CFR+** improvements over vanilla CFR:

1. **Regret floor**: Cumulative regrets never go below 0
   $$R^{t+1}(I, a) = \max(R^t(I, a) + r^{t+1}(I, a), 0)$$
   (Reset negative regrets to 0)

2. **Alternating updates**: Update one player at a time (not simultaneous)

3. **Linear averaging**: Weight recent iterations more
   $$w_t = t$$ (instead of uniform weighting)

**Result**:
- **Faster convergence**: $O(1/T^2)$ vs $O(1/T)$ for vanilla CFR
- Used in Libratus (poker AI)

**Why it works**: Prevents negative regret accumulation (interference).

### What's Monte Carlo CFR (sampling)?

**Problem**: Full tree traversal is expensive for large games.

**Solution**: **Sample** parts of tree instead of traversing all.

**Variants**:

1. **External Sampling**:
   - Sample: Chance events + opponent actions
   - Traverse: All own actions at visited infosets
   - Lower variance, moderate speedup

2. **Outcome Sampling**:
   - Sample: Entire trajectory (single path)
   - Fastest, highest variance

**Trade-off**: Speed vs variance
- Full CFR: Slow, deterministic
- Outcome sampling: Fast, noisy

**Practical**: External sampling common compromise.

### What's Deep CFR?

**Deep CFR**: Use neural networks to approximate regrets/strategies.

**Problem**: Tabular CFR doesn't scale to huge games (too many infosets).

**Solution**:
1. **Value network**: Approximate counterfactual values $v(I)$
2. **Regret network**: Approximate cumulative regrets $R(I, a)$
3. **Strategy network**: Approximate average strategy

**Training**:
- Run MCCFR to generate samples
- Train networks on sampled regrets/values
- Use networks instead of tables

**Result**: Scales to games too large for tabular (e.g., no-limit poker).

### Complexity of CFR per iteration?

**Time per iteration**: $O(|I|)$
- Linear in number of information sets
- Must visit each infoset

**Full game tree**: $O(|H|)$ where $H$ is all histories (nodes)
- Larger than $|I|$

**Convergence**: $O(1/\sqrt{T})$ to Nash
- After $T$ iterations, exploitability $\propto 1/\sqrt{T}$

**Total cost**: $O(|I| \cdot T)$ for $\epsilon$-Nash where $T \propto 1/\epsilon^2$

**Practical**: Millions of iterations for poker, but feasible.

### Why does CFR converge to Nash in two-player zero-sum?

**Proof sketch**:

1. **Regret minimization**: CFR guarantees average regret → 0
   $$\frac{1}{T}\sum_{t=1}^T r_i^t \to 0$$

2. **Zero-sum + regret minimization**: Folk theorem
   - If both players have no-regret, their average strategies form Nash

3. **Formal**: Average exploitability bounded by average regret
   $$\epsilon = \max_{\pi_i} u_i(\pi_i, \bar{\sigma}_{-i}) - u_i(\bar{\sigma}) \leq \frac{1}{T}\sum_{t=1}^T R_i^t$$

**Key insight**: Regret matching is no-regret algorithm → convergence.

**Extension**: Works for broader class (two-player general-sum with modifications).
