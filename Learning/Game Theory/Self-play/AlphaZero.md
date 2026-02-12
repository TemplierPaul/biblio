# AlphaZero

## Definition
AlphaZero is a landmark self-play reinforcement learning algorithm that achieves superhuman performance in Chess, Go, and Shogi through pure self-play, without human data or domain knowledge.

## Core Components

### 1. Neural Network (Dual-Head)
Single network with two outputs:

**Policy Head**: $\mathbf{p}$
$$p(a|s) = \text{softmax}(\text{PolicyNet}(s))$$
Predicts move probabilities

**Value Head**: $v$
$$v(s) \approx \mathbb{E}[z | s]$$
Predicts game outcome (-1: loss, 0: draw, +1: win)

**Architecture** (original):
- 20 residual blocks (Chess/Shogi) or 40 (Go)
- Convolutional neural network
- Input: Board state (multiple planes encoding position)

### 2. Monte Carlo Tree Search (MCTS)
**Purpose**: Planning algorithm that uses neural network to guide search

**Four Phases** (repeated until budget):

1. **Selection**: Traverse tree using PUCT
$$\text{PUCT}(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
   Where:
   - $Q(s, a)$: Mean action value
   - $P(s, a)$: Prior from policy network
   - $N(s)$: Visit count of state
   - $c_{puct}$: Exploration constant

2. **Expansion**: Add leaf node to tree

3. **Evaluation**: Use value network (no rollout!)
   - Classic MCTS: Random simulation
   - AlphaZero: $v_\theta(s)$ (learned value)

4. **Backup**: Update statistics along path
   - $N(s, a) \leftarrow N(s, a) + 1$
   - $Q(s, a) \leftarrow \frac{\sum_{visits} v}{N(s, a)}$

**Search Policy**:
After MCTS, select action proportional to visit counts:
$$\pi(a|s) \propto N(s, a)^{1/\tau}$$
where $\tau$ is temperature (1 for exploration, 0 for greedy)

### 3. Self-Play
**Data Generation**:
1. Play games against self using MCTS policy
2. Store $(s, \pi, z)$ tuples:
   - $s$: Board state
   - $\pi$: MCTS policy (visit counts)
   - $z$: Game outcome (+1/-1/0)

**Curriculum**: Learns by playing against increasingly strong versions of itself

### 4. Training
**Loss Function**:
$$\mathcal{L} = (z - v)^2 - \pi^T \log \mathbf{p} + c \|\theta\|^2$$

Three components:
1. **Value loss**: $(z - v)^2$ (MSE to game outcome)
2. **Policy loss**: $-\pi^T \log \mathbf{p}$ (cross-entropy to MCTS policy)
3. **Regularization**: $c \|\theta\|^2$ (L2 weight decay)

**Optimization**:
- Stochastic Gradient Descent (or Adam)
- Mini-batches sampled uniformly from replay buffer
- 700K steps (Chess/Shogi), 2M steps (Go)

## Training Loop

```
Initialize network θ randomly
Initialize replay buffer D = ∅

repeat
    // Self-play
    for N games in parallel:
        Play game using MCTS with π_θ
        Store (s, π, z) in D

    // Training
    for M mini-batches:
        Sample batch from D
        Compute loss: L = (z - v)^2 - π^T log p + c||θ||^2
        Update θ via SGD

until convergence
```

## Key Innovations

### 1. No Human Knowledge
- **Input**: Just game rules (legal moves)
- **No features**: Raw board representation
- **No openings**: Discovers strategies from scratch
- **Generality**: Same algorithm for Chess, Go, Shogi

### 2. No Rollouts (Value Network)
- Classic MCTS: Random simulations to terminal state
- AlphaZero: Learned value function $v_\theta(s)$
- **Advantage**: More accurate, computationally efficient

### 3. Single Neural Network
- AlphaGo: Separate policy and value networks
- AlphaZero: Shared representation
- **Advantage**: Efficiency, transfer between policy and value learning

### 4. Simplicity
- Removed handcrafted features from AlphaGo
- Removed supervised learning phase
- Pure self-play from random initialization

## Results

### Performance
- **Chess**: Defeated Stockfish 8 (155-6, 839 draws)
- **Shogi**: Defeated Elmo (91-8, 91 draws)
- **Go**: Surpassed AlphaGo Lee/Master
- **Training**: 9 hours (Chess), 12 hours (Shogi), 34 hours (Go) on TPUs

### Emergent Strategies
- Discovered known opening theory (and novel variations)
- Sacrificial play (long-term strategy)
- Positional understanding

## Hyperparameters

**MCTS**:
- Simulations per move: 800 (Chess/Shogi), 1600 (Go)
- $c_{puct}$: 2.5 (exploration)
- Dirichlet noise: $\alpha = 0.3$ at root (exploration)

**Training**:
- Batch size: 4096
- Learning rate: 0.2 (decay to 0.02)
- Weight decay: 1e-4
- Self-play games: 44 million (Go)

**Resignation**:
- Resign if $v < -0.9$ (value threshold)

## Limitations

### 1. Perfect Information Only
- Designed for deterministic games
- Known state (no hidden information)
- Two-player, zero-sum

### 2. Discrete Action Space
- Board games (finite moves)
- Not directly applicable to continuous control

### 3. Computational Cost
- Requires massive compute (5000 TPUs for Go)
- MCTS during inference (slower than direct policy)

### 4. Sample Efficiency
- Millions of self-play games
- Not suitable for real-world applications with limited interaction

## Extensions

### MuZero
- Learns environment model
- Applies to unknown dynamics (Atari, etc.)
- See [[MuZero]]

### Continuous Action Spaces
- Sampled AlphaZero: MCTS with sampled actions
- Progressive widening

### Multi-Player
- AlphaZero principles applied to multi-player games
- Requires equilibrium concept beyond minimax

## Interview Relevance

**Common Questions**:
1. **How does AlphaZero work?** Self-play + MCTS guided by neural network (policy + value)
2. **Why no rollouts?** Value network more accurate than random simulation
3. **PUCT formula?** $Q(s,a) + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)}$ (exploitation + exploration)
4. **Training objective?** Value loss + policy loss (distill MCTS to network)
5. **Key innovations?** No human data, single network, no handcrafted features
6. **AlphaGo vs AlphaZero?** AlphaGo: supervised pre-training, separate nets; AlphaZero: pure self-play, single net
7. **Limitations?** Perfect info, discrete actions, massive compute
8. **Why MCTS + neural net?** NN provides prior/value, MCTS improves policy via search
9. **Self-play convergence?** Learns by playing stronger versions of self (curriculum)

**Key Components**:
- **Dual-head network**: Policy + value
- **PUCT**: UCT with neural network prior
- **Self-play**: Generate training data
- **MCTS → Policy distillation**: Learn from search

**Key Insight**: AlphaZero demonstrates that superhuman performance can emerge from pure self-play with minimal domain knowledge, by combining learned intuition (neural network) with explicit planning (MCTS).
