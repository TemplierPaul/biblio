# Game Theory - Algorithm Reference

Quick reference for game theory concepts, equilibria, and multi-agent learning algorithms.

---

## Foundational Concepts

### Nash Equilibrium
**Paper**: "Non-Cooperative Games" (Nash, 1951)
**What**: Strategy profile where no player benefits from unilateral deviation
**Why**: Fundamental solution concept in game theory
**Types**: Pure, mixed, correlated

### Evolutionary Stable Strategy (ESS)
**Paper**: "The Logic of Animal Conflict" (Smith & Price, 1973)
**What**: Strategy that, if adopted by population, cannot be invaded by mutants
**Difference from Nash**: Population dynamics perspective, robustness to mutations

### Normal Form Games
**What**: Matrix representation of simultaneous-move games
**Key**: Payoff matrices, strategies, best responses

### Extensive Form Games
**What**: Tree representation of sequential games
**Key**: Information sets, subgame perfection, backward induction

---

## Classical Search Algorithms

### Minimax
**Paper**: "Programming a Computer for Playing Chess" (Shannon, 1950)
**What**: Optimal strategy for zero-sum perfect-information games
**How**: Recursively evaluate game tree assuming rational opponent
**Key innovation**: Alpha-beta pruning for efficiency
**When to use**: Perfect information, zero-sum, finite depth trees

### Monte Carlo Tree Search (MCTS)
**Paper**: "Bandit-based Monte-Carlo Planning" (Kocsis & Szepesvári, 2006)
**What**: Best-first search using random sampling for evaluation
**How**: Selection (UCT), expansion, simulation, backpropagation
**When to use**: Large branching factor, stochastic games, imperfect heuristics
**Difference from Minimax**: Anytime algorithm, handles stochasticity, asymmetric tree

---

## Self-Play Algorithms

### AlphaZero
**Paper**: "Mastering Chess and Shogi by Self-Play" (Silver et al., 2017)
**What**: Self-play + MCTS + deep neural networks for perfect-information games
**How**: PUCT search with NN policy/value, iterative self-play training
**Key innovation**: Tabula rasa learning (no human data), generalizes across games
**When to use**: Perfect information, deterministic, can simulate

### Fictitious Self-Play (FSP)
**Paper**: "Fictitious Play Property for Games with Identical Interests" (Monderer & Shapley, 1996)
**What**: Learn best response to opponent's average historical strategy
**How**: Maintain distribution over opponent's past actions, best-respond to it
**When to use**: Converges to Nash in 2-player zero-sum, simple baseline
**Difference from AlphaZero**: Model opponent explicitly, no tree search

---

## Policy-Space Response Oracles (PSRO)

### PSRO
**Paper**: "A Unified Game-Theoretic Approach to Multiagent RL" (Lanctot et al., 2017)
**What**: Iteratively expand strategy pool by training best responses
**How**: Meta-game (payoff matrix over strategies) → Nash → train BR → repeat
**When to use**: General-sum games, want diverse strategies, computational budget allows
**Difference from self-play**: Maintains population, explicit equilibrium computation

### JPSRO (Joint PSRO)
**What**: Cooperative PSRO where all players share policy pool
**Difference from PSRO**: Single shared repertoire vs. per-player repertoires
**When to use**: Symmetric games, want unified strategy space

### Pipeline PSRO
**Paper**: "Pipeline PSRO" (McAleer et al., 2020)
**What**: Parallel PSRO with asynchronous training and evaluation
**Key innovation**: Overlapping BR training phases, massive parallelization
**When to use**: Large-scale games, distributed compute available

### A-PSRO (Anytime PSRO)
**What**: PSRO variant with adaptive stopping (trains until improvement plateaus)
**Difference from PSRO**: Dynamic BR training duration vs. fixed budget

### SP-PSRO (Simplex-Projected PSRO)
**What**: PSRO enforcing fully-mixed meta-strategies (no pure exploitation)
**Difference from PSRO**: Always maintains diversity in meta-game

### SF-PSRO (Symmetric Full PSRO)
**What**: PSRO for symmetric games with full strategy sharing
**When to use**: Symmetric games like poker, sports

### PRD (Projected Replicator Dynamics)
**Paper**: "Computing Approximate Equilibria in Sequential Games" (Lanctot et al., 2012)
**What**: Fast iterative solver for Nash equilibria using replicator dynamics
**How**: Project strategy updates onto simplex, converges to Nash
**When to use**: Need fast approximate Nash, normal-form meta-games

### Rectified Nash
**What**: Nash equilibrium refinement that rectifies dominated strategies
**Difference from Nash**: Removes weakly dominated strategies first

---

## NeuPL (Neural Population Learning)

### NeuPL
**Paper**: "NeuPL: Neural Population Learning" (Bakhtin et al., 2022)
**What**: Learns distribution over strategies (population) instead of single policy
**How**: Parameterize strategy distribution with neural network, optimize for Nash
**Key innovation**: Infinite strategy populations, end-to-end differentiable
**Difference from PSRO**: Continuous distribution vs. discrete pool

### NeuPL-JPSRO
**What**: NeuPL combined with PSRO framework (joint training)
**When to use**: Want both population diversity and best-response training

### Simplex NeuPL
**What**: NeuPL with simplex-constrained meta-strategies
**Difference from NeuPL**: Enforces valid probability distributions explicitly

---

## Imperfect Information Games

### Counterfactual Regret Minimization (CFR)
**Paper**: "Regret Minimization in Games with Incomplete Information" (Zinkevich et al., 2007)
**What**: Iterative algorithm minimizing counterfactual regret for imperfect-info games
**How**: Traverse game tree, accumulate regrets, update strategy toward regret-matching
**When to use**: Imperfect information (poker), need exploitability guarantees
**Key variants**: CFR+, MCCFR (sampling), Deep CFR (function approximation)

---

## Social Dilemmas

### Prisoner's Dilemma
**Paper**: Classic game theory (Tucker, 1950)
**What**: Game where individual rationality leads to collective irrationality
**Key**: Cooperation vs. defection, Nash = mutual defect, social optimum = cooperate
**Extensions**: Iterated PD (Axelrod tournaments), spatial PD

---

## Summary by Use Case

| Use Case | Algorithm | Why |
|----------|-----------|-----|
| Perfect info, zero-sum | AlphaZero, Minimax | Optimal, self-contained |
| Imperfect info (poker) | CFR | Handles info sets, converges to Nash |
| General-sum, diverse strategies | PSRO | Population-based, explicit equilibria |
| Large-scale, parallel | Pipeline PSRO | Massive parallelization |
| Continuous populations | NeuPL | Infinite strategy space |
| Fast approximate Nash | PRD | Efficient solver for meta-games |
| Simple baseline | FSP | Easy to implement, works for some games |

---

## Relationships

- **Minimax** → **MCTS** → **AlphaZero**: Perfect info search evolution
- **Self-Play** → **PSRO**: Single policy → Population of policies
- **PSRO** → **NeuPL**: Discrete pool → Continuous distribution
- **PSRO** → **Pipeline PSRO**: Sequential → Parallel
- **Nash** → **PRD**: Exact → Approximate fast solver

---

**See individual files and subfolders for detailed implementations, convergence proofs, and code examples.**
