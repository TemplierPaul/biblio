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
**Paper**: "Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium Meta-Solvers" (Marris et al., 2021)
**What**: PSRO for n-player general-sum games using Correlated Equilibrium (CE) meta-solvers
**How**: Shared policy pool, optimize Maximum Gini CE (MGCE) for meta-strategy
**Key innovation**: Extends PSRO beyond 2-player zero-sum, uses Gini impurity (not entropy)
**When to use**: N-player general-sum games, symmetric games, want correlated equilibria
**Difference from PSRO**: CE/MGCE solver vs. Nash, supports general-sum

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

## Multi-Agent Evaluation

### α-Rank (AlphaRank)
**Paper**: "α-Rank: Multi-Agent Evaluation by Evolution" (Omidshafiei et al., DeepMind, 2019)
**What**: Evolutionary dynamics ranking method grounded in Markov-Conley chains (MCCs)
**How**: Compute stationary distribution π of discrete-time evolutionary Markov chain with ranking-intensity α
**Key innovation**: Dynamical solution concept (MCCs) vs. static (Nash), polynomial-time, unique ranking
**When to use**: Large-scale multi-agent evaluation, intransitive games, many agents, want evolutionary strengths
**Difference from Nash**: Descriptive (evolutionary) vs. prescriptive (rational), polynomial vs. PPAD-complete

---

## Game-Theoretic PSRO Methods

### PSRO-rN (Rectified Nash Response)
**Paper**: "Open-ended Learning in Symmetric Zero-sum Games" (Balduzzi et al., 2019)
**What**: PSRO variant using Rectified Nash for open-ended learning via game-theoretic niching
**How**: PRD meta-solver, response graph analysis, gamescapes framework (Hodge decomposition)
**Key innovation**: Nontransitive games enable open-endedness, avoid Nash convergence
**When to use**: Want continuous innovation, nontransitive domains, avoid equilibrium stagnation
**Difference from PSRO**: Explicit niching dynamics, gamescapes analysis, designed for nontransitivity

---

## NeuPL (Neural Population Learning)

### NeuPL
**Paper**: "Neural Population Learning beyond Symmetric Zero-sum Games" (Liu et al., 2021)
**What**: Shared conditional neural network representing entire policy population
**How**: Policy conditioned on strategy embedding ν, optimize for Coarse Correlated Equilibrium (CCE)
**Key innovation**: Efficient population via parameter sharing, continual learning, scales to general-sum
**When to use**: Large policy spaces, want compact representation, general-sum games
**Difference from PSRO**: Single shared network vs. discrete pool, CCE vs. Nash/CE

### NeuPL-JPSRO
**Paper**: "Neural Population Learning beyond Symmetric Zero-sum Games" (Liu et al., 2021)
**What**: NeuPL integrated with JPSRO for scalable n-player general-sum CCE convergence
**How**: Neural population (shared θ) + JPSRO outer loop (expand repertoire)
**Key innovation**: Combines compact neural representation with PSRO scalability
**When to use**: N-player general-sum games, need scalability, want CCE guarantees
**Difference from NeuPL**: Adds PSRO outer loop for iterative expansion

### Simplex NeuPL
**Paper**: "Simplex Neural Population Learning" (Liu et al., 2022)
**What**: NeuPL extension achieving any-mixture Bayes-optimality via simplex sampling
**How**: Sample opponent priors from Dirichlet distribution over population simplex (ε=0.5)
**Key innovation**: Optimal against ANY mixture (not just training set), implicit Bayesian inference
**When to use**: Test-time uncertainty, need adaptation, have subjective beliefs about opponents
**Difference from NeuPL**: Trains BR to all σ ∈ Δ^(N-1) vs. discrete set, Bayes-optimal behavior

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
| N-player general-sum | JPSRO, NeuPL-JPSRO | Correlated equilibria, scalable |
| Large-scale, parallel | Pipeline PSRO | Massive parallelization |
| Compact populations | NeuPL | Shared network, parameter-efficient |
| Test-time adaptation | Simplex NeuPL | Any-mixture optimality, Bayes-optimal |
| Open-ended learning | PSRO-rN | Nontransitive dynamics, avoid convergence |
| Multi-agent evaluation & ranking | α-Rank | Evolutionary strengths, polynomial-time, unique ranking |
| Fast approximate Nash | PRD | Efficient solver for meta-games |
| Simple baseline | FSP | Easy to implement, works for some games |

---

## Relationships

- **Minimax** → **MCTS** → **AlphaZero**: Perfect info search evolution
- **Self-Play** → **PSRO**: Single policy → Population of policies
- **PSRO** → **JPSRO**: 2-player zero-sum → N-player general-sum (CE)
- **PSRO** → **NeuPL**: Discrete pool → Shared neural network (CCE)
- **PSRO** → **PSRO-rN**: Nash convergence → Open-ended niching
- **PSRO** → **Pipeline PSRO**: Sequential → Parallel
- **NeuPL** → **NeuPL-JPSRO**: Standalone → PSRO integration
- **NeuPL** → **Simplex NeuPL**: Discrete mixtures → Any-mixture Bayes-optimality
- **Nash** → **PRD**: Exact → Approximate fast solver

---

**See individual files and subfolders for detailed implementations, convergence proofs, and code examples.**
