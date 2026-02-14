# α-Rank: Multi-Agent Evaluation by Evolution

## Overview

**α-Rank** is a principled evolutionary dynamics methodology for **evaluation** and **ranking** of agents in large-scale multi-agent interactions, grounded in a novel dynamical game-theoretic solution concept called **Markov-Conley chains** (MCCs).

**Authors**: Omidshafiei, Papadimitriou, Piliouras, Tuyls, Rowland, Lespiau, Czarnecki, Lanctot, Perolat, Munos (DeepMind, Columbia, SUTD)
**Paper**: "α-Rank: Multi-Agent Evaluation by Evolution"
**Venue**: Scientific Reports, 2019

## Core Problem

### Multi-Agent Evaluation Challenges

Evaluating agents in multi-agent settings is hard due to:

1. **Strategy space explosion**: Multi-robot systems, complex games
2. **Intransitive behaviors**: Cyclical best-responses (like Rock-Paper-Scissors at scale)
3. **Large number of agents**: Poker, team sports, many-player games
4. **Complex interactions**: Beyond pairwise (e.g., MuJoCo soccer)
5. **Asymmetric payoffs**: Different player roles (e.g., Scotland Yard board game)

### Limitations of Existing Approaches

**Nash Equilibrium**:
- ❌ Computing Nash is PPAD-complete (intractable)
- ❌ Equilibrium selection problem (many Nash equilibria)
- ❌ Static solution concept (incompatible with dynamics)
- ❌ No convergence guarantee for dynamical systems

**Replicator Dynamics** (continuous-time micro-model):
- ✅ Captures micro-dynamics, basins of attraction
- ❌ Limited to 3-4 agents (visualization complexity)
- ❌ Explodes in complexity for large/asymmetric games

**Existing discrete-time models**:
- ✅ Captures macro-dynamics
- ❌ Many evolutionary parameters
- ❌ Not grounded in game-theoretic solution concept

## α-Rank Solution

### Key Innovation: Markov-Conley Chains (MCCs)

**MCCs are a dynamical solution concept** based on:
- **Fixed points** (like Nash)
- **Recurrent sets, periodic orbits, limit cycles** (unlike Nash)
- Conley's Fundamental Theorem of Dynamical Systems

**Advantages over Nash**:
1. **Polynomial-time computable** (vs. PPAD-complete)
2. **Unique stationary distribution** (no equilibrium selection)
3. **Dynamical** (captures long-term evolution, not just static best-response)
4. **Descriptive** (evolutionary strengths) vs. prescriptive (rational play)

### The α-Rank Algorithm

```
1. Construct meta-game payoff table M^k for each population k
   (from multi-agent interaction data or simulations)

2. Compute transition matrix C with ranking-intensity α
   - Use large α to approximate MCC solution concept
   - Sweep α exponentially until ranking convergence

3. Compute unique stationary distribution π
   - Each element π_i = time spent in strategy profile i

4. Rank agents by ordered masses in π
   - Stationary distribution mass = agent "score"
```

**Ranking-intensity parameter α**:
- **Low α (α ≪ 1)**: Weak selection (sub-optimal can fixate)
- **High α (α → ∞)**: Strong selection (approximates MCC)
- **Practical**: Sweep exponentially, check convergence

---

## Mathematical Foundation

### Meta-Game Setup

**Meta-game (Empirical Game)**:
- Meta-strategies = learning agents (e.g., different AlphaGo variants)
- Payoffs = win/loss ratios from agent interactions
- Abstraction over primitive actions

**K-wise Interaction Normal Form Game**:
- $K$ players, strategy sets $S^k$
- Payoff functions $M^k: \prod_{i=1}^K S^i \to \mathbb{R}$
- Strategy profile $s = (s^1, \ldots, s^K)$

### Discrete-Time Evolutionary Model

**Multi-population finite-population dynamics**:
- $K$ populations, each with $m$ individuals
- Small mutation rate $\mu \to 0$ (at most 1 mutant population at a time)
- Fermi (logistic) selection function with ranking-intensity $\alpha$

**Fixation probability** (mutant $\tau$ in population of $\sigma$):

$$\rho^k_{\sigma,\tau}(s^{-k}) = \begin{cases}
\frac{1 - e^{-\alpha \Delta f}}{1 - e^{-m\alpha \Delta f}} & \text{if } \Delta f \neq 0 \\
\frac{1}{m} & \text{if } \Delta f = 0
\end{cases}$$

where $\Delta f = f^k(\tau, s^{-k}) - f^k(\sigma, s^{-k})$ (fitness difference)

**Markov chain transition matrix**:

$$C_{ij} = \begin{cases}
\eta \rho^k_{s^k_i, s^k_j}(s_i^{-k}) & \text{if } \exists k: s_i^k \neq s_j^k \land s_i^{-k} = s_j^{-k} \\
1 - \sum_{j \neq i} C_{ij} & \text{if } s_i = s_j \\
0 & \text{otherwise}
\end{cases}$$

where $\eta = \frac{1}{\sum_k (|S^k| - 1)}$ (normalization)

**Unique stationary distribution** $\pi$ (guaranteed to exist):
- $\pi^T C = \pi^T$, $\sum_i \pi_i = 1$
- $\pi_i$ = average time spent in monomorphic state $i$

---

## Markov-Conley Chains (MCCs)

### Response Graph

**Strictly/weakly better response**:
- $s_j$ is better response than $s_i$ for player $k$ if payoff at $s_j \geq$ payoff at $s_i$

**Response graph**:
- Vertices = pure strategy profiles $\prod_k S^k$
- Directed edge $s_i \to s_j$ if $s_j$ is weakly better response

**Sink strongly connected components (SCCs)**:
- Maximal subgraphs with paths between all vertices
- No outgoing edges
- If singleton → pure Nash equilibrium

### MCC Definition

**Markov-Conley chain**: Irreducible Markov chain over a sink SCC

**Transition probabilities**:
- Self-transition with some probability
- Remaining mass split equally among improving responses
- Strictly improving responses get higher probability than equal-payoff transitions

**Key Property**: Invariant under positive affine transformations of payoffs

### Connection to Continuous Dynamics

**Conley's Fundamental Theorem**:
- Any dynamical system decomposes into chain components + transient points
- Chain components = irreducible long-term behaviors
- Complete Lyapunov function guides transients to chain components

**Theorem** (α-Rank foundation):
- Every asymptotically stable sink chain component contains ≥ 1 MCC
- Each MCC contained in exactly 1 chain component
- **Finite number** of MCCs in any game

**Theorem** (Discrete-continuous correspondence):
- Limit $\alpha \to \infty$: Discrete-time macro-model → MCC
- Limit $m \to \infty$ (large population): Discrete-time → Replicator dynamics on simplex edges

---

## Conceptual Examples

### Rock-Paper-Scissors (Symmetric)

**Payoffs**: Standard cyclic game

**α-Rank result**:
- All 3 strategies: Rank 1, Score = 1/3
- Stationary distribution: $(1/3, 1/3, 1/3)$ for all $\alpha$
- **Agrees with Nash**: Symmetric mixed equilibrium

**Interpretation**: All strategies equally evolutionarily fit (intransitive cycle)

### Biased Rock-Paper-Scissors

**Payoffs**: Asymmetric payoffs favoring certain transitions

**Nash equilibrium**: $(1/16, 5/8, 5/16)$

**α-Rank result**: $(1/3, 1/3, 1/3)$

**Key difference**:
- **Nash** (prescriptive): "Play this mixture assuming rational opponents"
- **α-Rank** (descriptive): "All 3 strategies equally resistant to invasion"

### Battle of the Sexes (Asymmetric, 2-player)

**Payoffs**:
- (Opera, Opera): $(3, 2)$
- (Movie, Movie): $(2, 3)$
- Mismatch: $(0, 0)$

**α-Rank result**:
- States $(O, O)$ and $(M, M)$: Mass = 0.5 each
- States $(O, M)$ and $(M, O)$: Mass ≈ 0

**Interpretation**: Coordination states dominate, mismatches transient

---

## Applications & Results

### AlphaGo

**Agents evaluated**: Different AlphaGo variants (rollouts, value nets, policy nets)

**α-Rank insights**:
- Filters out transient/weak agents
- Identifies evolutionarily robust variants
- Rankings not always correlated with training time

### AlphaZero (Chess, Shogi, Go)

**Setup**: Different training checkpoints as meta-strategies

**Results**:
- Dramatic reduction in non-transient agents
- Support of $\pi$ much smaller than total agents
- Reveals which checkpoints are evolutionarily stable

### MuJoCo Soccer

**Complex multi-agent environment**: Team-based, continuous control

**α-Rank advantages**:
- Handles multi-agent interactions beyond pairwise
- Scales to large strategy spaces
- Captures team dynamics

### Poker (Leduc Hold'em)

**Large imperfect-information game**

**Results**:
- Tractable evaluation even for large game
- Avoids Nash computation (PPAD-hard)
- Rankings align with intuition/exploitability

---

## Comparison to Other Solution Concepts

| Aspect | Nash Equilibrium | Evolutionary Stable Strategy (ESS) | α-Rank (MCC) |
|--------|------------------|---------------------|---------------|
| **Type** | Static | Static refinement | Dynamical |
| **Computation** | PPAD-complete | Intractable | Polynomial-time |
| **Selection** | Many equilibria | Many ESS | Unique $\pi$ |
| **Dynamics** | No convergence guarantee | Limited | Grounded in dynamics |
| **Output** | Fixed points | Robust fixed points | Stationary distribution |
| **Philosophy** | Prescriptive (rational) | Prescriptive (stable) | Descriptive (evolutionary) |

---

## When to Use α-Rank

**✅ Use α-Rank when**:
- Large-scale multi-agent evaluation needed
- Many agents to rank (not just 2-4)
- Intransitive interactions expected
- Interested in evolutionary strengths
- Nash computation infeasible
- Need unique solution (no equilibrium selection)

**❌ Consider alternatives when**:
- Small games (2-4 agents) → Replicator dynamics visualization
- Want prescriptive equilibrium → Nash (if tractable)
- Need guarantees about rational play → Game-theoretic equilibria

---

## Key Insights

1. **Paradigm shift**: From static (Nash) to dynamical (MCC) solution concepts
2. **Tractability**: Polynomial-time vs. PPAD-complete
3. **No equilibrium selection**: Unique stationary distribution
4. **Evolutionary perspective**: Resistance to invasion, not just best-response
5. **Scalability**: Works for K-player, asymmetric, large games
6. **Unifying framework**: Links continuous (replicator) ↔ discrete (macro-model) ↔ MCCs

---

## Limitations

1. **Hyperparameter sweep**: Requires sweeping $\alpha$ to check convergence
2. **Meta-game construction**: Needs enough samples for accurate payoff estimates
3. **Descriptive not prescriptive**: Tells you evolutionary strengths, not what to play
4. **Single hyperparameter**: Choice of $\alpha$ affects results (though convergence detectable)

---

## Related

- [[AlphaRank_detailed]] — Implementation details, proofs, mathematical foundations
- [[PSRO]] / [[JPSRO]] — Population-based training methods (generate agents for α-Rank)
- [[NeuPL]] — Neural population learning (alternative to discrete populations)
- [[PSRO-rN]] — Rectified Nash Response (also uses response graphs)

## References

- **Paper**: Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution", Scientific Reports, 2019
- **Code**: Open-source implementation available
- **Related**: Replicator dynamics (Taylor & Jonker, 1978), Conley's Theorem (Conley, 1978), Evolutionary game theory (Weibull, 1997)
