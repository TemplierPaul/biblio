# Game-Theoretic Rating Methods: From Nash Averaging to Deviation Ratings

## Overview

All game-theoretic rating methods answer the same question: **given a table of interactions (who beats who, or who scores what on which task), how do you turn that into a fair ranking?** They differ on what kinds of games they handle, how they pick a unique answer when many exist, and whether they're robust to duplicates (clone invariance).

**Core problem**: Standard ratings (Elo, uniform averaging) break under two conditions: (1) **intransitive interactions** (rock-paper-scissors cycles), and (2) **redundant strategies/tasks** that inflate or deflate ratings when duplicated.

**Key property — Clone Invariance**: Adding a copy of an existing strategy should not change anyone else's rating. Equivalent to Balduzzi et al.'s "invariance to redundancy" (P1). This is the gold standard for fair evaluation, because it means you can be maximally inclusive with data — no curation needed, no cherry-picking possible.

---

## The Methods

### 1. Nash Equilibrium (NE)

**What it is**: A mixture over strategies where no player benefits from unilaterally deviating.

**How it rates**: Compute NE of the meta-game, use the equilibrium mixture weights or expected payoffs as ratings.

**Strengths**:
- Well-understood, foundational concept
- Tractable in 2-player zero-sum (linear program)
- Prescriptive: tells you what to play

**Weaknesses**:
- **Multiple equilibria**: Many NE exist, giving different rankings — no principled way to choose
- **Intractable in general**: PPAD-complete for N-player or general-sum games
- **Not clone-invariant** on its own (no selection mechanism to handle duplicates)
- **Misses cycles**: Elo (which approximates NE thinking) has zero predictive power for intransitive interactions

**Use in practice**: Base meta-solver in standard PSRO. Simple LP in 2-player zero-sum.

---

### 2. Maximum Entropy Nash Equilibrium (Maxent NE)

**Paper**: "Re-evaluating Evaluation" (Balduzzi, Tuyls, Perolat, Graepel, NeurIPS 2018)

**What it is**: Among all Nash equilibria, select the one with maximum Shannon entropy — the most "spread out" unbeatable mixture.

**How it rates**: Construct antisymmetric logit matrix A from interaction data. Define two-player zero-sum meta-game with payoff p⊺Aq. Find the unique symmetric NE (p*, p*) maximizing H(p) = −Σ pᵢ log pᵢ. The **Nash average** n_A = A · p* gives each strategy a score.

**Why maxent works**: NE in a symmetric zero-sum game form a compact, convex, symmetric polytope. Shannon entropy is strictly concave → unique maximum on any compact convex set. This gives exactly one answer.

**Key properties**:
- **P1 — Clone-invariant**: Duplicating agent C with mass 1/3 in a 3-player cycle → maxent NE assigns (1/3, 1/3, 1/6, 1/6), splitting C's mass equally across copies. Other agents unaffected.
- **P2 — Continuous**: Small data perturbations → small payoff changes (though NE support can jump discontinuously)
- **P3 — Interpretable**: Purely cyclic game (div(A) = 0) → uniform distribution. Purely transitive game (A = grad(r)) → mass on best player(s).

**Also introduces**:
- **Hodge decomposition**: Splits any interaction matrix into transitive component (captured by Elo/averages) + cyclic component (rock-paper-scissors structure). Elo = divergence of the logit matrix, sees only the transitive part.
- **mElo₂ₖ**: Multidimensional Elo augmenting scalar rating with 2k-vector for cyclic interactions. Prediction: p̂ᵢⱼ = σ(rᵢ − rⱼ + cᵢ⊺Ωcⱼ). Demonstrated on AlphaGo variants.
- **Agent-vs-Task (AvT) framework**: Score matrix S embedded into antisymmetric block matrix, bipartite meta-game solved for Nash distributions over agents and tasks simultaneously.

**Limitations**:
- **Only works in 2-player zero-sum**: Relies on A being antisymmetric, NE being convex and symmetric
- **Adversarial task selection**: In AvT, the task distribution can concentrate on very few tasks, ignoring available data
- **Requires score normalization**: Raw scores across tasks must be comparable (e.g., normalize to [0,1])
- **Not used as PSRO meta-solver in practice**: PRD is faster, "good enough" for training; maxent NE matters more for evaluation than training

**Atari result**: Under Nash averaging, human ties with best RL agents — the apparent super-human performance was an artifact of the ALE being skewed toward environments RL agents do well on.

---

### 3. α-Rank

**Paper**: "α-Rank: Multi-Agent Evaluation by Evolution" (Omidshafiei et al., Scientific Reports, 2019)

**What it is**: Evolutionary dynamics ranking. Instead of "what would rational players do?", asks "if strategies competed in a population, which survive?"

**How it rates**: Construct Markov chain where strategies invade each other based on payoff differences (controlled by ranking-intensity α). Compute unique stationary distribution π. Strategies with more mass = higher rated.

**Strengths**:
- **Works for any game**: N-player, general-sum, asymmetric — no restrictions
- **Unique answer**: Single stationary distribution, no equilibrium selection problem
- **Polynomial-time**: Tractable even for large games
- **Captures dynamics**: Sees limit cycles and recurrent sets, not just fixed points

**Weaknesses**:
- **Not clone-invariant**: Duplicating a strategy changes the evolutionary dynamics and shifts other strategies' ratings
- **Descriptive, not prescriptive**: Tells you evolutionary strengths, not what to play
- **Hyperparameter α**: Must sweep to check convergence (though convergence is detectable)

**Relationship to maxent NE**: Both solve the equilibrium selection problem (both give unique answers). α-Rank does it via evolutionary dynamics; maxent NE does it via entropy. α-Rank trades clone invariance for generality beyond 2-player zero-sum.

> See [[AlphaRank]] for full details.

---

### 4. Coarse Correlated Equilibrium (CCE)

**What it is**: A mediator suggests a joint action profile. No player wants to opt out *before* hearing their recommendation. Relaxation of both CE and NE.

**Hierarchy**: NE ⊆ CE ⊆ CCE (CCE is the broadest, most permissive concept)

**Why it matters for rating**: 
- **Convex polytope**: The set of CCEs is always convex, even in N-player general-sum games → amenable to optimization
- **Tractable**: Computable via linear program for any game
- **Joint distributions**: Captures coordination that factorized NE mixtures cannot (critical for general-sum games with cooperative components)
- **Many CCEs exist**: Like NE, need a selection rule to get a unique rating

**Used by**: JPSRO (meta-solver), payoff ratings, deviation ratings as the underlying solution concept.

> See [[JPSRO]] for CCE in the PSRO context.

---

### 5. MECCE (Maximum Entropy Coarse Correlated Equilibrium)

**Paper**: "Game Theoretic Rating in N-player general-sum games with Equilibria" (Marris, Lanctot, Gemp, Omidshafiei et al., 2022)

**What it is**: The maxent NE idea applied to CCE. Among all coarse correlated equilibria, pick the one with maximum entropy.

**How it rates**: Solve convex optimization: max H(σ) subject to σ ∈ CCE polytope. Use the resulting distribution's marginals to define "payoff ratings" for each strategy.

**Strengths**:
- **N-player general-sum**: Works where maxent NE cannot
- **Unique answer**: Strictly concave objective on convex set → one solution
- **Convex optimization**: Efficient to compute
- **Captures coordination**: Joint distributions handle cooperative game components

**Weaknesses**:
- **Not clone-invariant**: When you duplicate a strategy, the CCE polytope changes geometry, and the entropy-maximizing point shifts in ways that affect other strategies' ratings. This is the key gap — it inherits maxent NE's *spirit* but not its defining *property*.

**Relationship to maxent NE**: Direct generalization. In 2-player zero-sum, CCE = NE, so MECCE reduces to maxent NE. Outside that setting, MECCE extends the approach but loses clone invariance.

**Also used in**: JPSRO as one meta-solver option (MGCE — Maximum Gini CE — uses Gini impurity instead of Shannon entropy, is computationally simpler, and was recommended by Marris et al. 2021 for JPSRO).

---

### 6. Deviation Ratings

**Paper**: "Deviation Ratings: A General, Clone-Invariant Rating Method" (Marris, Liu, Gemp, Piliouras, Lanctot, Feb 2025)

**What it is**: The first N-player general-sum clone-invariant rating method. Closes the gap that maxent NE identified but couldn't fill beyond 2-player zero-sum.

**How it rates**: Based on "deviation gains" — how much each strategy benefits from the equilibrium relative to opting out. Solved via sequential linear programs that peel off layers of the equilibrium structure (conceptually similar to Nash clustering in SF-PSRO, but grounded in CCE).

**Key properties**:
- **Clone-invariant**: ✅ First method to achieve this in N-player general-sum
- **Unique**: Always exists, single answer
- **Offset-invariant**: Adding a constant to all payoffs doesn't change ratings
- **Dominance-preserving**: Dominated strategies rated lower
- **Efficient**: Sequential LPs, polynomial in game size

**Why it's significant**: Nash averaging (2018) established that clone invariance is the right desideratum. α-Rank (2019) gave uniqueness for general games but not clone invariance. MECCE (2022) gave entropy selection for general games but not clone invariance. Deviation ratings (2025) finally deliver all three: unique + clone-invariant + N-player general-sum.

**Practical relevance**: Paper explicitly discusses Chatbot Arena (LLM evaluation), where user-submitted prompts test overlapping skills — exactly the redundancy problem. Without clone-invariant ratings, the prompt distribution biases model rankings.

---

## Comparison Table

| Method | Beyond 2p zero-sum | Unique answer | Clone-invariant | Compute cost | Philosophy |
|--------|:---:|:---:|:---:|---|---|
| **NE** | ❌ (PPAD-hard) | ❌ (many equilibria) | N/A | LP (2p-ZS) | Rational play |
| **Maxent NE** | ❌ | ✅ | ✅ | LP + entropy opt | Rational + most spread |
| **α-Rank** | ✅ | ✅ | ❌ | Polynomial (Markov chain) | Evolutionary survival |
| **CCE** | ✅ | ❌ (many) | N/A | LP | Coordinated rational play |
| **MECCE** | ✅ | ✅ | ❌ | Convex optimization | Coordinated + most spread |
| **Deviation Ratings** | ✅ | ✅ | ✅ | Sequential LPs | Coordinated + clone-proof |

---

## Historical Trajectory

```
2018: Maxent NE (Balduzzi et al.)
       └─ Established clone invariance as key desideratum
       └─ Only works in 2-player zero-sum
       │
2019: α-Rank (Omidshafiei et al.)          PSRO-rN (Balduzzi et al.)
       └─ Unique for any game                └─ Uses Hodge decomposition
       └─ Not clone-invariant                 └─ Gamescapes framework
       │
2022: Payoff Ratings / MECCE (Marris et al.)
       └─ Extends maxent idea to CCE
       └─ N-player general-sum
       └─ Not clone-invariant (the gap)
       │
2025: Deviation Ratings (Marris et al.)     VasE / Social Choice (Lanctot et al.)
       └─ Clone-invariant + general-sum       └─ Voting-theory alternative
       └─ Closes the 7-year gap               └─ Notes Nash averaging limitations
```

---

## Conceptual Connections

### Clone Invariance Has Been Independently Discovered Multiple Times
The Deviation Ratings paper notes that the same principle appears under different names: "Nash averaging" (Balduzzi 2018), "maximal lotteries" (social choice theory, Fishburn 1984), and "Yao's Principle" (complexity theory, Yao 1977). All involve finding a minimax distribution over a payoff matrix that is invariant to duplicating rows/columns.

### Hodge Decomposition Remains Underused
The transitive-cyclic decomposition from the 2018 paper provides a diagnostic tool that most subsequent work hasn't fully exploited. It can quantify *how much* a game's interactions are cyclic (where clone-invariant methods matter most) vs. transitive (where even Elo works fine). Potentially valuable for diagnosing multi-task QD and adversarial self-play settings.

### mElo Is a Separate Contribution That Persists
The multidimensional Elo (mElo₂ₖ) from the same paper is orthogonal to Nash averaging. It extends Elo's scalar rating with vectors capturing cyclic interactions, using the Schur decomposition. Less cited than Nash averaging but potentially useful as a behavior descriptor in QD archives with intransitive agent interactions.

---

## Relevance to QD and Adversarial Self-Play

### For Evaluation
- **MTMB-ME score matrices** (solutions × tasks) are natural AvT data for Nash averaging or deviation ratings
- **GAME's tournament matrices** between adversarial populations are AvA data
- Naive aggregation (sum QD-scores, average coverage) is biased by redundant tasks — clone-invariant methods fix this

### For Training
- **Task sampling in MTMB-ME**: Nash distribution p*_e over tasks provides principled non-uniform curriculum (upweight discriminating tasks, downweight easy/universally hard ones)
- **PSRO meta-solver**: Maxent NE not widely adopted (PRD is faster), but deviation ratings could inform oracle selection in QD-PSRO hybrids
- **GAME task selection**: Ranking task selection approximates Nash averaging via k-means; replacing with maxent NE or deviation ratings would give formal clone invariance

### For Diagnosis
- **Hodge decomposition** on MTMB-ME output: large ‖rot(A)‖/‖A‖ = multi-task crossover is essential; small = independent optimization per task suffices
- **mElo vectors** as behavior descriptors for agents in QD archives that interact intransitively

---

## When to Use Which

**Evaluating agents in 2-player zero-sum** → Maxent NE (clone-invariant, interpretable)

**Evaluating agents in N-player or general-sum** → Deviation Ratings (clone-invariant, general)

**Quick ranking of large population, any game** → α-Rank (fast, unique, but not clone-invariant)

**PSRO meta-solver** → PRD (fast approximate Nash) or MECCE/MGCE (for JPSRO in general-sum)

**Diagnosing transitive vs. cyclic structure** → Hodge decomposition (from maxent NE paper)

**Rating agents with intransitive interactions** → mElo₂ₖ (extends Elo with cyclic component)

---

## Related

- [[Nash Equilibrium]] — Foundational concept
- [[AlphaRank]] — Evolutionary alternative to equilibrium-based rating
- [[PSRO]] / [[PSRO_detailed]] — Uses NE/PRD as meta-solver
- [[JPSRO]] / [[JPSRO_detailed]] — Uses CCE/MECCE as meta-solver
- [[PRD_Rectified_Nash]] — Practical meta-solver comparison
- [[PSRO_Rectified_Nash_Response]] — Uses Hodge decomposition and gamescapes
- [[GAME]] — Adversarial QD using tournament-based evaluation
- [[MTMB-ME]] — Multi-task QD where Nash averaging applies to evaluation

## References

- **Nash Averaging / Maxent NE / mElo / Hodge**: Balduzzi, Tuyls, Perolat, Graepel, "Re-evaluating Evaluation", NeurIPS 2018
- **α-Rank**: Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution", Scientific Reports, 2019
- **PSRO-rN / Gamescapes**: Balduzzi et al., "Open-ended Learning in Symmetric Zero-sum Games", ICML 2019
- **Payoff Ratings / MECCE**: Marris, Lanctot, Gemp et al., "Game Theoretic Rating in N-player general-sum games with Equilibria", 2022
- **Deviation Ratings**: Marris, Liu, Gemp, Piliouras, Lanctot, "Deviation Ratings: A General, Clone-Invariant Rating Method", 2025
- **Social Choice Evaluation**: Lanctot et al., "Evaluating Agents using Social Choice Theory", 2025
- **Maximum Entropy CE**: Ortiz, Schapire, Kakade, "Maximum Entropy Correlated Equilibria", AISTATS 2007
