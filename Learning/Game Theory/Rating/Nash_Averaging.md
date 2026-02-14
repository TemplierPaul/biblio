# Nash Averaging for Evaluation

**Quick reference**: [[Nash_Averaging_detailed]]

---

## Overview

**Nash Averaging** is an **invariant evaluation method** for multi-agent and multi-task settings. It automatically adapts to redundant agents/tasks, preventing bias from overrepresented evaluations.

**Authors**: David Balduzzi, Karl Tuyls, Julien Perolat, Thore Graepel (DeepMind)  
**Paper**: "Re-evaluating Evaluation", 2018  
**Key Innovation**: Maxent Nash equilibrium on meta-game automatically detects redundancy

---

## The Problem

### Current Evaluation Issues

**Problem 1**: Elo ratings fail on cyclic games
- Rock-paper-scissors: All get same Elo (no predictive power)
- AlphaGo variants: Elo can't handle intransitive interactions

**Problem 2**: Averages biased by redundancy
- Adding weak agents inflates Elo of agents that beat them
- Overrepresenting easy tasks inflates average performance
- Manual curation required (doesn't scale)

**Need**: Evaluation method that's **invariant to redundancy**

---

## Key Questions Addressed

**Q1**: Do tasks test what we think they test?  
→ Use Schur decomposition to find latent skills/tasks

**Q2**: When is a task/agent redundant?  
→ Maxent Nash automatically assigns low weight

**Q3**: Which tasks and agents matter most?  
→ Maxent Nash support = core agents/tasks

**Q4**: How should evaluations be evaluated?  
→ Meta-game on evaluation data

---

## Two Main Contributions

### 1. Multidimensional Elo (mElo)

**Problem**: Standard Elo assumes transitive skill (A > B, B > C ⇒ A > C)

**Solution**: Add cyclic component via low-rank approximation

```
Standard Elo: p̂_ij = σ(r_i - r_j)

mElo_2k: p̂_ij = σ(r_i - r_j + c_i^T Ω c_j)
         ├─ transitive ─┤  ├─ cyclic ─┤
```

**Result**: mElo₂ reduces error by 59% on AlphaGo predictions

### 2. Nash Averaging

**Core Idea**: Play meta-game on evaluation data
- Row player picks distribution over agents
- Column player picks distribution over tasks
- Maxent Nash equilibrium = unbeatable team

**Key Property**: **Invariant to redundancy**

Example:
```
Game: {A, B, C} vs Game: {A, B, C₁, C₂}
      (rock-paper-scissors)  (C duplicated)

Maxent Nash: (1/3, 1/3, 1/3)  vs  (1/3, 1/3, 1/6, 1/6)
                                    └─ C split evenly ─┘

Nash average: (0, 0, 0)  ==  (0, 0, 0, 0)
             └─ correctly detects no agent dominates ─┘
```

---

## When to Use Nash Averaging

✅ **Use when**:
- Evaluating many agents on many tasks/environments
- Risk of redundant tasks (e.g., ALE with 50+ games)
- Risk of redundant agents (e.g., many hyperparameter variants)
- Want **automatic** evaluation (no manual curation)
- Need **invariance** to duplication

❌ **Consider alternatives when**:
- Small number of agents/tasks (< 5)
- All tasks known to be diverse
- User has specific importance weights for tasks
- Need transitivity assumption (then use Elo)

---

## Algebraic Framework

### Hodge Decomposition

Any antisymmetric matrix decomposes into:

```
A = grad(r) + rot(A)
    └─ transitive ─┴─ cyclic ─┘
```

**Transitive component**: grad(r) = r·1^T - 1·r^T
- Elo ratings or averages
- A_ij = r_i - r_j

**Cyclic component**: rot(A)
- Rock-paper-scissors patterns
- Curl ≠ 0

**When Elo works**: curl(A) = 0 (purely transitive)

---

## Performance Summary

### AlphaGo Evaluation (8 variants)

| Metric | Elo | mElo₂ |
|--------|-----|-------|
| Frobenius error | 0.85 | **0.35** (-59%) |
| Logistic loss | 1.41 | **1.27** (-10%) |

**Key finding**: mElo₂ correctly predicts non-transitive interactions (α_value, α_policy, Zen)

### Atari Re-evaluation (20 agents, 54 games)

**Uniform averaging**: Rainbow >> Human (appears superhuman)

**Nash averaging**: Rainbow ≈ Human (tied!)

**Insight**: ALE skewed towards games current agents excel at; Nash rebalances to include human-specific skills

---

## Comparison to Related Methods

### vs Standard Elo
- Elo: Transitive ratings only, fails on cycles
- Nash: Handles cyclic + transitive via mElo

### vs Uniform Averaging
- Uniform: Biased by redundancy
- Nash: **Invariant** to redundancy

### vs AlphaRank
- AlphaRank: Ranks agents via evolutionary dynamics
- Nash Averaging: Evaluates via maxent Nash on meta-game
- Both handle cyclic interactions, complementary approaches

---

## Key Concepts

**Meta-game**: Game played on evaluation data
- Players: Row player (picks agents), Column player (picks tasks)
- Payoff: Expected log-odds under joint distribution
- Solution: Maxent Nash equilibrium

**Maxent Nash**: Unique symmetric Nash with maximum entropy
- Maximally indifferent between equal performers
- Automatically distributes mass over redundant players

**Nash average**: $n_A = A · p^*$
- Performance against maxent Nash team
- Invariant to redundancy

---

## Three Desired Properties

**P1. Invariant**: Adding redundant copies makes no difference

**P2. Continuous**: Robust to small data changes

**P3. Interpretable**:
- Cyclic game (div(A) = 0) → uniform Nash
- Transitive game (A = grad(r)) → Nash on best player(s)

---

## Advantages

1. **Automatic**: No manual task selection needed
2. **Scalable**: Benefit from including all available data
3. **Unbiased**: Easy tasks/weak agents don't skew results
4. **Interpretable**: Core agents/tasks = Nash support

---

## Limitations

1. **Garbage in, garbage out**: Quality depends on input data diversity
2. **Adversarial**: Nash takes harsh perspective (may not suit all contexts)
3. **Discontinuous**: Nash can jump at phase transitions
4. **Computational**: Requires solving for maxent Nash (LP solver)

---

## References

- **Paper**: Balduzzi et al., "Re-evaluating Evaluation" (DeepMind, 2018)
- **Related**: Elo ratings (1978), TrueSkill (Herbrich et al., 2007)
- **Theory**: Combinatorial Hodge theory (Jiang et al., 2011), Empirical game theory (Walsh et al., 2003)
- **Applications**: AlphaGo evaluation, Atari re-evaluation

---

**See `Nash_Averaging_detailed.md` for complete algorithms, Hodge decomposition mathematics, and worked examples.**
