# Multi-Task Multi-Behavior MAP-Elites (MTMB-ME)

**Quick reference**: [[MTMB-ME_detailed]]

---

## Overview

**MTMB-MAP-Elites** is a Quality-Diversity algorithm that finds **many diverse high-quality solutions for many tasks simultaneously**, combining task-level diversity (MAP-Elites) with multi-task efficiency (crossover between tasks).

**Authors**: Timothée Anne, Jean-Baptiste Mouret  
**Year**: 2023  
**Key Innovation**: Crossover between tasks to leverage similarity and find 2× more solutions than independent approaches

---

## The Problem

### Standard Approaches

| Method | Diversity | Multi-Task | Output |
|--------|-----------|------------|--------|
| **MAP-Elites** | ✅ Many diverse solutions | ❌ Single task | Archive for 1 task |
| **Multi-Task MAP-Elites** | ❌ One solution per task | ✅ Many tasks | 1 solution × n tasks |
| **MTMB-MAP-Elites** | ✅ Many diverse solutions | ✅ Many tasks | Archive for each task |

### Motivating Example: Robot Fault Recovery

**Scenario**: Humanoid robot with leg fault needs multiple contact strategies with a wall to regain balance.

**Why diversity?**
- Different faults require different solutions
- Repertoire of reflexes for robustness
- Handle partial observability (unexpected fault combinations)

**Challenge**: Find diverse solutions across 200 related tasks with limited evaluation budget (25,000 evals).

---

## Core Innovation: Cross-Task Crossover

### The Key Insight

**Problem**: Independent MAP-Elites on each task wastes evaluations in unpromising regions.

**Solution**: Use crossover between tasks that have already found good solutions.

```
Task 0 (successful): found solutions in region A
Task 1 (successful): found solutions in region B

Crossover: Combine from A and B → sample from distribution
centered on solution subspace → skip unpromising regions
```

**Result**: Indirect sampling that's 2× more efficient!

---

## Algorithm Overview

```
Initialize archives (one per task)
Random init until each task has ≥1 elite

Loop:
  1. Pick 2 random tasks with elites
  2. Get random elite from each task
  3. Crossover + mutate → offspring command
  4. Pick random task
  5. Evaluate offspring
  6. Update archive if better behavior or fitness
```

---

## When to Use MTMB-ME

✅ **Use when**:
- Need **diverse** solutions for **multiple** tasks
- Tasks have **similarity** (share structure)
- Limited budget per task
- Want **repertoire** of solutions (robot reflexes, grasping strategies)

❌ **Consider alternatives when**:
- Only need best solution per task → Multi-Task MAP-Elites
- Only have single task → MAP-Elites
- Tasks completely unrelated → Independent MAP-Elites

---

## Performance Summary

**Domain**: Humanoid robot fault recovery (200 tasks, 25k evals)

| Metric | MTMB-ME | Random | Grid | Task-Wise ME |
|--------|---------|--------|------|------|
| **Solved tasks %** | **67.8%** | 47.0% | 57.9% | 47.1% |
| **Solutions per task** | **10.2** | 4.9 | 3.4 | 4.9 |
| **Improvement** | — | +20.8% | +9.9% | **2×** |

**Key findings**:
- MTMB-ME leverages cross-task structure effectively
- 2× more solutions than independently-optimized tasks
- Handles limited budget per task (125 evals/task)

---

## Comparison to Related Methods

### vs MAP-Elites
- MAP-Elites: Single task, one archive
- MTMB-ME: Multiple tasks, one archive per task + cross-task crossover

### vs Multi-Task MAP-Elites
- MT-ME: Best single solution per task
- MTMB-ME: Multiple diverse solutions per task

### vs Independent MAP-Elites
- Independent: No sharing between tasks
- MTMB-ME: Crossover leverages similarity, 2× more solutions

---

## Key Concepts

**Archive Structure**: Dictionary of dictionaries
- `archives[task_id][behavior] = (command, fitness)`
- One archive per task
- Each archive stores elites (best per behavior)

**Crossover Strategy**:
- Uniform crossover: Each dimension from parent 1 or 2 (50/50)
- Mutation: Gaussian noise after crossover
- Task selection: Uniform random (no bias)

**Behavior Discretization**: Grid-based
- Continuous behaviors (e.g., robot hand positions) → discrete cells
- Grid size hyperparameter (e.g., 20cm squares)

---

## Advantages Over Baselines

1. **vs Random Search**: Exploits structure via crossover
2. **vs Grid Search**: Adaptive sampling on discovered solutions
3. **vs Task-Wise MAP-Elites**: With limited budget per task, crossover more efficient than single-task exploration

---

## References

- **Paper**: Anne & Mouret, "Multi-Task Multi-Behavior MAP-Elites", arXiv:2305.01264, 2023
- **Application**: Humanoid fault recovery (Talos robot in simulation)
- **Related**: MAP-Elites (Mouret & Clune, 2015), Multi-Task MAP-Elites (Pierrot et al., 2022)

---

**See `MTMB-ME_detailed.md` for complete implementation, worked examples, and all operators.**
