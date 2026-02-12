# Quality-Diversity for Multi-Task Optimization

Two progressive algorithms for finding diverse solutions across multiple (or infinite) tasks.

---

## Overview

This folder contains two related algorithms for multi-task optimization:

### 1. **MTMB-ME** (Multi-Task Multi-Behavior MAP-Elites)
- **Key idea**: Crossover between tasks to leverage similarity
- **Paper**: Anne & Mouret (2023)
- **Status**: [[MTMB-ME]] overview | [[MTMB-ME_detailed]] implementation
- **Best for**: Finite set of related tasks with limited budget per task
- **Key strength**: 2× more solutions than independent MAP-Elites

### 2. **PT-ME** (Parametric-Task MAP-Elites)
- **Key idea**: Local linear regression to exploit continuous task structure
- **Paper**: Anne & Mouret (2024)
- **Status**: [[PT-ME]] overview | [[PT-ME_detailed]] implementation
- **Best for**: Continuous task space (infinite possible task parameters)
- **Key strength**: Asymptotic coverage + distillable to neural network

---

## Algorithm Comparison

```
MAP-Elites (2015): Single task diversity
    ↓
Multi-Task MAP-Elites: Multiple tasks, single solution per task
    ↓
MTMB-ME (2023): Multiple tasks, multiple diverse solutions per task
    ↓ Cross-task crossover leverages similarity
    ↓
    Problem: Can only solve finite task set, arbitrary discretization
    ↓
PT-ME (2024): Continuous task space, infinite possible tasks
    ↓ Local linear regression + continuous sampling
    ↓ Distillable to neural network for any task
```

---

## Quick Comparison

| Feature | MTMB-ME | PT-ME |
|---------|---------|-------|
| **Task space** | Finite (fixed set) | **Continuous (∞)** |
| **Task sampling** | Fixed set | **New random task/iteration** |
| **Variation** | Crossover between tasks | SBX + **linear regression** |
| **Operators** | GA-like (2) | Hybrid (SBX + regression) |
| **Archive** | One per task | One with CVT cells |
| **Distillation** | Limited | **Neural network for any θ** |
| **Best for** | 10-100 related tasks | Continuous task space |

---

## When to Use Each

### MTMB-ME

**✅ Use when**:
- Have **fixed set of tasks** (e.g., 50-200 objects to grasp)
- Tasks are **similar** (crossover beneficial)
- Limited **budget per task**
- Want **diverse solutions** (multiple grasps per object)

**Strength**: Cross-task crossover 2× more efficient than independent optimization

**Example**: Robot fault recovery (200 task variations, 25k evals)

### PT-ME

**✅ Use when**:
- Task space is **continuous** (robot morphologies, hyperparameters)
- Need solution for **any** task parameter (not just sampled)
- Want to **distill** into continuous function (NN)
- Black-box fitness (no gradients)

**Strength**: Asymptotic coverage, local linear regression, NN distillation

**Example**: Walking gaits for continuous leg-length variations

---

## Algorithm Progression

### MTMB-ME: Exploit Task Similarity

**Problem**: Independent MAP-Elites on 200 tasks wastes budget in unpromising regions.

**Solution**: Crossover between tasks that found good solutions → indirect sampling from solution subspace distribution.

```
Archives[Task_0] = {behavior_a: solution_x, behavior_b: solution_y}
Archives[Task_1] = {behavior_c: solution_z}

Crossover(x, z) + mutate → new candidate
Evaluate on random task → reuse successful patterns
```

**Result**: 2× more solutions with same budget!

### PT-ME: Extend to Continuous Tasks

**Problem**: MTMB-ME only solves finite tasks; discretizing continuous space is arbitrary.

**Solution**: Sample new task each iteration + exploit local smoothness via linear regression.

```
Archive stores:  (theta_0, x_0), (theta_1, x_1), (theta_2, x_2)
Delaunay finds: Adjacent cells (neighbors)
Regression: x ≈ M · theta (from neighbors)
Predict: x_new = M · theta_new + noise
```

**Result**: Asymptotic coverage of continuous space + NN distillation!

---

## File Structure

### Overview Files (High-Level)
- **MTMB-ME.md** (~150 lines): Problem, core idea, algorithm overview, when to use
- **PT-ME.md** (~200 lines): Parametric problem, key innovations, operators, comparisons

### Detailed Files (Full Implementation)
- **MTMB-ME_detailed.md** (~350 lines): Complete algorithm, all operators, worked examples
- **PT-ME_detailed.md** (~550 lines): Complete algorithm, linear regression, distillation, experiments

### Folder Guide
- **README.md** (this file): Algorithm relationships and when to use each

---

## Key Concepts

### MTMB-ME: Cross-Task Crossover

**Uniform crossover**: Each dimension from parent 1 or 2 (50/50 probability)

```python
offspring = []
for d_i, d_j in zip(parent1, parent2):
    offspring.append(d_i if random() < 0.5 else d_j)
```

**Why it works**: Combines successful patterns from different tasks → discovers stepping stones

### PT-ME: Local Linear Regression

**Model**: $x \approx M \cdot \theta$ (linear mapping from adjacent cells)

```python
M = (Θ^T Θ)^{-1} Θ^T X
x = M @ theta + noise
```

**Why it works**: Exploits local smoothness (small task changes → small solution changes)

---

## Experimental Results

### MTMB-ME: Humanoid Fault Recovery

**Domain**: 200 task variations, 25,000 evaluations

| Metric | MTMB-ME | Task-Wise ME | Grid Search |
|--------|---------|-------------|------------|
| Solved tasks % | **67.8%** | 47.1% | 57.9% |
| Solutions per task | **10.2** | 4.9 | 3.4 |
| Improvement | — | **2.1×** | 1.4× |

### PT-ME: 10-DoF Arm (Continuous Task Space)

**Domain**: Continuous 2D target space [0,1]²

| Algorithm | QD-Score @ 200 cells | QD-Score @ 1000 cells |
|-----------|-----|-----|
| **PT-ME** | **78.5** | **72.3** |
| MT-ME | 65.2 | 48.1 |
| PPO | 68.3 | 62.1 |

**Key finding**: PT-ME generalizes better (smaller degradation at higher resolution)

---

## Cross-References

**Related algorithms**:
- [[MAP-Elites]] — Base single-task QD
- [[GAME]] — Adversarial multi-task variant
- [[DCG-ME]] — Multi-task with descriptor conditioning
- [[Evolutionary Optimization]] — Main folder with all algorithms

---

## How to Use These Files

### For Quick Understanding
1. Read **overview** file (MTMB-ME.md or PT-ME.md)
2. Review comparison table above
3. Decide which algorithm fits your problem

### For Implementation
1. Read **overview** file for conceptual understanding
2. Read **_detailed** file for complete algorithm + hyperparameters
3. Review worked examples in **_detailed** file
4. Implement from pseudocode + implementation notes

### For Troubleshooting
- Check architecture diagrams in detailed files
- Review worked examples showing step-by-step execution
- Compare to reference implementations (cite papers)

---

## Summary

**Quality-Diversity for multi-task optimization** has two key flavors:

1. **MTMB-ME** (2023): Exploit task similarity via cross-task crossover
   - Finite tasks, limited budget → 2× more solutions
   
2. **PT-ME** (2024): Extend to continuous tasks via local regression
   - Infinite tasks, scalable to any task parameter → NN distillation

Choose based on whether your task space is **finite** (MTMB-ME) or **continuous** (PT-ME).

---

## Papers & Code

- **MTMB-ME**: Anne & Mouret, "Multi-Task Multi-Behavior MAP-Elites" (2023)
- **PT-ME**: Anne & Mouret, "Parametric-Task MAP-Elites" (2024)
- **Code**: Distilled implementations in _detailed.md files

---
