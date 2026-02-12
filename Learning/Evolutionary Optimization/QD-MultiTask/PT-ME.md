# Parametric-Task MAP-Elites (PT-ME)

**Quick reference**: [[PT-ME_detailed]]

---

## Overview

**PT-ME** is a black-box algorithm for **continuous multi-task optimization**. It finds solutions for ANY task parameter in a continuous space, not just sampled tasks. Results can be distilled into a neural network that maps task parameters to optimal solutions.

**Authors**: Timothée Anne, Jean-Baptiste Mouret  
**Year**: 2024  
**Key Innovation**: Local linear regression operator that exploits task structure; continuous task coverage via sampling

---

## The Problem: Parametric-Task Optimization

### Standard Problem

Find a function $G: \Theta \to \mathcal{X}$ that maps **any** task parameter to its optimal solution:

$$G(\theta) = x^*_{\theta} = \text{best solution for task } \theta$$

where:
- $\Theta = [0,1]^{d_\theta}$ — **Continuous** task space (e.g., robot morphology dimensions)
- $\mathcal{X} = [0,1]^{d_x}$ — Solution space (e.g., controller parameters)
- $f(x, \theta)$ — Black-box fitness function

### Why Continuous Task Space Matters

**Robotics example**: Walking gaits for various leg lengths
- MT-ME discretizes to 2000 tasks → samples only 2000 points in continuous space
- PT-ME samples new task each iteration → asymptotically covers entire space
- With 100k evals: PT-ME visits 100k unique task parameters vs MT-ME's 2000

**Challenge**: Can't just run independent optimization per task (infinite tasks!)

---

## Key Innovation: Local Linear Regression

### The Insight

**SBX Crossover** alone doesn't exploit multi-task structure—it ignores which task you're solving.

**Solution**: Use **local linear regression** to predict solutions for nearby tasks:

```
Archive contains (task₀, solution₀), (task₁, solution₁)
New task ≈ [0.35, 0.45]

Linear model: solution ≈ M · task_parameter
Regression on adjacent cells: M = (Θᵀ Θ)⁻¹ ΘᵀX

Predict: x_pred = M · [0.35, 0.45]
Add noise: x = x_pred + σ_reg · N(0, variance)
Evaluate: Try this solution, likely good!
```

**Why it works**: Task structure is locally smooth (small changes in task → small changes in optimal solution)

---

## Algorithm Structure

### Two Complementary Operators (50/50 split)

**Operator 1: SBX with Tournament** (50%)
- Select 2 parents from archive
- Sample candidate tasks, pick closest to parent
- Apply Simulated Binary Crossover
- Tuned via bandit (no manual adjustment)

**Operator 2: Local Linear Regression** (50%) ✨
- Sample random task
- Find adjacent cells via Delaunay triangulation
- Build local linear model from neighbors
- Predict solution for new task
- Add exploration noise

### Archive

- Discretize task space $\Theta$ into $N$ cells (e.g., $N=200$)
- Each cell stores: (task_param, solution, fitness)
- **Key difference**: Evaluate on sampled tasks $\theta$, not just centroids

### Evaluation & Distillation

- Store **all evaluations** during optimization
- Re-archive at high resolution (10k cells)
- Train neural network: task → solution
- **Result**: Can query solution for ANY task parameter

---

## When to Use PT-ME

✅ **Use when**:
- Task space is **continuous** (not just 10-20 discrete tasks)
- Need solution for **any** task parameter (not pre-sampled)
- Black-box fitness (no gradients)
- Want to **distill** knowledge into continuous function (NN)
- Tasks have **smooth structure** (local similarity)

❌ **Use alternatives when**:
- Only finite set of tasks → MT-ME
- Function is differentiable → Parametric programming
- Need sequential decision-making → Deep RL
- Tasks completely unrelated → Independent optimization

---

## Performance Summary

**10-DoF Arm** (100k evaluations, continuous task space [0,1]²):

| Algorithm | QD-Score @ 200 cells | QD-Score @ 1000 cells |
|-----------|-----|-----|
| **PT-ME** | **78.5** | **72.3** |
| MT-ME (2000 tasks) | 65.2 | 48.1 |
| PPO | 68.3 | 62.1 |
| CMA-ES | 45.0 | 35.2 |

**Archery** (sparse rewards):
- PT-ME: Only method that reliably solves all task variants
- Others: Struggle with sparse reward signal

**Door-Pulling** (realistic robotics):
- PT-ME: Best performance on complex 9-DoF humanoid manipulation

---

## Key Differences from MT-ME

| Aspect | MT-ME | PT-ME |
|--------|-------|-------|
| Task sampling | Fixed centroids only | New random task each iteration |
| Task coverage | Finite (e.g., 2000) | Asymptotically complete (∞) |
| Selection pressure | ∝ 1/n_cells → degrades | Independent of n_cells ✅ |
| Variation | SBX + tournament | SBX + **linear regression** |
| Output | Discrete archive | Archive + **neural network** |
| Efficiency | Decreases with task count | Constant efficiency |

---

## Distillation: From Discrete to Continuous

### Why Needed

Archive has only ~200-10,000 cells, but task space is continuous (infinite possible tasks).

### Process

1. **Re-archive** all evaluations at high resolution (10k cells)
2. **Train neural network**: task parameter → solution
3. **Query**: Get solution for ANY task parameter
4. **Evaluate**: NN generalizes beyond training distribution

### Neural Network Architecture

Simple 2-layer MLP:
```
Input: task_parameter (d_θ dims)
  ↓
Linear(d_θ → 64) + Tanh
  ↓
Linear(64 → 64) + Tanh
  ↓
Linear(64 → d_x) + Sigmoid
  ↓
Output: solution (d_x dims)
```

---

## Hyperparameters (Robust Defaults)

- $B = 100{,}000$ — Evaluation budget
- $N = 200$ — Number of archive cells
- $S = \{1, 5, 10, 50, 100, 500\}$ — Tournament sizes for bandit
- $\sigma_{SBX} = 10.0$ — SBX mutation strength
- $\sigma_{reg} = 1.0$ — Regression noise factor

**Notes**:
- $N=200$ robust across task dimensions (not dimension-dependent)
- Bandit avoids tuning tournament size
- $\sigma_{reg}$ could benefit from per-problem tuning but default works well

---

## Comparison to Related Methods

### vs Multi-Task MAP-Elites (MT-ME)
- MT-ME: Discretizes continuous space (loses information)
- PT-ME: Asymptotically covers continuous space (better generalization)

### vs Parametric Programming
- Parametric Programming: Exact but requires differentiable functions
- PT-ME: Approximate but works on any black-box function

### vs PPO (Policy Gradient)
- PPO: Neural network trained via RL (needs environment interaction)
- PT-ME: Direct optimization + neural network distillation (sample efficient)

---

## Key Insights

1. **Continuous coverage** >> discretization (sampling new task each iteration)
2. **Local linear regression** >> blind crossover (exploit task structure)
3. **De-correlated selection pressure** (efficiency doesn't degrade with task count)
4. **Delaunay triangulation**: Automatic neighbor selection, no tuning
5. **Distillation**: Enable continuous function approximation from discrete evals

---

## Implementation Highlights

### Delaunay Triangulation (Automatic Neighbors)

Instead of tuning k-nearest neighbors or ε-ball radius:
```python
from scipy.spatial import Delaunay
delaunay = Delaunay(centroids)
neighbors = delaunay.neighbors[cell_idx]  # Automatic!
```

**Advantages**:
- ✅ Adapts to dimensionality automatically
- ✅ No hyperparameter to tune
- ✅ Precomputed once

### Linear Least Squares

Solve for linear mapping matrix $M$ from adjacent cells:
```python
# X ≈ Θ · M
M = np.linalg.lstsq(Theta_adj, X_adj, rcond=None)[0]
x_pred = M @ theta_new + noise
```

### UCB1 Bandit (Adaptive Tournament Size)

```
If SBX succeeds: increase likelihood of that tournament size
If SBX fails: try different tournament size
No manual tuning needed!
```

---

## References

- **Paper**: Anne & Mouret, "Parametric-Task MAP-Elites", arXiv:2402.01275, 2024
- **Code**: https://zenodo.org/doi/10.5281/zenodo.10926438
- **Related**: MT-ME (Pierrot et al., 2022), MAP-Elites (Mouret & Clune, 2015)
- **Comparison**: Parametric programming (Pistikopoulos et al., 2007), PPO (Schulman et al., 2017)

---

**See `PT-ME_detailed.md` for complete algorithm, worked examples, and all implementations.**
