# DNS: Dominated Novelty Search

## Overview

**DNS** (Dominated Novelty Search) is a Quality-Diversity algorithm that eliminates the need for predefined descriptor space bounds, grids, or distance thresholds. It reformulates QD as a Genetic Algorithm with specialized fitness transformation: solutions compete based on their behavioral distance to better-performing solutions.

**Core Innovation**: Instead of explicit archives or grids, DNS uses a **dominated novelty score** that naturally adapts to descriptor space topology, enabling QD in unsupervised, high-dimensional, and discontinuous spaces.

---

## Key Problem Solved

### Limitations of Existing QD Algorithms

**MAP-Elites**:
- ✅ Effective for well-defined, low-dimensional spaces
- ❌ Requires predefined descriptor bounds
- ❌ Fixed grid structure wastes cells on unreachable regions
- ❌ Exponential cell growth in high dimensions (curse of dimensionality)
- ❌ Cannot handle unsupervised/learned descriptors

**Threshold-Elites** (Unstructured Archives):
- ✅ No grid structure
- ❌ Requires tuning distance threshold $l$ per domain
- ❌ Threshold varies wildly across problems (0.01 to 3.0 in experiments)
- ❌ Difficult to set without domain knowledge

**DNS Solution**:
- No predefined bounds needed
- No grid structure or cell allocation
- No distance threshold to tune
- Naturally adapts to descriptor space topology
- Scales from 2D to 1000+ dimensions
- Works with unsupervised learned descriptors

---

## Key Concepts

### 1. Dominated Novelty Score

**Intuition**: A solution should be preserved if it's either:
- **High fitness**: Better than others (no superior competitors)
- **Behaviorally novel**: Far from better-performing solutions

**Formal Definition**:

For solution $i$ with fitness $f_i$ and descriptor $\mathbf{d}_i$:

```
1. Identify fitter solutions: D_i = {j : f_j > f_i}

2. If D_i is empty:
     competition_fitness_i = +∞  (guaranteed preservation)

3. Otherwise:
     - Compute distances to all fitter solutions: d_ij = ||d_i - d_j||
     - Find k nearest fitter solutions: K_i
     - competition_fitness_i = average distance to K_i
```

**Example**:
- Solution A: Low fitness, far from all better solutions → High competition fitness → Preserved (stepping stone)
- Solution B: Low fitness, close to better solutions → Low competition fitness → Eliminated (redundant)
- Solution C: Best fitness in local region → +∞ competition fitness → Guaranteed preservation

### 2. Quality-Diversity as Fitness Transformation

**Traditional View**: QD algorithms use archives/grids as collection mechanisms

**DNS View**: QD algorithms are Genetic Algorithms with specialized **competition functions** that transform raw fitness into competition fitness.

**Unified Framework**:

```
Standard GA: selection(fitness)
QD Algorithm: selection(competition(fitness, descriptors))
```

**Competition Function Types**:
- **MAP-Elites**: Grid-based competition (compete within cells)
- **Threshold-Elites**: Distance threshold competition
- **DNS**: Dominated novelty competition

**Benefits of This View**:
- Unifies disparate QD approaches
- Opens design space for new competition functions
- Reveals QD as local competition mechanism
- Enables principled algorithm design

### 3. Parameter-Free QD

**What DNS Eliminates**:
- Descriptor space bounds (min/max values)
- Grid cell size or structure
- Distance thresholds
- Centroid initialization

**Only Parameter**: $k$ = number of nearest fitter neighbors (default: 5)

**Robustness**: Ablation studies show minimal sensitivity to $k$ (tested 1, 2, 5, 10)

---

## Algorithm Overview

### Main Loop

```
1. Initialize population X with random policies
2. FOR each generation:
    a. Reproduction: Generate B offspring using mutation/crossover
    b. Combine: Add offspring to population
    c. Evaluation: Compute fitness f and descriptors d for all
    d. Competition: Transform fitness using dominated novelty score
       For each solution i:
           - Find fitter solutions D_i = {j : f_j > f_i}
           - If D_i empty: competition_fitness_i = +∞
           - Else: competition_fitness_i = mean distance to k-nearest in D_i
    e. Selection: Keep top-N by competition fitness
```

**Key Difference from Standard GA**: Step (d) adds descriptor-based local competition

**Key Difference from MAP-Elites**: No grid structure; competition emerges from fitness hierarchy

---

## Experimental Results

### When DNS Excels

**1. Discontinuous Descriptor Spaces (Ant Blocks)**
- Environment with obstacles → unreachable descriptor regions
- **Result**: DNS significantly outperforms MAP-Elites (p < 0.001)
- **Why**: Grid wastes cells on unreachable regions; DNS adapts naturally

**2. High-Dimensional Descriptors (Kheperax n-point trajectories)**

| Dimensions | DNS vs. MAP-Elites |
|------------|-------------------|
| 2D | Competitive |
| 5D | Significant win (p < 0.05) |
| 10D-100D | Consistent wins (p < 0.05) |
| 1000D | Significant win (p < 0.05) |

**Why**: Grid cells grow exponentially (2^d); DNS scales linearly

**3. Unsupervised Descriptors (AURORA-style learned embeddings)**
- Maze navigation with learned descriptors (no predefined bounds)
- **Result**: DNS significantly outperforms all methods (p < 0.05)
- **Why**: MAP-Elites cannot be applied without bounds; Threshold-Elites requires complex tuning

**4. Implicit Constraints (Walker foot contacts)**
- Many descriptor combinations physically infeasible
- **Result**: DNS outperforms MAP-Elites (p < 0.001)
- **Why**: Grid allocates cells to infeasible regions; DNS only competes in achievable space

### When DNS is Competitive

**Well-Defined, Low-Dimensional Spaces (2D Ant)**
- Simple environment, regular descriptor space
- **Result**: DNS ≈ MAP-Elites (no significant difference)
- **Why**: Grid perfectly aligned; no advantage to dynamic adaptation

---

## When to Use DNS

### ✅ Good Fit

**1. Unsupervised QD**
- Learned descriptors (AURORA, variational autoencoders)
- Descriptor bounds unknown ahead of time
- Example: Learning behavior embeddings from trajectories

**2. High-Dimensional Descriptors**
- Descriptor dimensionality > 5-10
- Grid becomes impractical
- Example: Multi-objective optimization with many objectives

**3. Discontinuous or Complex Descriptor Spaces**
- Obstacles creating unreachable regions
- Non-convex achievable spaces
- Example: Robot navigation with physical constraints

**4. Rapid Prototyping**
- No domain expertise for setting bounds/thresholds
- Want algorithm that "just works"
- Example: Research exploration of new domains

### ❌ Poor Fit (Use MAP-Elites Instead)

**1. Well-Aligned Low-Dimensional Spaces**
- 2D-3D descriptor space
- Known bounds, convex space
- Grid perfectly represents achievable space
- Example: Simple locomotion tasks

**2. When Grid Interpretability Matters**
- Need human-interpretable niche structure
- Visual exploration of solution space
- Example: Generative design for human inspection

**3. Memory-Constrained Scenarios**
- DNS maintains full population, not just elite per cell
- MAP-Elites more memory-efficient for sparse solutions
- Example: Embedded systems with limited RAM

---

## Key Innovations

### 1. Dominated Novelty as Competition Function

**Problem**: How to combine fitness and novelty without explicit archives?

**Solution**: Competition based on distance to **fitter** solutions, not all solutions

**Benefits**:
- Preserves high-fitness local optima (no fitter neighbors)
- Preserves behaviorally novel solutions (far from better solutions)
- Eliminates redundant solutions (close to better solutions)

### 2. Reformulation of QD Framework

**Contribution**: Unifies QD algorithms under fitness transformation lens

**MAP-Elites as Competition Function**:
```
competition(f, d) = max fitness in cell(d)
```

**Threshold-Elites as Competition Function**:
```
competition(f, d) = f if min_distance_to_archive(d) > threshold else 0
```

**DNS as Competition Function**:
```
competition(f, d) = mean_distance_to_k_nearest_fitter(f, d)
```

### 3. Parameter-Free Adaptation

**Problem**: QD algorithms require domain-specific tuning

**Solution**: Local competition emerges from fitness hierarchy alone

**Example** (Threshold-Elites tuning requirements):
- 2D Kheperax: $l$ = 0.01
- 5D Kheperax: $l$ = 0.1
- 100D Kheperax: $l$ = 1.0
- 1000D Kheperax: $l$ = 3.0

DNS requires **no tuning** across all these cases.

### 4. Scalability to High Dimensions

**Problem**: Grid-based QD suffers curse of dimensionality

**MAP-Elites cell count**: $c^d$ (c cells per dimension, d dimensions)
- 10 cells/dim, 2D: 100 cells
- 10 cells/dim, 10D: 10 billion cells

**DNS complexity**: $O(N \log N)$ (independent of descriptor dimensionality)

---

## Comparison to Related Methods

| Method | Bounds Needed? | Grid Structure? | Distance Threshold? | Scales to High-D? |
|--------|----------------|-----------------|---------------------|-------------------|
| **MAP-Elites** | Yes | Yes | No | No |
| **CVT-MAP-Elites** | Yes | Yes (adaptive) | No | No |
| **Threshold-Elites** | No | No | Yes (tuned) | Moderate |
| **DNS** | No | No | No | Yes |

**DNS's Unique Position**: Only algorithm that is both parameter-free AND scales to high dimensions.

---

## Interview Relevance

### For Research Scientists

**Common Questions**:
- "How does DNS differ from Novelty Search?"
  - DNS combines fitness and novelty; NS ignores fitness entirely. DNS uses distance to **fitter** solutions, not all solutions.
- "Why reformulate QD as fitness transformation?"
  - Unifies disparate approaches, reveals fundamental mechanism (local competition), enables principled design of new algorithms
- "What is the dominated novelty score's intuition?"
  - Solutions compete with better-performing neighbors; novel behaviors preserved even if low fitness

**Discussion Topics**:
- Local competition in evolutionary algorithms
- Curse of dimensionality in grid-based QD
- Unsupervised descriptor learning
- Trade-offs between explicit archives and implicit competition

### For ML Engineers

**Practical Considerations**:
- Framework: Drop-in replacement for MAP-Elites in QDax/pyribs
- Parameter tuning: Set $k$ = 5 and forget
- Memory: Maintains full population (more than MAP-Elites elite-per-cell)
- Computation: $O(N \log N)$ distance computations per generation

**Applications**:
- Neural architecture search with many objectives
- Hyperparameter optimization with learned embeddings
- Multi-objective RL with high-dimensional behavior spaces
- Generative model diversity (when features not predefined)

---

## Connection to Other Topics

- **[[Quality_Diversity]]**: DNS as alternative QD algorithm without grids
- **[[MAP_Elites]]**: Foundation algorithm DNS extends/replaces
- **[[Novelty_Search]]**: DNS combines novelty with fitness (NS ignores fitness)
- **[[Evolution_Strategies]]**: DNS as local competition mechanism for ES
- **[[AURORA]]**: DNS enables unsupervised QD with learned descriptors
- **[[Multi_Objective_Optimization]]**: DNS naturally handles many objectives

---

## Limitations

**Current Constraints**:
1. **Memory**: Maintains full population, not just elites per cell
2. **Interpretability**: No explicit grid structure for visualization
3. **Reproducibility**: Stochastic competition (no fixed cells)
4. **Analysis**: Harder to analyze descriptor space coverage

**When Not to Use**:
- Simple 2D-3D spaces with known bounds (MAP-Elites sufficient)
- Need human-interpretable grid structure
- Severe memory constraints

---

## Quick Summary

**What**: QD algorithm using dominated novelty score for parameter-free local competition

**Why**: Enables QD without predefined bounds, grids, or distance thresholds

**How**:
1. Compute fitness and descriptors for all solutions
2. For each solution, find fitter neighbors
3. Competition fitness = average distance to k-nearest fitter neighbors
4. Select solutions with high competition fitness

**Key Results**:
- Outperforms MAP-Elites in high-D (5D-1000D), discontinuous, and unsupervised settings
- Competitive with MAP-Elites in standard low-D settings
- No parameter tuning required (k = 5 default)

---

**See [[DNS_detailed]] for implementation details, pseudocode, distance computations, and parameter sensitivity analysis.**
