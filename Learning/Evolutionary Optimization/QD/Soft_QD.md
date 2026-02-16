# Soft Quality-Diversity (Soft-QD)

**Paper**: "Soft Quality-Diversity Optimization" (Hedayatian & Nikolaidis, USC)
**Key Innovation**: Continuous QD formulation that avoids discretizing behavior space

---

## Core Concept

**Soft-QD** rethinks Quality-Diversity optimization by eliminating the need to partition behavior space into discrete cells (archive tessellation). Instead, it treats each solution as a "light source" that illuminates the behavior space with intensity proportional to its quality, with influence decaying smoothly based on distance.

### The Problem with Traditional QD

Traditional QD algorithms (like MAP-Elites) partition behavior space into discrete cells and find the best solution per cell. This approach has two fundamental limitations:

1. **Not differentiable**: Discrete tessellations prevent direct gradient-based optimization
2. **Curse of dimensionality**: In high-dimensional behavior spaces:
   - Grid archives: number of cells grows exponentially
   - CVT archives: volume per cell grows exponentially

### The Soft-QD Solution

Instead of discrete cells, Soft-QD uses a **continuous formulation** where solutions contribute smoothly to multiple regions of behavior space.

---

## Key Components

### 1. Behavior Value Function

At any point **b** in behavior space, the **behavior value** induced by population θ is:

```
v_θ(b) = max_n [f_n · exp(-||b - b_n||² / 2σ²)]
```

Where:
- `f_n` = quality of solution n
- `b_n` = behavior descriptor of solution n
- `σ` = kernel width parameter (controls influence radius)

**Intuition**: Measures the quality of the best available solution for target behavior **b**, discounted exponentially by distance.

### 2. Soft QD Score

The total illumination over the entire behavior space:

```
S(θ) = ∫_B v_θ(b) db
```

This integral measures how well the population covers the behavior space with high-quality solutions.

### 3. Theoretical Properties

**Theorem**: Soft QD Score satisfies:

1. **Monotonicity**: Adding solutions or improving quality never decreases the score
2. **Submodularity**: Diminishing returns property (adding solutions to sparse regions helps more than dense regions)
3. **Limiting Equivalence**: As σ→0, converges to traditional QD Score

---

## SQUAD Algorithm

**SQUAD** (Soft QD Using Approximated Diversity) makes Soft QD Score tractable for optimization.

### The Objective

SQUAD optimizes a **lower bound** of Soft QD Score:

```
Objective = Σ f_n - (1/2) Σ_(i,j∈neighbors) √(f_i·f_j) · exp(-||b_i - b_j||²/γ²)
```

Two forces in equilibrium:
- **Attractive force** (Σf_n): Pushes solutions toward higher quality
- **Repulsive force** (pairwise term): Spreads solutions across behavior space

### Key Design Choices

1. **K-nearest neighbors**: Only compute repulsion between nearby solutions (exponential decay makes distant solutions negligible)
2. **Mini-batch updates**: Update population in batches for memory/computational efficiency
3. **Bounded spaces**: Use logit transformation b' = log(b/(1-b)) to map bounded→unbounded

### Computational Efficiency

- **Fast on simple domains**: <1 minute for 1000 iterations on Rastrigin (16-d behavior space)
- **Scalable to complex domains**: Works with differentiable renderers (Image Composition) and large networks (StyleGAN+CLIP)
- **Quick convergence**: Surpasses baselines in <200 iterations (can stop early in practice)

---

## When to Use Soft-QD

**Best for:**
- ✅ High-dimensional behavior spaces (where discretization suffers from curse of dimensionality)
- ✅ Differentiable domains (DQD settings with gradient information)
- ✅ Creative/generative tasks (image generation, design optimization)
- ✅ When you want end-to-end gradient-based optimization

**Traditional MAP-Elites better for:**
- ❌ Non-differentiable domains (evolutionary mutations work better)
- ❌ Low-dimensional behavior spaces (discretization is fine)
- ❌ When interpretability of discrete cells is important

---

## Comparison to Related Methods

### vs. Traditional MAP-Elites
- **MAP-Elites**: Discrete cells, evolutionary mutations, works in any domain
- **Soft-QD**: Continuous formulation, gradient-based, requires differentiability

### vs. SVGD (Stein Variational Gradient Descent)
- **SVGD**: Approximates probability distributions, no behavior space concept
- **Soft-QD**: Optimizes quality + diversity in behavior space, shares repulsive mechanism

### vs. Continuous QD Score (metric)
- **Continuous QD Score**: Non-smooth kernel, Monte Carlo estimation, evaluation only
- **Soft QD Score**: Smooth Gaussian kernel, analytical approximation, used for optimization

---

## Results Summary

SQUAD outperforms strong baselines (CMA-MAEGA, CMA-MEGA, DNS, DNS-G, Sep-CMA-MAE, GA-ME) on:

1. **Rastrigin** (benchmark): Better quality and coverage across all difficulties
2. **Image Composition**: Higher QD Score and QVS (quality-value score)
3. **Latent Space Illumination**: Superior diversity (Vendi Score), coverage, and mean quality

### Key Metrics
- **QD Score**: Sum of qualities across archive (or integral for Soft-QD)
- **QVS**: Quality × Vendi Score (quality × diversity)
- **Vendi Score**: Entropy-based diversity measure
- **Coverage**: Percentage of behavior space occupied

---

## Practical Insights

1. **Hyperparameters are robust**: Algorithm works well across range of k (neighbors) and batch sizes
2. **Logit transformation is critical**: For bounded behavior spaces, this preprocessing is essential
3. **Early stopping works**: Don't need full 1000 iterations, can stop when performance plateaus
4. **Gradient cost dominates runtime**: Use efficient backpropagation pipelines for complex domains

---

## Connection to Other Concepts

- **Quality-Diversity**: Core paradigm - optimize quality AND diversity simultaneously
- **Differentiable QD**: Assumes differentiable quality and descriptor functions
- **Gradient Arborescence**: Family of methods using gradient information in QD
- **Variational Inference**: Borrows repulsive force mechanism from SVGD
- **Kernel Methods**: Uses Gaussian kernels for smooth influence functions

---

## Key Takeaway

**Soft-QD** liberates Quality-Diversity from discrete archives, enabling **end-to-end differentiable optimization** in high-dimensional behavior spaces. SQUAD makes this practical through an efficient lower-bound approximation with attractive and repulsive forces.

**The big picture**: You can now do QD optimization the same way you do any other gradient-based ML optimization - no special archive management needed.
