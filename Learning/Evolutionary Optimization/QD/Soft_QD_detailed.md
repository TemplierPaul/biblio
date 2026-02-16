# Soft Quality-Diversity: Technical Implementation Details

Complete technical reference for implementing Soft-QD and SQUAD from scratch.

---

## Problem Formulation

### Standard QD Problem

Given:
- Solution space: Θ
- Quality function: f: Θ → ℝ
- Behavior descriptor: desc: Θ → B (behavior space)

Goal: Find diverse set of high-quality solutions {θ_b}_{b∈B} maximizing:
```
∫_B f(θ_b) db
```

### Traditional Discrete Approach

Partition behavior space into n cells A = {c₁, ..., cₙ} (archive/tessellation).

QD Score:
```
max_θ QD_Score_A(θ) = Σ_{c∈A} max{f(θ) : θ∈θ, desc(θ)∈c}
```

**Limitations**:
1. Grid archives: |A| grows exponentially with dim(B)
2. CVT archives: volume per cell grows exponentially with dim(B)
3. Non-differentiable (can't use gradient descent directly)

---

## Soft QD Formulation

### Behavior Value Function

For population θ = {θ₁, ..., θ_N}, define behavior value at point **b**:

```
v_θ(b) = max_{1≤n≤N} f_n · exp(-||b - b_n||² / 2σ²)
```

Where:
- f_n = f(θ_n) (quality of solution n)
- b_n = desc(θ_n) (behavior descriptor of solution n)
- σ² = kernel bandwidth parameter

**Interpretation**: Gaussian kernel centered at each solution's behavior, weighted by quality. Take max across all solutions.

### Soft QD Score

```
S(θ) = ∫_B v_θ(b) db
```

**Properties**:

**Theorem 1** (Monotonicity):
- Adding solutions: S(θ ∪ {θ'}) ≥ S(θ)
- Improving quality: If f(θ'_i) ≥ f(θ_i), then S(θ') ≥ S(θ)

**Theorem 2** (Submodularity):
For populations θ₁ ⊆ θ₂ and solution θ:
```
S(θ₁ ∪ {θ}) - S(θ₁) ≥ S(θ₂ ∪ {θ}) - S(θ₂)
```
(Diminishing returns: adding to sparse regions helps more)

**Theorem 3** (Limiting Equivalence):
As σ→0:
```
S(θ) → (2πσ²)^(d/2) · QD_Score_A(θ)
```
where A is fine-grained archive

---

## SQUAD: Tractable Optimization

### The Challenge

Computing S(θ) = ∫_B max_n [f_n·exp(-||b-b_n||²/2σ²)] db is intractable.

### Lower Bound Derivation

**Step 1**: Max-to-sum relaxation (Jensen's inequality)
```
max_n [f_n·exp(-||b-b_n||²/2σ²)] ≥ Σ_n f_n·exp(-||b-b_n||²/2σ²) - correction
```

**Step 2**: Integrate and apply bounds to get approximation:
```
S(θ) ≥ (2πσ²)^(d/2) · [Σ_n f_n - (1/2)Σ_{i≠j} √(f_i·f_j)·exp(-||b_i-b_j||²/8σ²)]
```

**Step 3**: Reparameterize γ² = 8σ²/d for clarity:
```
Ṽ(θ) = (2πσ²)^(d/2) · [Σ_n f_n - (1/2)Σ_{i≠j} √(f_i·f_j)·exp(-||b_i-b_j||²/γ²)]
```

### SQUAD Objective

Maximize lower bound Ṽ(θ):

```
Objective = Σ_n f_n - (1/2) Σ_{i,j∈neighbors} √(f_i·f_j) · exp(-||b_i-b_j||²/γ²)
           └─ Quality term ─┘  └──────────────── Diversity (repulsion) term ────────────────┘
```

**Interpretation**:
- **Quality term**: Attracts solutions toward high fitness
- **Repulsion term**: Pushes solutions apart in behavior space
- **√(f_i·f_j)**: Stronger repulsion between high-quality solutions
- **exp(...)**: Repulsion decays exponentially with distance

### Computational Optimizations

#### 1. K-Nearest Neighbors Approximation

Only compute repulsion for k-nearest neighbors in behavior space:
```
Objective ≈ Σ_n f_n - (1/2) Σ_{i∈batch, j∈N_i} √(f_i·f_j)·exp(-||b_i-b_j||²/γ²)
```

**Justification**: Exponential decay makes distant solutions contribute negligibly.

**Typical k**: 5-20 (see ablation studies)

#### 2. Mini-Batch Updates

Update population in batches of size M (rather than all N at once):
- Reduces memory footprint
- Allows larger populations
- Typical batch size: 32-256

#### 3. Bounded Behavior Spaces

If B = [0,1]^d (bounded), apply **logit transformation**:
```
b' = log(b / (1-b))  ∈ ℝ^d  (unbounded)
```

**Critical**: Experiments show this transformation is essential for success.

---

## SQUAD Algorithm (Complete Pseudocode)

```
Algorithm: SQUAD (Soft QD Using Approximated Diversity)

Input:
  - Optimizer O (e.g., Adam, SGD)
  - Learning rate η
  - Population size N
  - Batch size M
  - Number of neighbors K
  - Max epochs T_max
  - Kernel bandwidth γ²
  - Evaluation function Eval(θ) → (quality f, descriptor b)

Initialize:
  - Population θ = {θ_1, ..., θ_N}
  - Evaluations (F, B) ← Eval(θ)  # qualities and behaviors
  - Optimizer state S ← O.init(θ)

For t = 1 to T_max:
    For each batch of indices I ⊆ {1, ..., N}:
        # Find k-nearest neighbors for each solution in batch
        For each i ∈ I:
            N_i ← K-Nearest-Neighbors(b_i, B)

        # Compute objective for batch
        S_I(θ) = Σ_{i∈I} f_i - (1/2) Σ_{i∈I, j∈N_i} √(f_i·f_j)·exp(-||b_i-b_j||²/γ²)

        # Compute gradients
        G_I ← ∇_{θ_I} S_I(θ)

        # Update parameters
        (θ_I, S_I) ← O.update(θ_I, G_I, S_I, η)

        # Re-evaluate updated solutions
        (F_I, B_I) ← Eval(θ_I)

Return: Final population θ
```

### Gradient Computation Details

The gradient ∇_θ S_I(θ) requires:

1. **Quality gradient**: ∇_{θ_i} f_i (from backprop through f)
2. **Descriptor gradient**: ∇_{θ_i} b_i (from backprop through desc)
3. **Chain rule through repulsion term**:
   ```
   ∂/∂θ_i [√(f_i·f_j)·exp(-||b_i-b_j||²/γ²)]
   ```

**Requires**: Both f and desc must be differentiable w.r.t. θ (Differentiable QD)

---

## Hyperparameter Selection

### Kernel Bandwidth γ²

Controls repulsion range in behavior space.

**Heuristic**: Set based on expected behavior space spread
```
γ ≈ (typical distance between diverse solutions) / 2
```

**Effect**:
- Too small: Solutions cluster (weak repulsion)
- Too large: Uniform repulsion (ignores behavior geometry)

**Typical range**: 0.1 - 2.0 (normalized behavior space)

### Number of Neighbors K

**Ablation results**: Algorithm robust to K ∈ [5, 50]

**Recommendation**:
- Start with K = 10
- Increase for denser populations
- Computational cost scales linearly with K

### Batch Size M

**Trade-off**:
- Larger M: More stable gradients, higher memory
- Smaller M: More frequent updates, lower memory

**Recommendation**:
- M = 32-128 for most problems
- Adjust based on available memory

### Population Size N

**Depends on**:
- Dimensionality of behavior space
- Desired coverage

**Typical**: N = 100-1000

**Rule of thumb**: Higher-dimensional behavior spaces need larger N

### Learning Rate η

**Depends on optimizer**:
- Adam: η = 1e-3 to 1e-2
- SGD: η = 1e-1 to 1

**Recommendation**: Use adaptive optimizers (Adam, RMSprop) for stability

---

## Implementation Considerations

### 1. K-Nearest Neighbors Computation

**Efficient methods**:
- KD-tree (if d < 20)
- Ball-tree (if d < 50)
- Approximate methods (FAISS, Annoy) for high dimensions

**Update frequency**:
- Recompute every iteration (behaviors change)
- Can cache if behaviors change slowly

### 2. Bounded Behavior Space Handling

**Standard normalization** (if bounds known):
```python
# Map [a, b] → [0, 1]
b_normalized = (b - a) / (b - a)

# Apply logit transform
b_unbounded = log(b_normalized / (1 - b_normalized))
```

**Clipping**: Avoid numerical issues near 0 and 1
```python
b_clipped = clip(b_normalized, ε, 1-ε)  # ε = 1e-6
```

### 3. Gradient Stability

**Issues**:
- √(f_i·f_j): Undefined gradient if f ≤ 0
- exp(-||b_i-b_j||²/γ²): Can vanish for distant solutions

**Solutions**:
- Ensure f > 0 (use softplus or add offset)
- Use mixed precision training for numerical stability

### 4. Initialization

**Quality-aware initialization**:
1. Random initialization
2. Pre-train with quality objective only (no repulsion)
3. Then optimize full SQUAD objective

**Benefit**: Ensures population has reasonable quality before diversifying

---

## Evaluation Metrics

### 1. Soft QD Score (S(θ))

The optimization objective itself (via lower bound).

**Use**: Track during training to verify improvement

### 2. Traditional Metrics (for comparison)

If comparing to archive-based methods:

**QD Score** (on CVT archive):
```
QD_Score = Σ_{c∈A} max{f(θ) : desc(θ)∈c}
```

**Coverage**:
```
Coverage = (# occupied cells / # total cells) × 100%
```

### 3. Diversity Metrics

**Vendi Score**: Entropy-based diversity in behavior space
```
VS(θ) = exp(H(eigenvalues of K))
```
where K is kernel matrix K_ij = exp(-||b_i-b_j||²/σ²)

**QVS** (Quality-Value Score):
```
QVS = mean_quality × Vendi_Score
```

**Mean/Max Objective**:
```
Mean = (Σ f_i) / N
Max = max_i f_i
```

---

## Domain-Specific Tips

### Image Generation / Creative Tasks

**Quality function**: Use pretrained models (CLIP, aesthetic predictors)

**Behavior descriptor**:
- Latent space coordinates (StyleGAN w space)
- Semantic features (CLIP embeddings)
- Hand-designed (color, composition)

**Tip**: Normalize descriptors to [0,1] before logit transform

### Robotics / Control

**Quality function**: Task reward (cumulative return)

**Behavior descriptor**:
- Final state features
- Trajectory statistics
- Visited positions (QD for locomotion)

**Tip**: May need larger populations (N=500-1000) for complex spaces

### Optimization Benchmarks (Rastrigin, etc.)

**Quality function**: -f(x) where f is function to minimize

**Behavior descriptor**:
- Subset of decision variables
- PCA of decision variables
- Function-specific features

**Tip**: These are fast - can run 1000s of iterations quickly

---

## Ablation Studies (Key Findings)

### Effect of K (number of neighbors)

- K=1: Unstable, poor coverage
- K=5-50: Robust performance
- K=100+: Diminishing returns, higher cost

**Conclusion**: K=10-20 is sweet spot

### Effect of Batch Size M

- M=16: More stochastic, slower convergence
- M=32-128: Good balance
- M=256+: Marginal improvement, higher memory

**Conclusion**: M=64 recommended default

### Effect of Logit Transformation

**Without logit** (bounded spaces): Poor diversity, solutions cluster near boundaries

**With logit**: Uniform exploration, much better coverage

**Conclusion**: Always use logit for bounded spaces

### Effect of γ (kernel bandwidth)

**Too small** (γ=0.01): Solutions cluster
**Moderate** (γ=0.1-1.0): Best results
**Too large** (γ=10): Weak diversity pressure

**Conclusion**: Tune γ to behavior space scale

---

## Comparison to Baselines

### vs. CMA-ME / CMA-MEGA / CMA-MAEGA

**CMA variants**: Use CMA-ES for local optimization within cells

**SQUAD advantages**:
- No archive management
- Higher-dimensional behavior spaces
- Better quality-diversity trade-off

**When CMA wins**: Very low-dimensional behavior spaces (d≤3)

### vs. DNS (Dominated Novelty Search)

**DNS**: Fitness transformation + novelty archive

**SQUAD advantages**:
- Continuous formulation
- Better scaling to high dimensions
- Faster convergence (fewer iterations)

### vs. Sep-CMA-MAE

**Sep-CMA-MAE**: Separable CMA variant

**SQUAD advantages**:
- Much better diversity (5-10× higher Vendi Score)
- Scales to complex domains (DNS-G comparable on simple functions)

### vs. GA-ME (Genetic Algorithm MAP-Elites)

**GA-ME**: Classic evolutionary MAP-Elites

**SQUAD advantages**:
- Gradient-based (much faster in differentiable domains)
- Better quality (2-3× higher mean objective)

---

## Advanced Techniques

### 1. Adaptive Kernel Bandwidth

Instead of fixed γ, adapt based on local density:
```
γ_i = median(||b_i - b_j|| for j ∈ N_i)
```

**Benefit**: Better adaptation to varying density

### 2. Annealing Schedule

Start with large γ (explore), decrease over time (exploit):
```
γ(t) = γ_0 · (1 - t/T_max)^α
```

**Benefit**: Better coverage early, refinement later

### 3. Multi-Objective Formulation

Combine SQUAD with other objectives:
```
L = α·SQUAD_objective + β·auxiliary_loss
```

**Example**: Add novelty objective for open-ended search

### 4. Hierarchical Populations

Multiple sub-populations with different γ values:
- High γ: Coarse diversity
- Low γ: Fine-grained diversity

**Benefit**: Multi-scale exploration

---

## Debugging Checklist

### Poor Diversity (Solutions Cluster)

- [ ] Increase γ (kernel bandwidth)
- [ ] Check logit transformation is applied (bounded spaces)
- [ ] Increase K (number of neighbors)
- [ ] Verify descriptor gradients are non-zero

### Poor Quality

- [ ] Decrease γ (too much repulsion)
- [ ] Check learning rate (may be too high/low)
- [ ] Verify quality gradients are correct
- [ ] Try quality-only pre-training

### Slow Convergence

- [ ] Increase learning rate
- [ ] Use adaptive optimizer (Adam)
- [ ] Increase batch size
- [ ] Check gradient magnitudes (scale if needed)

### Numerical Instability

- [ ] Clip behavior descriptors before logit
- [ ] Ensure f > 0 (use softplus)
- [ ] Use mixed precision training
- [ ] Reduce learning rate

---

## Code Structure Template

```python
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

class SQUAD:
    def __init__(self, population_size, batch_size, k_neighbors, gamma, lr):
        self.N = population_size
        self.M = batch_size
        self.K = k_neighbors
        self.gamma_sq = gamma ** 2
        self.lr = lr

        # Initialize population (domain-specific)
        self.population = self.initialize_population()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.population, lr=lr)

    def initialize_population(self):
        # Domain-specific initialization
        return [...]

    def eval_solution(self, theta):
        # Returns (quality, behavior_descriptor)
        quality = self.quality_function(theta)
        behavior = self.descriptor_function(theta)
        return quality, behavior

    def logit_transform(self, b):
        # For bounded behavior spaces [0,1]
        b_clipped = torch.clip(b, 1e-6, 1-1e-6)
        return torch.log(b_clipped / (1 - b_clipped))

    def compute_objective(self, indices, F, B):
        # F: qualities, B: behaviors
        objective = 0.0

        # Quality term
        objective += F[indices].sum()

        # Repulsion term
        for i in indices:
            # Find k-nearest neighbors
            nn = NearestNeighbors(n_neighbors=self.K)
            nn.fit(B.detach().cpu().numpy())
            _, neighbor_idx = nn.kneighbors([B[i].detach().cpu().numpy()])

            for j in neighbor_idx[0]:
                if i != j:
                    dist_sq = torch.sum((B[i] - B[j]) ** 2)
                    repulsion = torch.sqrt(F[i] * F[j]) * torch.exp(-dist_sq / self.gamma_sq)
                    objective -= 0.5 * repulsion

        return objective

    def step(self):
        # Evaluate population
        F, B = [], []
        for theta in self.population:
            f, b = self.eval_solution(theta)
            F.append(f)
            B.append(self.logit_transform(b))

        F = torch.stack(F)
        B = torch.stack(B)

        # Mini-batch update
        for batch_start in range(0, self.N, self.M):
            batch_indices = range(batch_start, min(batch_start + self.M, self.N))

            self.optimizer.zero_grad()
            objective = self.compute_objective(batch_indices, F, B)
            loss = -objective  # Maximize objective = minimize negative
            loss.backward()
            self.optimizer.step()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Population updated")
```

---

## Key References

1. **MAP-Elites**: Mouret & Clune (2015) - Original QD algorithm
2. **CVT-MAP-Elites**: Vassiliades et al. (2018) - Centroidal Voronoi Tessellation
3. **Differentiable QD**: Fontaine et al. (2021) - DQD formulation
4. **CMA-MEGA**: Fontaine & Nikolaidis (2021) - CMA-ES for QD
5. **SVGD**: Liu & Wang (2016) - Stein Variational Gradient Descent
6. **DNS**: Faldor et al. (2023) - Dominated Novelty Search

---

## Summary: Implementation Checklist

### To implement SQUAD from scratch:

1. [ ] Define quality function f(θ) (differentiable)
2. [ ] Define behavior descriptor desc(θ) (differentiable)
3. [ ] Apply logit transform if behavior space is bounded
4. [ ] Initialize population of size N
5. [ ] Set hyperparameters: K, M, γ, η
6. [ ] Implement k-NN search in behavior space
7. [ ] Compute SQUAD objective (quality + repulsion terms)
8. [ ] Backpropagate gradients through objective
9. [ ] Update population using optimizer (Adam recommended)
10. [ ] Track metrics: Soft QD Score, diversity, quality
11. [ ] Iterate until convergence (or max epochs)

**That's it!** No archive management, cell allocation, or discretization needed.
