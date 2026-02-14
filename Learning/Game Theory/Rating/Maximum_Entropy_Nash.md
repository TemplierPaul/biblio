# Maximum Entropy Nash Equilibrium

**Related**: [[Nash_Averaging]], [[Nash_Averaging_detailed]]

---

## Overview

**Maximum Entropy Nash Equilibrium (Maxent NE)** is a **unique refinement** of Nash equilibrium for zero-sum games that selects the equilibrium with **maximum Shannon entropy**.

**Key Property**: For antisymmetric matrices (zero-sum games), maxent NE is:
1. **Unique** (resolves non-uniqueness of Nash)
2. **Symmetric** (both players use same strategy)
3. **Maximally indifferent** (spreads probability over equivalent strategies)

---

## Motivation

### Problem: Nash is Not Unique

In zero-sum games, multiple Nash equilibria often exist.

**Example**: Rock-paper-scissors

```
Game matrix A (antisymmetric):
     R    P    S
R [  0    1   -1 ]
P [ -1    0    1 ]
S [  1   -1    0 ]

Nash equilibria:
- (1/3, 1/3, 1/3)  ← maxent
- (1/2, 1/2, 0)
- (1/2, 0, 1/2)
- (0, 1/2, 1/2)
- ... infinitely many!
```

**Challenge**: Which Nash to use for evaluation?

### Solution: Maximum Entropy Principle

**Idea**: Among all Nash equilibria, pick the one with **maximum entropy**.

**Entropy**: $H(p) = -\sum_i p_i \log p_i$

**Interpretation**:
- High entropy = spread out, indifferent, uncertain
- Low entropy = concentrated, confident, deterministic
- **Maxent** = maximally non-committal given constraints

---

## Mathematical Formulation

### For Antisymmetric Matrices

**Theorem** (Balduzzi et al., 2018):

For antisymmetric matrix $A$ (i.e., $A^T = -A$), there exists a **unique** symmetric Nash equilibrium $(p^*, p^*)$ solving:

$$\max_{p \in \Delta_n} \min_{q \in \Delta_n} p^T A q$$

that has **strictly greater entropy** than any other Nash equilibrium.

**Simplex**: $\Delta_n = \{p \in \mathbb{R}^n : p \geq 0, \sum_i p_i = 1\}$

### Optimization Problem

**Maxent NE** solves:

$$\begin{align}
\max_{p \in \Delta_n} \quad & H(p) = -\sum_i p_i \log p_i \\
\text{subject to} \quad & p^T A p = 0 \quad \text{(Nash condition)} \\
& p \geq 0 \\
& \sum_i p_i = 1
\end{align}$$

**Equivalent formulation** (via value of game):

$$\begin{align}
\max_{p \in \Delta_n} \quad & H(p) \\
\text{subject to} \quad & A \cdot p \geq v \cdot 1 \quad \text{for some } v \in \mathbb{R} \\
& p \geq 0, \quad \sum_i p_i = 1
\end{align}$$

For zero-sum antisymmetric games, the value $v = 0$.

---

## Properties

### 1. Uniqueness

**Standard Nash**: Often infinitely many equilibria  
**Maxent Nash**: Exactly one (for antisymmetric matrices)

**Reason**: Entropy is strictly concave → unique maximum

### 2. Symmetry

For antisymmetric $A$:
- Row player's maxent NE = Column player's maxent NE
- $(p^*, p^*)$ is a symmetric equilibrium

**Reason**: Zero-sum game is symmetric ($A^T = -A$)

### 3. Maximum Entropy

Among all Nash equilibria $\{(p_i, q_i)\}$:

$$H(p^*) > H(p_i) \quad \forall i$$

**Interpretation**: Maxent Nash is maximally non-committal

### 4. Invariance to Redundancy

Adding redundant strategies (duplicating rows/columns) doesn't change evaluation.

**Example**:
```
Original: A, B, C  →  Maxent: (1/3, 1/3, 1/3)
Add copy of C: A, B, C₁, C₂  →  Maxent: (1/3, 1/3, 1/6, 1/6)
                                         └─ splits C's mass ─┘
```

**Nash average** (performance):
- Original: $A \cdot (1/3, 1/3, 1/3) = (0, 0, 0)$
- With copy: $A' \cdot (1/3, 1/3, 1/6, 1/6) = (0, 0, 0, 0)$
- **Invariant**: Same relative evaluation ✓

---

## Interpretation

### Indifference Principle

**Nash equilibrium property**: Strategies in support have equal payoff.

For maxent Nash:
- All $i$ with $p^*_i > 0$ have same expected payoff
- $(A \cdot p^*)_i = (A \cdot p^*)_j$ for all $i,j$ in support
- **Nash average**: $(A \cdot p^*)_i = 0$ for all $i$ in support (for zero-sum games)

**Maxent spreads probability uniformly** over all strategies with equal performance.

### Special Cases

**Pure cyclic** (rock-paper-scissors):
- $\text{div}(A) = 0$ (no agent better on average)
- Maxent Nash: **uniform** distribution $p^* = (1/n, \ldots, 1/n)$

**Pure transitive** (total ordering):
- $A = \text{grad}(r)$ (skill ratings)
- Maxent Nash: **uniform over best** player(s)
- If unique best: $p^* = (1, 0, \ldots, 0)$
- If tie for best: $p^* = (\frac{1}{k}, \ldots, \frac{1}{k}, 0, \ldots, 0)$ for $k$ tied players

---

## Computation

### Linear Programming Formulation

**Primal problem** (find Nash):

```
Variables: p ∈ ℝⁿ, u ∈ ℝ

Maximize: u
Subject to:
  A · p ≥ u · 1    (row player best response)
  p ≥ 0
  sum(p) = 1
```

**Solution**: $p^*$ is a Nash equilibrium

**To get maxent Nash**: Among all Nash (same value $u$), pick the one with max $H(p)$.

### Entropic Regularization

**Smooth approximation**:

$$\max_{p \in \Delta_n} \quad u + \lambda H(p) \quad \text{s.t.} \quad A \cdot p \geq u \cdot 1$$

As $\lambda \to \infty$, converges to maxent Nash.

**Advantage**: Smooth, differentiable, easier optimization

### Projected Gradient Ascent

```python
import numpy as np

def compute_maxent_nash_gradient(A, n_iters=10000, lr=0.01, lambda_entropy=1.0):
    """
    Compute maxent Nash via gradient ascent with entropic regularization.
    
    Args:
        A: Antisymmetric matrix
        n_iters: Number of iterations
        lr: Learning rate
        lambda_entropy: Entropy regularization strength
    
    Returns:
        p_star: Maxent Nash distribution
    """
    n = A.shape[0]
    p = np.ones(n) / n  # Initialize uniform
    
    for _ in range(n_iters):
        # Gradient of objective: u + λH(p)
        # Nash constraint: A·p ≥ u·1 (Lagrange multipliers)
        
        # Simplified: gradient ascent on entropy-regularized payoff
        payoff = A @ p  # Expected payoff against current p
        
        # Gradient: ∇H(p) = -log(p) - 1
        entropy_grad = -np.log(p + 1e-10) - 1
        
        # Combined gradient
        grad = payoff + lambda_entropy * entropy_grad
        
        # Update
        p = p + lr * grad
        
        # Project onto simplex
        p = np.maximum(p, 0)  # Non-negative
        p = p / np.sum(p)  # Normalize
    
    return p
```

### Exact via Support Enumeration

**For small games** (n ≤ 10):

1. Enumerate all possible supports $S \subseteq [n]$
2. For each support:
   - Check if uniform on $S$ is Nash
   - Compute entropy $H(p_S)$
3. Return support with maximum entropy

**Complexity**: $O(2^n)$ (exponential)

---

## Worked Examples

### Example 1: Rock-Paper-Scissors

**Game matrix**:
```
A = [[ 0,  1, -1],
     [-1,  0,  1],
     [ 1, -1,  0]]
```

**All Nash equilibria**:
- Any $(p_R, p_P, p_S)$ with $p_R = p_S$
- Examples: $(1/3, 1/3, 1/3)$, $(1/2, 0, 1/2)$, $(0, 1, 0)$, ...

**Maxent Nash**:
$$p^* = (1/3, 1/3, 1/3)$$

**Entropy**:
- $H(p^*) = -3 \cdot \frac{1}{3} \log \frac{1}{3} = \log 3 \approx 1.099$
- $H(1/2, 0, 1/2) = -2 \cdot \frac{1}{2} \log \frac{1}{2} = \log 2 \approx 0.693$
- $H(0, 1, 0) = 0$

**Maxent is uniform** ✓

### Example 2: Cyclic + Transitive Mix

**Game matrix**:
$$A(\epsilon) = \begin{pmatrix}
0 & 1+\epsilon & -1-\epsilon \\
-1-\epsilon & 0 & 1+\epsilon \\
1+\epsilon & -1-\epsilon & 0
\end{pmatrix}$$

**Analysis**:

- $\epsilon = 0$ (pure cyclic): $p^* = (1/3, 1/3, 1/3)$
- $0 < \epsilon < 1/2$ (weak transitive): $p^* = (\frac{1+\epsilon}{3}, \frac{1-2\epsilon}{3}, \frac{1+\epsilon}{3})$
- $\epsilon > 1/2$ (strong transitive): $p^* = (1, 0, 0)$

**Phase transition** at $\epsilon = 1/2$:
- Below: Distributed over players 1 and 3
- Above: Concentrated on player 1 (dominates)

### Example 3: Redundant Agent

**Original game** (A, B, C):
```
A = [[ 0,  2, -2],
     [-2,  0,  2],
     [ 2, -2,  0]]
```

Maxent Nash: $(1/3, 1/3, 1/3)$  
Nash average: $(0, 0, 0)$

**Add copy of C** (A, B, C₁, C₂):
```
A' = [[ 0,  2, -2, -2],
      [-2,  0,  2,  2],
      [ 2, -2,  0,  0],
      [ 2, -2,  0,  0]]
```

**Nash equilibria of A'**:
- $(\frac{1}{3}, \frac{1}{3}, \frac{\alpha}{3}, \frac{1-\alpha}{3})$ for any $\alpha \in [0,1]$
- **Continuum** of Nash!

**Maxent Nash**:
$$p^* = (1/3, 1/3, 1/6, 1/6)$$

(Uniform over equivalentstrategies)

**Entropy**:
- $H(1/3, 1/3, 1/6, 1/6) = -\frac{2}{3} \log \frac{1}{3} - \frac{2}{6} \log \frac{1}{6} \approx 1.329$
- $H(1/3, 1/3, 1/3, 0) = \log 3 \approx 1.099$
- **Maxent spreads over redundant strategies** ✓

**Nash average**:
$$A' \cdot p^* = (0, 0, 0, 0)$$

**Invariance**: Same relative evaluation as original ✓

---

## Comparison to Alternatives

### vs Uniform Nash (any Nash)

**Uniform Nash**: Pick any Nash equilibrium  
**Maxent Nash**: Pick the unique maximum entropy one

**Advantage**: Removes ambiguity, provides canonical choice

### vs Utilitarian Social Welfare

**Utilitarian**: $\max_{p} \sum_i u_i(p)$ (maximize total utility)  
**Maxent Nash**: $\max_{p: \text{Nash}} H(p)$ (max entropy subject to Nash)

**Difference**: Maxent Nash is **game-theoretic** (adversarial), not cooperative

### vs Quantal Response Equilibrium

**QRE**: $p_i \propto \exp(\lambda \cdot \text{payoff}_i(p))$ (bounded rationality)  
**Maxent Nash**: Rational Nash with max entropy

**Connection**: As $\lambda \to \infty$, QRE → maxent Nash

---

## Applications

### 1. Multi-Agent Evaluation

**Use case**: Rank agents that interact cyclically (e.g., AlphaGo variants)

**Method**:
- Compute win-loss matrix $P$
- Logit transform: $A = \log \frac{P}{1-P}$
- Find maxent Nash $p^*$
- Nash average: $n = A \cdot p^*$
- **Core agents**: Support of $p^*$

### 2. Benchmark Suite Design

**Use case**: Select important tasks for evaluation

**Method** (Agent vs Task):
- Score matrix $S$ (agents × tasks)
- Antisymmetrize
- Find maxent Nash $(p_a^*, p_e^*)$
- **Core tasks**: Support of $p_e^*$
- Reweight tasks by $p_e^*$ in evaluation

### 3. Population-Based Training

**Use case**: Guide search via tournament outcomes

**Method**:
- Tournament matrix $A$ (current population)
- Maxent Nash $p^*$ = importance weights
- Sample new candidates from distribution $p^*$
- **Avoid**: Oversampling weak agents (inflation)

---

## Key Insights

1. **Uniqueness**: Resolves non-uniqueness of Nash via entropy maximization
2. **Invariance**: Automatically handles redundant strategies
3. **Symmetry**: Equal treatment of equivalent strategies
4. **Interpretability**: 
   - Pure cyclic → uniform
   - Pure transitive → best player(s)
5. **Practical**: Computable via LP or gradient methods
6. **Connection to QRE**: Limiting case of quantal response equilibrium

---

## References

- **Paper**: Balduzzi et al., "Re-evaluating Evaluation", ICML 2018
- **Theory**: Maximum entropy principle (Jaynes, 1957), Nash equilibrium (Nash, 1950)
- **Related**: Quantal response equilibrium (McKelvey & Palfrey, 1995), von Neumann winners (Dudík et al., 2015)
- **Computation**: Linear programming (Dantzig, 1963), Entropic regularization (Ortiz & Schaeffer, 2007)
