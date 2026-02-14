# Nash Averaging - Detailed Implementation

**Paper**: Balduzzi et al., "Re-evaluating Evaluation" (DeepMind, 2018)

---

## Mathematical Foundations

### Antisymmetric Matrices

**Definition**: Matrix $A$ is antisymmetric if $A + A^T = 0$

**Properties**:
- Even rank
- Imaginary eigenvalues: $\{\pm i\lambda_j\}_{j=1}^{\text{rank}(A)/2}$
- Real Schur decomposition exists

**Schur Decomposition**:
```
A_{n×n} = Q_{n×n} · Λ_{n×n} · Q^T_{n×n}
```

where $Q$ is orthogonal and $\Lambda$ has 2×2 diagonal blocks:

$$\Lambda = \begin{pmatrix}
0 & \lambda_j \\
-\lambda_j & 0
\end{pmatrix}$$

---

## Combinatorial Hodge Theory

### Core Operators

**Gradient**: $\text{grad}(r) = r \cdot 1^T - 1 \cdot r^T$
- Produces flow from vector $r$
- $\text{grad}(r)_{ij} = r_i - r_j$

**Divergence**: $\text{div}(A) = \frac{1}{n} A \cdot 1$
- Measures contribution of each vertex as source
- In AvA: Recovers Elo ratings
- In AvT: Recovers average performance

**Curl**: $\text{curl}(A)_{ijk} = A_{ij} + A_{jk} - A_{ik}$
- Detects cyclic patterns (rock-paper-scissors)
- $\text{curl} = 0$ ⟺ transitive

**Rotation**: $\text{rot}(A)_{ij} = \frac{1}{n} \sum_k \text{curl}(A)_{ijk}$

### Hodge Decomposition Theorem

**Any antisymmetric matrix decomposes uniquely**:

$$A = \text{grad}(r) + \text{rot}(A)$$

where:
- $\text{grad}(r)$ = transitive component (captured by Elo/averages)
- $\text{rot}(A)$ = cyclic component (intransitive interactions)
- $r = \text{div}(A)$ (recovers ratings/averages)

**Orthogonality**:
- $\langle \text{grad}(r), \text{rot}(A) \rangle = 0$
- $\text{div} \circ \text{rot}(A) = 0$ (rotation has no divergence)
- $\text{rot} \circ \text{grad}(r) = 0$ (gradient has no rotation)

---

## Agent vs Agent (AvA)

### Data Representation

**Input**: Win-loss probabilities $P_{ij}$ (probability agent $i$ beats $j$)

**Logit transformation**:
$$A_{ij} = \log \frac{p_{ij}}{1 - p_{ij}} = \log \frac{p_{ij}}{p_{ji}}$$

**Properties**:
- $A$ is antisymmetric: $A_{ij} = -A_{ji}$
- $p_{ij} = \sigma(A_{ij})$ where $\sigma(x) = \frac{1}{1 + e^{-x}}$

### Standard Elo

**Prediction**: $\hat{p}_{ij} = \sigma(r_i - r_j)$

**Loss**: $\ell_{\text{Elo}}(p_{ij}, \hat{p}_{ij}) = -p_{ij} \log \hat{p}_{ij} - (1-p_{ij}) \log(1-\hat{p}_{ij})$

**Online update**:
$$r_i^{t+1} \leftarrow r_i^t + \eta \cdot (S_{ij}^t - \hat{p}_{ij}^t)$$

where $S_{ij}^t \in \{0,1\}$ is match outcome, $\eta$ is learning rate (16 or 32).

**Fixed point condition**:
$$\sum_j p_{ij} = \sum_j \hat{p}_{ij} \quad \forall i$$

(Row sums of empirical and predicted probabilities match)

**When Elo works**:
- $\text{curl}(\log \frac{p_{ij}}{p_{ji}}) = 0$ for all $i,j,k$
- Equivalently: $\log \frac{p_{ij}}{p_{ji}} + \log \frac{p_{jk}}{p_{kj}} + \log \frac{p_{ki}}{p_{ik}} = 0$
- **Interpretation**: Purely transitive interactions

### Multidimensional Elo (mElo)

**Problem**: Standard Elo has no predictive power for rock-paper-scissors

**Solution**: Add low-rank cyclic approximation

$$A \approx \text{grad}(r) + C^T \Omega C$$

where:
- $C_{n×2k}$ = matrix of cyclic features
- $\Omega_{2k×2k} = \sum_{i=1}^k (e_{2i-1} e_{2i}^T - e_{2i} e_{2i-1}^T)$ (antisymmetric structure)
- Rows of $C$ orthogonal to each other, to $r$, and to $1$

**mElo$_{2k}$ prediction**:

$$\hat{p}_{ij} = \sigma(r_i - r_j + c_i^T \Omega c_j)$$

**Parameters**:
- Agent $i$ has: Elo rating $r_i$ + feature vector $c_i \in \mathbb{R}^{2k}$
- Total: $n(1 + 2k)$ parameters
- mElo$_0$ = standard Elo
- mElo$_2$ = simplest cyclic extension

**Training**: Gradient descent on logistic loss with orthogonality constraints

---

## Nash Averaging (AvA)

### Meta-Game Construction

**Players**:
- Row meta-player picks distribution $p$ over agents
- Column meta-player picks distribution $q$ over agents

**Payoffs**:
- Row player: $\mu_1(p, q) = p^T A q$ (expected log-odds)
- Column player: $\mu_2(p, q) = p^T B q$ where $B = A^T = -A$

**Game properties**:
- Symmetric: $B = A^T$
- Zero-sum: $B = -A$
- Value = 0
- Nash equilibria = unbeatable teams

### Maxent Nash Equilibrium

**Theorem**: For antisymmetric $A$, there exists a unique symmetric Nash equilibrium $(p^*, p^*)$ with maximum entropy.

**Computation**:
$$p^* = \arg\max_{p \in \Delta_n} H(p) \quad \text{s.t.} \quad p^T A p = 0$$

where $H(p) = -\sum_i p_i \log p_i$ is Shannon entropy.

**Alternative formulation** (via linear programming):

```
Maximize: u
Subject to:
  A · p ≥ u · 1  (row player's constraints)
  p ≥ 0
  sum(p) = 1
```

The maxent Nash is the uniform distribution over the support of any Nash equilibrium with maximum support.

### Nash Average

**Definition**: $n_A = A \cdot p^*$

**Interpretation**:
- $n_A[i]$ = agent $i$'s performance against maxent Nash team
- Agents in Nash support have equal performance: $n_A[i] = 0$ if $p^*[i] > 0$
- By indifference principle for Nash equilibria

---

## Invariance to Redundancy

### Formal Definition

An evaluation method $\mathcal{E}: \{\text{antisymmetric matrices}\} \to [\{\text{players}\} \to \mathbb{R}]$ is **invariant** if:

Adding redundant copies of player $i$ (duplicating row/column) doesn't change evaluation of any player.

### Theorem: Nash Averaging is Invariant

**Proof sketch**:
1. If agent $C$ is duplicated as $C_1, C_2$ in matrix $A'$
2. Then $(p_A^*, p_B^*, \alpha p_C^*, (1-\alpha) p_C^*)$ is Nash for $A'$ for any $\alpha \in [0,1]$
3. Maxent Nash distributes $C$'s mass uniformly: $(p_A^*, p_B^*, \frac{p_C^*}{2}, \frac{p_C^*}{2})$
4. Nash average: $n_{A'}[C_1] = n_{A'}[C_2] = n_A[C]$

### Example: Rock-Paper-Scissors with Redundancy

**Matrix $A$** (original):
```
     A    B    C
A [  0   4.6  -4.6]
B [-4.6   0   4.6]
C [ 4.6 -4.6   0 ]
```

Maxent Nash: $p^* = (1/3, 1/3, 1/3)$  
Nash average: $n = (0, 0, 0)$

**Matrix $A'$** (C duplicated):
```
     A    B    C₁   C₂
A [  0   4.6  -4.6  -4.6]
B [-4.6   0   4.6  4.6]
C₁[ 4.6 -4.6   0    0  ]
C₂[ 4.6 -4.6   0    0  ]
```

Maxent Nash: $p^* = (1/3, 1/3, 1/6, 1/6)$  
Nash average: $n = (0, 0, 0, 0)$

**Comparison**:
- Uniform average (divergence):
  - $\text{div}(A) = (0, 0, 0)$ ✓
  - $\text{div}(A') = (-1.15, 1.15, 0, 0)$ ✗ (falsely suggests B superior)
- Nash average:
  - Both give zero ✓ (correctly detects no dominance)

---

## Agent vs Task (AvT)

### Data Representation

**Input**: Score matrix $S_{m×n}$ (m agents, n tasks)
- Rows = agents
- Columns = tasks
- Entries = scores (accuracy, reward, etc.)

**Preprocessing**:
1. Subtract total mean: $\sum_{ij} S_{ij} = 0$
2. Compute averages:
   - $s = \frac{1}{n} S \cdot 1$ (average skill per agent)
   - $d = -\frac{1}{m} S^T \cdot 1$ (average difficulty per task)

**Antisymmetrize**:
$$A_{(m+n)×(m+n)} = \begin{pmatrix}
\text{grad}(s) & S \\
-S^T & \text{grad}(d)
\end{pmatrix}$$

**Interpretation**:
- Top-left: Agents compared by average skill
- Top-right: Agent performance on tasks
- Bottom-left: Task difficulty for agents
- Bottom-right: Tasks compared by average difficulty

### SVD and Latent Structure

**Residual matrix**: $\tilde{S} = S - (s \cdot 1^T - 1 \cdot d^T)$

**SVD**: $\tilde{S} = U D V^T$

**Interpretation**:
- Rows of $U$: Latent abilities exhibited by agents
- Rows of $V$: Latent problems posed by tasks
- Singular values in $D$: Importance of each latent dimension

**Connection to Schur**:
$$\tilde{A} = \begin{pmatrix}
0 & \tilde{S} \\
-\tilde{S}^T & 0
\end{pmatrix}$$

has Schur decomposition with singular values = $\pm$ singular values of $\tilde{S}$

---

## Nash Averaging (AvT)

### Meta-Game

**Players**:
- Row meta-player picks $p_a$ over agents
- Column meta-player picks $p_e$ over environments

**Payoffs**:
- Row: $\mu_1(p_a, p_e) = p_a^T S p_e$ (expected score)
- Column: $\mu_2(p_a, p_e) = -p_a^T S p_e$ (expected difficulty)

**Solution**: Find maxent Nash $(p_a^*, p_e^*)$

### Nash Averages

**Agent skill** (Nash average over environments):
$$\text{skill}_i = (S \cdot p_e^*)_i$$

**Task difficulty** (Nash average over agents):
$$\text{difficulty}_j = -(S^T \cdot p_a^*)_j$$

**Core agents/tasks**: Support of $p_a^*, p_e^*$

---

## Worked Example: 3-Player Cyclic + Transitive Mix

**Setup**: Combine cyclic and transitive components

$$C = \begin{pmatrix}
0 & 1 & -1 \\
-1 & 0 & 1 \\
1 & -1 & 0
\end{pmatrix}, \quad
T = \begin{pmatrix}
0 & 1 & 2 \\
-1 & 0 & 1 \\
-2 & -1 & 0
\end{pmatrix}$$

**Mixed game**: $A(\epsilon) = C + \epsilon T$

**Analysis**:

1. **Pure cyclic** ($\epsilon = 0$):
   - Maxent Nash: $(1/3, 1/3, 1/3)$
   - Nash average: $(0, 0, 0)$
   - All agents equally good

2. **Weak transitive** ($0 < \epsilon < 1/2$):
   - Maxent Nash: $(\frac{1+\epsilon}{3}, \frac{1-2\epsilon}{3}, \frac{1+\epsilon}{3})$
   - Nash average: $(0, 0, 0)$
   - Agent 2 slightly worse, but still tied overall

3. **Strong transitive** ($\epsilon > 1/2$):
   - Maxent Nash: $(1, 0, 0)$
   - Nash average: $(0, -1-\epsilon, 1-2\epsilon)$
   - Agent 1 dominates

**Phase transition** at $\epsilon = 1/2$:
- Nash jumps discontinuously
- But Nash average changes continuously
- Payoff robust to perturbations (continuity property)

---

## Implementation

### Computing Maxent Nash (Linear Programming)

```python
import numpy as np
from scipy.optimize import linprog

def compute_maxent_nash(A):
    """
    Compute maxent Nash equilibrium for antisymmetric matrix A.
    
    Args:
        A: (n×n) antisymmetric matrix (logit of win probabilities)
    
    Returns:
        p_star: Maxent Nash distribution
        n_A: Nash average
    """
    n = A.shape[0]
    
    # LP formulation: max u s.t. A·p ≥ u·1, p≥0, sum(p)=1
    # Convert to min -u
    
    # Variables: [p_1, ..., p_n, u]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # Maximize u
    
    # Inequality constraints: -A·p + u·1 ≤ 0
    # (or equivalently: A·p ≥ u·1)
    A_ub = np.hstack([-A, np.ones((n, 1))])
    b_ub = np.zeros(n)
    
    # Equality constraint: sum(p) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: p ≥ 0, u unbounded
    bounds = [(0, None)] * n + [(None, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("LP solver failed")
    
    p_star = result.x[:n]
    
    # Compute Nash average
    n_A = A @ p_star
    
    return p_star, n_A


def compute_nash_from_probabilities(P):
    """
    Compute Nash averaging from win-loss probability matrix.
    
    Args:
        P: (n×n) win-loss probability matrix
    
    Returns:
        p_star: Maxent Nash distribution
        n_A: Nash average
    """
    # Logit transformation
    eps = 1e-10
    P_clipped = np.clip(P, eps, 1 - eps)
    A = np.log(P_clipped / (1 - P_clipped))
    
    # Ensure antisymmetry (average to remove numerical errors)
    A = 0.5 * (A - A.T)
    
    return compute_maxent_nash(A)
```

### Computing mElo

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_melo(P, k=1, n_epochs=1000, lr=0.01):
    """
    Train mElo_{2k} model.
    
    Args:
        P: (n×n) win-loss probability matrix
        k: Cyclic dimension (mElo uses 2k features)
        n_epochs: Training iterations
        lr: Learning rate
    
    Returns:
        r: Elo ratings
        C: Cyclic feature matrix (n×2k)
    """
    n = P.shape[0]
    
    # Parameters
    r = nn.Parameter(torch.zeros(n))
    C = nn.Parameter(torch.randn(n, 2 * k) * 0.1)
    
    # Omega matrix (antisymmetric structure)
    Omega = torch.zeros(2 * k, 2 * k)
    for i in range(k):
        Omega[2*i, 2*i+1] = 1.0
        Omega[2*i+1, 2*i] = -1.0
    
    optimizer = optim.Adam([r, C], lr=lr)
    
    P_tensor = torch.FloatTensor(P)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute predictions
        r_diff = r.unsqueeze(1) - r.unsqueeze(0)  # (n×n)
        cyclic_part = C @ Omega @ C.T  # (n×n)
        logits = r_diff + cyclic_part
        p_hat = torch.sigmoid(logits)
        
        # Binary cross-entropy loss
        loss = -torch.mean(P_tensor * torch.log(p_hat + 1e-10) +
                          (1 - P_tensor) * torch.log(1 - p_hat + 1e-10))
        
        loss.backward()
        optimizer.step()
        
        # Enforce constraints
        with torch.no_grad():
            # Zero-sum ratings
            r -= r.mean()
            
            # Orthogonalize C (Gram-Schmidt)
            for i in range(2 * k):
                # Orthogonal to r
                C[:, i] -= (C[:, i] @ r) / (r @ r + 1e-10) * r
                # Orthogonal to 1
                C[:, i] -= C[:, i].mean()
                # Orthogonal to previous columns
                for j in range(i):
                    C[:, i] -= (C[:, i] @ C[:, j]) / (C[:, j] @ C[:, j] + 1e-10) * C[:, j]
                # Normalize
                C[:, i] /= (torch.norm(C[:, i]) + 1e-10)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return r.detach().numpy(), C.detach().numpy()
```

---

## Experimental Results

### AlphaGo Variants

**8 Go agents** from Silver et al. 2016:
- 7 AlphaGo variants (value net, policy net, MCTS, etc.)
- Zen (baseline)

**Results**:
| Metric | Elo | mElo₂ | Improvement |
|--------|-----|-------|-------------|
| Frobenius error | 0.85 | 0.35 | -59% |
| Logistic loss | 1.41 | 1.27 | -10% |

**Key finding**: $\alpha_v$ (value), $\alpha_p$ (policy), Zen form cyclic pattern

**Elo predictions** (poor):
```
     α_v   α_p   Zen
α_v   -   0.41  0.58
α_p  0.59  -    0.67
Zen  0.42 0.33   -
```

**Empirical** (ground truth):
```
     α_v   α_p   Zen
α_v   -   0.7   0.4
α_p  0.3   -    1.0
Zen  0.6  0.0    -
```

**mElo₂ predictions** (accurate):
```
     α_v   α_p   Zen
α_v   -   0.72  0.46
α_p  0.28  -    0.98
Zen  0.55 0.02   -
```

### Atari Games (ALE)

**Setup**:
- 20 agents (Rainbow, DQN, Dueling, Human, Random, etc.)
- 54 Atari games
- Score matrix $S_{20×54}$ normalized per-game to [0,1]

**Uniform averaging**:
- Rainbow skill: 0.68
- Human skill: 0.52
- **Conclusion**: Rainbow >> Human (superhuman)

**Nash averaging** (meta-game solution):
- Core agents: 6 agents with $p_a^* > 0$ (including Rainbow, Human)
- Core environments: 18 games with $p_e^* > 0$
- Rainbow skill: 0.60
- Human skill: 0.60
- **Conclusion**: Rainbow ≈ Human (tied!)

**Interpretation**: ALE skewed towards games current RL agents excel at; rebalancing via Nash reveals humans still competitive

---

## Key Takeaways

1. **Hodge decomposition** separates transitive (Elo) from cyclic (rock-paper-scissors) interactions
2. **mElo** extends Elo to handle intransitive abilities via low-rank cyclic approximation
3. **Nash averaging** is invariant to redundant agents/tasks → automatic, scalable evaluation
4. **Maxent Nash** provides unique, interpretable solution with maximum entropy
5. **Meta-game perspective** treats evaluation as adversarial game on data
6. **AlphaGo**: mElo₂ captures cyclic patterns Elo misses
7. **Atari**: Nash averaging suggests human performance not yet surpassed

---

## References

- **Paper**: Balduzzi et al., "Re-evaluating Evaluation", ICML 2018
- **Code**: Appendix includes Python implementation
- **Theory**: Hodge decomposition (Jiang et al., 2011), Schur decomposition (standard linear algebra)
- **Related**: AlphaRank (Omidshafiei et al., 2019), Empirical game theory (Walsh et al., 2003)
