# Deviation Ratings - Detailed Implementation

**Paper**: Marris et al., "Deviation Ratings: A general, clone invariant rating method" (ICLR 2025)

---

## Mathematical Foundations

### Normal-Form Games (NFGs)

**Definition**: $N$-player simultaneous-action game

**Components**:
- Players: $p \in [1, N]$
- Strategy sets: $a_p \in \mathcal{A}_p = \{a_p^1, \ldots, a_p^{|\mathcal{A}_p|}\}$
- Joint strategy: $a = (a_1, \ldots, a_N) \in \mathcal{A} = \bigotimes_p \mathcal{A}_p$
- Payoff functions: $G_p: \mathcal{A} \mapsto \mathbb{R}$ for each player $p$
- Joint distributions: $\sigma \in \Delta^{|\mathcal{A}|-1}$ (probability simplex)

**Notation**: $a_{-p}$ = strategies of all players except $p$

---

## Equilibrium Concepts

### Deviation Gains

**Definition**: Expected payoff change when deviating to $a'_p$ from recommended $a''_p$ under joint distribution $\sigma$:

$$\delta^{\sigma}_p(a'_p, a''_p) = \sum_{a_{-p}} \sigma(a''_p, a_{-p}) [G_p(a'_p, a_{-p}) - G_p(a''_p, a_{-p})]$$

### Hierarchy of Equilibria

**Well-Supported Correlated Equilibrium ($\epsilon$-WSCE)**:
$$\delta^{\sigma}_p(a'_p, a''_p) \leq \sigma_p(a''_p) \epsilon \quad \forall p, a'_p, a''_p$$

**Correlated Equilibrium ($\epsilon$-CE)**:
$$\delta^{\sigma}_p(a'_p, a''_p) \leq \epsilon \quad \forall p, a'_p, a''_p$$

**Coarse Correlated Equilibrium ($\epsilon$-CCE)**:
$$\sum_{a''_p} \delta^{\sigma}_p(a'_p, a''_p) \leq \epsilon \quad \forall p, a'_p$$

**Inclusion**: WSNE ⊆ NE ⊆ WSCE ⊆ CE ⊆ CCE

**Why CCE**:
- Convex (LP-computable)
- Always exists
- General (N-player, general-sum)
- Sufficient for clone invariance

### CCE Deviation Gains (Simplified)

$$\delta^{\sigma}_p(a'_p) = \sum_{a''_p} \delta^{\sigma}_p(a'_p, a''_p) = \sum_a \sigma(a) [G_p(a'_p, a_{-p}) - G_p(a)]$$

**CCE condition**: $\delta^{\sigma}_p(a'_p) \leq 0 \quad \forall p, a'_p$

**Interpretation**: No incentive to deviate to any pure strategy

---

## Complete Algorithm

```python
import numpy as np
from scipy.optimize import linprog

def compute_deviation_ratings(G, players):
    """
    Compute deviation ratings for N-player general-sum game.
    
    Args:
        G: Payoff tensor (shape: |A_1| × |A_2| × ... × |A_N| × N)
        players: List of player indices
    
    Returns:
        ratings: Dict mapping (player, strategy) → rating
        sigma_star: CCE distribution
    """
    N = len(players)
    strategy_counts = [G.shape[i] for i in range(N)]
    total_joint_strategies = np.prod(strategy_counts)
    
    # Initialize
    frozen = {p: set() for p in players}
    ratings = {p: np.zeros(strategy_counts[p]) for p in players}
    
    # Iterative algorithm
    while not all(len(frozen[p]) == strategy_counts[p] for p in players):
        # Build LP for current iteration
        sigma, active_constraints = solve_iteration_lp(
            G, players, strategy_counts, frozen, ratings
        )
        
        # Update frozen sets and ratings
        for p in players:
            for a_p in active_constraints[p]:
                if a_p not in frozen[p]:
                    frozen[p].add(a_p)
                    ratings[p][a_p] = compute_deviation_gain(
                        G, sigma, p, a_p, strategy_counts
                    )
    
    return ratings, sigma


def solve_iteration_lp(G, players, strategy_counts, frozen, ratings):
    """
    Solve LP for one iteration: min max deviation gain.
    
    Returns:
        sigma: Optimal joint distribution
        active: Dict of active max constraints per player
    """
    N = len(players)
    total_joint = np.prod(strategy_counts)
    
    # Variables: [σ(a₁,...,aₙ) for all joint strategies, slack]
    n_vars = total_joint + 1
    
    # Objective: minimize slack variable (represents max deviation)
    c = np.zeros(n_vars)
    c[-1] = 1.0  # Minimize slack
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # For each unfrozen strategy: deviation ≤ slack
    for p in players:
        for a_p in range(strategy_counts[p]):
            if a_p not in frozen[p]:
                # Build constraint row
                row = build_deviation_constraint_row(
                    G, p, a_p, strategy_counts
                )
                row = np.append(row, -1.0)  # -slack
                A_ub.append(row)
                b_ub.append(0.0)
            else:
                # Frozen constraint: deviation = previous rating
                row = build_deviation_constraint_row(
                    G, p, a_p, strategy_counts
                )
                row = np.append(row, 0.0)
                A_ub.append(row)
                b_ub.append(ratings[p][a_p])
                A_ub.append(-row)
                b_ub.append(-ratings[p][a_p])
    
    # Equality constraint: sum(σ) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :total_joint] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: σ ≥ 0, slack unbounded
    bounds = [(0, None)] * total_joint + [(None, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("LP solver failed")
    
    sigma = result.x[:total_joint].reshape(strategy_counts)
    
    # Identify active constraints (dual variables)
    active = identify_active_constraints(result, players, frozen, strategy_counts)
    
    return sigma, active


def build_deviation_constraint_row(G, p, a_p, strategy_counts):
    """
    Build constraint row for deviation gain δ_p(a'_p).
    
    Returns: coefficient vector for σ
    """
    N = len(strategy_counts)
    total_joint = np.prod(strategy_counts)
    row = np.zeros(total_joint)
    
    # Enumerate all joint strategies
    for joint_idx in range(total_joint):
        # Convert flat index to multi-index
        joint_strategy = np.unravel_index(joint_idx, strategy_counts)
        
        # Coefficient: G_p(a'_p, a_{-p}) - G_p(a)
        a_prime = list(joint_strategy)
        a_prime[p] = a_p
        
        payoff_deviation = G[tuple(a_prime)][p]
        payoff_original = G[joint_strategy][p]
        
        row[joint_idx] = payoff_deviation - payoff_original
    
    return row


def compute_deviation_gain(G, sigma, p, a_p, strategy_counts):
    """Compute deviation gain for strategy a_p of player p."""
    delta = 0.0
    total_joint = np.prod(strategy_counts)
    
    for joint_idx in range(total_joint):
        joint_strategy = np.unravel_index(joint_idx, strategy_counts)
        
        a_prime = list(joint_strategy)
        a_prime[p] = a_p
        
        deviation = G[tuple(a_prime)][p] - G[joint_strategy][p]
        delta += sigma.flat[joint_idx] * deviation
    
    return delta
```

---

## Formal Properties & Proofs

### Property 1: Existence

**Theorem**: Deviation ratings always exist.

**Proof**:
- CCEs are supersets of Nash equilibria
- Nash equilibria always exist for finite NFGs (Nash, 1951)
- Therefore CCEs always exist
- Deviation ratings computed from CCEs → always exist ∎

### Property 2: Uniqueness

**Theorem**: Deviation ratings are unique.

**Proof**:
- Problem (Equation 1) is convex LP
- Optimal objective value is unique (convexity)
- Rating = optimal objective value (not parameters $\sigma$)
- Therefore rating is unique ∎

**Note**: Many equilibria $\sigma^*$ may give same ratings (non-unique equilibria, unique ratings)

### Property 3: Bounds

**Theorem**: Deviation ratings are bounded:

$$\min_a [G_p(a'_p, a_{-p}) - G_p(a)] \leq r_p(a'_p) \leq 0$$

**Proof**:
- Upper bound: $\epsilon = 0$ CCEs always exist → $r_p(a'_p) \leq 0$
- Lower bound: By definition of deviation gains ∎

**Interpretation**: Ratings are non-positive (deviation losses)

### Property 4: Dominance Preserving

**Theorem**: If strategy $\tilde{a}_p$ dominates $\hat{a}_p$, then $r_p(\tilde{a}_p) \geq r_p(\hat{a}_p)$.

**Proof**:
$$\begin{align}
G_p(\tilde{a}_p, a_{-p}) &\geq G_p(\hat{a}_p, a_{-p}) \quad \forall a_{-p} \\
\Rightarrow G_p(\tilde{a}_p, a_{-p}) - G_p(a) &\geq G_p(\hat{a}_p, a_{-p}) - G_p(a) \quad \forall a \\
\Rightarrow \sum_a \sigma(a) [G_p(\tilde{a}_p, a_{-p}) - G_p(a)] &\geq \sum_a \sigma(a) [G_p(\hat{a}_p, a_{-p}) - G_p(a)] \\
\Rightarrow \delta^{\sigma}_p(\tilde{a}_p) &\geq \delta^{\sigma}_p(\hat{a}_p) \\
\Rightarrow r_p(\tilde{a}_p) &\geq r_p(\hat{a}_p) \quad \text{∎}
\end{align}$$

### Property 5: Offset Invariance

**Theorem**: Adding offsets $\tilde{G}_p(a) = G_p(a) + b_p(a_{-p})$ doesn't change ratings.

**Proof**:
$$\begin{align}
\tilde{G}_p(a'_p, a_{-p}) - \tilde{G}_p(a) &= [G_p(a'_p, a_{-p}) + b_p(a_{-p})] - [G_p(a) + b_p(a_{-p})] \\
&= G_p(a'_p, a_{-p}) - G_p(a)
\end{align}$$

Deviation gains unchanged → equilibria unchanged → ratings unchanged ∎

### Property 6: Clone Invariance

**Theorem**: Adding a clone strategy doesn't change ratings.

**Proof** (via constraint matrix):

CCE defined by linear constraints: $A\sigma \leq 0$

**Original matrix** $A$: shape $[\sum_p |\mathcal{A}_p|, |\mathcal{A}|]$

**Cloned matrix** $\hat{A}$ (adding clone of $a_p^i$):

$$A = \begin{bmatrix}
A[\neg a_p^i, :] \\
A[a_p^i, :]
\end{bmatrix}, \quad
\hat{A} = \begin{bmatrix}
A[\neg a_p^i, :] & A[\neg a_p^i, \text{columns with } \hat{a}_p^i] \\
A[a_p^i, :] & 0 \\
A[a_p^i, :] & 0
\end{bmatrix}$$

**Key observation**: Clone creates **identical row** (redundant constraint)

**Equilibria**: Continuum in cloned game (mixture over clones) but same deviation gain values

**Any selection method** over deviation gains → clone invariant ∎

### Property 7: Mixture Invariance

**Theorem**: Adding mixed strategy $\tilde{a}_p = \sum_{a_p} \tilde{\sigma}(a_p) a_p$ results in rating:

$$r_p(\tilde{a}_p) = \sum_{a_p} \tilde{\sigma}(a_p) r_p(a_p)$$

**Proof**:
Mixed strategy creates mixed constraint (linear combination):

$$\delta^{\sigma}_p(\tilde{a}_p) = \sum_{a_p} \tilde{\sigma}(a_p) \delta^{\sigma}_p(a_p)$$

Therefore rating is mixture of component ratings ∎

### Property 8: Connection to Nash Averaging

**Theorem**: In 2-player zero-sum games, deviation ratings generalize Nash averaging:

$$r_p^{\text{CCE}}(a'_p) = r_p^{\text{NA}}(a'_p) - \sum_a \sigma(a) G_p(a)$$

**Proof**:
- 2p zero-sum: NE = CCE (sets are equal)
- All equilibria have equal value $v$
- Nash averaging: $r_p^{\text{NA}}(a'_p) = \sum_{a_{-p}} \sigma_{-p}(a_{-p}) G_p(a'_p, a_{-p})$
- Deviation rating: $r_p^{\text{CCE}}(a'_p) = \sum_a \sigma(a) [G_p(a'_p, a_{-p}) - G_p(a)]$
- Difference = constant offset ∎

---

## Worked Examples

### Example 1: Rock-Paper-Scissors (Symmetric)

**Game**:
```
     R     P     S
R [  0,0   -1,1   1,-1 ]
P [  1,-1   0,0  -1,1  ]
S [ -1,1   1,-1   0,0  ]
```

**CCE**: Uniform $(1/3, 1/3, 1/3) \otimes (1/3, 1/3, 1/3)$

**Deviation gains** (player 1):
$$\begin{align}
\delta_1(R) &= \frac{1}{9}(0-0 + (-1)-(-1) + 1-1 + 1-1 + 0-0 + (-1)-(-1) + (-1)-(-1) + 1-1 + 0-0) = 0 \\
\delta_1(P) &= 0 \\
\delta_1(S) &= 0
\end{align}$$

**Ratings**: $r_1(R) = r_1(P) = r_1(S) = 0$ ✓ (All equal, as expected for cyclic game)

### Example 2: Biased Shapley's Game

**Game** (with Nash strategy N):
```
     R           P           S           N
R [ -8,-8      -2,+2       +4,-4       -2.82,-2.95 ]
P [ +2,-2      -8,-8       -1,+1       -2.82,-3.82 ]
S [ -4,+4      +1,-1       -8,-8       -2.82,-0.76 ]
N [ -2.95,-2.82 -3.82,-2.82 -0.76,-2.82 -2.82,-2.82 ]
```

**Uniform rating**: R > P > N > S (biased by asymmetry)

**Deviation rating**: R = P = S = N = -2.82 (all equal despite biases)

**Interpretation**:
- Cycle: R ≻ S ≻ P ≻ R → no transitive dominance
- Nash strategy also equal (mixture invariance)
- Game-theoretically: all strategies equally good

### Example 3: Clone Attack

**Original** (Rock-Paper-Scissors):
```
Ratings: R = P = S = 0
```

**Cloned** (duplicate Rock 100 times):
```
Ratings: R₁ = R₂ = ... = R₁₀₀ = P = S = 0
```

**Uniform rating** (vulnerable):
- Inflates importance of counters to Rock (Paper)
- Biased result

**Deviation rating** (resistant):
- All ratings unchanged ✓
- Clone invariance property

---

## LLM Evaluation Case Study

### Livebench Dataset

**Data**: 92 models × 18 tasks → model-task scores $T(m, t)$

**Game formulation** (3-player):
- Players: Model A, Model B, Prompt
- Strategies: 92 models each, 18 tasks
- Payoffs:
  - $G_A(m_A, m_B, t) = T(m_A, t) - T(m_B, t)$ (zero-sum between models)
  - $G_B = -G_A$
  - $G_P = |G_A|$ (prompt favors distinguishing prompts)

### Results

**Uniform/Elo rankings** (transitive):
```
1. claude-3-5-sonnet
2. gemini-1.5-pro
3. gpt-4o
4. Llama-3.1-405B
...
```

**Deviation ratings** (game-theoretic grouping):
```
Tier 1 (tied): claude-3-5-sonnet, gemini-1.5-pro, gpt-4o, Llama-3.1-405B
Tier 2: ...
```

### Task Contribution Analysis

**Decompose ratings** by task:
$$r_A(m_A') = \sum_t c(m_A', t)$$

where $c(m_A', t) = \sum_{m_A, m_B} \sigma^*(m_A', m_B, t) [G_A(m_A', m_B, t) - G_A(m_A, m_B, t)]$

**Specializations found**:
- **claude-3-5-sonnet**: Strong at LCB generation
- **gemini-1.5-pro**: Strong at summarization
- **Llama-3.1-405B**: Strong at other tasks
- **gpt-4o**: Strong at connections

**Interpretation**: Each top model excels at different task subsets → no single dominant model → group at top

---

## Implementation Notes

### Efficiency Improvements

**Sparse constraint matrix**: Only store non-zero entries

**Incremental LP**: Warm-start from previous iteration

**Early termination**: If all active constraints frozen

**Parallel evaluation**: Compute deviation gains in parallel

### Numerical Stability

**Scale payoffs**: Normalize to [-1, 1] range

**LP solver tolerance**: Set appropriate optimality gap

**Dual variable threshold**: Consider constraint active if dual > ε

### Practical Considerations

**Large games**: Use column generation for huge strategy spaces

**Approximate ratings**: Early stop for ε-approximate CCE

**Data requirements**: Need full payoff matrix (all N-tuples)

---

## Comparison to Other Rating Methods

### Computational Complexity

| Method | Complexity | Convex? |
|--------|-----------|---------|
| Uniform | O($|\mathcal{A}|$) | N/A |
| Elo | O($|\mathcal{A}|^2$) updates | N/A |
| Nash Averaging (2p) | O($|\mathcal{A}|^3$) LP | ✅ |
| α-Rank | O($|\mathcal{A}|^3$) eigenvalue | ❌ |
| Deviation Ratings | O($|\mathcal{A}|^3 \cdot \sum_p |\mathcal{A}_p|$) LP | ✅ |

### Property Comparison

| Property | Deviation | Nash Avg | Elo | α-Rank |
|----------|-----------|----------|-----|--------|
| Clone invariant | ✅ | ✅ | ❌ | ❌ |
| N-player | ✅ | ❌ | ❌ | ✅ |
| General-sum | ✅ | ❌ | ❌ | ✅ |
| Offset invariant | ✅ | ❌ | ❌ | ✅ |
| Unique | ✅ | ✅ | ✅ | ✅ |

---

## Key Takeaways

1. **First N-player general-sum clone-invariant rating** via CCE deviation gains
2. **Iterative strictest equilibrium** selection preserves invariance
3. **LP-computable** (polynomial time, convex optimization)
4. **Multiple invariances**: Clone, mixture, offset
5. **Game-theoretic grouping**: Ties strategies with complementary strengths
6. **LLM evaluation**: Reveals model specializations ignored by Elo
7. **Resilient to attacks**: Clone-spamming doesn't bias ratings
8. **Directional improvement**: Hill-climbing drives holistic progress

---

## References

- **Paper**: Marris et al., "Deviation Ratings: A general, clone invariant rating method", ICLR 2025
- **Theory**: CCE (Hannan, 1957; Moulin, 1978), Nash (1951)
- **Clone invariance**: Independence of clones (Tideman, 1987), Maximal lotteries (Fishburn, 1984)
- **Related**: Nash averaging (Balduzzi et al., 2018), α-Rank (Omidshafiei et al., 2019)
- **Applications**: Chatbot Arena (Chiang et al., 2024), Livebench (White et al., 2024)
