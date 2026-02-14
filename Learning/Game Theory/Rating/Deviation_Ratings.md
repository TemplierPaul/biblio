# Deviation Ratings

**Quick reference**: [[Deviation_Ratings_detailed]]

---

## Overview

**Deviation Ratings** is the **first N-player general-sum clone-invariant rating method**. It assigns ratings to strategies in multi-agent games by computing the strictest Coarse Correlated Equilibrium (CCE), ensuring ratings are invariant to redundant strategies.

**Authors**: Luke Marris, Siqi Liu, Ian Gemp, Georgios Piliouras, Marc Lanctot (Google DeepMind)  
**Paper**: "Deviation Ratings: A general, clone invariant rating method" (ICLR 2025)  
**Key Innovation**: Iteratively minimize maximum deviation gains → unique, clone-invariant ratings

---

## The Problem

### Limitations of Existing Ratings

**Elo**:
- ❌ Not clone invariant (vulnerable to cloning attacks)
- ❌ Only 2-player zero-sum
- ❌ Distribution-dependent (biased by repeated similar strategies)

**Nash Averaging** (Balduzzi et al., 2018):
- ✅ Clone invariant
- ❌ Only 2-player zero-sum (NE selection problem in N-player)
- ❌ Non-convex optimization for general-sum games

**Need**: Clone-invariant rating for **N-player general-sum** games

---

## Key Innovation: Deviation Gains Selection

### Core Idea

Instead of selecting a single equilibrium (hard for N-player), select **unique deviation gains**.

**Deviation gain**: Expected payoff change from deviating to strategy $a'_p$:

$$\delta^{\sigma}_p(a'_p) = \sum_a \sigma(a) [G_p(a'_p, a_{-p}) - G_p(a)]$$

**Deviation Rating**:

$$r_p(a'_p) = \delta^{\sigma^*}_p(a'_p)$$

where $\sigma^*$ is selected by **iteratively minimizing maximum deviation gains**.

---

## Algorithm: Iterative Strictest Equilibrium

```
Initialize: frozen_set = ∅, ratings = 0

While not all strategies frozen:
  1. Solve LP: min max deviation_gain
             subject to: frozen deviations = previous ratings
  
  2. Active constraints → newly frozen strategies
  
  3. Update ratings for newly frozen strategies
  
  4. Add to frozen_set

Return: ratings
```

**Key properties**:
- Each iteration freezes ≥1 strategy → terminates in ≤ Σ|strategies| iterations
- LP is convex → unique optimal deviation gains
- Selects **strictest** (most stable) equilibrium

---

## When to Use Deviation Ratings

✅ **Use when**:
- **N-player general-sum** games (e.g., model vs model vs prompt)
- Risk of **clone attacks** (repeated similar strategies)
- Need **scalable** evaluation (no manual curation)
- Want **directional** improvement (game-theoretic hill-climbing)
- Have strategic interactions (adversarial, cooperative, or mixed)

❌ **Consider alternatives when**:
- Only 2-player zero-sum → Nash Averaging (simpler)
- No strategic interactions → Uniform averaging
- Need interpretable win probabilities → Elo (despite limitations)

---

## Properties Summary

| Property | Deviation Ratings | Nash Averaging | Elo | Uniform |
|----------|-------------------|----------------|-----|---------|
| **Clone Invariant** | ✅ | ✅ | ❌ | ❌ |
| **Mixture Invariant** | ✅ | ✅ | ❌ | ❌ |
| **Offset Invariant** | ✅ | ❌ | ❌ | ❌ |
| **Dominance Preserving** | ✅ | ✅ | ✅ | ✅ |
| **N-Player** | ✅ | ❌ | ❌ | ✅ |
| **General-Sum** | ✅ | ❌ | ❌ | ✅ |
| **Unique** | ✅ | ✅ | ✅ | ✅ |
| **Always Exists** | ✅ | ✅ | ✅ | ✅ |

---

## Key Desiderata

### 1. Clone Invariance

Adding a copy of strategy $i$ doesn't change any ratings:

```
Original game: {A, B, C}
Cloned game: {A, B, C₁, C₂}

Result: r(C₁) = r(C₂) = r_original(C)
        r(A) = r_original(A), r(B) = r_original(B)
```

**Why important**:
- **Scalable**: Can include all data without curation
- **Resilient**: Immune to clone attacks (spam similar strategies)
- **Directional**: Hill-climbing leads to holistic improvement

### 2. Mixture Invariance

Adding a mixed strategy rates it as the mixture of component ratings:

$$r(\alpha A + (1-\alpha) B) = \alpha \cdot r(A) + (1-\alpha) \cdot r(B)$$

### 3. Offset Invariance

Adding offsets $G'_p(a) = G_p(a) + b_p(a_{-p})$ doesn't change ratings.

**Reason**: Deviations gains invariant to offsets:
$$G'_p(a'_p, a_{-p}) - G'_p(a) = G_p(a'_p, a_{-p}) - G_p(a)$$

---

## Applications

### 1. LLM Evaluation (Livebench)

**Problem**: Chatbot Arena uses Elo (2-player), ignores prompt player

**Solution**: 3-player game (Model A vs Model B vs Prompt)

**Payoffs**:
- Models: $G_A(m_A, m_B, t) = T(m_A, t) - T(m_B, t)$
- Prompts: $G_P = |G_A|$ (favor prompts that separate models)

**Results** (Livebench dataset):
- **Elo/Uniform**: Near-identical rankings
- **Deviation Ratings**: Top 4 models **tied**: `claude-3-5-sonnet`, `gemini-1.5-pro`, `Llama-3.1-405B`, `gpt-4o`
- **Insight**: Each model excels on different task subsets
  - Claude: LCB generation
  - Gemini: Summarization
  - Llama: Other tasks
  - GPT-4o: Connections

**Interpretation**: No single dominant model → group top performers together

### 2. Cyclic Games (Shapley's Game)

**Game**: Rock-paper-scissors variant with anti-coordination (penalty for matching)

**Uniform rating**: Ranks R > P > S (ignores cycles)

**Deviation rating**: R = P = S = N (Nash strategy also equal)

**Clone resilience**:
- Biased population (many Paper strategies)
- Uniform: Favors Scissors (counters overrepresented Paper)
- Deviation: Still rates R = P = S ✓

---

## Comparison to Related Methods

### vs Nash Averaging
- **Nash Avg**: 2-player zero-sum only (maxent NE)
- **Deviation**: N-player general-sum (strictest CCE)
- **Connection**: In 2p zero-sum, equivalent up to constant offset

### vs α-Rank
- **α-Rank**: Mass rating (stationary distribution of evolutionary dynamics)
- **Deviation**: Payoff rating (deviation gains under CCE)
- **Both**: Handle N-player general-sum, but different approaches

### vs Voting Methods
- **Voting**: Ordinal rankings (clone invariant via social choice)
- **Deviation**: Cardinal ratings (preserve payoff quantification)

---

## Computational Complexity

**Per iteration**: Solve LP with $|\mathcal{A}|$ variables, $\sum_p |\mathcal{A}_p|$ constraints

**Total iterations**: At most $\sum_p |\mathcal{A}_p|$ (one per strategy)

**Tractability**: Polynomial time (LP is convex)

**Comparison**:
- Nash Equilibrium: PPAD-hard (no FPTAS unless PPAD ⊆ P)
- CCE: Polynomial (linear programming)

---

## Key Insights

1. **Side-step equilibrium selection**: Select deviation gains, not equilibria
2. **Iterative strictness**: Minimize max deviation gains → strictest equilibrium
3. **Clone invariance from CCE**: Cloning creates redundant constraints
4. **Mixture invariance from linearity**: Mixed strategies = mixed constraints
5. **Offset invariance from deviations**: Deviations immune to offsets
6. **Grouping at top**: Strategies with complementary strengths rate equally

---

## Limitations

1. **Computational**: Requires iterative LP solving (still polynomial)
2. **Interpretation**: Negative ratings (deviation losses) less intuitive than Elo
3. **Data requirements**: Needs N-player payoff data (not always available)
4. **Grouping**: May tie many strategies at top (less discriminative than Elo)

---

## References

- **Paper**: Marris et al., "Deviation Ratings: A general, clone invariant rating method", ICLR 2025
- **Code**: TBD (ICLR 2025 submission)
- **Related**: Nash Averaging (Balduzzi et al., 2018), α-Rank (Omidshafiei et al., 2019)
- **Theory**: CCE (Hannan, 1957; Moulin, 1978), Clone independence (Tideman, 1987)

---

**See `Deviation_Ratings_detailed.md` for complete algorithm, proofs, worked examples, and LLM evaluation details.**
