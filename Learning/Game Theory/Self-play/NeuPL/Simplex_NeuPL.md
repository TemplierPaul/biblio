# Simplex Neural Population Learning

## Definition
Simplex-NeuPL extends NeuPL to achieve **any-mixture Bayes-optimality**: the ability to play optimally against arbitrary mixtures over a diverse policy population, using a single conditional network.

## Core Problem

**Standard NeuPL/PSRO limitation**:
- Can only BR to specific mixture policies $\{\Pi^{\sigma_0}, \ldots, \Pi^{\sigma_{N-1}}\}$ enumerated during training
- Cannot adapt to arbitrary opponent mixtures at test time
- Must either execute fixed policy or play Nash equilibrium mixture (conservative)

**Example failure**: Even if opponent uses same population and publicly declares strategy, cannot guarantee optimal play

## Population Simplex

### Geometric Interpretation

**Definition**: Simplex $\Delta^{N-1}$ where:
- **Vertices**: Basis policies $\{\pi_0, \pi_1, \ldots, \pi_{N-1}\}$
- **Points**: Barycentric coordinates $\sigma \in \Delta^{N-1}$
- **Mixture policies**: $\Pi^\sigma = \sum_i \sigma_i \pi_i$

**Traditional PSRO**: Iteratively expands simplex
- Selects discrete $\sigma_i \in \Delta^{i-1}$ via meta-solver
- Adds BR($\Pi^{\sigma_i}$) as new vertex → $\Delta^i$
- **Issue**: Only trains BR to specific $\sigma_i$, not all points

**Simplex-NeuPL**: Trains BR to ALL $\sigma \in \Delta^{N-1}$

## Algorithm

### Simplex Sampling

With probability $\epsilon$, sample opponent prior from **entire simplex**:

```
Initialize θ, Σ

While training:
    # Simplex sampling
    if rand() < ε:
        σ ~ Dirichlet(α_≤)           # Uniform over simplex
    else:
        σ ~ Uniform(UniqueRows(Σ))   # From meta-graph

    # Train best-response
    Π_θ(·|o_≤t, σ) ← ABR(Π_θ(·|o_≤t, σ), Π^σ_{θ,Σ})

    # Update meta-graph (optional)
    U ← Eval(Π_θ)
    Σ ← MetaGraphSolver(U)
```

**Dirichlet distribution**: $\text{Dir}(\alpha, \ldots, \alpha)$
- Symmetric Dirichlet with equal concentration
- Samples uniformly over simplex when $\alpha = 1$
- Applied to unique policies only

### Training Objective

Simultaneously optimize:
1. **Discrete BRs**: To meta-graph strategies $\Sigma$ (standard NeuPL)
2. **Continuum of BRs**: To all $\sigma \in \Delta^{N-1}$ (Bayes-optimal)

## Bayes-Optimal Policies

### Connection to Meta-Learning

Simplex-NeuPL optimizes continuum of Bayes-optimal objectives:

$$\max_\theta \mathbb{E}_{\sigma \sim \text{Dir}(\alpha)} [J(\Pi_\theta(\cdot|o_{\leq t}, \sigma), \Pi^\sigma_{\theta, \Sigma})]$$

**Interpretation**: Meta-learning over tasks defined by opponent priors

### Implicit Posterior Inference

**Key capability**: Policy performs Bayesian inference over opponent identity

**Process**:
1. Start with prior $\sigma$ (initial belief)
2. Observe opponent actions through interaction
3. Update posterior belief about which $\pi_i$ opponent is playing
4. Adapt policy accordingly

**Evidence**: Auxiliary readout head can predict opponent identity from history

## Experimental Results

### Goofspiel (Tractable Game)

**Setup**: Imperfect information bidding card game
- 5 point cards, descending order
- Players don't observe opponent's bids
- Known strategic cycles
- Analytical BR and posterior computable

**Results**:
- **Informed** $\Pi_\theta(\cdot | o_{\leq t}, \sigma)$: Approaches exact BR
- **Uninformed** $\Pi_\theta(\cdot | o_{\leq t}, \bar{\sigma})$ (uniform prior): High performance
- Both **significantly outperform** NE mixture policy
- Gap closes as prior becomes less informative (higher entropy)

**Posterior inference**:
- At $t=0$: Implicit posterior matches prior $\sigma$
- Through interaction: Converges to true opponent
- Uninformed policy: Quickly infers opponent from observations alone

### Running-with-Scissors (Complex Domain)

**Setup**: Partially-observed spatiotemporal strategy game
- Collect rock/paper/scissors items
- Confront at end (500 timesteps)
- Requires vision, memory, inference

**Test-time policy choices**:
1. **NE mixture**: ~0.05 (safe but suboptimal)
2. **Uniform mixture**: ~-0.02 (commits early, no adaptation)
3. **Uninformed policy**: ~0.15 ⭐ (adapts through interaction)
4. **Informed policy**: ~0.25 ⭐ (best with perfect prior)

**Key insight**: Temporal games enable inference → don't commit upfront

## Any-Mixture Optimality

### Definition

Conditional policy has **any-mixture optimality** if:

$$\forall \sigma \in \Delta^{N-1}: \Pi_\theta(\cdot | o_{\leq t}, \sigma) \approx \text{BR}(\Pi^\sigma)$$

### Test-Time Flexibility

**Options at deployment**:
1. **Informed**: If have prior belief → use $\Pi_\theta(\cdot | o_{\leq t}, \sigma)$
2. **Uninformed**: If no prior → use $\Pi_\theta(\cdot | o_{\leq t}, \bar{\sigma})$ (uniform)
3. **NE mixture**: If want safety → use $\Pi^{\sigma_{NE}}_{\theta, \Sigma}$

## Skill Transfer Benefits

### Auxiliary Task Effect

Simplex sampling acts as **auxiliary task**:
- Training to BR across simplex improves base population
- Transfer across continuum of BRs accelerates learning
- **Ablation**: Simplex-NeuPL produces better populations than vanilla NeuPL

## Hyperparameters

- **Simplex sampling rate**: $\epsilon = 0.5$ (50% of training)
- **Dirichlet concentration**: $\alpha = 1.0$ (uniform over simplex)

## Comparison

| Method | Any-mixture | Bayes-optimal | Adaptation | Game Type |
|--------|-------------|---------------|------------|-----------|
| **PSRO** | ✗ | ✗ | ✗ | Symmetric ZS |
| **NeuPL** | ✗ | ✗ | ✗ | Symmetric ZS |
| **Simplex-NeuPL** | ✓ | ✓ | ✓ | Symmetric ZS |

## Interview Relevance

**Q: NeuPL vs Simplex-NeuPL?**
- A: NeuPL trains BRs to discrete set of mixtures; Simplex trains BRs to **any mixture** over population

**Q: Why Bayes-optimal?**
- A: When opponent distribution is uncertain, optimal policy integrates over all possible opponents weighted by prior

**Q: How does it infer opponent?**
- A: Implicitly performs Bayesian posterior update through observation history representation (shown via auxiliary readout head)

**Q: When to use?**
- A: When need test-time adaptation, have subjective beliefs about opponents, or game allows extended interaction

**Q: Limitation?**
- A: Currently designed for symmetric zero-sum games (NeuPL-JPSRO extends NeuPL to general-sum but Simplex variant not yet developed)

## Key Takeaways

1. **Any-mixture optimality**: Can BR to arbitrary opponent mixtures, not just training set
2. **Bayes-optimal behavior**: Incorporates prior beliefs effectively
3. **Implicit inference**: Learns to infer opponent from observations automatically
4. **Test-time flexibility**: Choose informed/uninformed/NE based on scenario
5. **Skill transfer**: Simplex sampling improves population as auxiliary task

## Related
- [[Simplex_NeuPL_detailed]] — Implementation details
- [[NeuPL]] / [[NeuPL_detailed]] — Base framework
- [[NeuPL_JPSRO]] — General-sum extension (without simplex)
- [[PSRO]] — Population-based predecessor
