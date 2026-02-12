# Neural Population Learning (NeuPL)

## Definition
NeuPL represents an entire population of diverse policies within a **single conditional neural network**, enabling efficient skill transfer and scalability to complex games.

## Core Innovation

**Traditional population learning** (PSRO, JPSRO):
- Train $N$ independent networks for $N$ strategies
- No skill transfer between policies
- Memory: $O(N \times M)$ parameters
- Computationally expensive (~100x self-play cost)

**NeuPL solution**:
- Single conditional network: $\Pi_\theta(\cdot | o_{\leq t}, \nu)$
  - $\theta$: Shared network parameters
  - $\nu$: Strategy embedding vector (learnable)
- Memory: $O(M)$ parameters
- Training cost: ~1x self-play

## Architecture

### Conditional Policy Network
```
Components:
1. Observation encoder: h_obs = φ(o_≤t)   # LSTM/GRU for history
2. Strategy embedding: ν ∈ ℝ^d            # Learnable per-policy
3. Concatenation: h = [h_obs; ν]
4. Policy head: π = softmax(ψ(h))
```

### Strategy Embeddings
- **Initialization**: Random Gaussian $\nu_i \sim \mathcal{N}(0, I)$
- **Optimization**: End-to-end with $\theta$ via gradient descent
- **Semantics**: Encode strategic differences between policies

## Game Setting: POSG

**Partially-Observed Stochastic Games**:
- $(\mathcal{S}, \mathcal{O}, \mathcal{X}, \mathcal{A}, \mathcal{P}, \mathcal{R})$
- Partial observations: $\mathcal{X}: \mathcal{S} \to \mathcal{O} \times \mathcal{O}$
- Policy: $\pi(\cdot | o_{\leq t})$ maps observation history to actions
- Symmetric zero-sum: $\mathcal{R}(s) = (r, -r)$

## Neural Population

**Population**: $\Pi_{\theta, \Sigma} = \{\Pi_\theta(\cdot | o_{\leq t}, \sigma_i) : \sigma_i \in \Sigma\}$

**Interaction graph** $\Sigma = \{\sigma_i \in \Delta^{N-1}\}^{N-1}_{i=0}$:
- Each $\sigma_i$ defines which mixture policy to best-respond to
- Adjacency matrix of training matchups
- Effective population size: $|\text{UniqueRows}(\Sigma)| \leq N$

## Algorithm

```
Initialize θ, {νᵢ}, Σ

While training:
    # Sample meta-strategy
    σ ~ Uniform(UniqueRows(Σ))

    # Approximate best-response via RL
    Π_θ(·|o_≤t, σ) ← ABR(Π_θ(·|o_≤t, σ), Π^σ_{θ,Σ})

    # Update empirical payoffs (optional if adaptive)
    U ← Eval(Π_θ)

    # Update interaction graph (optional if adaptive)
    Σ ← MetaGraphSolver(U)
```

## Continual Learning Challenge

**Problem with shared representation**:
- Updating $\theta$ affects ALL policies simultaneously
- May cause catastrophic forgetting
- Can violate convergence assumptions

**Solution**: KL-divergence regularization
- Policy update: $\pi'(\cdot|s) \gets \pi(\cdot|s)$ implemented as:
  $$\min_{\theta, \nu'} D_{KL}[\pi(\cdot|s) \| \pi'_{\theta,\nu'}(\cdot|s)]$$
- Prevents forgetting while allowing improvement

## Convergence (NeuPL-JPSRO)

**Theorem**: NeuPL-JPSRO converges to CCE when:
1. Approximate BR improves: $J(\hat{\pi}, \pi') \geq J(\pi, \pi')$
2. KL minimization preserves behavior
3. Meta-graph follows JPSRO dynamics

## Skill Transfer Benefits

**Automatic transfer** via shared $\theta$:
- Perception (vision, state encoding)
- Locomotion (motor control)
- Memory (recurrent state)
- Strategic variations build on common foundation

**Empirical speedup**:
- MuJoCo control: Faster convergence than independent policies
- Capture-the-Flag: Transfer of visual features

## Comparison

| Method | Representation | Players | Game Type | Convergence |
|--------|----------------|---------|-----------|-------------|
| **PSRO** | Independent | 2 | Zero-sum | Nash |
| **JPSRO** | Independent | n | General-sum | CE/CCE |
| **NeuPL (2022)** | Shared | 2 | Symm. zero-sum | Empirical |
| **NeuPL-JPSRO** | Shared | n | General-sum | CCE (proven) |

## Extensions

- **NeuPL-JPSRO**: General-sum games, CCE convergence → [[NeuPL_JPSRO]]
- **Simplex NeuPL**: Any-mixture Bayes-optimality → [[Simplex_NeuPL]]

## Key Advantages

1. **~100x cheaper** than independent policy training
2. **Automatic skill transfer** across population
3. **Constant memory** w.r.t. population size
4. **Online adaptation** via conditioning on opponent prior
5. **Scalable** to complex domains (vision, RL)

## Interview Relevance

**Q: Why not just train N independent networks?**
- A: No skill transfer, memory $O(N \times M)$, computationally prohibitive for complex domains

**Q: How does one network represent multiple strategies?**
- A: Strategy embeddings $\nu$ condition the network, learned end-to-end

**Q: Doesn't shared representation break convergence?**
- A: Yes, but continual learning (KL regularization) fixes it → proven CCE convergence in NeuPL-JPSRO

**Q: Original NeuPL vs NeuPL-JPSRO?**
- A: Original (2022): symmetric zero-sum only, no formal convergence guarantees
- NeuPL-JPSRO (2024): n-player general-sum, proven CCE convergence

## Related
- [[NeuPL_detailed]] — Implementation details
- [[NeuPL_JPSRO]] — General-sum extension
- [[Simplex_NeuPL]] — Any-mixture optimality
- [[PSRO]] / [[JPSRO]] — Base algorithms
