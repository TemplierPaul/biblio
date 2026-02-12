# Descriptor-Conditioned RL MAP-Elites (DCRL-ME)

**Quick reference**: [[DCRL-ME_detailed]]

---

## Overview

**DCRL-MAP-Elites** extends DCG-ME by using the descriptor-conditioned actor as a **generative model** to produce diverse specialized policies. This eliminates actor evaluation, achieving **2× sample efficiency improvement** over DCG-ME while maintaining all benefits of descriptor conditioning and archive distillation.

**Authors**: Faldor, Chalumeau, Flageat, Cully  
**Year**: 2024  
**Key Innovation**: Actor Injection mechanism that transforms descriptor-conditioned actor into unconditioned archive-compatible policies

---

## The Problem with DCG-ME

**DCG-ME requirement**: Evaluate actor on sampled descriptors to generate negative samples
- Cost: 256 extra evaluations per iteration
- Purpose: Generate (d,d') pairs where d ≠ d' for robust critic training
- Impact: Reduces sample efficiency by ~20%

**Question**: Can we eliminate this while keeping descriptor conditioning benefits?

---

## Solution: Actor as Generative Model

### Core Insight

The descriptor-conditioned actor π_φ(s|d) can **generate any behavior** by varying descriptor d.

For a fixed d:
```
π_ψ_d(s) = π_φ(s | d=fixed)
```

This is an unconditioned policy with different architecture.

**Challenge**: Archive requires unconditioned policies, actor outputs descriptor-conditioned actions.

**Solution**: **Extract** the unconditioned policy from the descriptor-conditioned actor.

---

## Actor Injection: The Key Innovation

### The Transformation

**Problem**: First layer mismatch

```
Actor first layer:     [s || d]^T W + b   (state + descriptor inputs)
Archive first layer:   [s]^T W' + b'       (state only)
```

**Solution**: Decompose weights

```
(s || d)^T W + b = s^T W_1 + (d^T W_2 + b)
                    ↓           ↓
                 state part   descriptor part

For descriptor d = [0.3, 0.7]:
New policy has:
  W' = W_1  (state weights, unchanged)
  b' = d^T W_2 + b  (descriptor "baked into" bias!)
  
Result: Unconditioned policy π_d(s) specialized for d
```

### How It Works

```python
def actor_injection(actor, descriptor_space, batch_size=64):
    policies = []
    for d in sample_descriptors(descriptor_space, batch_size):
        # Extract state weights
        W_state = actor.weights[:, :state_dim]
        
        # Compute specialized bias
        b_specialized = d @ actor.weights[:, state_dim:].T + actor.bias
        
        # Create policy with matching archive architecture
        policy = create_policy(W_state, b_specialized)
        policies.append(policy)
    return policies
```

---

## Three Complementary Variation Operators

### 1. GA Variation (25%)

Random Gaussian mutations. **Role**: Exploration

### 2. PG Variation (50%)

Descriptor-conditioned policy gradient (same as DCG-ME). **Role**: Exploitation toward specific descriptors

### 3. Actor Injection (25%) ✨ NEW

Transform actor into specialized policies for sampled descriptors. **Role**: Inject high-quality diverse solutions from RL training

---

## Algorithm Loop

```
while evaluations < budget:
    # Train descriptor-conditioned actor-critic
    train_actor_critic(actor, critic, replay_buffer)
    
    # Selection
    parents = sample_from_archive(b)
    
    # THREE variation operators (instead of 2 + actor eval)
    offspring_GA = variation_ga(parents[:64])
    offspring_PG = variation_pg(parents[64:192], critic)
    offspring_AI = actor_injection(actor, b=64)
    
    # Evaluation (same as DCG-ME)
    for offspring in offspring_GA + offspring_PG + offspring_AI:
        fitness, d_actual, transitions = evaluate(offspring)
        
        # Assign target descriptor based on operator
        d_target = d_actual if GA else parent_d if PG else sampled_d
        
        replay_buffer.add(transitions with (d_actual, d_target))
        archive.update(d_actual, fitness)
    
    # NO ACTOR EVALUATION NEEDED!
```

---

## Sample Efficiency: The Key Advantage

**DCG-ME evaluations per iteration**:
```
GA offspring:     64 evals
PG offspring:     128 evals
Actor evaluation: 256 evals  ← EXPENSIVE!
─────────────────────────────
Total:           448 evals + 256 extra = 704 evals/iter
```

**DCRL-ME evaluations per iteration**:
```
GA offspring:     64 evals
PG offspring:     128 evals
AI offspring:     64 evals   ← Weight extraction, no evaluation!
─────────────────────────────
Total:           256 evals/iter (direct benefit)
```

**Improvement**: **2.75× more efficient** than DCG-ME!

---

## Automatic Curriculum Learning

**Early training**: Many AI offspring fail to achieve d_target
- Lots of d_actual ≠ d_target pairs
- Critic learns robust negative examples
- Challenging learning signal

**Late training**: AI offspring improve at achieving targets
- Fewer d_actual ≠ d_target pairs
- Curriculum automatically adjusts
- Focuses on harder problems

No handcrafted curriculum needed!

---

## Performance & Results

| Task | DCG-ME | DCRL-ME | Speedup |
|------|--------|---------|---------|
| Ant-Omni | 870k | 900k | **+3%** |
| AntTrap-Omni | 800k | 850k | **+6%** |
| Hexapod-Omni | 680k | 700k | **+3%** |
| **Evaluations needed** | 1M | 500k | **2×** |

DCRL-ME achieves equal or better results with **50% fewer evaluations**!

---

## When to Use DCRL-ME

✅ **Use when**:
- DCG-ME works but evaluation budget is tight
- Want **best sample efficiency**
- Sufficient computational budget for actor training
- Need **archive distillation**

❌ **Use DCG-ME instead when**:
- Evaluation is cheap, computation is bottleneck
- Actor training stability is concern

❌ **Use PGA-ME when**:
- Simple unidirectional tasks
- No descriptor conditioning needed

---

## Advantages Over DCG-ME

| Aspect | DCG-ME | DCRL-ME |
|--------|--------|---------|
| Evaluations/iter | 704 | 256 |
| Actor evaluation | ✅ (overhead) | ❌ (eliminated) |
| Variation operators | 2 + eval | 3 integrated |
| Curriculum | Manual | Automatic |
| Sample efficiency | Baseline | **+175%** |

---

## Key Innovations

1. **Weight decomposition**: Separate state and descriptor components in first layer
2. **Generative model**: Use actor as policy generator via descriptor conditioning
3. **Implicit specialization**: Descriptor "baked into" bias, state inputs unchanged
4. **Eliminate actor eval**: Direct injection replaces expensive evaluation
5. **Automatic curriculum**: AI offspring provide natural difficulty progression

---

## The Evolution of QD-RL

```
MAP-Elites (2015)
    ↓
PGA-ME (2021): Add policy gradients
    ↓
DCG-ME (2023): Add descriptor conditioning
    ↓
DCRL-ME (2024): Add actor injection, eliminate actor evaluation
```

Each step solves a limitation of the previous approach.

---

## References

- **Paper**: Faldor et al., "Synergizing Quality-Diversity with Descriptor-Conditioned RL" (TELO 2024)
- **Building on**: DCG-ME (Faldor et al. 2023), PGA-ME (Nilsson & Cully 2021)
- **Related**: Actor-Critic methods (Fujimoto et al., Lillicrap et al.)
