# DCRL-MAP-Elites - Detailed Implementation

> **Quick overview**: [[DCRL-ME]]

## Paper Information

**Title**: Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning
**Authors**: Faldor, Chalumeau, Flageat, Cully
**Venue**: TELO 2024
**Key Innovation**: Use descriptor-conditioned actor as generative model for diverse policies

---

## Core Problem Solved

**DCG-ME requirement**: Actor evaluation to generate negative samples (d ≠ d')
- Cost: b_eval = 128 extra evaluations per iteration
- Reduces sample efficiency by ~20%

**DCRL-ME solution**: Use actor π_φ(s|d) as generative model to produce specialized unconditioned policies
- **No actor evaluation needed**
- Three variation operators: GA + PG + AI
- Natural negative samples from AI offspring
- **2× sample efficiency improvement**

---

## Architecture Transformation: The Key Innovation

### The Challenge

**Archive policies**: π_ψ(s) : S → A (takes only state)
**Descriptor-conditioned actor**: π_φ(s|d) : S × D → A (takes state + descriptor)

**Problem**: Can't directly inject π_φ into archive (different I/O architectures).

### The Solution: Weight Extraction

**Key observation**: First layer concatenates inputs: [s || d]

```
(s || d)^T W + b = s^T W_1 + (d^T W_2 + b)
         ↓              ↓
    First layer    Decomposed form
```

Where:
- W ∈ ℝ^{(dim(S)+dim(D)) × h} is actor's first layer weight
- W_1 ∈ ℝ^{dim(S) × h} (state part)
- W_2 ∈ ℝ^{dim(D) × h} (descriptor part)
- b ∈ ℝ^h (bias)

**Extraction**: For a specific descriptor d:
```
New policy first layer:
  W' = W_1  (state weights, unchanged)
  b' = d^T W_2 + b  (descriptor "baked into" bias!)
  
Result: Policy with same architecture as archive policies!
```

### Implementation

```python
def actor_injection(actor, descriptor_space, batch_size=64):
    """
    Transform descriptor-conditioned actor into specialized unconditioned policies.
    
    For each sampled descriptor d:
    1. Extract state weights W_1 from first layer
    2. Compute modified bias b' = d^T W_2 + b
    3. Create new policy with (W_1, b') and copy remaining layers
    4. Result: π_d(s) that achieves descriptor d
    """
    policies = []
    
    # Sample descriptors uniformly
    descriptors = sample_uniform(descriptor_space, batch_size)
    
    for d in descriptors:
        # Extract from actor's first layer
        W_full = actor.net[0].weight  # Shape: [256, state_dim + desc_dim]
        b_full = actor.net[0].bias
        
        # Decompose into state and descriptor parts
        W_state = W_full[:, :state_dim]      # [256, state_dim]
        W_desc = W_full[:, state_dim:]       # [256, desc_dim]
        
        # Compute specialized bias
        d_tensor = torch.tensor(d, dtype=torch.float32)
        b_new = torch.matmul(d_tensor, W_desc.t()) + b_full
        
        # Create specialized policy
        policy = UnconditionedPolicy(state_dim, action_dim)
        
        # Set first layer
        policy.net[0].weight.data = W_state
        policy.net[0].bias.data = b_new
        
        # Copy remaining layers unchanged
        for i in range(1, len(actor.net)):
            if hasattr(actor.net[i], 'weight'):
                policy.net[i].weight.data = actor.net[i].weight.data.clone()
            if hasattr(actor.net[i], 'bias'):
                policy.net[i].bias.data = actor.net[i].bias.data.clone()
        
        policies.append(policy)
    
    return policies
```

### Visual Example

```
DESCRIPTOR-CONDITIONED ACTOR:
┌──────────────────────────────────────┐
│ Input: [s || d]  (state + descriptor)│
├──────────────────────────────────────┤
│ First layer: [s || d]^T W + b        │
│   W: (state_dim + desc_dim) × 256    │
│   b: 256                             │
└──────────────────────────────────────┘

AFTER TRANSFORMATION FOR d = [0.3, 0.7]:
┌──────────────────────────────────────┐
│ Input: [s]  (state only!)            │
├──────────────────────────────────────┤
│ First layer: s^T W_1 + b'            │
│   W_1: state_dim × 256  (from W)     │
│   b' = [0.3, 0.7]^T W_desc + b       │
│      (descriptor compiled into bias) │
└──────────────────────────────────────┘

Result: Unconditioned policy that inherently targets d=[0.3, 0.7]
```

---

## Three Variation Operators

### Operator 1: GA Variation (25%)

```python
def variation_ga(parents):
    """Same as DCG-ME: random Gaussian mutations."""
    offspring = []
    for policy in parents:
        new_p = copy.deepcopy(policy)
        for param in new_p.parameters():
            sigma = 0.005 if random.random() < 0.5 else 0.05
            param.data += torch.randn_like(param) * sigma
        offspring.append(new_p)
    return offspring
```

**Role**: Exploration, discovers new regions

### Operator 2: PG Variation (50%)

```python
def variation_pg_dcg(parents, critic, replay_buffer):
    """Descriptor-conditioned PG: same as DCG-ME."""
    offspring = []
    for policy in parents:
        new_p = copy.deepcopy(policy)
        d_target = policy.descriptor  # Parent's descriptor
        opt = optim.Adam(new_p.parameters())
        
        for _ in range(150):
            s, a, _, _, _, _ = sample_batch(replay_buffer, 100)
            loss = -critic(s, new_p(s), d_target).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        offspring.append(new_p)
    return offspring
```

**Role**: Exploitation, improves fitness while maintaining descriptor

### Operator 3: Actor Injection (25%) ✨ NEW

```python
def actor_injection(actor, descriptor_space, batch_size=64):
    """
    Transform descriptor-conditioned actor into specialized policies.
    
    This REPLACES actor evaluation!
    - No need to run π_φ(·|d') to generate negative samples
    - Instead, create policies that implicitly target specific d
    - Much more sample efficient
    """
    return transform_actor_to_policies(actor, descriptor_space, batch_size)
```

**Role**: 
- Injects high-quality solutions from RL training
- Generates diverse solutions by varying descriptor d
- Natural negative samples from evaluation (d_sampled ≠ d_actual)
- **Eliminates need for separate actor evaluation**

---

## Main Loop: Three Operators

```python
b_GA = 64    # 25%
b_PG = 128   # 50%
b_AI = 64    # 25% (was 0 in DCG-ME)

while evaluations < budget:
    # Train actor-critic (same as DCG-ME)
    train_actor_critic_dcg(actor, critic_1, critic_2, ...)
    
    # Selection
    parents = sample_batch_from_archive(b)
    
    # THREE Variation operators
    offspring_GA = variation_ga(parents[:b_GA])
    offspring_PG = variation_pg_dcg(parents[b_GA:b_GA+b_PG], critic_1, replay_buffer)
    
    # New! Actor injection
    offspring_AI = actor_injection(actor, descriptor_space, b_AI)
    
    # Evaluation with implicit descriptor sampling
    for offspring in offspring_GA + offspring_PG + offspring_AI:
        fitness, d_actual, transitions = evaluate(offspring)
        
        # Determine target descriptor based on variation operator
        if offspring in offspring_GA:
            d_target = d_actual  # GA: observe what you got
        elif offspring in offspring_PG:
            d_target = parent_descriptor  # PG: target parent's descriptor
        else:  # offspring in offspring_AI
            d_target = sampled_descriptor  # AI: target sampled descriptor
        
        # Augment transitions
        augmented = [(s, a, r, s', d_actual, d_target) for s, a, r, s' in transitions]
        replay_buffer.extend(augmented)
        
        # Update archive
        cell_idx = closest_centroid(d_actual)
        if archive[cell_idx]['fitness'] < fitness:
            archive[cell_idx] = {'policy': offspring, 'fitness': fitness}
        
        evaluations += 1
    
    # NO ACTOR EVALUATION NEEDED!
```

---

## Implicit Descriptor Sampling Strategy

**Automatic curriculum learning** without handcrafted sampling:

**Early training**:
- Many AI offspring fail to achieve d_target
- Lots of d_actual ≠ d_target transitions
- Critic learns "what NOT to do"

**Late training**:
- AI offspring improve at achieving targets
- Fewer d_actual ≠ d_target transitions
- Curriculum automatically adjusts difficulty

---

## Sample Efficiency Comparison

```
DCG-ME evaluation cost per iteration:
  b_GA offspring:   64 evals
  b_PG offspring:   128 evals
  Actor evaluation: 256 evals (EXPENSIVE!)
  ────────────────
  Total:            448 evals + 256 = 704 evals/iter

DCRL-ME evaluation cost per iteration:
  b_GA offspring:   64 evals
  b_PG offspring:   128 evals
  AI offspring:     64 evals
  ────────────────
  Total:            256 evals/iter (no actor evaluation!)

Improvement: 2.75× fewer evaluations!
```

---

## Hyperparameters

Exactly same as DCG-ME, except:

```python
# Archive batch breakdown
b = 256
b_GA = 64      # Changed from DCG-ME's 128
b_PG = 128     # Changed from DCG-ME's 128
b_AI = 64      # NEW (was 0)

# Everything else identical
L = 0.008
sigma_1 = 0.005
sigma_2 = 0.05
# ... etc
```

---

## Worked Example: Actor Injection Step-by-Step

**Suppose**:
- state_dim = 4, action_dim = 2, desc_dim = 2
- Actor first layer: W ∈ ℝ^{6×256}, b ∈ ℝ^{256}

**Want**: Specialize for d = [0.6, 0.4]

**Step 1**: Extract components
```
W_state = W[:, 0:4]    # State part
W_desc = W[:, 4:6]     # Descriptor part
```

**Step 2**: Compute new bias
```
d = [0.6, 0.4]
W_desc has shape [256, 2]
d^T W_desc has shape [256]

b_new = d^T W_desc + b = [0.6, 0.4] @ W_desc^T + b
```

**Step 3**: Create policy with matching architecture
```
New first layer:
  Weight: W_state (shape [256, 4])
  Bias: b_new (shape [256])

Remaining layers: identical to actor
```

**Step 4**: Evaluate
```
π_d(s) = actor.net[1:](W_state @ s + b_new)
       = policy specialized for d=[0.6, 0.4]
```

---

## Advantages Over DCG-ME

1. **Sample efficiency**: 2× fewer evaluations (no actor eval)
2. **Three operators**: Better exploration + exploitation balance
3. **Simpler training**: Implicit sampling handles negative examples naturally
4. **Same performance**: Often equal or slightly better than DCG-ME
5. **Practical**: No extra hyperparameter for actor evaluation

---

## Key Insights

1. **Generative model**: Descriptor-conditioned actor generates diverse policies
2. **Weight decomposition**: Bake descriptor into bias, keep state weights
3. **No architecture mismatch**: Specialized policies match archive architecture
4. **Natural curriculum**: AI offspring difficulty increases automatically
5. **Unified framework**: Single actor-critic produces both exploration (GA) and exploitation (PG/AI)
