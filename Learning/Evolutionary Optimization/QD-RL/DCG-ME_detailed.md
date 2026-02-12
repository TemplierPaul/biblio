# DCG-MAP-Elites - Detailed Implementation

> **Quick overview**: [[DCG-ME]]

## Paper Information

**Title**: MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation
**Authors**: Faldor, Chalumeau, Flageat, Cully
**Venue**: GECCO 2023
**Key Innovation**: Descriptor-conditioned critic Q(s,a|d) + Archive distillation

---

## Core Innovation: Descriptor-Conditioned Critic

### Mathematical Foundation

**Problem**: PGA-ME's critic Q(s,a) maximizes global fitness → ignores descriptors → diversity collapses on omnidirectional tasks.

**Solution**: Descriptor-conditioned Q(s,a|d) that estimates return when achieving target descriptor d.

**Formulation** (with smoothing for continuity):

```
S(d, d') = exp(-||d - d'|| / L)  where L is length scale

Q(s,a|d') = S(d,d') · Q(s,a)
          = E_π[Σ γ^t · S(d,d') · r_t | s,a]
```

**Effect**: Reward scales by descriptor similarity. If d≈d' then S≈1 (learn to achieve d'), if d≠d' then S≈0.

### Descriptor-Conditioned Critic Network

```python
class DescriptorConditionedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, descriptor_dim):
        super().__init__()
        # Concatenate action AND descriptor at input
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + descriptor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action, descriptor):
        return self.net(torch.cat([state, action, descriptor], dim=-1))
```

### Descriptor-Conditioned Actor Network

```python
class DescriptorConditionedActor(nn.Module):
    def __init__(self, state_dim, action_dim, descriptor_dim):
        super().__init__()
        # Concatenate descriptor at input
        self.net = nn.Sequential(
            nn.Linear(state_dim + descriptor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, descriptor):
        return self.net(torch.cat([state, descriptor], dim=-1))
```

### TD3 with Descriptor Conditioning

```python
def train_actor_critic_dcg(actor, critic_1, critic_2,
                          actor_target, critic_1_target, critic_2_target,
                          replay_buffer, optimizers, n_critic=3000, L=0.008):
    """
    TD3 with descriptor conditioning.
    
    Transitions now include: (s, a, r, s', d, d')
    where d = observed descriptor, d' = target descriptor
    """
    
    for t in range(n_critic):
        # Sample batch with descriptors
        s, a, r, s_next, d, d_target = sample_batch_dcg(replay_buffer, N)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.clamp(torch.randn_like(a) * sigma_noise,
                               -noise_clip, noise_clip)
            
            # Target actor is descriptor-conditioned
            a_target = torch.clamp(
                actor_target(s_next, d_target) + noise, -1, 1
            )
            
            # Compute similarity scaling
            dist = torch.norm(d - d_target, dim=-1, keepdim=True)
            S = torch.exp(-dist / L)
            
            # Descriptor-conditioned Q-targets
            q1_target = critic_1_target(s_next, a_target, d_target)
            q2_target = critic_2_target(s_next, a_target, d_target)
            
            # Scale reward by similarity!
            y = S * r + gamma * torch.min(q1_target, q2_target)
        
        # Update critics (same as TD3)
        loss_q1 = nn.MSELoss()(critic_1(s, a, d_target), y)
        loss_q2 = nn.MSELoss()(critic_2(s, a, d_target), y)
        
        optimizers[1].zero_grad()
        loss_q1.backward()
        optimizers[1].step()
        
        optimizers[2].zero_grad()
        loss_q2.backward()
        optimizers[2].step()
        
        # Delayed actor update
        if t % Delta == 0:
            # Maximize Q(s, π(s|d), d)
            a_actor = actor(s, d_target)
            loss_actor = -critic_1(s, a_actor, d_target).mean()
            
            optimizers[0].zero_grad()
            loss_actor.backward()
            optimizers[0].step()
            
            # Soft updates
            for tp, p in zip(actor_target.parameters(), actor.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)
            # ... same for critics
```

### Implicit Descriptor Sampling Strategy

**Key insight**: No need for handcrafted descriptor sampling. Use variation operators to generate (d, d') pairs naturally:

**For PG offspring** (parent has descriptor d_ψ):
- PG mutates parent toward improving fitness **while maintaining** d_ψ
- Set d_target = d_ψ (implicit target)
- Most samples: d ≈ d_ψ → **positive samples**

**For GA offspring** (random mutation):
- GA produces random behaviors with descriptor d
- Set d_target = d (observed)
- Most samples: d = d_target → **positive samples**

**For actor evaluation** (active learning):
- Sample batch of descriptors from archive: d'₁, ..., d'_b
- Evaluate π_φ(·|d'_i), observe actual descriptor d_i
- Transitions: (s, a, r, s', d_i, d'_i)
- Many d_i ≠ d'_i → **negative samples** for robust training

### Actor Evaluation (Negative Sampling)

```python
def evaluate_actor(actor, archive, b_eval=256, sigma_d=0.0004):
    """
    Evaluate actor on sampled descriptors to generate negative samples.
    """
    transitions = []
    
    # Sample descriptors from archive with noise
    descriptors = []
    for _ in range(b_eval):
        elite = random.choice(list(archive.values()))
        d_sampled = elite['descriptor'] + np.random.normal(0, sigma_d)
        descriptors.append(d_sampled)
    
    # Evaluate actor on each descriptor
    for d_target in descriptors:
        d_tensor = torch.tensor(d_target, dtype=torch.float32)
        policy = actor(d_tensor)  # Specialized policy for this d
        
        fitness, d_actual, trans = evaluate(policy)
        
        # Augment transitions: add (d_actual, d_target)
        for (s, a, r, s_next) in trans:
            transitions.append((s, a, r, s_next, d_actual, d_target))
    
    return transitions
```

---

## Main Loop with Actor Evaluation

```python
while evaluations < budget:
    # Train descriptor-conditioned actor-critic
    train_actor_critic_dcg(actor, critic_1, critic_2, ...)
    
    # Selection
    parents = sample_batch_from_archive(archive, b)
    
    # Variation (same as PGA-ME)
    offspring_GA = variation_ga(parents[:b_GA])
    offspring_PG = variation_pg_dcg(parents[b_GA:b_GA+b_PG], critic_1, replay_buffer)
    
    # Evaluation with descriptor augmentation
    for offspring in offspring_GA:
        fitness, descriptor, transitions = evaluate(offspring)
        
        # GA: target descriptor = observed
        d_target = descriptor
        
        augmented = [(s, a, r, s', descriptor, d_target) for s, a, r, s' in transitions]
        replay_buffer.extend(augmented)
        
        cell_idx = closest_centroid(descriptor)
        if archive[cell_idx]['fitness'] < fitness:
            archive[cell_idx] = {'policy': offspring, 'fitness': fitness, 
                                'descriptor': descriptor}
    
    # Actor evaluation (generates negative samples)
    actor_trans = evaluate_actor(actor, archive, b_eval=b_GA)
    replay_buffer.extend(actor_trans)
    
    # Actor injection (optional)
    for d_sampled in sample_descriptors(archive, b_AI):
        policy_specialized = specialize_actor(actor, d_sampled)
        fitness, descriptor, transitions = evaluate(policy_specialized)
        
        cell_idx = closest_centroid(descriptor)
        if archive[cell_idx]['fitness'] < fitness:
            archive[cell_idx] = ...
        
        evaluations += 1
```

---

## Archive Distillation Bonus

During actor-critic training, the actor π_φ(s|d) **naturally distills** archive capabilities:

1. Training samples come from diverse behaviors (all archive elites)
2. Actor learns to execute **any behavior** by conditioning on d
3. Result: Single policy that can reproduce entire archive

**Usage**:

```python
# For any descriptor in archive
d = archive[123]['descriptor']

# Get specialized policy
policy = actor(state, descriptor=d)

# This policy approximately achieves descriptor d while maximizing fitness
# (on 50% of tasks, QD-score of policy ≈ QD-score of archive)
```

---

## Hyperparameters

Same as PGA-ME plus:

```python
# Descriptor conditioning
L = 0.008        # Similarity length scale
sigma_d = 0.0004 # Descriptor noise for actor evaluation
b_eval = b_GA    # Actor evaluation batch size

# Everything else unchanged from PGA-ME
```

---

## Worked Example: Why Descriptor Conditioning Fixes Omnidirectional Tasks

**Scenario**: Ant walks in any direction to minimize energy

**Without descriptor conditioning (PGA-ME)**:
```
Critic learns: Q(s,a) = E[energy savings]
Best action: DON'T MOVE (minimum energy)
PG mutation: All offspring → stationary (0,0)
Archive: Collapses to single point
Failure!
```

**With descriptor conditioning (DCG-ME)**:
```
Critic learns: Q(s,a|d) = E[energy while achieving position d]
Best action for d=(1,0): Move right efficiently
Best action for d=(0,1): Move up efficiently
Best action for d=(0,0): Stand still
PG mutation: Improves fitness FOR EACH descriptor
Archive: Maintains diversity across space!
Success: 82% improvement over PGA-ME
```

---

## Key Takeaways

1. **Reward scaling by similarity**: Descriptor conditioning ≡ scaling reward by S(d,d')
2. **Implicit sampling**: Variation operators naturally generate positive/negative samples
3. **Actor evaluation**: Active learning generates negative samples where d ≠ d'
4. **Archive distillation**: Actor learns to execute all archive behaviors
5. **Robust to deceptive fitness**: Works on omnidirectional where PGA-ME fails
