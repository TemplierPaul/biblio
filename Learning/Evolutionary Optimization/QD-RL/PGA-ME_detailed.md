# PGA-MAP-Elites - Detailed Implementation

> **Quick overview**: [[PGA-ME]]

## Paper Information

**Title**: Policy Gradient Assisted MAP-Elites
**Authors**: Nilsson & Cully  
**Venue**: GECCO 2021
**Domain**: Neuroevolution for continuous control

---

## Core Algorithm: TD3-based Hybrid Variation

PGA-ME combines:
- **GA Variation** (50%): Random mutations for exploration
- **PG Variation** (50%): Policy gradient for exploitation
- **Actor Injection** (1): Trained RL actor for high-quality solutions

### Hyperparameters

```python
# Archive & Batch Configuration
b = 256              # Total batch size per iteration
b_GA = 128           # GA offspring
b_PG = 127           # PG offspring
b_AI = 1             # Actor injection

n_centroids = 1024   # CVT grid cells
budget = 1_000_000   # Total evaluations

# GA Parameters
sigma_1 = 0.005  # Small mutations (σ₁)
sigma_2 = 0.05   # Large mutations (σ₂)

# TD3 Parameters
N = 100           # Batch size for training
n_critic = 3000   # Critic updates per iteration
m_pg = 150        # PG steps per offspring
gamma = 0.99      # Discount factor
Delta = 2         # Actor update delay
tau = 0.005       # Soft update rate

# Noise parameters
sigma_noise = 0.2 # Target policy smoothing std
noise_clip = 0.5  # Clip smoothing noise

# Learning rates
lr_policy = 5e-3
lr_actor = 3e-4
lr_critic = 3e-4

replay_buffer_size = 1_000_000
```

### Variation Operators

#### GA Variation: Random Mutations

```python
def variation_ga(policies, sigma_1=0.005, sigma_2=0.05):
    """Gaussian mutations with bimodal distribution."""
    offspring = []
    for policy in policies:
        new_policy = copy.deepcopy(policy)
        for param in new_policy.parameters():
            sigma = sigma_1 if np.random.random() < 0.5 else sigma_2
            param.data += torch.randn_like(param) * sigma
        offspring.append(new_policy)
    return offspring
```

#### PG Variation: Deterministic Policy Gradient

```python
def variation_pg(policies, critic, replay_buffer, m_pg=150):
    """Gradient-based mutations using critic guidance."""
    offspring = []
    for policy in policies:
        new_policy = copy.deepcopy(policy)
        opt = optim.Adam(new_policy.parameters(), lr=lr_policy)
        
        for _ in range(m_pg):
            batch = sample_batch(replay_buffer, N)
            states, _, _, _ = batch
            
            # Maximize Q(s, π(s))
            loss = -critic(states, new_policy(states)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        offspring.append(new_policy)
    return offspring
```

#### Actor Injection

```python
def actor_injection(actor):
    """Add trained RL actor to archive."""
    return copy.deepcopy(actor)
```

### TD3 Training

```python
def train_actor_critic(actor, critic_1, critic_2,
                      actor_target, critic_1_target, critic_2_target,
                      replay_buffer, optimizers, n_critic=3000):
    """Standard TD3: Clipped Double Q-learning with delayed actor updates."""
    
    for t in range(n_critic):
        s, a, r, s_next = sample_batch(replay_buffer, N)
        
        # Critic target with policy smoothing
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(a) * sigma_noise, 
                               -noise_clip, noise_clip)
            a_target = torch.clamp(actor_target(s_next) + noise, -1, 1)
            
            q1_target = critic_1_target(s_next, a_target)
            q2_target = critic_2_target(s_next, a_target)
            y = r + gamma * torch.min(q1_target, q2_target)
        
        # Update critics
        loss_q1 = nn.MSELoss()(critic_1(s, a), y)
        loss_q2 = nn.MSELoss()(critic_2(s, a), y)
        
        optimizers[1].zero_grad()
        loss_q1.backward()
        optimizers[1].step()
        
        optimizers[2].zero_grad()
        loss_q2.backward()
        optimizers[2].step()
        
        # Delayed actor update
        if t % Delta == 0:
            loss_actor = -critic_1(s, actor(s)).mean()
            optimizers[0].zero_grad()
            loss_actor.backward()
            optimizers[0].step()
            
            # Soft update target networks
            for tp, p in zip(actor_target.parameters(), actor.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)
            for tp, p in zip(critic_1_target.parameters(), critic_1.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)
            for tp, p in zip(critic_2_target.parameters(), critic_2.parameters()):
                tp.data.copy_(tau * p.data + (1-tau) * tp.data)
```

### Main Loop

```python
while evaluations < budget:
    # Train critic
    train_actor_critic(actor, critic_1, critic_2, ...)
    
    # Selection
    parent_indices = np.random.choice(n_centroids, b)
    parents = [archive[i]['policy'] for i in parent_indices]
    
    # Variation (50% GA, 50% PG, 1 AI)
    offspring_GA = variation_ga(parents[:b_GA])
    offspring_PG = variation_pg(parents[b_GA:b_GA+b_PG], critic_1, replay_buffer)
    offspring_AI = [actor_injection(actor)]
    
    # Evaluation & Addition
    for offspring in offspring_GA + offspring_PG + offspring_AI:
        fitness, descriptor, transitions = evaluate(offspring)
        replay_buffer.extend(transitions)
        
        cell_idx = closest_centroid(descriptor, centroids)
        if archive[cell_idx]['fitness'] < fitness:
            archive[cell_idx] = {'policy': offspring, 'fitness': fitness}
        
        evaluations += 1
```

---

## Worked Example

**Setup**: 4 cells, 1 iteration

**Initial state**:
```
Cell 0: π₀ (fitness=5.0)
Cell 1: π₁ (fitness=4.5)  
Cell 2: empty
Cell 3: empty
```

**Iteration 1**:

1. **Sample parents**: [π₀, π₁] (with replacement)

2. **GA on π₀**: Add noise σ₂=0.05
   ```
   offspring_GA = [π₀ + 0.05·N(0,I)]
   ```

3. **PG on π₁**: 150 gradient steps
   ```
   For i=1..150:
       loss = -Q(s, π(s))
       ∇loss → update π
   offspring_PG = improved π₁
   ```

4. **Actor injection**:
   ```
   offspring_AI = [actor.clone()]
   ```

5. **Evaluate all 3**:
   ```
   fitness_GA = 4.8, descriptor_GA = [0.1, 0.9]
   → Cell 1, 4.8 < 4.5? No
   
   fitness_PG = 5.2, descriptor_PG = [0.8, 0.2]
   → Cell 3, empty? Yes → ADD
   
   fitness_AI = 5.5, descriptor_AI = [0.5, 0.5]
   → Cell 2, empty? Yes → ADD
   ```

**Final**:
```
Cell 0: π₀ (5.0)
Cell 1: π₁ (4.5)
Cell 2: offspring_AI (5.5) ✓ NEW
Cell 3: offspring_PG (5.2) ✓ NEW
```

---

## Key Insights

1. **Bimodal GA mutation**: σ₁ for fine-tuning, σ₂ for exploration
2. **PG leverages critic**: Exploits fitness landscape structure
3. **Actor injection**: Adds high-quality solution from RL training
4. **TD3 is standard**: No modifications to the RL algorithm
5. **Sample efficient**: 10× faster than MAP-Elites

---

## Limitations

- **Omnidirectional tasks fail**: Critic optimizes global fitness, ignores descriptors
- **No descriptor conditioning**: Cannot guide toward specific behaviors
- **Discrete archive**: Outputs 1024 solutions, not continuous function

→ Solution: **DCG-ME** adds descriptor-conditioned critic
