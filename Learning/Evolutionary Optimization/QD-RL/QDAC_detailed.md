# QDAC - Detailed Implementation

> **Quick overview**: [[QDAC]]

## Paper Information

**Title**: Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics
**Authors**: Luca Grillotti, Maxence Faldor, Borja González León, Antoine Cully
**Affiliations**: Imperial College London, Iconic AI
**Venue**: ICML 2024
**Code**: https://github.com/adaptive-intelligent-robotics/QDAC
**Website**: https://adaptive-intelligent-robotics.github.io/QDAC/

---

## Problem Formulation

### Quality-Diversity as Constrained Optimization

**Goal**: For all skills z ∈ Z, learn π(a|s,z) that:

**Problem 1 (P1) - Original**:
```
maximize E_π_z [Σ_{i=0}^∞ γ^i r_{t+i}]
subject to: φ̄ = z

where φ̄ = lim_{T→∞} (1/T) Σ_{t=0}^{T-1} φ_t
      φ_t = φ(s_t, a_t)  (features provided by environment)
```

**Problem 2 (P2) - Relaxed**:
```
maximize E_π_z [Σ_{i=0}^∞ γ^i r_{t+i}]
subject to: ||φ̄ - z|| ≤ ε
```

**Problem 3 (P3) - Tractable (via Proposition 1)**:
```
maximize E_π_z [V(s,z)]
subject to: E_π_z [||(1-γ)ψ(s,z) - z||] ≤ ε

where:
  V(s,z) = E[Σ γ^i r_{t+i} | s_t = s]        (value function)
  ψ(s,z) = E[Σ γ^i φ_{t+i} | s_t = s]        (successor features)
```

---

## Theoretical Foundation

### Proposition 1: Upper Bound via Successor Features

**Statement**: Under policy π_z, the distance between expected features and skill is bounded:

```
||φ̄ - z|| ≤ (1-γ) E_ρ^π_z [||ψ(s,z) - z||]

where ρ^π_z(s) = lim_{t→∞} P(s_t = s | s_0, π_z)  (stationary distribution)
```

**Proof sketch** (see Appendix for full proof):
1. Expand φ̄ as infinite sum: φ̄ = (1-γ) Σ_{t=0}^∞ γ^t φ_t
2. Use Bellman equation for ψ: ψ(s) = E[φ_t + γψ(s_{t+1}) | s_t = s]
3. Apply triangle inequality and geometric series bound

**Significance**: This justifies replacing the intractable constraint on φ̄ with a tractable constraint on ψ.

### Corollary: Constraint Satisfaction

If ||(1-γ)ψ(s,z) - z|| ≤ ε for all s ~ ρ^π_z, then ||φ̄ - z|| ≤ ε.

**Interpretation**: Minimizing the bound ensures constraint satisfaction.

---

## Algorithm Architecture

### Network Components

**1. Skill-Conditioned Actor π_θ_π (a | s, z)**:
```python
class SkillConditionedActor(nn.Module):
    def __init__(self, state_dim, action_dim, skill_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, skill):
        x = self.encoder(torch.cat([state, skill], dim=-1))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state, skill):
        mean, log_std = self.forward(state, skill)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()  # reparameterization trick
        action = torch.tanh(x)

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
```

**2. Value Function Critic V_θ_V (s, z)**:
```python
class ValueCritic(nn.Module):
    def __init__(self, state_dim, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, skill):
        return self.net(torch.cat([state, skill], dim=-1))
```

**3. Successor Features Critic ψ_θ_ψ (s, z)**:
```python
class SuccessorFeaturesCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)  # outputs feature vector
        )

    def forward(self, state, skill):
        return self.net(torch.cat([state, skill], dim=-1))
```

**4. Lagrange Multiplier Network λ_θ_λ (s, z)**:
```python
class LagrangeMultiplier(nn.Module):
    def __init__(self, state_dim, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # ensures λ ∈ [0,1]
        )

    def forward(self, state, skill):
        return self.net(torch.cat([state, skill], dim=-1))
```

---

## Training Procedure

### Complete QDAC Algorithm

```python
def qdac(env, num_iterations, episode_length, batch_size):
    # Initialize networks
    actor = SkillConditionedActor(state_dim, action_dim, skill_dim)
    value_critic = ValueCritic(state_dim, skill_dim)
    sf_critic = SuccessorFeaturesCritic(state_dim, skill_dim, feature_dim)
    lagrange = LagrangeMultiplier(state_dim, skill_dim)

    # Target networks for stability
    value_critic_target = deepcopy(value_critic)
    sf_critic_target = deepcopy(sf_critic)

    # Optimizers
    actor_opt = Adam(actor.parameters(), lr=3e-4)
    value_opt = Adam(value_critic.parameters(), lr=3e-4)
    sf_opt = Adam(sf_critic.parameters(), lr=3e-4)
    lagrange_opt = Adam(lagrange.parameters(), lr=3e-4)

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=1_000_000)

    for iteration in range(num_iterations):
        # 1. Sample skill uniformly
        skill = sample_uniform(skill_space)

        # 2. Collect episode
        state = env.reset()
        for t in range(episode_length):
            action, _ = actor.sample(state, skill)
            next_state, reward, done, info = env.step(action)
            features = info['features']  # φ(s,a) from environment

            # Store transition with skill
            replay_buffer.add((state, action, reward, features, next_state, skill))

            state = next_state
            if done:
                break

        # 3. Training steps (multiple per environment step)
        for _ in range(num_train_steps):
            # Sample mini-batch
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, features, next_states, skills = batch

            # SKILL RE-LABELING: Augment batch with random skills
            new_skills = sample_uniform(skill_space, size=batch_size)
            states = torch.cat([states, states], dim=0)
            actions = torch.cat([actions, actions], dim=0)
            rewards = torch.cat([rewards, rewards], dim=0)
            features = torch.cat([features, features], dim=0)
            next_states = torch.cat([next_states, next_states], dim=0)
            skills = torch.cat([skills, new_skills], dim=0)

            # 3a. Update Lagrange multiplier
            lambda_loss = update_lagrange(lagrange, lagrange_opt,
                                         states, skills, sf_critic, threshold)

            # 3b. Update value critic
            value_loss = update_value_critic(value_critic, value_critic_target,
                                            value_opt, states, rewards,
                                            next_states, skills, gamma)

            # 3c. Update successor features critic
            sf_loss = update_sf_critic(sf_critic, sf_critic_target, sf_opt,
                                       states, features, next_states,
                                       skills, gamma)

            # 3d. Update actor
            actor_loss = update_actor(actor, actor_opt, value_critic,
                                     sf_critic, lagrange, states, skills)

            # 3e. Update target networks (soft update)
            soft_update(value_critic_target, value_critic, tau=0.005)
            soft_update(sf_critic_target, sf_critic, tau=0.005)

    return actor
```

### Update Functions

**1. Lagrange Multiplier Update**:
```python
def update_lagrange(lagrange, optimizer, states, skills, sf_critic, threshold):
    """
    Update λ to increase when constraint violated, decrease otherwise.

    Binary cross-entropy loss:
      y = 1 if ||(1-γ)ψ(s,z) - z|| > ε  (constraint violated)
      y = 0 otherwise                     (constraint satisfied)

    Loss = -[(1-y)log(1-λ) + y·log(λ)]
    """
    with torch.no_grad():
        sf = sf_critic(states, skills)
        distance = torch.norm((1 - gamma) * sf - skills, dim=-1)
        y = (distance > threshold).float()  # binary labels

    lambda_pred = lagrange(states, skills)

    # Binary cross-entropy
    loss = -(
        (1 - y) * torch.log(1 - lambda_pred + 1e-8) +
        y * torch.log(lambda_pred + 1e-8)
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**2. Value Critic Update**:
```python
def update_value_critic(critic, target_critic, optimizer,
                       states, rewards, next_states, skills, gamma):
    """
    Minimize Bellman error:
      L_V = E[(V(s,z) - (r + γ·V(s',z)))²]
    """
    with torch.no_grad():
        target_value = rewards + gamma * target_critic(next_states, skills)

    predicted_value = critic(states, skills)
    loss = F.mse_loss(predicted_value, target_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**3. Successor Features Critic Update**:
```python
def update_sf_critic(critic, target_critic, optimizer,
                    states, features, next_states, skills, gamma):
    """
    Minimize Bellman error for successor features:
      L_ψ = E[||ψ(s,z) - (φ + γ·ψ(s',z))||²]

    Note: φ plays the role of reward
    """
    with torch.no_grad():
        target_sf = features + gamma * target_critic(next_states, skills)

    predicted_sf = critic(states, skills)
    loss = F.mse_loss(predicted_sf, target_sf)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**4. Actor Update**:
```python
def update_actor(actor, optimizer, value_critic, sf_critic,
                lagrange, states, skills):
    """
    Maximize Lagrangian objective:
      L = (1 - λ(s,z))·V(s,z) - λ(s,z)·||(1-γ)ψ(s,z) - z||

    Combined with SAC entropy term:
      J_π = E[L + α·H(π(·|s,z))]
    """
    # Sample action from current policy
    actions, log_probs = actor.sample(states, skills)

    # Compute Lagrangian components
    lambda_val = lagrange(states, skills).detach()  # don't backprop through λ

    value = value_critic(states, skills)
    sf = sf_critic(states, skills)
    sf_distance = torch.norm((1 - gamma) * sf - skills, dim=-1, keepdim=True)

    # Lagrangian objective
    lagrangian = (1 - lambda_val) * value - lambda_val * sf_distance

    # SAC entropy regularization
    alpha = 0.1  # temperature parameter
    entropy = -alpha * log_probs

    # Total objective (negative because we minimize)
    loss = -(lagrangian + entropy).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**5. Soft Target Update**:
```python
def soft_update(target_net, source_net, tau=0.005):
    """
    Soft update: θ_target ← τ·θ + (1-τ)·θ_target
    """
    for target_param, param in zip(target_net.parameters(),
                                   source_net.parameters()):
        target_param.data.copy_(tau * param.data +
                               (1 - tau) * target_param.data)
```

---

## Skill Sampling Strategy

### Uniform Sampling

```python
def sample_uniform(skill_space, size=1):
    """
    Sample skills uniformly from skill space.

    For feet contact: skill_space = [0,1]^n_feet
    For velocity: skill_space = [-v_max, v_max]²
    For jump: skill_space = [0, h_max]
    For angle: skill_space = unit circle (cos θ, sin θ)
    """
    if skill_space.type == 'hypercube':
        return np.random.uniform(skill_space.low, skill_space.high,
                                size=(size, skill_space.dim))

    elif skill_space.type == 'circle':
        theta = np.random.uniform(0, 2*np.pi, size=size)
        return np.stack([np.cos(theta), np.sin(theta)], axis=-1)
```

### Archive-Based Sampling (Optional Enhancement)

```python
def sample_from_archive(archive, skill_space, epsilon=0.1):
    """
    Sample skills near archive gaps to improve coverage.

    With probability ε: sample uniformly
    With probability 1-ε: sample near least-visited regions
    """
    if np.random.rand() < epsilon:
        return sample_uniform(skill_space)
    else:
        # Find least-visited region
        coverage = archive.compute_coverage()
        low_coverage_region = coverage.argmin()
        return archive.sample_near(low_coverage_region)
```

---

## Feature Functions

### Example Feature Definitions

**1. Feet Contact (Ant: 4 feet, Humanoid: 2 feet)**:
```python
def feet_contact_features(state, action, info):
    """
    Binary indicator for each foot's ground contact.

    Returns: [c_1, c_2, ..., c_n] where c_i ∈ {0,1}
    Average features = proportion of time each foot touches ground
    """
    contacts = info['contact_forces']  # from physics engine
    return (contacts > 0.1).astype(float)
```

**2. Velocity (2D)**:
```python
def velocity_features(state, action, info):
    """
    Velocity in xy-plane.

    Returns: [v_x, v_y]
    Average features = average velocity
    """
    return info['velocity'][:2]
```

**3. Jump (minimum foot height)**:
```python
def jump_features(state, action, info):
    """
    Height of lowest foot above ground.

    Returns: [min(h_1, h_2, ..., h_n)]
    Average features = average min height
    """
    foot_heights = info['foot_heights']
    return [np.min(foot_heights)]
```

**4. Angle (body orientation)**:
```python
def angle_features(state, action, info):
    """
    Body angle about z-axis (encoded as unit circle point).

    Returns: [cos(α), sin(α)]
    Average features = average direction (circular mean)
    """
    quaternion = info['orientation']
    angle = quaternion_to_euler(quaternion)[2]  # yaw angle
    return [np.cos(angle), np.sin(angle)]
```

---

## Hyperparameters

### Model-Free QDAC (based on SAC)

```python
HYPERPARAMETERS = {
    # Network architecture
    'hidden_dim': 256,
    'num_hidden_layers': 2,

    # Training
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 3e-4,
    'learning_rate_sf': 3e-4,
    'learning_rate_lagrange': 3e-4,
    'batch_size': 256,
    'replay_buffer_size': 1_000_000,

    # RL parameters
    'gamma': 0.99,
    'tau': 0.005,  # target network update rate
    'alpha': 0.1,  # SAC temperature (or auto-tune)

    # QDAC-specific
    'threshold': 0.1,  # ε for constraint relaxation
    'skill_relabel_fraction': 0.5,  # fraction of batch to relabel

    # Episode
    'episode_length': 1000,
    'num_train_steps_per_env_step': 1,
}
```

### Task-Specific Thresholds

```
Feet Contact: ε = 0.05
Velocity: ε = 0.5
Jump: ε = 0.02
Angle: ε = 0.1
```

---

## Model-Based Variant (QDAC-MB)

### Architecture Differences

Built on **DreamerV2** instead of SAC:

**1. World Model**:
```python
class WorldModel(nn.Module):
    def __init__(self):
        self.encoder = ImageEncoder()  # or StateEncoder
        self.rssm = RecurrentSSM()     # recurrent state-space model
        self.reward_predictor = RewardPredictor()
        self.feature_predictor = FeaturePredictor()  # predict φ
        self.continue_predictor = ContinuePredictor()
```

**2. Training in Imagination**:
```python
def train_qdac_mb(world_model, actor, critics, replay_buffer):
    # 1. Update world model from replay buffer
    train_world_model(world_model, replay_buffer)

    # 2. Imagine trajectories
    with torch.no_grad():
        imagined_trajectories = imagine_trajectories(world_model, actor,
                                                     horizon=15)

    # 3. Train actor and critics on imagined data
    train_actor_critics(actor, critics, imagined_trajectories)
```

**Advantages**:
- Better on complex features (Jump with min operator)
- Can learn from imagined rollouts
- More sample efficient

**When to use**: Complex feature functions, limited real-world samples

---

## Evaluation Metrics

### 1. Distance to Skill Metrics

**Expected Distance to Skill d(z)**:
```python
def expected_distance_to_skill(policy, env, skill, num_rollouts=10):
    """
    Estimate E[||φ̄ - z||] for a given skill z.
    """
    distances = []
    for _ in range(num_rollouts):
        features = []
        state = env.reset()
        done = False

        while not done:
            action, _ = policy.sample(state, skill)
            next_state, reward, done, info = env.step(action)
            features.append(info['features'])
            state = next_state

        avg_features = np.mean(features, axis=0)
        distances.append(np.linalg.norm(avg_features - skill))

    return np.mean(distances)
```

**Distance Profile**:
```python
def distance_profile(policy, env, skill_grid):
    """
    For each distance threshold d, compute proportion of skills
    with expected distance < d.

    Returns: function d ↦ (1/N)Σ 𝟙[d(z_i) < d]
    """
    distances = [expected_distance_to_skill(policy, env, z)
                for z in skill_grid]

    def profile(d):
        return np.mean([dist < d for dist in distances])

    return profile, distances
```

**Distance Score**:
```python
def distance_score(policy, env, skill_grid):
    """
    Aggregate metric: average negative distance.
    Higher is better (less negative).

    Returns: (1/N)Σ -d(z_i)
    """
    distances = [expected_distance_to_skill(policy, env, z)
                for z in skill_grid]
    return -np.mean(distances)
```

### 2. Performance Metrics

**Expected Return R(z)**:
```python
def expected_return(policy, env, skill, num_rollouts=10):
    """
    Estimate E[Σ r_t] for a given skill z (undiscounted).
    """
    returns = []
    for _ in range(num_rollouts):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            action, _ = policy.sample(state, skill)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            state = next_state

        returns.append(episode_return)

    return np.mean(returns)
```

**Performance Profile**:
```python
def performance_profile(policy, env, skill_grid, d_eval=0.1):
    """
    For each return threshold R, compute proportion of ACHIEVED skills
    with expected return > R.

    Filters out unachieved skills: d(z) > d_eval

    Returns: function R ↦ (1/N)Σ 𝟙[d(z_i)<d_eval ∧ R(z_i)>R]
    """
    data = []
    for z in skill_grid:
        d = expected_distance_to_skill(policy, env, z)
        R = expected_return(policy, env, z)
        data.append((d, R))

    def profile(R_thresh):
        return np.mean([R > R_thresh for d, R in data if d < d_eval])

    return profile, data
```

**Performance Score**:
```python
def performance_score(policy, env, skill_grid, d_eval=0.1):
    """
    Aggregate metric: average return of achieved skills.

    Returns: (1/N)Σ R(z_i)·𝟙[d(z_i) < d_eval]
    """
    data = []
    for z in skill_grid:
        d = expected_distance_to_skill(policy, env, z)
        R = expected_return(policy, env, z)
        if d < d_eval:
            data.append(R)

    return np.mean(data) if data else 0.0
```

---

## Implementation Tips

### 1. Skill Re-labeling

**Why**: Improve sample efficiency by reusing transitions for multiple skills.

**How**: For each transition (s,a,r,φ,s',z), create new transition with random z'.

```python
# Naive approach: resample each transition
for (s,a,r,phi,s_next,z) in batch:
    z_new = sample_uniform(skill_space)
    batch_augmented.add((s,a,r,phi,s_next,z_new))

# Efficient approach: vectorized
skills_original = batch['skills']
skills_relabeled = sample_uniform(skill_space, size=batch_size)
batch_augmented = {
    'states': torch.cat([batch['states'], batch['states']], 0),
    'actions': torch.cat([batch['actions'], batch['actions']], 0),
    'rewards': torch.cat([batch['rewards'], batch['rewards']], 0),
    'features': torch.cat([batch['features'], batch['features']], 0),
    'next_states': torch.cat([batch['next_states'], batch['next_states']], 0),
    'skills': torch.cat([skills_original, skills_relabeled], 0)
}
```

### 2. Threshold Selection

**Rule of thumb**: ε should be ~5-10% of skill space diameter.

```python
# For hypercube [low, high]^d
diameter = np.linalg.norm(high - low)
threshold = 0.05 * diameter

# For unit circle (angle features)
threshold = 0.1  # ~6 degrees
```

### 3. Normalization

**Features**: Normalize to similar scales for successor features learning.

```python
# Compute statistics during initial exploration
feature_mean = running_mean(features)
feature_std = running_std(features)

# Normalize features
features_normalized = (features - feature_mean) / (feature_std + 1e-8)
```

### 4. Debugging

**Check constraint satisfaction**:
```python
def check_constraint_satisfaction(policy, env, skill_grid):
    violations = 0
    for z in skill_grid:
        d = expected_distance_to_skill(policy, env, z)
        if d > threshold:
            violations += 1

    print(f"Constraint violations: {violations}/{len(skill_grid)}")
    print(f"Violation rate: {violations/len(skill_grid)*100:.1f}%")
```

**Visualize λ distribution**:
```python
import matplotlib.pyplot as plt

lambdas = []
for z in skill_grid:
    state = env.reset()
    lambda_val = lagrange(state, z).item()
    lambdas.append(lambda_val)

plt.hist(lambdas, bins=50)
plt.xlabel('λ(s,z)')
plt.ylabel('Frequency')
plt.title('Lagrange Multiplier Distribution')
plt.show()
```

### 5. Common Pitfalls

❌ **Forgetting to detach λ** when computing actor loss → backprop through λ
✅ Use `lambda_val = lagrange(states, skills).detach()`

❌ **Wrong scale for features** → successor features can't learn effectively
✅ Normalize features to [0,1] or standardize

❌ **Threshold too small** → all skills violated, λ→1, no quality improvement
✅ Start with larger ε and anneal if needed

❌ **No skill re-labeling** → poor sample efficiency
✅ Always augment batch with relabeled skills

---

## Comparison with Baselines

### vs. DCG-ME

**DCG-ME**: Population-based, descriptor-conditioned critic

**Advantages of QDAC**:
- Single policy (no population management)
- Better performance on conflicting skills (negative velocity)
- Adaptive trade-off (vs. fixed archive selection pressure)

**Implementation difference**:
```python
# DCG-ME: Q(s,a|d) with similarity scaling
Q_dcg = S(d, d_achieved) * Q(s,a)
where S(d,d') = exp(-||d-d'|| / L)

# QDAC: Lagrangian with V(s,z) and ψ(s,z)
L_qdac = (1-λ)·V(s,z) - λ·||(1-γ)ψ(s,z) - z||
```

### vs. SMERL

**SMERL**: Maximize MI + near-optimality threshold

**Advantages of QDAC**:
- Explicit skill conditioning (SMERL discovers, doesn't execute targets)
- Constrained optimization (SMERL uses reward threshold)
- No discriminator needed

**Implementation difference**:
```python
# SMERL reward
r_smerl = r_env + α·𝟙[R ≥ R*-ε]·log q(z|s)
where q(z|s) is learned discriminator (DIAYN)

# QDAC objective
L_qdac = (1-λ)·V(s,z) - λ·||(1-γ)ψ(s,z) - z||
where λ adapts to constraint satisfaction
```

### vs. UVFA

**UVFA**: Skill-conditioned value function only

**Advantages of QDAC**:
- Successor features for trajectory-level skills
- Lagrangian for quality-diversity trade-off
- Can execute non-Markovian skills (feet contact proportions)

**Implementation difference**:
```python
# UVFA: maximize V(s,z) with naive distance penalty
J_uvfa = V(s,z) - β·Σ γ^t ||φ_t - z||²

# QDAC: constrained optimization with successor features
J_qdac = (1-λ)·V(s,z) - λ·||(1-γ)ψ(s,z) - z||
```

---

## Extensions & Variants

### 1. Continuous Skill Space

Already supported! Skill sampling:
```python
# Hypercube
skill = np.random.uniform(low, high, size=skill_dim)

# Simplex (e.g., for feet contact with Σz_i = 1 constraint)
skill = np.random.dirichlet(alpha=np.ones(skill_dim))

# Sphere surface
skill = np.random.randn(skill_dim)
skill /= np.linalg.norm(skill)
```

### 2. Hierarchical QDAC

**Use learned skills for hierarchical RL**:
```python
class MetaController:
    def __init__(self, qdac_policy):
        self.skill_policy = qdac_policy  # fixed
        self.meta_policy = SAC()  # learns to select skills

    def step(self, state):
        # Meta-controller selects skill
        skill = self.meta_policy.select_skill(state)

        # Low-level policy executes skill
        action = self.skill_policy.select_action(state, skill)

        return action, skill
```

### 3. Transfer Learning

**Zero-shot transfer via successor features**:
```python
# Task 1: learn ψ_1(s,z)
qdac_1 = train_qdac(env_1)

# Task 2: initialize with ψ_1
qdac_2 = QDAC()
qdac_2.sf_critic.load_state_dict(qdac_1.sf_critic.state_dict())

# Fine-tune on new task
train_qdac(env_2, qdac=qdac_2)
```

### 4. Curriculum Learning

**Gradually increase skill difficulty**:
```python
def curriculum_skill_sampling(iteration, skill_space):
    # Start with easy skills (center of space)
    difficulty = min(1.0, iteration / 10000)

    center = (skill_space.high + skill_space.low) / 2
    radius = difficulty * np.linalg.norm(skill_space.high - center)

    # Sample from expanding sphere around center
    direction = np.random.randn(skill_space.dim)
    direction /= np.linalg.norm(direction)

    skill = center + radius * np.random.uniform(0, 1) * direction
    return np.clip(skill, skill_space.low, skill_space.high)
```

---

## Ablation Studies

### QDAC-SepSkill (No Successor Features)

Replace successor features term with naive distance:

```python
# Original QDAC
L = (1-λ)·V(s,z) - λ·||(1-γ)ψ(s,z) - z||

# QDAC-SepSkill
L = (1-λ)·V(s,z) - λ·Σ γ^t ||φ_t - z||
```

**Result**: Fails on feet contact (can only achieve corners where φ_t = z always).

### QDAC-FixedLambda (No Adaptive Trade-off)

Use fixed λ instead of learned:

```python
# Original QDAC
lambda_val = lagrange_network(state, skill)

# QDAC-FixedLambda
lambda_val = 0.5  # or tuned hyperparameter
```

**Result**: Worse coverage (can't reach skill space edges), fails on hard tasks (Jump).

### UVFA (Both Ablations)

Naive distance + fixed trade-off:

```python
# UVFA
L = V(s,z) - β·||φ_t - z||
```

**Result**: Worst performance, can't execute trajectory-level skills.

---

## Key Takeaways for Implementation

1. **Use successor features** for skills defined by average features over trajectories
2. **Adaptive λ is crucial** for balancing quality-diversity automatically
3. **Skill re-labeling** significantly improves sample efficiency
4. **Normalize features** to similar scales for stable learning
5. **Threshold ε** should be ~5-10% of skill space diameter
6. **Target networks** (τ=0.005) stabilize critic learning
7. **Model-based variant** helps with complex feature functions
8. **Batch augmentation** doubles effective batch size at minimal cost

**When to use QDAC**:
- Need diverse high-performing behaviors
- Want single policy (not population)
- Skills defined by trajectory statistics (not instantaneous state)
- Have hand-designed feature function φ(s,a)
- Need adaptation/transfer/hierarchical RL capabilities
