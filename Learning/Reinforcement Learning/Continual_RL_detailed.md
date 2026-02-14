# Continual Reinforcement Learning: Detailed Implementation Guide

**Quick reference**: [[Continual_RL]]

---

## 1. Formal Definitions and Problem Settings

### 1.1 What Is Continual RL?

**Definition** (Abel et al., 2024): Learning in a sequence of MDPs $\mathcal{M}_1, \mathcal{M}_2, \ldots$ where:
- The agent has a single policy (or policy-generating mechanism) that persists across tasks
- The agent may or may not know when tasks switch
- The agent cannot (or should not need to) retrain from scratch on each task

### 1.2 Scenario Settings

| Scenario | State Space | Action Space | Reward Function | Task Identity |
|----------|-------------|--------------|-----------------|---------------|
| **Task-incremental** | May change | May change | Changes | Known at test time |
| **Domain-incremental** | Changes (e.g., visual) | Same | Same | Unknown at test time |
| **Reward-incremental** | Same | Same | Changes | Unknown at test time |

**Boundary-free continual RL**: Agent must detect distributional shifts without explicit task signals (harder, more realistic).

### 1.3 Formal Setup

Task sequence $\{T_k\}_{k=1}^{N}$ where each $T_k = (\mathcal{S}_k, \mathcal{A}_k, P_k, R_k, \gamma_k)$ is an MDP.

**Objective**: Minimize total loss across all tasks:
$$\mathcal{L}_t(\theta) \triangleq \sum_{i=1}^t \mathbb{E}_{(s,a) \sim T_i} \left[ \ell \left( \pi_\theta(a|s), a^*_i \right) \right]$$

**Constraints**:
- Limited access to data from $T_1, \ldots, T_{k-1}$ when training on $T_k$
- Bounded compute and memory budget

---

## 2. The Plasticity Loss Problem

**Critical finding**: Distinct from catastrophic forgetting. Networks progressively lose ability to learn *anything new*.

### 2.1 The Phenomenon

**Plasticity loss** = gradual degradation of learning capacity over sequential tasks, even with unlimited data.

**Measurement**: Learning speed on new task degrades over time.
$$\text{Plasticity}(T_k) = \frac{\Delta \text{Performance}}{\Delta \text{Steps}} \text{ on task } T_k$$

Typically: $\text{Plasticity}(T_{10}) \ll \text{Plasticity}(T_1)$

### 2.2 Causes

1. **Dead/dormant neurons**: ReLU units saturate at zero
   $$\text{Dead if } \forall (s,a): \ h_i(s,a) = \max(0, w_i^\top x) = 0$$
   
2. **Feature rank collapse**: Effective dimensionality shrinks
   $$\text{rank}(H_{T_k}) < \text{rank}(H_{T_1})$$
   where $H$ is the feature matrix
   
3. **Weight magnitude growth**: Weights drift to large values
   $$\|\theta_{T_k}\| \gg \|\theta_{T_1}\|$$

### 2.3 Solutions

| Method | Mechanism | Implementation |
|--------|-----------|----------------|
| **Continual Backpropagation** | Reinitialize dormant neurons | Detect utility < threshold, reset |
| **Neuron Recycling** | Reset dead units periodically | $\text{if } \|\nabla_\theta \mathcal{L}\|_i < \epsilon: \ \theta_i \sim \mathcal{N}(0, \sigma^2)$ |
| **Layer Normalization** | Normalize activations per layer | $h' = \frac{h - \mu}{\sigma}$ |
| **Shrink & Perturb** | Interpolate toward initialization | $\theta \leftarrow \alpha \theta + (1-\alpha)\theta_{init}$ |
| **CReLU** | Preserve gradient flow | $\text{CReLU}(x) = [\max(0,x), \max(0,-x)]$ |

**Key insight**: Solving forgetting without addressing plasticity loss is incomplete.

---

## 3. Reset and Reinitialization Literature

### 3.1 Primacy Bias

**Nikishin et al. (2022)**: Networks trained on early data become biased toward early features.

**Solution**: Periodic reset of network (while keeping replay buffer):
```python
if iteration % reset_interval == 0:
    θ = initialize_network()  # Reset weights
    # Keep replay buffer intact
```

### 3.2 Shrink and Perturb (S&P)

**Ash & Adams (2020)**: Interpolate weights toward initialization.

```python
def shrink_and_perturb(θ, θ_init, α=0.9):
    """
    Shrink weights toward initialization.
    
    Args:
        θ: Current parameters
        θ_init: Initial parameters
        α: Interpolation factor (0.9 → 90% current, 10% init)
    
    Returns:
        Updated parameters
    """
    return α * θ + (1 - α) * θ_init
```

**When to apply**: Every $K$ iterations or when plasticity drops below threshold.

### 3.3 UPGD (Utility-based Perturbed Gradient Descent)

**Elsayed & Mahmood (2024)**: Perturb low-utility parameters.

```python
def upgd_step(θ, utility, η, σ):
    """
    Perturb parameters with low utility.
    
    Args:
        θ: Parameters
        utility: Per-parameter utility metric (e.g., gradient magnitude)
        η: Learning rate
        σ: Perturbation noise std
    
    Returns:
        Updated parameters
    """
    # Identify low-utility parameters (bottom 10%)
    threshold = np.percentile(utility, 10)
    low_utility_mask = utility < threshold
    
    # Standard gradient update
    θ_new = θ - η * ∇L(θ)
    
    # Add perturbation to low-utility params
    noise = np.random.normal(0, σ, size=θ.shape)
    θ_new[low_utility_mask] += noise[low_utility_mask]
    
    return θ_new
```

---

## 4. Detailed Taxonomy with Algorithms

### 4.1 Policy-Focused Methods

#### A. Policy Reuse

**MAXQINIT**: Initialize Q-values with max across previous tasks.

```python
def maxqinit(Q_history, s, a):
    """
    Initialize Q-value for new task.
    
    Args:
        Q_history: List of Q-functions from previous tasks
        s: State
        a: Action
    
    Returns:
        Initialized Q-value
    """
    if not Q_history:
        return 0.0
    return max(Q_k(s, a) for Q_k in Q_history)
```

**CSP (Continual Subspace of Policies)**: Maintain convex hull of policy parameters.

```python
def csp_policy(θ_history, s):
    """
    Policy from convex combination of past policies.
    
    Args:
        θ_history: List of policy parameters from previous tasks
        s: State
    
    Returns:
        Action from convex combination
    """
    # Learn mixing weights α
    α = softmax(mixing_network(s))  # Sum to 1
    
    # Compute weighted average policy
    θ_combined = sum(α[i] * θ_history[i] for i in range(len(θ_history)))
    
    return π(a|s; θ_combined)
```

#### B. Policy Decomposition

**Factor Decomposition**: $\theta_k = L \cdot s_k$

```python
class FactorDecomposedPolicy:
    """Policy decomposition with shared latent basis."""
    
    def __init__(self, d, m):
        """
        Args:
            d: Policy parameter dimension
            m: Latent basis dimension (m << d)
        """
        self.L = np.random.randn(d, m)  # Shared basis
        self.task_coefficients = {}  # s_k per task
    
    def add_task(self, task_id):
        """Initialize task-specific coefficients."""
        self.task_coefficients[task_id] = np.random.randn(self.L.shape[1])
    
    def get_policy(self, task_id):
        """Reconstruct policy for task."""
        s_k = self.task_coefficients[task_id]
        θ_k = self.L @ s_k
        return θ_k
    
    def update_basis(self, gradient):
        """Update shared basis L."""
        self.L -= learning_rate * gradient
```

**PNN (Progressive Neural Networks)**:

```python
class ProgressiveNN:
    """Progressive Neural Networks with lateral connections."""
    
    def __init__(self):
        self.columns = []  # List of frozen columns
        self.adapters = []  # Lateral connection weights
    
    def add_task(self):
        """Add new column for new task."""
        new_column = initialize_network()
        
        # Create lateral adapters from all previous columns
        new_adapters = [
            initialize_adapter(prev_col, new_column)
            for prev_col in self.columns
        ]
        
        self.columns.append(new_column)
        self.adapters.append(new_adapters)
    
    def forward(self, x, task_id):
        """Forward pass with lateral connections."""
        h = x
        for layer in range(num_layers):
            # Current column's layer
            h_current = self.columns[task_id].layer[layer](h)
            
            # Lateral connections from previous columns
            h_lateral = sum(
                adapter[layer](self.columns[j].layer[layer](x))
                for j, adapter in enumerate(self.adapters[task_id])
            )
            
            # Combine
            h = activation(h_current + h_lateral)
        
        return h
```

#### C. Policy Merging

**EWC (Elastic Weight Consolidation)**:

```python
def ewc_loss(θ, θ_old, F, λ):
    """
    EWC regularized loss.
    
    Args:
        θ: Current parameters
        θ_old: Parameters from previous task
        F: Fisher Information Matrix (diagonal)
        λ: Regularization strength
    
    Returns:
        Regularization penalty
    """
    penalty = 0.5 * λ * sum(
        F[i] * (θ[i] - θ_old[i])**2
        for i in range(len(θ))
    )
    return penalty

def compute_fisher(θ, data):
    """
    Compute diagonal Fisher Information Matrix.
    
    Args:
        θ: Policy parameters
        data: Batch of (state, action) pairs
    
    Returns:
        Diagonal Fisher Information
    """
    F = np.zeros_like(θ)
    
    for s, a in data:
        # Compute log probability gradient
        log_prob = log π(a|s; θ)
        grad = compute_gradient(log_prob, θ)
        
        # Fisher is expectation of squared gradient
        F += grad ** 2
    
    F /= len(data)
    return F
```

**Full EWC Algorithm**:

```python
def train_with_ewc(tasks, λ=1000):
    """
    Train with EWC regularization.
    
    Args:
        tasks: List of tasks
        λ: EWC regularization strength
    
    Returns:
        Final policy parameters
    """
    θ = initialize_policy()
    θ_old_list = []
    F_list = []
    
    for task in tasks:
        # Train on current task
        for epoch in range(num_epochs):
            batch = sample_batch(task)
            
            # Task loss
            loss_task = compute_loss(θ, batch)
            
            # EWC penalty (sum over all previous tasks)
            loss_ewc = sum(
                ewc_loss(θ, θ_old, F, λ)
                for θ_old, F in zip(θ_old_list, F_list)
            )
            
            # Total loss
            loss = loss_task + loss_ewc
            
            # Update
            θ -= learning_rate * gradient(loss, θ)
        
        # Store parameters and Fisher after task
        θ_old_list.append(θ.copy())
        F_list.append(compute_fisher(θ, task.data))
    
    return θ
```

---

### 4.2 Experience-Focused Methods

#### A. Direct Replay

**CLEAR (Continual Learning with Experience And Replay)**:

```python
class CLEAR:
    """
    Complementary learning system with behavior cloning.
    """
    
    def __init__(self, buffer_size_short, buffer_size_long):
        self.short_term = ReplayBuffer(buffer_size_short)  # Hippocampus
        self.long_term = ReplayBuffer(buffer_size_long)    # Neocortex
        self.π = initialize_policy()
    
    def train_step(self, current_batch):
        """Single training step."""
        # Sample from both buffers
        batch_short = self.short_term.sample(batch_size // 2)
        batch_long = self.long_term.sample(batch_size // 2)
        
        # Policy gradient on current data
        loss_pg = compute_policy_gradient_loss(self.π, current_batch)
        
        # Behavior cloning on replayed data
        replayed = batch_short + batch_long
        loss_bc = compute_behavior_cloning_loss(self.π, replayed)
        
        # Combined loss
        loss = loss_pg + λ_bc * loss_bc
        
        # Update policy
        self.π.update(loss)
    
    def add_to_buffers(self, transitions):
        """Add new experiences to buffers."""
        # Add to short-term (recent task)
        self.short_term.add(transitions)
        
        # Selectively add to long-term (reservoir sampling)
        for trans in transitions:
            if should_add_to_long_term(trans):
                self.long_term.add(trans)
    
    def task_switch(self):
        """Called when task changes."""
        # Transfer important experiences from short to long term
        important = self.short_term.get_high_reward_samples()
        self.long_term.add(important)
        
        # Clear short-term buffer
        self.short_term.clear()
```

**Behavior Cloning Loss**:

```python
def compute_behavior_cloning_loss(π, batch):
    """
    Supervised loss on past behavior.
    
    Args:
        π: Current policy
        batch: Replay buffer samples (s, a, r, s')
    
    Returns:
        BC loss
    """
    loss = 0
    for s, a, r, s_next in batch:
        # Negative log likelihood of past action
        log_prob = log π(a|s)
        loss -= log_prob
    
    return loss / len(batch)
```

#### B. Generative Replay

**RePR (Reinforcement Pseudo-Rehearsal)**:

```python
class RePR:
    """GAN-based generative replay."""
    
    def __init__(self):
        self.generator = GAN()  # Generates (s, a, r, s') tuples
        self.π = initialize_policy()
        self.replay_buffer = []  # Store small set of real data
    
    def train_step(self, real_batch):
        """Training step with generative replay."""
        # Generate synthetic past experiences
        synthetic_batch = self.generator.sample(batch_size // 2)
        
        # Mix real and synthetic
        mixed_batch = real_batch + synthetic_batch
        
        # Train policy on mixed batch
        loss = compute_loss(self.π, mixed_batch)
        self.π.update(loss)
        
        # Update generator to match real data distribution
        self.generator.train(self.replay_buffer)
    
    def task_switch(self, task_data):
        """Update generator when task changes."""
        # Add representative samples to replay buffer
        self.replay_buffer.extend(sample_diverse(task_data))
        
        # Train generator on all past task data
        self.generator.train(self.replay_buffer)
```

---

### 4.3 Dynamic-Focused Methods

#### A. Direct Modeling

**HyperCRL (Hypernetwork Continual RL)**:

```python
class HyperCRL:
    """
    Hypernetwork generates task-conditional dynamics model.
    """
    
    def __init__(self, z_dim, model_dim):
        self.hypernetwork = Hypernetwork(z_dim, model_dim)
        self.task_embeddings = {}  # z_k per task
    
    def get_dynamics_model(self, task_id):
        """Get dynamics model for task."""
        z = self.task_embeddings[task_id]
        θ_dynamics = self.hypernetwork(z)
        return DynamicsModel(θ_dynamics)
    
    def train_on_task(self, task_id, data):
        """Learn dynamics for new task."""
        # Initialize task embedding
        z = np.random.randn(self.z_dim)
        self.task_embeddings[task_id] = z
        
        # Train to predict transitions
        for epoch in range(num_epochs):
            for s, a, s_next in data:
                # Get task-conditional model
                model = self.get_dynamics_model(task_id)
                
                # Predict next state
                s_pred = model.predict(s, a)
                
                # Prediction loss
                loss = ||s_pred - s_next||²
                
                # Update hypernetwork (generates all models)
                self.hypernetwork.update(loss)
```

**LLIRL (Lifelong Incremental RL with CRP)**:

```python
class LLIRL:
    """
    Chinese Restaurant Process for dynamic model instantiation.
    """
    
    def __init__(self, α):
        self.models = []  # List of dynamics models
        self.α = α  # CRP concentration parameter
    
    def select_or_create_model(self, task_data):
        """
        Assign task to existing model or create new one.
        
        CRP probability:
        p(assign to model k) = n_k / (n + α)
        p(create new model) = α / (n + α)
        """
        n = len(self.models)
        
        if not self.models:
            # First task: create model
            return self.create_model(task_data)
        
        # Evaluate likelihood under each existing model
        likelihoods = [
            model.likelihood(task_data) * len(model.tasks)
            for model in self.models
        ]
        likelihoods.append(self.α)  # New model probability
        
        # Sample assignment
        probs = np.array(likelihoods) / sum(likelihoods)
        choice = np.random.choice(len(probs), p=probs)
        
        if choice < len(self.models):
            # Assign to existing model
            self.models[choice].add_task(task_data)
            return self.models[choice]
        else:
            # Create new model
            return self.create_model(task_data)
    
    def create_model(self, task_data):
        """Instantiate new dynamics model."""
        model = DynamicsModel()
        model.train(task_data)
        self.models.append(model)
        return model
```

#### B. Indirect/Latent Modeling

**LILAC (Lifelong Latent Actor-Critic)**:

```python
class LILAC:
    """
    Latent variable model for task-dependent dynamics.
    """
    
    def __init__(self, z_dim):
        self.encoder = StateEncoder()  # s → z
        self.latent_dynamics = LatentDynamics()  # (z, a) → z'
        self.decoder = StateDecoder()  # z → s
        self.z_dim = z_dim
    
    def predict_next_state(self, s, a):
        """Predict s' in latent space."""
        # Encode current state
        z = self.encoder(s)
        
        # Predict next latent state
        z_next = self.latent_dynamics(z, a)
        
        # Decode to observation space
        s_next = self.decoder(z_next)
        
        return s_next
    
    def train(self, transitions):
        """Train latent dynamics model."""
        for s, a, s_next in transitions:
            # Encode
            z = self.encoder(s)
            z_next_true = self.encoder(s_next)
            
            # Predict in latent space
            z_next_pred = self.latent_dynamics(z, a)
            
            # Latent dynamics loss
            loss_dynamics = ||z_next_pred - z_next_true||²
            
            # Reconstruction loss
            s_recon = self.decoder(z)
            loss_recon = ||s_recon - s||²
            
            # Total loss
            loss = loss_dynamics + λ * loss_recon
            
            # Update all components
            update(self.encoder, self.latent_dynamics, self.decoder, loss)
```

---

### 4.4 Reward-Focused Methods

#### A. Reward Shaping

**Potential-Based Shaping**:

```python
def potential_based_shaping(s, s_next, Φ, γ):
    """
    Potential-based reward shaping.
    
    Args:
        s: Current state
        s_next: Next state
        Φ: Potential function
        γ: Discount factor
    
    Returns:
        Shaping bonus (preserves optimal policy)
    """
    return γ * Φ(s_next) - Φ(s)
```

**SR-LLRL (Shaping Rewards for Lifelong RL)**:

```python
class SR_LLRL:
    """Reward shaping based on cumulative visit counts."""
    
    def __init__(self):
        self.visit_counts = {}  # Cumulative across all tasks
    
    def shaped_reward(self, s, r):
        """Add exploration bonus."""
        # Base reward
        r_shaped = r
        
        # Intrinsic reward (inverse sqrt of visits)
        count = self.visit_counts.get(s, 0)
        r_intrinsic = 1.0 / np.sqrt(count + 1)
        
        # Combined
        r_shaped += α * r_intrinsic
        
        # Update count
        self.visit_counts[s] = count + 1
        
        return r_shaped
```

#### B. Intrinsic Motivation

**IML (Intrinsically Motivated Lifelong Exploration)**:

```python
class IML:
    """Multi-timescale intrinsic motivation."""
    
    def __init__(self):
        self.short_term_novelty = {}  # Current task
        self.long_term_novelty = {}   # Across all tasks
    
    def intrinsic_reward(self, s, task_id):
        """Compute intrinsic reward."""
        # Short-term: Novel for current task
        count_short = self.short_term_novelty.get((task_id, s), 0)
        r_short = 1.0 / np.sqrt(count_short + 1)
        
        # Long-term: Novel across all tasks
        count_long = self.long_term_novelty.get(s, 0)
        r_long = 1.0 / np.sqrt(count_long + 1)
        
        # Weighted combination
        r_intrinsic = β₁ * r_short + β₂ * r_long
        
        # Update counts
        self.short_term_novelty[(task_id, s)] = count_short + 1
        self.long_term_novelty[s] = count_long + 1
        
        return r_intrinsic
    
    def task_switch(self, task_id):
        """Reset short-term novelty for new task."""
        keys_to_remove = [k for k in self.short_term_novelty if k[0] == task_id]
        for k in keys_to_remove:
            del self.short_term_novelty[k]
```

---

## 5. Successor Features and GPI

### 5.1 Successor Features

**Key idea**: Decompose value function as:
$$Q^\pi(s,a) = \psi^\pi(s,a)^\top \mathbf{w}$$

where:
- $\psi^\pi(s,a) = \mathbb{E}^\pi \left[ \sum_{t=0}^\infty \gamma^t \phi(s_t, a_t) \mid s_0=s, a_0=a \right]$ (successor features)
- $r(s,a,s') = \phi(s,a,s')^\top \mathbf{w}$ (reward is linear in features)

**For CRL**: If dynamics are shared but rewards differ, learn $\psi$ once, only update $\mathbf{w}$ per task.

```python
class SuccessorFeatures:
    """Successor features for zero-shot transfer."""
    
    def __init__(self, φ_dim):
        self.ψ = {}  # SF per policy: π → ψ^π(s,a)
        self.w = {}  # Reward weights per task
        self.φ_dim = φ_dim
    
    def learn_task(self, task_id, π, data):
        """Learn SF and reward weights for task."""
        # Learn successor features for policy π
        ψ_π = self.learn_successor_features(π, data)
        self.ψ[task_id] = ψ_π
        
        # Learn reward weights (linear regression)
        φ_samples = [self.φ(s, a) for s, a, r in data]
        r_samples = [r for s, a, r in data]
        w = linear_regression(φ_samples, r_samples)
        self.w[task_id] = w
    
    def learn_successor_features(self, π, data):
        """Learn ψ^π via TD learning."""
        ψ = initialize_sf_network()
        
        for epoch in range(num_epochs):
            for s, a, s_next in data:
                # Current SF
                ψ_current = ψ(s, a)
                
                # Target SF (Bellman equation for SF)
                φ_current = self.φ(s, a)
                a_next = π(s_next)
                ψ_next = ψ(s_next, a_next)
                ψ_target = φ_current + γ * ψ_next
                
                # TD loss
                loss = ||ψ_current - ψ_target||²
                ψ.update(loss)
        
        return ψ
    
    def zero_shot_transfer(self, new_task_w):
        """
        Transfer to new task with reward weights w.
        No retraining needed!
        """
        # Use SF from previous tasks with new reward weights
        def Q_new(s, a, policy_id):
            ψ = self.ψ[policy_id](s, a)
            return ψ @ new_task_w
        
        return Q_new
```

### 5.2 Generalized Policy Improvement (GPI)

```python
def gpi_policy(ψ_policies, w_new, s):
    """
    Generalized Policy Improvement.
    
    Args:
        ψ_policies: List of SF for previous policies
        w_new: Reward weights for new task
        s: Current state
    
    Returns:
        Best action according to GPI
    """
    best_action = None
    best_value = -np.inf
    
    # For each action
    for a in action_space:
        # Evaluate under all previous policies
        values = [
            ψ_π(s, a) @ w_new
            for ψ_π in ψ_policies
        ]
        
        # Take max over policies
        value = max(values)
        
        if value > best_value:
            best_value = value
            best_action = a
    
    return best_action
```

**Result**: Zero-shot transfer to new reward functions without any training!

---

## 6. Meta-RL as Continual Learning Alternative

### 6.1 MAML (Model-Agnostic Meta-Learning)

```python
def maml(tasks, α_inner, α_outer, K):
    """
    MAML for fast adaptation.
    
    Args:
        tasks: Distribution of tasks
        α_inner: Inner loop learning rate
        α_outer: Outer loop (meta) learning rate
        K: Number of inner gradient steps
    
    Returns:
        Meta-initialized parameters θ*
    """
    θ = initialize_parameters()
    
    for iteration in range(num_iterations):
        # Sample batch of tasks
        task_batch = sample_tasks(tasks, batch_size)
        
        meta_gradient = 0
        
        for task in task_batch:
            # Inner loop: Adapt to task
            θ_task = θ.copy()
            for k in range(K):
                batch = sample_data(task)
                loss = compute_loss(θ_task, batch)
                θ_task -= α_inner * gradient(loss, θ_task)
            
            # Outer loop: Meta-gradient
            batch_test = sample_data(task)
            loss_test = compute_loss(θ_task, batch_test)
            meta_gradient += gradient(loss_test, θ)  # Gradient wrt original θ!
        
        # Meta-update
        θ -= α_outer * meta_gradient / len(task_batch)
    
    return θ
```

**For new task**: Start from $\theta^*$, take $K$ gradient steps → quick adaptation.

### 6.2 RL² (RL-Squared)

```python
class RL2:
    """RNN-based in-context learning."""
    
    def __init__(self, hidden_size):
        self.rnn = RNN(hidden_size)
        self.policy_head = PolicyNetwork()
    
    def train_meta(self, task_distribution):
        """Train RNN to encode learning algorithm."""
        for episode in range(num_episodes):
            # Sample task
            task = sample_task(task_distribution)
            
            # Reset RNN hidden state
            h = self.rnn.initial_state()
            
            # Collect full episode with RNN state
            trajectory = []
            s = task.reset()
            done = False
            
            while not done:
                # RNN processes (s, a_prev, r_prev, h_prev)
                a, h = self.step(s, a_prev, r_prev, h)
                
                # Environment step
                s_next, r, done = task.step(a)
                
                trajectory.append((s, a, r, h))
                s, a_prev, r_prev = s_next, a, r
            
            # Update RNN + policy to maximize return
            loss = -sum(r for s, a, r, h in trajectory)
            self.rnn.update(loss)
            self.policy_head.update(loss)
    
    def step(self, s, a_prev, r_prev, h):
        """Single step with RNN state."""
        # Concatenate inputs
        x = concat(s, a_prev, r_prev)
        
        # RNN forward
        h_next = self.rnn(x, h)
        
        # Policy
        a = self.policy_head(h_next)
        
        return a, h_next
    
    def adapt_to_new_task(self, task):
        """
        Adapt to new task through RNN hidden state.
        No gradient updates needed!
        """
        h = self.rnn.initial_state()
        s = task.reset()
        done = False
        
        while not done:
            a, h = self.step(s, a_prev, r_prev, h)
            s_next, r, done = task.step(a)
            s, a_prev, r_prev = s_next, a, r
```

**Key idea**: Learning algorithm is in RNN weights. Adaptation happens via hidden state evolution.

---

## 7. Evaluation: Full Formulations

### 7.1 Performance Matrix

**Definition**: $P_{ij}$ = performance on task $i$ after training on task $j$.

Full matrix for $N$ tasks:
$$P = \begin{bmatrix}
p_{1,1} & p_{1,2} & \cdots & p_{1,N} \\
- & p_{2,2} & \cdots & p_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
- & - & \cdots & p_{N,N}
\end{bmatrix}$$

**Diagonal** ($p_{i,i}$): Performance immediately after training on task $i$

**Last column** ($p_{i,N}$): Final performance on task $i$ after all training

### 7.2 Complete Metrics

**Average Performance**:
$$A_N = \frac{1}{N} \sum_{i=1}^{N} p_{N,i}$$

**Forgetting** (per task):
$$CF_i = \max(p_{i,i} - p_{N,i}, 0)$$

**Average Forgetting**:
$$FG = \frac{1}{N-1} \sum_{i=1}^{N-1} CF_i$$

**Forward Transfer** (task $i$ benefits from task $j$, where $j < i$):
$$FT_{i,j} = \frac{p_{i,j} - p_{i,0}}{|p_{i,\text{max}}|}$$

**Backward Transfer** (task $i$ improves after task $j$, where $j > i$):
$$BT_{i,j} = p_{i,j} - p_{i,i}$$

**Average Backward Transfer**:
$$BT = \frac{1}{N-1} \sum_{i=1}^{N-1} (p_{N,i} - p_{i,i})$$

Positive BT = new knowledge improved old tasks (rare, desirable)

### 7.3 Efficiency Metrics

**Model Size Growth**:
$$\text{Size}(N) = \text{number of parameters after } N \text{ tasks}$$

Ideally: $\text{Size}(N) = O(1)$ (constant) or $O(\log N)$ (sub-linear)

**Sample Complexity** (per task):
$$S_i = \text{environment interactions to reach threshold on task } i$$

**Memory Footprint**:
$$M(N) = \text{replay buffer size} + \text{stored models} + \text{other storage}$$

---

## 8. Curriculum and Task Ordering

### 8.1 Sensitivity Analysis

**Wolczyk et al. (2022)** decomposition:

Total performance change = Forward Transfer + Backward Transfer - Forgetting

All three components **depend on task ordering**.

### 8.2 Ordering Strategies

1. **Random**: Shuffle tasks randomly (baseline)
2. **Difficulty**: Easy → hard progression
3. **Similarity**: Group similar tasks together
4. **Adversarial**: Hard → easy (stress test)
5. **Adaptive**: Agent chooses curriculum

**Recommendation**: Evaluate on at least random + adversarial orderings.

---

## 9. Safety and Constrained CRL

### 9.1 Constrained MDPs

**Formulation**: 
$$\max_\pi \mathbb{E}_\pi \left[ \sum_t \gamma^t r_t \right]$$
$$\text{s.t. } \mathbb{E}_\pi \left[ \sum_t \gamma^t c_t \right] \leq d$$

where $c_t$ is cost (e.g., distance to humans).

### 9.2 Safe Transfer

**Problem**: Conservative policy from warehouse may be too conservative for hospital. Exploratory policy from hospital may be unsafe in warehouse.

**Solution**: Uncertainty-aware transfer
```python
def safe_transfer(π_old, task_new, uncertainty_threshold):
    """Transfer with safety constraints."""
    π_new = π_old.copy()  # Start conservative
    
    for s in task_new.states:
        # Estimate uncertainty in new task
        uncertainty = estimate_uncertainty(s, task_new)
        
        if uncertainty < uncertainty_threshold:
            # Low uncertainty → safe to explore
            π_new.update(s)
        else:
            # High uncertainty → use conservative old policy
            π_new[s] = π_old[s]
    
    return π_new
```

---

## 10. Multi-Agent Continual RL

### 10.1 Endogenous Non-Stationarity

**Key difference**: Environment changes because other agents change strategies (not external task switches).

**Catastrophic forgetting in multi-agent**: Losing ability to counter old strategies → spinning top problem (cycling through counter-strategies).

### 10.2 PSRO as Implicit CRL

**PSRO** (Policy-Space Response Oracles):
```python
def psro(game, num_iterations):
    """PSRO iteratively adds best responses."""
    population = [random_policy()]
    meta_strategies = [uniform_distribution()]
    
    for t in range(num_iterations):
        # Compute meta-strategy (Nash of meta-game)
        meta_strategy = solve_meta_game(population)
        
        # Train best response to meta-strategy
        π_br = train_best_response(game, population, meta_strategy)
        
        # Add to population (never remove!)
        population.append(π_br)
        
        # This is continual learning: maintain competence across
        # expanding population while adding new strategies
    
    return population, meta_strategy
```

**CRL view**: Each iteration is a "task" (counter a new meta-strategy). Must maintain competence against all previous meta-strategies.

---

## 11. Theory (What We Know and Don't Know)

### 11.1 Known Results

**Pentina & Lampert (2014)** — PAC-Bayes bounds for lifelong learning:

Sample complexity for task $t$ given $t-1$ previous tasks:
$$\tilde{O}\left( \frac{d}{m} \log t \right)$$
where $d$ is VC dimension, $m$ is samples per task.

**Assumption**: Tasks are related (shared structure).

**SF/GPI** — Zero-shot transfer guarantees when:
- Reward is linear in features: $r = \phi^\top w$
- Feature extractor $\phi$ is fixed

Then GPI achieves near-optimal performance on new task without any training.

### 11.2 Open Questions

1. **Capacity limits**: How many tasks can network of size $n$ support?
   - Conjecture: $O(n / \log n)$ tasks before interference dominates

2. **Fundamental trade-off**: Is there an impossibility result for simultaneously achieving:
   - Forgetting $< \epsilon$
   - Transfer $> \delta$
   - Model size $< B$?

3. **Sample complexity**: What is the sample complexity of CRL in general?

4. **Plasticity loss**: Why and when does it occur? Theoretical characterization lacking.

---

## 12. Implementation Checklist

### Research Paper Checklist

- [ ] Report performance matrix $P_{ij}$
- [ ] Compute and report: Average performance, Forgetting, FT, BT
- [ ] Report model size and memory footprint over tasks
- [ ] Include baselines: Fine-tuning, MTL, Periodic Reset + Replay
- [ ] Test on ≥2 benchmarks from different domains
- [ ] Evaluate on ≥2 task orderings (random + adversarial)
- [ ] Use ≥5 random seeds, report mean ± std
- [ ] Include learning curves (not just final numbers)
- [ ] Ablation studies for each component
- [ ] Release code for reproducibility

### Production System Checklist

- [ ] Monitor plasticity: Track learning speed on new tasks
- [ ] Monitor forgetting: Periodically evaluate on old tasks
- [ ] Set memory budget: Decide replay buffer size upfront
- [ ] Choose method based on constraints (memory/compute/privacy)
- [ ] Implement fallback: If catastrophic forgetting detected, trigger mitigation (replay, reset)
- [ ] Log task boundaries: Even if agent doesn't use them, log for debugging
- [ ] Versioning: Save checkpoints after each task for rollback
- [ ] Safety checks: Validate performance on critical old tasks before deployment

---

## References

**See `Continual_RL.md` for consolidated reference list.**

**Additional Technical References**:
- Barreto et al. (2017), *Successor Features for Transfer in Reinforcement Learning*
- Barreto et al. (2018), *Transfer in Deep RL Using Successor Features and GPI*
- Dohare et al. (2024), *Loss of Plasticity in Deep Continual Learning* (Nature)
- Lyle et al. (2023), *Understanding Plasticity in Neural Networks*
- Lyle et al. (2024), *Disentangling the Causes of Plasticity Loss*
- Nikishin et al. (2022), *The Primacy Bias in Deep Reinforcement Learning*
- Ash & Adams (2020), *On Warm-Starting Neural Network Training*
- Elsayed & Mahmood (2024), *Utility-based Perturbed Gradient Descent*
- Finn et al. (2017), *Model-Agnostic Meta-Learning for Fast Adaptation*
- Duan et al. (2016), *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning*
- Pentina & Lampert (2014), *A PAC-Bayesian Bound for Lifelong Learning*
- Wolczyk et al. (2022), *Disentangling Transfer in Continual Reinforcement Learning*
