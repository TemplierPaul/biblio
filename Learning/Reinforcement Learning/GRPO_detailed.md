# GRPO: Detailed Implementation Guide

## Algorithm Overview

GRPO (Group Relative Policy Optimization) eliminates the critic network from PPO by using group-relative advantage estimation. This guide provides complete implementation details for LLM alignment tasks.

---

## Core GRPO Algorithm

### Pseudocode

```python
# Initialization
policy_model = load_sft_model()
reference_model = copy(policy_model)
reward_model = load_trained_reward_model()

# Hyperparameters
G = 64  # Group size
β = 0.04  # KL coefficient
ε = 0.1  # Clipping threshold
lr = 1e-6  # Learning rate
μ = 1  # Policy update steps per batch

for iteration in range(num_iterations):
    # Step 1: Set reference model to current policy
    reference_model = copy(policy_model)

    for batch in training_data:
        questions = batch  # List of questions

        # Step 2: Sample group outputs
        all_outputs = []
        all_rewards = []
        for q in questions:
            outputs = []
            for _ in range(G):
                o = policy_model.sample(q, old_policy=True)
                outputs.append(o)

            # Step 3: Get rewards from reward model
            rewards = [reward_model(q, o) for o in outputs]

            # Step 4: Normalize rewards within group
            r_mean = mean(rewards)
            r_std = std(rewards)
            normalized_rewards = [(r - r_mean) / r_std for r in rewards]

            all_outputs.extend(outputs)
            all_rewards.extend(normalized_rewards)

        # Step 5: Update policy μ times
        for _ in range(μ):
            loss = compute_grpo_loss(
                policy_model, reference_model,
                all_outputs, all_rewards,
                β, ε
            )
            loss.backward()
            optimizer.step()

    # Step 6: Update reward model with new samples
    update_reward_model(reward_model, all_outputs, replay_ratio=0.1)
```

---

## GRPO Loss Function

### Mathematical Formulation

```python
def compute_grpo_loss(policy, reference, outputs, advantages, β, ε):
    """
    Compute GRPO loss for batch of outputs.

    Args:
        policy: Current policy model πθ
        reference: Reference policy model πref
        outputs: List of (question, output) tuples
        advantages: Normalized group advantages
        β: KL coefficient
        ε: Clipping threshold

    Returns:
        loss: Scalar GRPO loss
    """
    total_loss = 0

    for (q, o), advantage in zip(outputs, advantages):
        # Get log probabilities
        log_probs_new = policy.log_prob(o, q)  # πθ(o|q)
        log_probs_old = policy.log_prob_old(o, q)  # πθ_old(o|q)
        log_probs_ref = reference.log_prob(o, q)  # πref(o|q)

        # Compute probability ratios
        ratio = torch.exp(log_probs_new - log_probs_old)
        ratio_ref = torch.exp(log_probs_ref - log_probs_new)

        # Clipped surrogate objective (per token)
        clipped_ratio = torch.clamp(ratio, 1 - ε, 1 + ε)
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        policy_loss = torch.min(surrogate1, surrogate2)

        # KL divergence term (unbiased estimator)
        kl_term = ratio_ref - torch.log(ratio_ref) - 1
        kl_loss = β * kl_term

        # Combined loss (per token)
        token_loss = -(policy_loss - kl_loss)

        # Average over tokens in output
        total_loss += token_loss.mean()

    # Average over all outputs
    return total_loss / len(outputs)
```

---

## Advantage Computation

### Outcome Supervision (OS)

```python
def compute_advantages_os(questions, outputs, reward_model, G):
    """
    Compute group-relative advantages with outcome supervision.

    Args:
        questions: List of questions
        outputs: List of G outputs per question
        reward_model: Reward model
        G: Group size

    Returns:
        advantages: List of advantages (one per output)
    """
    advantages = []

    for i, q in enumerate(questions):
        # Get outputs for this question
        group_outputs = outputs[i*G : (i+1)*G]

        # Compute rewards
        rewards = [reward_model(q, o) for o in group_outputs]

        # Normalize within group
        r_mean = np.mean(rewards)
        r_std = np.std(rewards) + 1e-8  # Avoid division by zero

        normalized_rewards = [(r - r_mean) / r_std for r in rewards]

        # Advantage = normalized reward for all tokens
        for r_norm in normalized_rewards:
            advantages.append(r_norm)

    return advantages
```

### Process Supervision (PS)

```python
def compute_advantages_ps(questions, outputs, reward_model, G):
    """
    Compute group-relative advantages with process supervision.

    Args:
        questions: List of questions
        outputs: List of G outputs per question (with step annotations)
        reward_model: Process reward model
        G: Group size

    Returns:
        advantages: List of advantages (per token, cumulative)
    """
    advantages = []

    for i, q in enumerate(questions):
        # Get outputs for this question
        group_outputs = outputs[i*G : (i+1)*G]

        # Compute step-level rewards
        step_rewards_list = []
        for o in group_outputs:
            # Get rewards at each reasoning step
            step_rewards = reward_model.step_rewards(q, o)
            step_rewards_list.append(step_rewards)

        # Normalize rewards at each step across group
        num_steps = len(step_rewards_list[0])
        normalized_step_rewards = []

        for step_idx in range(num_steps):
            # Rewards for this step across all outputs
            step_rewards_at_t = [sr[step_idx] for sr in step_rewards_list]

            # Normalize
            r_mean = np.mean(step_rewards_at_t)
            r_std = np.std(step_rewards_at_t) + 1e-8

            normalized = [(r - r_mean) / r_std for r in step_rewards_at_t]
            normalized_step_rewards.append(normalized)

        # Compute cumulative advantages (sum from t to T)
        for output_idx in range(G):
            output_advantages = []
            for step_idx in range(num_steps):
                # Sum of normalized rewards from this step onward
                cumulative = sum(
                    normalized_step_rewards[j][output_idx]
                    for j in range(step_idx, num_steps)
                )
                output_advantages.append(cumulative)

            advantages.append(output_advantages)

    return advantages
```

---

## Gradient Coefficient Analysis

### Computing Gradient Coefficients

```python
def compute_gradient_coefficient(advantage, ratio_ref, β):
    """
    Compute GRPO gradient coefficient.

    Args:
        advantage: Normalized group advantage
        ratio_ref: πref / πθ
        β: KL coefficient

    Returns:
        gc: Gradient coefficient
    """
    gc = advantage + β * (ratio_ref - 1)
    return gc
```

### Interpretation

```python
# Example gradient coefficients for different scenarios:

# High reward output
advantage_high = 2.0  # 2 std above mean
ratio_ref = 1.0  # No divergence from reference
gc_high = 2.0 + 0.04 * (1.0 - 1.0) = 2.0  # Strong positive reinforcement

# Average reward output
advantage_avg = 0.0  # At mean
ratio_ref = 1.0
gc_avg = 0.0 + 0.04 * (1.0 - 1.0) = 0.0  # Neutral (no update)

# Low reward output
advantage_low = -2.0  # 2 std below mean
ratio_ref = 1.0
gc_low = -2.0 + 0.04 * (1.0 - 1.0) = -2.0  # Strong negative (penalize)

# Diverged from reference
advantage = 1.0
ratio_ref = 2.0  # Policy diverged from reference
gc_diverged = 1.0 + 0.04 * (2.0 - 1.0) = 1.04  # Slightly reduced (KL penalty)
```

---

## Iterative Training with Reward Model Updates

### Complete Training Loop

```python
def train_grpo_iterative(
    policy_model,
    reward_model,
    training_questions,
    num_iterations=3,
    reward_update_interval=1000
):
    """
    Iterative GRPO training with reward model updates.

    Args:
        policy_model: Initial SFT model
        reward_model: Initial reward model
        training_questions: List of questions
        num_iterations: Number of RL iterations
        reward_update_interval: Steps between RM updates

    Returns:
        trained_policy: Final policy model
    """
    # Hyperparameters
    G = 64
    β = 0.04
    ε = 0.1
    lr = 1e-6
    replay_ratio = 0.1

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    # Historical data for replay
    historical_data = []

    for iteration in range(num_iterations):
        print(f"RL Iteration {iteration + 1}/{num_iterations}")

        # Set reference model
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()

        step_count = 0

        for batch_idx, questions_batch in enumerate(training_questions):
            # Sample outputs
            outputs, rewards = sample_and_reward(
                policy_model, reward_model,
                questions_batch, G
            )

            # Compute advantages
            advantages = compute_advantages_os(
                questions_batch, outputs, reward_model, G
            )

            # Update policy
            loss = compute_grpo_loss(
                policy_model, reference_model,
                outputs, advantages, β, ε
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            # Store for reward model update
            historical_data.append((questions_batch, outputs, rewards))

            step_count += 1

            # Update reward model periodically
            if step_count % reward_update_interval == 0:
                update_reward_model_with_replay(
                    reward_model,
                    historical_data,
                    replay_ratio
                )

        # Evaluate after each iteration
        evaluate(policy_model, val_questions)

    return policy_model


def update_reward_model_with_replay(reward_model, historical_data, replay_ratio):
    """
    Update reward model with recent data + replay.

    Args:
        reward_model: Reward model to update
        historical_data: List of (questions, outputs, rewards)
        replay_ratio: Fraction of historical data to include
    """
    # Sample recent data
    recent_size = int(len(historical_data) * (1 - replay_ratio))
    recent_data = historical_data[-recent_size:]

    # Sample from history
    replay_size = int(len(historical_data) * replay_ratio)
    replay_data = random.sample(historical_data, replay_size)

    # Combine
    training_data = recent_data + replay_data

    # Create pairwise preference data
    preference_pairs = create_preference_pairs(training_data)

    # Train reward model
    train_reward_model(reward_model, preference_pairs)
```

---

## Hyperparameters

### DeepSeek-Math Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Policy Learning Rate** | 1e-6 | Very small for stability |
| **KL Coefficient (β)** | 0.04 | Regularization strength |
| **Clipping Threshold (ε)** | 0.01-0.1 | PPO clipping range |
| **Group Size (G)** | 64 | Outputs per question |
| **Max Output Length** | 1024 | Token limit |
| **Training Batch Size** | 1024 | Total tokens per batch |
| **Policy Updates (μ)** | 1 | Updates per batch |
| **Reward Model LR** | 2e-5 | RM update learning rate |
| **Replay Ratio** | 10% (0.1) | Historical data fraction |
| **Training Questions** | 144K | GSM8K + MATH CoT subset |
| **RL Iterations** | 2-3 | Full training cycles |
| **Gradient Clipping** | 1.0 | Max gradient norm |

### Hyperparameter Sensitivity

**KL Coefficient (β)**:
- Too low (< 0.01): Policy diverges from reference
- Optimal (0.04): Balance between exploration and stability
- Too high (> 0.1): Over-regularized, slow learning

**Group Size (G)**:
- Too small (< 16): High variance in advantages
- Optimal (64): Good balance
- Too large (> 128): Diminishing returns, higher sampling cost

**Clipping Threshold (ε)**:
- Standard PPO: 0.1-0.2
- GRPO for LLMs: 0.01-0.1 (tighter clipping for stability)

---

## Implementation Tips

### 1. Efficient Sampling

```python
def parallel_sample_outputs(policy_model, questions, G, num_gpus=4):
    """
    Parallelize output sampling across GPUs.

    Args:
        policy_model: Policy model
        questions: List of questions
        G: Group size
        num_gpus: Number of GPUs available

    Returns:
        outputs: List of G outputs per question
    """
    # Distribute questions across GPUs
    questions_per_gpu = len(questions) // num_gpus

    with torch.no_grad():
        # Parallel sampling
        outputs = torch.multiprocessing.spawn(
            sample_on_gpu,
            args=(policy_model, questions, G, questions_per_gpu),
            nprocs=num_gpus,
            join=True
        )

    return outputs


def sample_on_gpu(gpu_id, policy_model, questions, G, batch_size):
    """Sample outputs on single GPU."""
    torch.cuda.set_device(gpu_id)

    start_idx = gpu_id * batch_size
    end_idx = start_idx + batch_size
    gpu_questions = questions[start_idx:end_idx]

    outputs = []
    for q in gpu_questions:
        for _ in range(G):
            o = policy_model.generate(q, max_length=1024)
            outputs.append(o)

    return outputs
```

### 2. Memory Optimization

```python
def gradient_checkpointing(policy_model):
    """
    Enable gradient checkpointing to save memory.
    """
    # For HuggingFace models
    policy_model.gradient_checkpointing_enable()

    # Reduces memory by recomputing activations during backward pass
    # Trade-off: ~20% slower but 30-40% less memory


def mixed_precision_training(policy_model, optimizer):
    """
    Use mixed precision (FP16/BF16) for memory savings.
    """
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    for batch in training_data:
        with autocast(dtype=torch.bfloat16):
            loss = compute_grpo_loss(...)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

### 3. Advantage Normalization

```python
def normalize_advantages_robust(advantages):
    """
    Robust advantage normalization handling edge cases.
    """
    advantages = np.array(advantages)

    # Handle constant advantages (all same)
    if np.std(advantages) < 1e-8:
        return np.zeros_like(advantages)

    # Z-score normalization
    mean = np.mean(advantages)
    std = np.std(advantages) + 1e-8
    normalized = (advantages - mean) / std

    # Clip extreme values (optional)
    normalized = np.clip(normalized, -5, 5)

    return normalized
```

### 4. Logging and Debugging

```python
def log_grpo_metrics(outputs, rewards, advantages, policy_loss, kl_div):
    """
    Log key metrics for monitoring training.
    """
    metrics = {
        # Reward statistics
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'reward_min': np.min(rewards),
        'reward_max': np.max(rewards),

        # Advantage statistics
        'advantage_mean': np.mean(advantages),
        'advantage_std': np.std(advantages),

        # Loss components
        'policy_loss': policy_loss.item(),
        'kl_divergence': kl_div.item(),

        # Gradient coefficient distribution
        'gc_mean': np.mean(advantages),  # Simplified
        'gc_positive_ratio': (np.array(advantages) > 0).mean(),

        # Output statistics
        'output_length_mean': np.mean([len(o) for o in outputs]),
    }

    return metrics
```

---

## Comparison: PPO vs GRPO Implementation

### PPO Implementation (for reference)

```python
class PPO:
    def __init__(self, policy, critic):
        self.policy = policy
        self.critic = critic  # Separate value network

    def update(self, rollouts):
        # Compute advantages using GAE
        advantages = self.compute_gae(
            rollouts.states,
            rollouts.rewards,
            rollouts.values
        )

        # Update policy
        policy_loss = self.policy_loss(
            rollouts.actions,
            advantages
        )

        # Update critic
        value_loss = self.value_loss(
            rollouts.states,
            rollouts.returns
        )

        total_loss = policy_loss + 0.5 * value_loss
        total_loss.backward()
```

### GRPO Implementation (simplified)

```python
class GRPO:
    def __init__(self, policy):
        self.policy = policy  # No critic needed!

    def update(self, questions, group_outputs, group_rewards):
        # Compute group-relative advantages
        advantages = self.normalize_group_rewards(group_rewards)

        # Update policy only
        policy_loss = self.grpo_loss(
            group_outputs,
            advantages
        )

        policy_loss.backward()  # No value loss!
```

**Key Difference**: GRPO eliminates entire critic update pipeline.

---

## Common Pitfalls

### 1. Reward Scale Mismatch

**Problem**: Reward model outputs unbounded values

**Solution**: Always normalize within groups
```python
# Bad
advantages = rewards  # Unbounded scale

# Good
advantages = (rewards - mean(rewards)) / std(rewards)
```

### 2. Group Size Too Small

**Problem**: High variance in advantages with G < 16

**Solution**: Use G ≥ 64 for stable gradients

### 3. KL Coefficient Too Low

**Problem**: Policy diverges from reference, unstable training

**Solution**: Start with β = 0.04, increase if divergence issues

### 4. Forgetting Old Policy

**Problem**: Not storing old_log_probs before updating

**Solution**: Detach and store old log probabilities
```python
with torch.no_grad():
    old_log_probs = policy.log_prob(outputs, questions).detach()
```

### 5. Reward Model Overfitting

**Problem**: Reward model exploited by policy

**Solution**:
- Use replay buffer (10% historical data)
- Regular reward model updates
- Monitor out-of-distribution reward scores

---

## Evaluation

### Metrics to Track

```python
def evaluate_grpo(policy_model, val_questions, reward_model):
    """
    Comprehensive GRPO evaluation.
    """
    metrics = {}

    # 1. Maj@K (Majority Voting)
    for K in [1, 4, 8, 16]:
        accuracy = majority_voting_accuracy(
            policy_model, val_questions, K
        )
        metrics[f'maj@{K}'] = accuracy

    # 2. Pass@K (Single Sample)
    for K in [1, 10, 100]:
        pass_rate = pass_at_k(
            policy_model, val_questions, K
        )
        metrics[f'pass@{K}'] = pass_rate

    # 3. Average Reward
    rewards = []
    for q in val_questions:
        output = policy_model.sample(q)
        reward = reward_model(q, output)
        rewards.append(reward)

    metrics['avg_reward'] = np.mean(rewards)

    # 4. Output Diversity
    outputs_per_question = []
    for q in val_questions[:100]:  # Sample
        outputs = [policy_model.sample(q) for _ in range(10)]
        diversity = compute_diversity(outputs)
        outputs_per_question.append(diversity)

    metrics['output_diversity'] = np.mean(outputs_per_question)

    return metrics
```

---

## Extensions

### 1. Multi-Turn Dialogue

```python
def grpo_for_dialogue(policy_model, conversations, G):
    """
    Extend GRPO to multi-turn dialogue.
    """
    for conv in conversations:
        # Sample G completions for entire conversation
        completions = []
        for _ in range(G):
            completion = policy_model.complete_conversation(conv)
            completions.append(completion)

        # Reward entire conversation
        rewards = [reward_model_dialogue(conv, c) for c in completions]

        # Normalize and update
        advantages = normalize(rewards)
        update_policy(completions, advantages)
```

### 2. Hierarchical GRPO

```python
def hierarchical_grpo(policy_model, questions, G_coarse=8, G_fine=8):
    """
    Two-level GRPO: coarse then fine sampling.
    """
    # Coarse sampling
    coarse_outputs = [policy_model.sample(q) for _ in range(G_coarse)]
    coarse_rewards = [reward_model(q, o) for o in coarse_outputs]

    # Select top K
    top_k_indices = np.argsort(coarse_rewards)[-G_fine:]

    # Fine sampling around top candidates
    fine_outputs = []
    for idx in top_k_indices:
        # Generate variations
        for _ in range(G_fine // len(top_k_indices)):
            variation = policy_model.sample_near(coarse_outputs[idx])
            fine_outputs.append(variation)

    # GRPO update on fine samples
    fine_rewards = [reward_model(q, o) for o in fine_outputs]
    advantages = normalize(fine_rewards)
    update_policy(fine_outputs, advantages)
```

---

## Summary

GRPO implementation requires:
1. **Group sampling**: Generate G outputs per question
2. **Reward evaluation**: Score with reward model
3. **Advantage normalization**: Normalize within groups
4. **Policy update**: Standard PPO objective with group advantages
5. **No critic training**: Eliminates value network entirely

**Key Implementation Insight**: GRPO is PPO minus critic, plus group normalization. Most PPO codebases can be adapted by removing critic components and changing advantage computation.

**Practical Recommendation**: Start with small-scale experiments (G=16, single iteration) to validate pipeline, then scale to full GRPO (G=64, iterative training with RM updates).

---

**See [[GRPO]] for high-level overview and when to use GRPO vs PPO.**
