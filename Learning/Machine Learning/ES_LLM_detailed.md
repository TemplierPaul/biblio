# Evolution Strategies for LLM Fine-Tuning - Detailed Technical Guide

**Paper**: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (Qiu et al., 2024)

This file contains complete technical details, algorithms, mathematical formulations, and implementation guidance to implement ES-based LLM fine-tuning from scratch.

---

## Table of Contents

- [Mathematical Foundations](#mathematical-foundations)
- [Detailed Algorithm](#detailed-algorithm)
- [Implementation Details](#implementation-details)
- [Scaling Techniques](#scaling-techniques)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Pseudocode](#pseudocode)
- [Practical Implementation](#practical-implementation)

---

## Mathematical Foundations

### Natural Evolution Strategies (NES)

**Goal**: Maximize expected reward over a distribution of parameters

$$\max_{\theta} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [R(\theta + \sigma \epsilon)]$$

where:
- $\theta$ = model parameters
- $\sigma$ = noise scale (exploration magnitude)
- $\epsilon$ = standard normal noise
- $R(\cdot)$ = reward function (scalar)

### Gradient Estimation via Perturbation

**Natural gradient** of expected reward:

$$\nabla_\theta \mathbb{E}[R(\theta + \sigma \epsilon)] = \frac{1}{\sigma} \mathbb{E}[\nabla_\epsilon R(\theta + \sigma \epsilon) \cdot \epsilon]$$

**Simplification** (likelihood ratio trick):

$$\nabla_\theta \mathbb{E}[R(\theta + \sigma \epsilon)] = \mathbb{E}[R(\theta + \sigma \epsilon) \cdot \epsilon]$$

**Intuition**: Weight each perturbation $\epsilon$ by how good it was (reward), then move in that direction.

### Parameter Update Rule

**Standard form**:
$$\theta_t = \theta_{t-1} + \alpha \cdot \frac{1}{\sigma} \cdot \frac{1}{N} \sum_{n=1}^N R_n \epsilon_n$$

where:
- $\alpha$ = learning rate
- $N$ = population size
- $R_n$ = reward for perturbation $n$
- $\epsilon_n$ = noise vector for perturbation $n$

**Simplified form** (this paper):
$$\theta_t = \theta_{t-1} + \alpha \cdot \frac{1}{N} \sum_{n=1}^N R_n \epsilon_n$$

(The factor $\frac{1}{\sigma}$ is absorbed into the learning rate $\alpha$)

### Reward Normalization

**Z-score normalization** (per iteration):
$$\tilde{R}_n = \frac{R_n - \text{mean}(R)}{\text{std}(R) + \epsilon}$$

where:
- mean$(R)$ = average reward across population
- std$(R)$ = standard deviation of rewards
- $\epsilon$ = small constant for numerical stability

**Why normalize**:
- Rewards can vary wildly across iterations
- Normalization ensures consistent gradient magnitude
- Improves stability and convergence

### Variance of Gradient Estimator

**Key advantage over RL** (policy gradient variance):

**RL (REINFORCE)**:
$$\text{Var}[\hat{g}_{RL}] \propto \frac{1}{N} \cdot H^4$$

where $H$ is episode horizon (strongly depends on sequence length)

**ES**:
$$\text{Var}[\hat{g}_{ES}] \propto \frac{1}{N} \cdot D$$

where $D$ is dimension of parameter space (does NOT strongly depend on horizon)

**Implication**: ES gradient variance doesn't explode with long sequences, while RL does.

---

## Detailed Algorithm

### Algorithm 1: Basic ES for LLM Fine-Tuning

```
Input:
  - θ₀: Pretrained LLM parameters
  - R: Reward function that takes parameter set and returns scalar
  - T: Number of iterations
  - N: Population size (typically 30)
  - σ: Noise scale (typically 0.001)
  - α: Learning rate (typically 5×10⁻⁴)

Output:
  - θ_final: Fine-tuned parameters

Initialize:
  θ ← θ₀

For t = 1 to T:
    rewards ← []
    perturbations ← []

    # Phase 1: Generate perturbations and evaluate
    For n = 1 to N:
        ε_n ~ Normal(0, I)  // Sample noise from standard normal
        θ'_n ← θ + σ·ε_n      // Perturb parameters
        R_n ← R(θ'_n)          // Evaluate perturbed parameters
        rewards.append(R_n)
        perturbations.append(ε_n)

    # Phase 2: Normalize rewards
    mean_R ← mean(rewards)
    std_R ← std(rewards)
    normalized_rewards ← (rewards - mean_R) / (std_R + 1e-8)

    # Phase 3: Update parameters
    gradient ← 0
    For n = 1 to N:
        gradient ← gradient + normalized_rewards[n] × perturbations[n]
    gradient ← gradient / N

    θ ← θ + α × gradient  // Parameter update

Return θ
```

### Algorithm 2: Distributed ES (Multi-GPU)

```
Input:
  - P: Number of processes
  - All parameters from Algorithm 1

Initialize on each process:
  θ_local ← θ₀

For t = 1 to T:
    # Each process generates and evaluates own perturbation
    ε_local ~ Normal(0, I)
    θ'_local ← θ_local + σ·ε_local
    R_local ← R(θ'_local)

    # Gather all rewards and perturbations to main process
    all_rewards ← MPI_Gather(R_local, root=0)
    all_perturbations ← MPI_Gather(ε_local, root=0)

    # Main process computes update (at process 0)
    If rank == 0:
        mean_R ← mean(all_rewards)
        std_R ← std(all_rewards)
        normalized_rewards ← (all_rewards - mean_R) / (std_R + 1e-8)

        gradient ← 0
        For n = 1 to N:
            gradient ← gradient + normalized_rewards[n] × all_perturbations[n]
        gradient ← gradient / N

        θ ← θ + α × gradient

    # Broadcast updated parameters to all processes
    θ_local ← MPI_Bcast(θ, root=0)

Return θ
```

### Algorithm 3: Memory-Efficient In-Place Perturbation

```
Input:
  - All parameters from Algorithm 1
  - model: PyTorch neural network
  - random_seed: Reproducible random number generator

For t = 1 to T:
    rewards ← []
    seeds ← []

    For n = 1 to N:
        seed_n ← random_seed.generate_seed()
        seeds.append(seed_n)

        # Perturb in-place layer by layer
        For each layer L in model:
            rng.seed(seed_n)  // Reproducible noise
            noise ← Normal(0, I) [size = L.parameter.shape]
            L.parameter += σ × noise

        # Evaluate (forward pass only)
        R_n ← R(model)
        rewards.append(R_n)

        # Restore in-place layer by layer
        For each layer L in model:
            rng.seed(seed_n)
            noise ← Normal(0, I) [size = L.parameter.shape]
            L.parameter -= σ × noise  // Undo perturbation

    # Normalize rewards
    mean_R ← mean(rewards)
    std_R ← std(rewards)
    normalized_rewards ← (rewards - mean_R) / (std_R + 1e-8)

    # Update parameters in-place, layer by layer
    update ← 0
    For each layer L in model:
        layer_update ← 0
        For n = 1 to N:
            rng.seed(seeds[n])
            noise ← Normal(0, I) [size = L.parameter.shape]
            layer_update += normalized_rewards[n] × noise
        layer_update ← layer_update / N
        L.parameter += α × layer_update
```

---

## Implementation Details

### 1. Noise Retrieval with Random Seeds

**Why**: Storing N×M noise values (M = model size in billions) is memory-prohibitive.

**Solution**: Store only random seeds, regenerate noise on-the-fly.

```python
import numpy as np

class DeterministicRNG:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def generate_noise(self, shape):
        """Generate noise deterministically from seed"""
        return self.rng.normal(0, 1, shape)

# Example:
rng = DeterministicRNG(seed=42)
noise1 = rng.generate_noise((1000, 1000))  # Same noise every time

# For distributed setting:
seed_list = [123, 124, 125, ..., 152]  # 30 seeds for 30 population members
# Send only 30 seeds (1KB) instead of 30 × 8B parameters (240GB)
```

**Cost**:
- Storage: N × log₂(2³²) bits = 30 × 32 bits = 120 bytes
- Computation: ~1% overhead to regenerate noise

### 2. Parallel Evaluations

**Setup**:
```python
import torch.distributed as dist

def parallel_evaluate(model, reward_fn, seed, rank, world_size):
    """Each rank evaluates one perturbation"""
    # Each rank gets same model, different seed
    rng.seed(seed)
    noise = torch.randn_like(model.parameters())

    # Perturb model
    theta_perturbed = model.parameters() + sigma * noise

    # Evaluate
    reward = reward_fn(model_with_params(model, theta_perturbed))

    # Gather all rewards
    rewards = [None] * world_size
    dist.all_gather_object(rewards, reward)

    return rewards, noise
```

**Efficiency**:
- With 8 GPUs: 8 parallel evaluations per iteration
- Wall-clock speedup: ~8× (minus communication overhead)

### 3. Layer-Level In-Place Perturbation

**Goal**: Reduce peak GPU memory

```python
def perturb_and_evaluate_inplace(model, sigma, seed, reward_fn):
    """
    Perturb model in-place layer by layer.
    This avoids storing a full copy of the model.
    """
    rng = np.random.RandomState(seed)
    reward_accumulator = []

    for layer_idx, param in enumerate(model.parameters()):
        # Store original values
        original = param.data.clone()

        # Generate noise for this layer
        noise = torch.from_numpy(rng.normal(0, 1, param.shape))
        param.data += sigma * noise

        # Evaluate (only forward pass needed)
        if layer_idx == len(list(model.parameters())) - 1:
            # Last layer: evaluate full model
            reward = reward_fn(model)
            reward_accumulator.append((noise, reward))

        # Restore layer
        param.data = original

    return reward_accumulator

# Memory saving:
# Naive: N × M (model size × population) = 30 × 8GB = 240GB
# In-place: M + layer_size = 8GB + 0.1GB = 8.1GB
# Savings: 29× reduction
```

### 4. Reward Normalization

**Implementation**:
```python
import numpy as np

def normalize_rewards(rewards):
    """Z-score normalization of rewards"""
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if std_reward < 1e-8:
        # All rewards are same, avoid division by zero
        return np.zeros_like(rewards)

    normalized = (rewards - mean_reward) / (std_reward + 1e-8)
    return normalized

# Example:
rewards = [0.5, 0.8, 0.3, 0.7, 0.6]
normalized = normalize_rewards(rewards)
# Result: roughly [-0.5, 1.0, -1.5, 0.8, 0.3]
# Mean ≈ 0, Std ≈ 1
```

**Why normalize**:
- Prevents gradient magnitude from varying wildly
- Ensures stable learning rate
- Handles tasks with different reward ranges

### 5. Greedy Decoding

**Setup**:
```python
import torch

def evaluate_with_greedy_decoding(model, prompt, max_length):
    """
    Generate response deterministically using greedy decoding.
    Ensures all variance comes from parameter perturbations, not sampling.
    """
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(prompt)

        for _ in range(max_length):
            logits = model(tokens)
            # Greedy: always select highest probability token
            next_token = torch.argmax(logits[-1, :], dim=-1)
            tokens.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(tokens)
    reward = evaluate_response(response)  # e.g., correctness score
    return reward
```

**Why greedy**:
- Ensures reproducibility
- All variance from parameters, not randomness
- Isolates effect of parameter perturbations

### 6. Decomposed Parameter Updates

**Goal**: Reduce peak memory during update

```python
def update_parameters_decomposed(model, sigma, seeds, rewards, learning_rate):
    """
    Update parameters by processing each seed/reward pair one at a time.
    Avoids storing full gradient tensor.
    """
    num_perturbations = len(seeds)

    for param in model.parameters():
        # Initialize update for this parameter
        param_update = torch.zeros_like(param)

        # Process each perturbation one at a time
        for idx, (seed, reward) in enumerate(zip(seeds, rewards)):
            rng = np.random.RandomState(seed)
            noise = torch.from_numpy(rng.normal(0, 1, param.shape))

            # Accumulate update
            param_update += (reward / num_perturbations) * noise

        # Apply update
        param.data += learning_rate * param_update

# Memory:
# Naive: Store gradient for all N perturbations = N × M
# Decomposed: Process one at a time = M (reuse memory)
```

### 7. Learning Rate Digestion

**Standard form**:
$$\theta_t = \theta_{t-1} + \alpha \cdot \frac{1}{\sigma} \cdot \frac{1}{N} \sum R_n \epsilon_n$$

**Simplified form**:
$$\theta_t = \theta_{t-1} + \alpha' \cdot \frac{1}{N} \sum R_n \epsilon_n$$

where $\alpha' = \frac{\alpha}{\sigma}$

**Benefit**: Eliminates division by $\sigma$, simplifies hyperparameter tuning.

---

## Scaling Techniques

### Horizontal Scaling (More GPUs)

**Setup**: Distribute population across GPUs

```python
import torch.distributed as dist

def distributed_es_step(model, reward_fn, N, sigma, alpha, rank, world_size):
    """
    Distributed ES where each rank evaluates one population member.
    """
    # Ensure N is divisible by world_size
    assert N % world_size == 0

    members_per_rank = N // world_size

    local_rewards = []
    local_perturbations = []

    # Each rank handles its own population members
    for idx in range(members_per_rank):
        seed = rank * members_per_rank + idx
        rng = np.random.RandomState(seed)

        # Perturb and evaluate
        noise = torch.from_numpy(rng.normal(0, 1, model_size))
        theta_perturbed = model.parameters() + sigma * noise
        reward = reward_fn(theta_perturbed)

        local_rewards.append(reward)
        local_perturbations.append(noise)

    # Gather all rewards and perturbations from all ranks
    all_rewards = [None] * world_size
    all_perturbations = [None] * world_size

    dist.all_gather_object(all_rewards, local_rewards)
    dist.all_gather_object(all_perturbations, local_perturbations)

    # Compute update (done on all ranks in parallel)
    if rank == 0:
        all_rewards_flat = sum(all_rewards, [])
        all_perturbations_flat = sum(all_perturbations, [])

        mean_R = np.mean(all_rewards_flat)
        std_R = np.std(all_rewards_flat)
        normalized = [(r - mean_R) / (std_R + 1e-8) for r in all_rewards_flat]

        gradient = sum(n * p for n, p in zip(normalized, all_perturbations_flat)) / N
        model.parameters().data += alpha * gradient

    # Broadcast updated parameters
    dist.broadcast(model.parameters(), src=0)
```

**Scaling properties**:
- N=30, 8 GPUs → ~3.75 members per GPU
- Communication: Only rewards (scalars) + seeds (minimal)
- Communication time: ~1% of total (fast)
- Wall-clock speedup: ~7.5× (nearly linear)

### Vertical Scaling (Larger Models)

**Challenge**: 8B model parameters = 32GB (fp32) or 16GB (fp16)

**Solution**: Layer-wise in-place operations

```python
def process_large_model_layer_by_layer(model, sigma, seeds, reward_fn):
    """
    For very large models, process one layer at a time to avoid
    memory overhead from storing full model copies.
    """
    all_rewards = []

    for seed_idx, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)

        for layer_idx, layer in enumerate(model):
            # Create noise only for this layer
            noise = torch.from_numpy(rng.normal(0, 1, layer.weight.shape))

            # Perturb in-place
            layer.weight.data += sigma * noise

        # Evaluate entire model (after all layers perturbed)
        reward = reward_fn(model)
        all_rewards.append(reward)

        # Restore all layers
        rng = np.random.RandomState(seed)  # Recreate noise with same seed
        for layer_idx, layer in enumerate(model):
            noise = torch.from_numpy(rng.normal(0, 1, layer.weight.shape))
            layer.weight.data -= sigma * noise  # Undo perturbation

    return all_rewards
```

**Memory breakdown**:
- Model parameters: 16GB (fp16)
- Single layer perturbation: 0.5GB
- Temporary buffers: 1GB
- Total: ~17.5GB (fits on 20GB GPU)

Without optimization: Need 16GB × 30 = 480GB (impossible)

---

## Hyperparameter Tuning

### Population Size (N)

**Effect**: Variance vs computation trade-off

$$\text{Var}[\hat{g}] \propto \frac{1}{N}$$

**Practical guidance**:
| Use Case | N | Notes |
|----------|---|-------|
| Prototyping | 10-20 | Fast, noisy updates |
| Production | 30-50 | Good balance (paper uses 30) |
| Large model | 50-100 | More stability for very large θ |
| Small model | 10-20 | Faster, sufficient variance reduction |

**Tuning procedure**:
```python
for N in [10, 20, 30, 50, 100]:
    rewards = []
    for iteration in range(10):
        # Run ES with this N
        final_reward = run_es(N=N, num_iterations=50)
        rewards.append(final_reward)

    print(f"N={N}: mean_reward={np.mean(rewards)}, std={np.std(rewards)}")
    # Look for where accuracy plateaus
```

### Noise Scale (σ)

**Effect**: Exploration magnitude

$$\theta' = \theta + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

**If σ too small**: Insufficient exploration, converges to local optima
**If σ too large**: Too much noise, can't find good solutions

**Typical range**: 10⁻⁴ to 10⁻²

**Paper's choice**: σ = 10⁻³ = 0.001

**Tuning procedure**:
```python
# Start with σ = 0.001, monitor parameter changes
for sigma in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    model = initialize()
    for t in range(100):
        # Run ES step
        run_es_step(sigma=sigma, ...)

        # Monitor parameter drift from initial
        param_diff = torch.norm(model.parameters() - initial_params)
        print(f"sigma={sigma}, iter={t}, param_change={param_diff.item()}")
```

### Learning Rate (α)

**Effect**: Update magnitude per iteration

$$\theta_t = \theta_{t-1} + \alpha \cdot \text{gradient}$$

**Relationship**: Often set as $\alpha = \frac{\sigma}{2}$ or $\alpha = 0.5 \times \sigma$

**Paper's choice**: $\alpha = 5 \times 10^{-4}$ with $\sigma = 0.001$

So $\alpha = 0.5 \times \sigma$ ✓

**Typical range**: $10^{-4}$ to $10^{-2}$

**Tuning procedure**:
```python
# Learning rate schedule: decay over time
def get_learning_rate(iteration, initial_alpha=5e-4, decay_rate=0.95):
    return initial_alpha * (decay_rate ** (iteration / 100))

# Or adaptive: scale by reward improvement
for t in range(num_iterations):
    current_reward = evaluate(model)
    if current_reward > best_reward:
        best_reward = current_reward
        no_improve_counter = 0
    else:
        no_improve_counter += 1

    if no_improve_counter > 10:
        alpha = alpha * 0.5  # Decay learning rate
```

### Number of Iterations (T)

**Effect**: Total training time and convergence

**Guidance**:
- Watch reward curve: Stop when plateau
- Typically: 20-50 iterations for most tasks
- Can use early stopping

```python
best_reward = -inf
no_improve_counter = 0
patience = 10

for t in range(max_iterations):
    avg_reward = run_es_step()
    print(f"Iteration {t}: avg_reward={avg_reward}")

    if avg_reward > best_reward:
        best_reward = avg_reward
        no_improve_counter = 0
        save_checkpoint(model)
    else:
        no_improve_counter += 1

    if no_improve_counter > patience:
        print(f"Early stopping at iteration {t}")
        break
```

---

## Evaluation Metrics

### 1. Task Performance

**Downstream task score**: Accuracy, F1, BLEU, etc. on evaluation set

```python
def evaluate_task_performance(model, test_set):
    """Evaluate fine-tuned model on actual task"""
    correct = 0
    for example in test_set:
        prediction = model.generate(example["input"])
        if prediction == example["target"]:
            correct += 1
    accuracy = correct / len(test_set)
    return accuracy
```

### 2. Behavioral Alignment (KL Divergence from Base)

**Measure**: How much did fine-tuning change model behavior?

$$\text{KL}(P_{\text{base}} || P_{\text{fine-tuned}}) = \sum_i P_{\text{base}} \log \frac{P_{\text{base}}}{P_{\text{fine-tuned}}}$$

**Low KL**: Model maintains base behavior ✓
**High KL**: Model has drifted significantly (possible misalignment)

```python
def compute_kl_divergence(base_model, fine_tuned_model, prompt):
    """Compute KL divergence between logits"""
    with torch.no_grad():
        base_logits = base_model(prompt)
        tuned_logits = fine_tuned_model(prompt)

        # Convert to probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        tuned_probs = F.softmax(tuned_logits, dim=-1)

        # Compute KL divergence
        kl = torch.sum(base_probs * (torch.log(base_probs) - torch.log(tuned_probs)))
    return kl.item()
```

### 3. Stability Across Runs

**Measure**: Does same fine-tuning process with different seeds produce consistent results?

```python
def evaluate_stability(model_fn, num_seeds=5):
    """Run ES multiple times, check consistency"""
    results = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = initialize()
        model = run_es(model, num_iterations=50)
        final_reward = evaluate(model)
        results.append(final_reward)

    mean_reward = np.mean(results)
    std_reward = np.std(results)
    cv = std_reward / mean_reward  # Coefficient of variation

    print(f"Mean: {mean_reward:.3f}, Std: {std_reward:.3f}, CV: {cv:.3f}")
    # Lower CV = more stable (better)
```

### 4. Sample Efficiency

**Measure**: Total function evaluations vs final performance

```python
def compute_sample_efficiency(model, num_iterations, population_size):
    """How many evaluations needed to reach target performance?"""
    total_evaluations = num_iterations * population_size
    return total_evaluations

# Example:
# ES: 50 iterations × 30 population = 1,500 evaluations
# RL: 50 iterations × 50 rollouts × 10 gradient steps = 25,000 evaluations
# ES is 16× more sample-efficient!
```

---

## Pseudocode

### Main Training Loop

```
def es_fine_tune_llm(
    model: LLM,
    reward_function: callable,
    num_iterations: int,
    population_size: int,
    noise_scale: float,
    learning_rate: float,
    device: str
):
    """
    Fine-tune LLM using Evolution Strategies.

    Args:
        model: Pretrained LLM
        reward_function: Takes model and returns scalar reward
        num_iterations: Number of ES iterations
        population_size: N in Algorithm 1
        noise_scale: σ in Algorithm 1
        learning_rate: α in Algorithm 1
        device: "cuda:0" or "cpu"

    Returns:
        fine_tuned_model: Updated LLM parameters
    """

    model.to(device)
    model.eval()  # No batch norm training

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")

        all_rewards = []
        all_perturbations = []

        # Evaluate population
        for pop_idx in range(population_size):
            # Generate reproducible noise from seed
            seed = iteration * population_size + pop_idx
            rng = np.random.RandomState(seed)

            # Perturb parameters
            perturbation = torch.from_numpy(
                rng.normal(0, 1, total_param_count)
            ).to(device)

            # In practice: perturb layer-by-layer (see implementation)
            perturbed_params = unflatten(
                flatten(model.parameters()) + noise_scale * perturbation
            )

            # Evaluate
            with torch.no_grad():
                reward = reward_function(model)

            all_rewards.append(reward)
            all_perturbations.append(perturbation)

        # Normalize rewards
        rewards_array = np.array(all_rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)

        if std_reward > 1e-8:
            normalized_rewards = (rewards_array - mean_reward) / std_reward
        else:
            normalized_rewards = np.zeros_like(rewards_array)

        # Compute gradient
        gradient = torch.zeros(total_param_count).to(device)
        for rew, pert in zip(normalized_rewards, all_perturbations):
            gradient += rew * pert
        gradient = gradient / population_size

        # Update parameters
        with torch.no_grad():
            param_flat = flatten(model.parameters())
            param_flat += learning_rate * gradient

        # Log progress
        print(f"  Mean reward: {mean_reward:.3f}, Std: {std_reward:.3f}")

    return model


def reward_function(model, test_data):
    """
    Example reward function: accuracy on test set.
    """
    correct = 0
    for example in test_data:
        with torch.no_grad():
            prediction = model.generate(example["input"])
        if prediction == example["target"]:
            correct += 1
    return correct / len(test_data)
```

---

## Practical Implementation

### Step 1: Setup

```python
import torch
import numpy as np
from typing import Callable, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
```

### Step 2: Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Freeze batch norm, turn off gradients
model.eval()
for param in model.parameters():
    param.requires_grad = False  # ES doesn't compute gradients

model.to(device)
logger.info(f"Model size: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
```

### Step 3: Define Reward Function

```python
def task_reward(model, test_examples, max_tokens=100):
    """
    Reward function for reasoning task.
    Returns average correctness across examples.
    """
    correct = 0

    for example in test_examples:
        prompt = example["prompt"]
        target = example["target"]

        # Generate with greedy decoding
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                do_sample=False,  # Greedy
                temperature=1.0,
                top_p=1.0
            )

        generated_text = tokenizer.decode(outputs[0])
        extracted_answer = extract_answer(generated_text)

        if extracted_answer == target:
            correct += 1

    return correct / len(test_examples)


def extract_answer(text):
    """Extract final answer from model output"""
    # Task-specific extraction logic
    # For math: extract number after "Answer:" or "Final Answer:"
    # For yes/no: extract yes/no
    # Customize per application
    pass
```

### Step 4: Implement ES

```python
class ESFinetuner:
    def __init__(
        self,
        model,
        reward_fn: Callable,
        noise_scale: float = 0.001,
        learning_rate: float = 5e-4,
        population_size: int = 30,
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.sigma = noise_scale
        self.alpha = learning_rate
        self.N = population_size
        self.device = next(model.parameters()).device

        # Flatten all parameters once
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_count = sum(p.numel() for p in model.parameters())

    def flatten_params(self):
        """Flatten all model parameters into 1D vector"""
        params = torch.cat([p.data.flatten() for p in self.model.parameters()])
        return params

    def unflatten_params(self, params_flat):
        """Convert flat vector back to parameter dict"""
        offset = 0
        params_dict = {}
        for name, param in self.model.named_parameters():
            size = param.numel()
            params_dict[name] = params_flat[offset:offset+size].reshape(param.shape)
            offset += size
        return params_dict

    def set_params(self, params_flat):
        """Set model parameters from flat vector"""
        offset = 0
        for name, param in self.model.named_parameters():
            size = param.numel()
            param.data = params_flat[offset:offset+size].reshape(param.shape)
            offset += size

    def optimize(self, num_iterations: int):
        """Run ES optimization"""
        params_original = self.flatten_params().clone()

        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration+1}/{num_iterations} ===")

            all_rewards = []
            all_perturbations = []

            # Evaluate population
            for pop_idx in range(self.N):
                seed = iteration * self.N + pop_idx
                rng = np.random.RandomState(seed)

                # Generate noise
                noise = torch.from_numpy(
                    rng.normal(0, 1, self.param_count)
                ).float().to(self.device)

                # Perturb
                params_perturbed = params_original + self.sigma * noise

                # Set model to perturbed params
                self.set_params(params_perturbed)

                # Evaluate
                with torch.no_grad():
                    reward = self.reward_fn(self.model)

                all_rewards.append(reward)
                all_perturbations.append(noise)

                logger.info(f"  Pop {pop_idx}: reward={reward:.4f}")

            # Normalize rewards
            rewards_array = np.array(all_rewards)
            mean_reward = np.mean(rewards_array)
            std_reward = np.std(rewards_array)
            logger.info(f"Mean reward: {mean_reward:.4f}, Std: {std_reward:.4f}")

            if std_reward > 1e-8:
                normalized_rewards = (rewards_array - mean_reward) / std_reward
            else:
                logger.warning("All rewards identical, skipping update")
                continue

            # Compute gradient
            gradient = torch.zeros(self.param_count).to(self.device)
            for rew, pert in zip(normalized_rewards, all_perturbations):
                gradient += rew * pert
            gradient = gradient / self.N

            # Update
            params_original = params_original + self.alpha * gradient
            self.set_params(params_original)

            # Periodic evaluation
            if (iteration + 1) % 5 == 0:
                with torch.no_grad():
                    final_reward = self.reward_fn(self.model)
                logger.info(f"Final reward at iter {iteration+1}: {final_reward:.4f}")

        return self.model
```

### Step 5: Run Fine-tuning

```python
# Prepare test data
test_examples = [
    {"prompt": "What is 2+2?", "target": "4"},
    {"prompt": "What is 10*5?", "target": "50"},
    # ... more examples
]

# Create finetuner
finetuner = ESFinetuner(
    model=model,
    reward_fn=lambda m: task_reward(m, test_examples),
    noise_scale=0.001,
    learning_rate=5e-4,
    population_size=30,
)

# Run optimization
fine_tuned_model = finetuner.optimize(num_iterations=30)

# Save
torch.save(fine_tuned_model.state_dict(), "es_finetuned_model.pt")
logger.info("Fine-tuning complete!")
```

---

## Comparison with RL Implementation

### Key Differences

| Aspect | ES | RL (PPO/GRPO) |
|--------|-----|---|
| **Gradients** | No (inference-only) | Yes (autograd) |
| **Memory** | ~16GB (model only) | ~32GB (model + opt + KV cache) |
| **Parallelization** | Trivial (different seeds) | Complex (gradient sync) |
| **Hyperparameter Tuning** | Minimal (single α, σ) | Extensive (KL penalty β, lr, etc.) |
| **Reward Signal** | Outcome-only ✓ | Token-level preferred |
| **Long Horizon** | Natural ✓ | Challenging |
| **Implementation Complexity** | ~200 lines | ~1000+ lines |

---

## Advanced Topics

### Multi-Objective Optimization

Fine-tune for multiple rewards simultaneously:

$$\text{reward} = w_1 \cdot \text{accuracy} + w_2 \cdot \text{conciseness} - w_3 \cdot \text{divergence}$$

```python
def multi_objective_reward(model, test_examples, weights):
    accuracy = compute_accuracy(model, test_examples)
    conciseness = compute_conciseness(model, test_examples)
    kl_div = compute_kl_divergence(base_model, model)

    reward = (
        weights["accuracy"] * accuracy +
        weights["conciseness"] * conciseness -
        weights["divergence"] * kl_div
    )
    return reward
```

### Adaptive Noise Scale

Adjust σ based on reward improvement:

```python
def adaptive_sigma(iteration, best_reward, current_reward):
    if current_reward < best_reward * 0.9:
        # No improvement, increase exploration
        return sigma * 1.5
    elif current_reward > best_reward:
        # Improvement found, can reduce exploration slightly
        return sigma * 0.95
    else:
        return sigma
```

---

## Conclusion & Key Takeaways

1. **ES scales**: Contrary to expectations, ES works on billion-parameter LLMs with only N=30
2. **Inference-only**: No gradients = lower memory, simpler implementation
3. **Robust**: Works across model families without retuning
4. **Outcome-only**: Perfect for sparse, long-horizon rewards
5. **Stable**: Consistent results, less variance than RL

This implementation provides the foundation to apply ES fine-tuning to your own LLMs and tasks.
