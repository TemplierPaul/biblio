# GRPO: Group Relative Policy Optimization

## Overview

**GRPO** (Group Relative Policy Optimization) is a memory-efficient variant of PPO designed specifically for LLM alignment. It eliminates the critic (value function) model by using group-relative advantage estimation, reducing memory by 40-50% while maintaining or improving performance on reasoning tasks.

**Core Innovation**: Replace the learned value function with average reward from multiple sampled outputs on the same question, computing advantages relative to the group.

---

## Key Problem Solved

### PPO's Memory Bottleneck for LLMs

**PPO Requirements**:
- Policy network (e.g., 7B parameters)
- Value network (typically same size as policy: 7B parameters)
- Total: ~14B parameters in memory during training

**Impact**: Memory constraints limit batch sizes, model sizes, and training efficiency.

**GRPO Solution**: Eliminate the value network entirely, using group statistics for advantage estimation.

---

## Key Concepts

### 1. Group Relative Advantages

**Intuition**: Instead of learning "how good is this state?", ask "how good is this output compared to other outputs for the same question?"

**Process**:
1. Sample group of G outputs from old policy for same question
2. Get rewards r₁, r₂, ..., r_G from reward model
3. Normalize: r̃ᵢ = (rᵢ - mean(r)) / std(r)
4. Advantage Âᵢ = r̃ᵢ (for each output)

**Why It Works**:
- Aligns with how reward models are trained (pairwise comparisons)
- Group statistics provide natural baseline
- No need to learn value function

### 2. Outcome vs. Process Supervision

**Outcome Supervision (OS)**:
- Single reward at end of output
- All tokens get same advantage: Âᵢ,t = r̃ᵢ
- Simpler, requires only outcome reward model

**Process Supervision (PS)**:
- Reward at each reasoning step
- Cumulative advantage: Âᵢ,t = Σⱼ₌ₜ^T r̃ᵢⱼ
- More fine-grained training signal
- Requires step-level reward annotations

**Result**: GRPO+PS > GRPO+OS > Online RFT on math benchmarks

### 3. Dynamic Gradient Coefficients

**Key Insight**: GRPO's gradient coefficient adapts to reward magnitude.

**Gradient Coefficient**:
```
GC(q, o, t) = Âᵢ,t + β(π_ref/π_θ - 1)
```

**Interpretation**:
- High reward → Large positive coefficient → Strong reinforcement
- Low reward → Negative coefficient → Penalize
- Neutral reward → Small coefficient → Minimal update

**Contrast with RFT** (Rejection Fine-Tuning):
- RFT: GC = +1 (correct) or -1 (incorrect)
- GRPO: GC proportional to reward value (continuous reinforcement)

---

## Algorithm Comparison

### PPO
```
For each batch:
    1. Collect rollouts with old policy
    2. Compute values with critic network
    3. Compute advantages using GAE
    4. Update policy with clipped objective
    5. Update critic to minimize value loss
```

**Memory**: Policy + Critic (~2× model size)

### GRPO
```
For each batch:
    1. Sample G outputs per question with old policy
    2. Get rewards from reward model
    3. Compute group-relative advantages (normalize)
    4. Update policy with clipped objective + KL term
```

**Memory**: Policy only (~1× model size)

**Key Difference**: Steps 2-3-5 eliminated in GRPO

---

## GRPO Objective

### Mathematical Formulation

```
J_GRPO(θ) = E_q,{oᵢ}[
    1/G Σᵢ 1/|oᵢ| Σₜ {
        min[
            πθ(oᵢ,t | q, oᵢ,<t) / πθ_old(...) · Âᵢ,t,
            clip(πθ/πθ_old, 1-ε, 1+ε) · Âᵢ,t
        ]
        - β D_KL[πθ || πref]
    }
]
```

**Components**:
- **Clipped ratio**: Same as PPO (prevents large policy changes)
- **Group advantage**: Âᵢ,t from group normalization
- **KL divergence**: Added to loss (not reward) for regularization

### KL Divergence Term

**Unbiased Estimator**:
```
D_KL[πθ || πref] = πref(oᵢ,t|...) / πθ(...) - log(πref/πθ) - 1
```

**Properties**:
- Always positive (guaranteed)
- Penalizes deviation from reference policy
- Added to loss directly (vs. reward in standard PPO)

---

## Experimental Results

### DeepSeek-Math 7B (Mathematical Reasoning)

| Benchmark | Instruct (SFT) | GRPO | Improvement |
|-----------|----------------|------|-------------|
| GSM8K (CoT) | 82.9% | **88.2%** | +5.3% |
| MATH (CoT) | 46.8% | **51.7%** | +4.9% |
| CMATH | 84.6% | **88.8%** | +4.2% |

### Key Findings

**1. Maj@K Improvement** (not Pass@K)
- GRPO boosts majority voting accuracy
- Single sample performance unchanged
- **Interpretation**: RL improves output distribution robustness, not fundamental capabilities

**2. GRPO > Online RFT**
- Both use neural reward models
- GRPO's dynamic gradient coefficients outperform RFT's binary reinforcement
- Process supervision (GRPO+PS) best overall

**3. Out-of-Domain Generalization**
- GRPO improves on benchmarks without in-domain training data
- Reward model quality critical for generalization

**4. Iterative RL Effects**
- First iteration: Significant gains
- Subsequent iterations: Diminishing returns
- Continuous reward model updates beneficial

---

## When to Use GRPO vs PPO

### ✅ Use GRPO When

**1. LLM Alignment Tasks**
- Fine-tuning large language models (7B+)
- RLHF for instruction following
- Mathematical reasoning, code generation

**2. Memory Constraints**
- Limited VRAM (can't fit policy + critic)
- Want larger batch sizes
- Training on consumer hardware

**3. Reward Models Available**
- Have or can train neural reward models
- Reward model outputs meaningful magnitudes (not just rankings)

**4. Process Supervision Possible**
- Have step-level reward annotations
- Want fine-grained training signal

**5. Group Sampling Affordable**
- Can sample multiple outputs per prompt
- Inference cost acceptable for training

### ❌ Use PPO When

**1. General RL Tasks**
- Continuous control (robotics)
- Game playing (Atari, board games)
- Non-language domains

**2. Sparse Rewards**
- Limited reward signal
- Need value function for credit assignment

**3. Immediate Rewards**
- Reward available at each timestep
- GAE beneficial for variance reduction

**4. Simpler Integration**
- Existing PPO implementations work
- Don't need memory optimization

**5. Multi-Task RL**
- Shared value function across tasks
- General-purpose RL agent

---

## Advantages of GRPO

### 1. Memory Efficiency (Primary)
- **40-50% memory reduction** (no critic)
- Enables training larger models
- Allows larger batch sizes

### 2. Computational Efficiency
- No critic forward/backward pass
- Simpler advantage computation
- Faster training iterations

### 3. Conceptual Alignment
- Group advantages match reward model training (pairwise)
- Natural for ranking/comparison tasks
- Better reward signal utilization

### 4. Performance
- Comparable or better than PPO on reasoning tasks
- Improves output distribution robustness
- Scales to larger models

### 5. Dynamic Reinforcement
- Gradient coefficients proportional to reward
- Differential treatment based on quality
- More nuanced than binary accept/reject

---

## Limitations

### 1. Requires Group Sampling
- Higher inference cost during training
- Need G outputs per question (typically G=64)
- Not suitable if sampling expensive

### 2. Reward Model Dependency
- Quality depends on reward model accuracy
- Noisy rewards can degrade performance
- Out-of-distribution reward model errors propagate

### 3. Limited to LLM Domains
- Designed for language tasks
- May not generalize to other RL domains
- PPO more general-purpose

### 4. Maj@K Improvement Only
- Doesn't improve single-sample performance
- Requires ensemble for benefits
- May not help real-time applications

### 5. Hyperparameter Sensitivity
- KL coefficient (β) tuning important
- Group size (G) affects variance
- Clipping threshold (ε) needs adjustment

---

## Connection to Other Methods

### Unified RL Framework

All methods as: **Data Source → Reward → Gradient Coefficient → Update**

| Method | Reward Source | Gradient Coefficient |
|--------|---------------|---------------------|
| **SFT** | None | GC = 1 (uniform) |
| **RFT** | Rule-based | GC = ±1 (binary) |
| **DPO** | Pairwise rules | GC from preference pairs |
| **Online RFT** | Neural RM | GC = ±1 (binary) |
| **PPO** | Neural RM | GC = GAE advantages |
| **GRPO** | Neural RM | GC = group advantages + KL |

**GRPO's Position**: Between Online RFT (too coarse) and PPO (too complex), with dynamic gradient coefficients.

---

## Interview Relevance

### For Research Scientists

**Common Questions**:
- "How does GRPO differ from PPO?"
  - Eliminates critic, uses group-relative advantages, reduces memory 40-50%
- "Why does GRPO work for LLMs?"
  - Reward models trained on comparisons align with group advantages
- "What's the trade-off?"
  - Memory savings vs. need for group sampling (higher inference cost)

**Discussion Topics**:
- Value function necessity in policy gradient methods
- Reward model alignment with advantage estimation
- Memory-efficient RL for large models
- Process vs. outcome supervision

### For ML Engineers

**Practical Considerations**:
- Framework: Can adapt existing PPO implementations
- Memory: ~50% reduction in training memory
- Sampling: Need to generate G outputs per prompt
- Hyperparameters: Very small learning rates (1e-6), KL coefficient tuning

**Applications**:
- RLHF for chatbots
- Code generation with execution feedback
- Mathematical problem solving
- Any LLM task with reward model

---

## Connection to Other Topics

- **[[PPO]]**: Foundation algorithm GRPO extends
- **[[RLHF]]**: GRPO as alternative to PPO in alignment pipeline
- **[[DPO]]**: Both avoid critic, but DPO offline vs. GRPO online
- **[[World_Models]]**: GRPO for language, world models for vision/control
- **[[RNN_Policies]]**: GRPO works with any policy architecture

---

## Quick Summary

**What**: Memory-efficient PPO variant using group-relative advantages

**Why**: Eliminate critic model (40-50% memory savings) for LLM training

**How**:
1. Sample group of outputs per question
2. Normalize rewards within group
3. Use normalized rewards as advantages
4. Update policy with clipped objective + KL

**Key Result**: Matches or beats PPO on math reasoning with half the memory

---

**See [[GRPO_detailed]] for implementation details, hyperparameters, and iterative training algorithm.**
