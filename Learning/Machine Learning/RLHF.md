# Reinforcement Learning from Human Feedback (RLHF)

## Definition
RLHF is a technique to align language models with human preferences by training a reward model from human comparisons, then using reinforcement learning to optimize the policy against this learned reward.

## Motivation
- Pre-trained LLMs predict next token but don't necessarily produce helpful/safe/honest outputs
- Hard to specify "good" output with loss function
- Humans can compare outputs (easier than rating absolutely)

## Three-Stage Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)
**Purpose**: Get model in the right format (instruction following)

**Process**:
1. Collect high-quality demonstrations
2. Fine-tune base model on these examples
3. Creates SFT model (baseline policy)

**Data**: ~10K-100K instruction-response pairs

### Stage 2: Reward Model Training
**Purpose**: Learn what humans prefer

**Data Collection**:
1. Sample prompts from dataset
2. Generate 4-9 outputs from SFT model (varying temperature/sampling)
3. Humans rank outputs: best to worst
4. Result: Preference pairs $(y_w, y_l)$ for prompt $x$ (winner, loser)

**Training Objective** (Bradley-Terry model):
$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

Where:
- $r_\theta(x, y)$ is scalar reward for response $y$ to prompt $x$
- Higher reward for preferred response
- $\sigma$ is sigmoid function

**Architecture**:
- Start from SFT model
- Replace final token prediction head with scalar output head
- Typical: 6B parameter reward model (same size or smaller than policy)

**Dataset Size**: ~50K-100K comparisons (InstructGPT used 33K)

### Stage 3: RL Fine-tuning

**Purpose**: Optimize policy to maximize reward model

**Objective**:
$$\mathcal{L}_{RL} = \mathbb{E}_{x \sim D, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \mathbb{D}_{KL}[\pi_\theta(y|x) \| \pi_{ref}(y|x)]$$

Where:
- $r_\phi(x, y)$: Learned reward model (frozen)
- $\pi_\theta$: Policy being optimized
- $\pi_{ref}$: Reference policy (SFT model, frozen)
- $\beta \mathbb{D}_{KL}$: KL penalty to prevent diverging too far from reference
- $\beta$: Coefficient (typically 0.01-0.1)

**Why KL Penalty?**
- Prevents reward hacking (gaming the reward model)
- Maintains language quality (reference model is coherent)
- Exploration-exploitation balance

#### Option A: PPO (Proximal Policy Optimization)

**Standard Approach** - Used in InstructGPT, ChatGPT

**Process**:
- Sample batch of prompts
- Generate responses with current policy
- Compute rewards (reward model - KL penalty)
- Update policy AND value function with PPO objective

**Components**:
- Policy network (actor): The LLM being trained
- Value network (critic): Estimates expected reward (same size as policy)
- Total memory: ~2× policy size

**Hyperparameters**:
- Learning rate: ~1e-6 (very small)
- Batch size: 512-1024 prompts
- PPO epochs per batch: 1-4
- KL coefficient ($\beta$): 0.01-0.02
- Clipping threshold ($\epsilon$): 0.1-0.2
- Training: ~256K prompts total

#### Option B: GRPO (Group Relative Policy Optimization)

**Memory-Efficient Alternative** - Used in DeepSeek-Math, DeepSeek-R1

**Key Difference**: Eliminates value network, uses group-relative advantages instead

**Process**:
1. Sample G outputs (typically 64) per prompt from current policy
2. Score all outputs with reward model
3. Normalize rewards within each group: $\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
4. Use normalized rewards as advantages (no learned value function)
5. Update policy with clipped objective

**Advantages over PPO**:
- **40-50% memory reduction**: No critic network
- **Faster training**: No value function updates
- **Dynamic gradient coefficients**: Reinforcement proportional to reward magnitude
- **Group alignment**: Matches how reward models are trained (comparisons)

**Disadvantages**:
- **Higher sampling cost**: Need G outputs per prompt (vs. 1 in PPO)
- **LLM-specific**: Designed for language tasks, not general RL

**When to use GRPO over PPO**:
- Training large LLMs (>7B parameters) with memory constraints
- Have good reward models
- Can afford higher inference cost during training
- Want faster convergence on reasoning tasks

**Results** (DeepSeek-Math 7B):
- GSM8K: 82.9% → 88.2% (+5.3% over SFT)
- MATH: 46.8% → 51.7% (+4.9% over SFT)
- Comparable or better than PPO with half the memory

## Key Challenges

### 1. Reward Model Quality
- **Overoptimization**: Policy exploits reward model errors at high KL
- **Solution**: Early stopping based on KL divergence, human eval
- **Typical**: Stop when KL from reference > 10-20

### 2. Distribution Shift
- **Problem**: Policy generates responses outside reward model's training distribution
- **Solution**: KL penalty, adversarial training, iterative RLHF

### 3. Reward Hacking
- **Examples**:
  - Generating very long responses (if reward model biased toward length)
  - Repeating text that gets high reward
  - Using specific phrases reward model likes
- **Solutions**: Better reward model, diverse prompts, KL penalty

### 4. Compute Cost

**With PPO**:
- Need to run multiple models during training:
  - Policy model (being trained)
  - Reference model (for KL)
  - Reward model (for scoring)
  - Value model (critic, typically same size as policy)
- **Memory**: ~2× policy size (policy + value)
- **Compute**: Typically 4× more expensive than SFT

**With GRPO**:
- Models needed:
  - Policy model (being trained)
  - Reference model (for KL)
  - Reward model (for scoring)
  - ~~Value model~~ (eliminated!)
- **Memory**: ~1× policy size (40-50% savings vs. PPO)
- **Compute**: Higher sampling cost (G outputs per prompt), but fewer model updates
- **Trade-off**: Memory for inference (more samples during training)

## Alternatives to RLHF

### 1. DPO (Direct Preference Optimization)
**Key Idea**: Skip reward model, directly optimize policy on preferences

**Objective**:
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**Advantages**:
- Simpler: no reward model or RL
- More stable training
- Less compute (only need policy and reference)
- Often comparable performance to RLHF

**Used in**: Zephyr, Llama-2, many open-source models

### 2. RLAIF (RL from AI Feedback)
- Replace human preferences with AI model preferences
- Use GPT-4 or Claude to rank outputs
- Much cheaper than human labeling
- Quality depends on AI judge quality

### 3. Constitutional AI (Anthropic)
- **Critique**: AI generates critiques of its own outputs
- **Revision**: AI revises based on critiques
- Self-improvement loop
- Reduces need for human feedback

### 4. RAFT (Reward rAnked FineTuning)
- Rank model's own outputs by reward
- Fine-tune on top-ranked outputs
- Simpler than PPO
- Iterative improvement

## Evaluation

### Reward Model
- **Accuracy**: % correctly ranked pairs on held-out data
- **Typical**: 60-70% accuracy (human agreement ~70-75%)

### Policy After RLHF
- **Win Rate**: Human preference vs SFT baseline
- **Helpfulness**: Task completion, instruction following
- **Harmlessness**: Reduced toxic/biased outputs
- **KL Divergence**: Distance from reference policy

## InstructGPT Results (OpenAI)
- 1.3B RLHF model preferred over 175B SFT model
- Significant reduction in toxicity
- Better instruction following
- Some reduction in performance on public NLP benchmarks (tradeoff)

## Interview Relevance

**Common Questions**:

### General RLHF
1. **Why RLHF?** Align with human preferences; hard to define "good" output with loss function
2. **Why three stages?** (1) SFT: get right format, (2) RM: learn preferences, (3) RL: optimize for preferences
3. **Why KL penalty?** Prevent reward hacking and maintain language quality
4. **Reward hacking examples?** Length exploitation, phrase repetition, gaming reward model
5. **RLHF vs DPO?** RLHF: two-stage (RM + RL), more compute; DPO: direct optimization, simpler
6. **Overoptimization issue?** Policy exploits RM errors at high KL; solution: early stopping

### PPO vs GRPO

7. **What is GRPO?** Group Relative Policy Optimization - memory-efficient PPO variant that eliminates value network
   - **Answer**: GRPO samples multiple outputs per prompt, normalizes rewards within groups, uses normalized rewards as advantages. Saves 40-50% memory by not needing critic.

8. **How does GRPO compute advantages without a value function?**
   - **Answer**: Sample G outputs (e.g., 64) per prompt, score with reward model, normalize: $\tilde{r}_i = (r_i - \mu) / \sigma$ within group. Use $\tilde{r}_i$ as advantage. Group statistics provide natural baseline.

9. **Why does GRPO work for LLMs?**
   - **Answer**: (1) Reward models trained on pairwise comparisons → group-relative advantages align naturally, (2) LLMs can generate many samples cheaply, (3) Memory often more limiting than inference for large models

10. **GRPO vs PPO: when to use which?**
    - **PPO**: General RL, continuous control, need value function, sparse rewards
    - **GRPO**: LLM alignment, memory constraints, mathematical reasoning, have good reward models
    - **Key trade-off**: GRPO trades sampling cost (G outputs) for memory savings (no critic)

11. **What are GRPO's gradient coefficients?**
    - **Answer**: $GC = \hat{A}_i + \beta(\pi_{ref}/\pi_\theta - 1)$ where $\hat{A}_i$ is normalized group advantage. Unlike binary reinforcement (RFT: ±1), GRPO's coefficients are continuous and proportional to reward magnitude → differential reinforcement of varying intensities.

12. **What is outcome vs process supervision in GRPO?**
    - **Outcome Supervision (OS)**: Single reward at end, all tokens get same advantage $\hat{A}_i = \tilde{r}_i$
    - **Process Supervision (PS)**: Step-level rewards, cumulative advantage $\hat{A}_{i,t} = \sum_{j=t}^T \tilde{r}_{i,j}$
    - **Result**: GRPO+PS > GRPO+OS on math reasoning (DeepSeek-Math)

13. **Why doesn't GRPO improve Pass@1, only Maj@K?**
    - **Answer**: GRPO improves output distribution robustness (ensemble accuracy) but doesn't fundamentally expand model capabilities. It refines the sampling distribution to favor correct solutions more often, benefiting majority voting.

14. **Compute requirements: PPO vs GRPO?**
    - **PPO**: 4 models (policy, reference, reward, value), 2× policy memory
    - **GRPO**: 3 models (policy, reference, reward), 1× policy memory, but G× sampling cost
    - **Trade-off**: GRPO saves memory but increases inference during training

15. **Can GRPO replace PPO in all RLHF pipelines?**
    - **Answer**: No - GRPO is LLM-specific. PPO still needed for: (1) Continuous control, (2) Sparse reward environments, (3) Non-language domains, (4) When value function provides significant variance reduction

### Advanced GRPO Topics

16. **How does iterative GRPO work?**
    - **Answer**: After each RL iteration, update reward model with new samples (10% historical replay prevents catastrophic forgetting). Set reference model to current policy. Continue training. Enables co-evolution of policy and reward model.

17. **What causes reward model exploitation in GRPO?**
    - **Answer**: Same as PPO - policy finds spurious patterns reward model associates with high scores. Mitigation: (1) KL penalty, (2) Iterative reward model updates, (3) Diverse training data, (4) Early stopping based on held-out metrics

18. **GRPO vs Online RFT (Rejection Fine-Tuning)?**
    - **Answer**: Both use neural reward models, but:
    - **RFT**: Binary gradient coefficients (GC = +1 correct, -1 incorrect)
    - **GRPO**: Continuous coefficients proportional to normalized reward
    - **Result**: GRPO outperforms Online RFT on math reasoning benchmarks

**Key Concepts**:
- **Bradley-Terry model**: Probabilistic model for pairwise preferences
- **KL divergence**: Measures how much policy deviates from reference
- **Reward hacking**: Policy exploits flaws in reward model
- **Distribution shift**: Policy generates out-of-distribution samples
- **Group-relative advantages**: Advantages computed from group statistics, not learned value function
- **Gradient coefficients**: Weights for policy gradient updates; GRPO makes them dynamic and proportional to reward

**Key Insights**:
- RLHF transforms LLMs from "text completers" to "helpful assistants" by incorporating human preferences
- PPO is standard but memory-intensive; GRPO offers 40-50% memory savings for LLM-specific tasks
- GRPO's group-relative advantages align with reward model training (pairwise comparisons)
- Choice between PPO/GRPO depends on: domain (general RL vs LLMs), resources (compute vs memory), and task (reasoning benefits more from GRPO)

**See Also**:
- **[[GRPO]]**: Detailed GRPO overview and comparison
- **[[GRPO_detailed]]**: Implementation guide with code examples
- **[[PPO]]**: Standard PPO algorithm (referenced in multiple files)
