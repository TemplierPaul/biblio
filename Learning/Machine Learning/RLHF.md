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

### Stage 3: RL Fine-tuning (PPO)
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

**Algorithm**: Proximal Policy Optimization (PPO)
- Sample batch of prompts
- Generate responses with current policy
- Compute rewards (reward model - KL penalty)
- Update policy with PPO objective

**Hyperparameters**:
- Learning rate: ~1e-6 (very small)
- Batch size: 512-1024 prompts
- PPO epochs per batch: 1-4
- KL coefficient ($\beta$): 0.01-0.02
- Training: ~256K prompts total

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
- Need to run multiple models during training:
  - Policy model (being trained)
  - Reference model (for KL)
  - Reward model (for scoring)
  - Value model (for PPO, sometimes separate)
- Typically 4x more expensive than SFT

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
1. **Why RLHF?** Align with human preferences; hard to define "good" output with loss function
2. **Why three stages?** (1) SFT: get right format, (2) RM: learn preferences, (3) RL: optimize for preferences
3. **Why KL penalty?** Prevent reward hacking and maintain language quality
4. **Reward hacking examples?** Length exploitation, phrase repetition, gaming reward model
5. **RLHF vs DPO?** RLHF: two-stage (RM + RL), more compute; DPO: direct optimization, simpler
6. **Why PPO?** On-policy algorithm, stable for LLM fine-tuning
7. **Overoptimization issue?** Policy exploits RM errors at high KL; solution: early stopping
8. **Compute requirements?** 4 models running (policy, reference, reward, value), ~4x SFT cost

**Key Concepts**:
- **Bradley-Terry model**: Probabilistic model for pairwise preferences
- **KL divergence**: Measures how much policy deviates from reference
- **Reward hacking**: Policy exploits flaws in reward model
- **Distribution shift**: Policy generates out-of-distribution samples

**Key Insight**: RLHF transforms LLMs from "text completers" to "helpful assistants" by incorporating human preferences, but requires careful tuning to avoid reward hacking.
