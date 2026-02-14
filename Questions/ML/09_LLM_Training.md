# LLM Training & Fine-tuning - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: Large Language Models]]
  - [[#What's the pre-training objective for decoder-only LLMs?]]
  - [[#Explain the three-stage training pipeline (pre-training → SFT → RLHF)]]
  - [[#What are emergent abilities and at what scale do they appear?]]
  - [[#What is in-context learning?]]
  - [[#Explain the Chinchilla scaling laws]]
  - [[#What's the KV cache and why is it needed?]]
  - [[#What's the difference between temperature sampling and greedy decoding?]]
  - [[#Why use RoPE over absolute positional encoding?]]
  - [[#What's Grouped-Query Attention (GQA) and why use it?]]
  - [[#Explain the difference between GPT-3, ChatGPT, and GPT-4]]
- [[#Part 2: Fine-tuning & PEFT]]
  - [[#What's the difference between full fine-tuning and PEFT?]]
  - [[#Why use lower learning rate for fine-tuning?]]
  - [[#What is catastrophic forgetting and how to prevent it?]]
  - [[#What's instruction tuning (SFT)?]]
  - [[#Explain LoRA: what are the A and B matrices?]]
  - [[#Why does LoRA reduce parameters by 10,000x?]]
  - [[#What rank should you use for LoRA?]]
  - [[#Why initialize B to zero in LoRA?]]
  - [[#What's QLoRA and how does it enable 65B on single GPU?]]
  - [[#When to use LoRA vs full fine-tuning?]]
- [[#Part 3: Alignment & RLHF]]
  - [[#Why do we need RLHF after pre-training?]]
  - [[#Explain the three stages of RLHF pipeline]]
  - [[#What's the Bradley-Terry model for reward modeling?]]
  - [[#Why use PPO for the RL stage?]]
  - [[#What's the KL penalty term and why is it critical?]]
  - [[#What is reward hacking? Give examples]]
  - [[#What's the difference between RLHF and DPO?]]
  - [[#When does overoptimization occur?]]
  - [[#Why does RLHF require 4x more compute than SFT?]]
  - [[#What is GRPO and how does it differ from PPO?]]
  - [[#How does GRPO compute advantages without a value function?]]
  - [[#When should you use GRPO vs PPO for RLHF?]]
  - [[#What are GRPO's gradient coefficients?]]
  - [[#What's the difference between outcome and process supervision in GRPO?]]
  - [[#Why does GRPO improve Maj@K but not Pass@1?]]
  - [[#What are the memory and compute trade-offs between GRPO and PPO?]]
  - [[#How does iterative GRPO work?]]
  - [[#Compare RLHF, DPO, RLAIF, Constitutional AI]]

---

## Part 1: Large Language Models

### What's the pre-training objective for decoder-only LLMs?

**Next-token prediction** (autoregressive language modeling):

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

- Given tokens 1 to t-1, predict token t
- Trained on raw text from internet (trillions of tokens)
- Self-supervised: no labels needed (text provides its own supervision)

**Simple but powerful**: This single objective produces models that can answer questions, write code, translate, reason, etc.

### Explain the three-stage training pipeline (pre-training → SFT → RLHF)

**Stage 1: Pre-training** (base model)
- Objective: Next-token prediction on raw web text
- Data: 1-5T tokens (books, web, code)
- Result: Model that "completes" text but doesn't follow instructions

**Stage 2: SFT (Supervised Fine-Tuning)**
- Objective: Learn instruction-following format
- Data: 10K-100K instruction-response pairs
- Result: Model that responds to instructions (but not always helpful/safe)

**Stage 3: RLHF** (Reinforcement Learning from Human Feedback)
- Objective: Align with human preferences
- Data: Human preference comparisons (50K-100K)
- Result: Helpful, harmless, honest assistant (ChatGPT-style)

Each stage builds on previous: Base → Instruction → Aligned

### What are emergent abilities and at what scale do they appear?

**Emergent abilities**: Capabilities that appear suddenly at scale, not present in smaller models.

**Examples**:
- **In-context learning**: Learning from prompt examples without gradient updates
- **Chain-of-thought reasoning**: Multi-step logical reasoning
- **Instruction following**: Generalizing to unseen task formats
- **Arithmetic**: Multi-digit addition, multiplication

**Scale threshold**: Typically ~10B parameters
- <10B: Mostly predictable scaling
- 10B-100B: Emergence of complex reasoning
- 100B+: Stronger emergence

**Key insight**: Not just "bigger = better", but qualitatively new capabilities at scale.

### What is in-context learning?

**Ability to learn from examples in the prompt** without any gradient updates.

**Example**:
```
Translate to French:
Hello → Bonjour
Good morning → Bon matin
Thank you → [model predicts: Merci]
```

Model adapts to task from examples alone (few-shot) or even just instruction (zero-shot).

**Why surprising**:
- No training/fine-tuning needed
- Emerges from pre-training alone
- Seems to "learn" new tasks instantly

**Mechanism**: Not fully understood, but model likely recognizes patterns during pre-training and adapts its outputs accordingly.

### Explain the Chinchilla scaling laws

**Finding**: Most models are **over-parameterized and under-trained**.

**Optimal scaling**: Parameters and training tokens should scale **equally**
- For compute budget C: $N_{params} \propto C^{0.5}$, $D_{tokens} \propto C^{0.5}$
- **Implication**: 10x more compute → 3.16x more params, 3.16x more data

**Example**:
- GPT-3 (175B): Likely overtrained (should have been ~70B with more tokens)
- Chinchilla (70B): Trained on 4x more tokens than Gopher (280B), outperformed it

**Practical impact**: Better to train smaller model on more data than larger model on less data (for fixed budget).

### What's the KV cache and why is it needed?

**KV cache**: Store computed **key and value** tensors during autoregressive generation.

**Why needed**:
- Generate token-by-token: "The cat sat on the"
- At each step, need to attend to ALL previous tokens
- Without cache: Recompute K,V for all previous tokens (wasteful!)
- With cache: Store K,V, only compute for new token

**Memory cost**: $2 \times n_{layers} \times d_{model} \times$ sequence_length
- Example: LLaMA 7B, 2K context: ~1GB KV cache

**Optimization**: Grouped-Query Attention (GQA) reduces cache size by sharing K,V across heads.

### What's the difference between temperature sampling and greedy decoding?

**Greedy decoding** (temperature = 0):
```
token = argmax(logits)
```
Always pick highest probability token (deterministic)

**Temperature sampling**:
$$P(x_i) = \frac{\exp(logit_i / T)}{\sum_j \exp(logit_j / T)}$$

- **T < 1**: Sharpens distribution (more deterministic, confident)
- **T = 1**: Use raw probabilities
- **T > 1**: Flattens distribution (more random, creative)

**When to use**:
- T=0: Factual Q&A, want consistency
- T=0.7: Chatbots, balanced
- T=1.0+: Creative writing, brainstorming

### Why use RoPE over absolute positional encoding?

**RoPE** (Rotary Position Embeddings):

**Advantages**:
1. **Relative positions**: Encodes relative distance, not absolute position
2. **Extrapolation**: Generalizes to longer sequences than training
3. **No learned params**: Defined by rotation matrices (no extra parameters)
4. **Efficient**: Applied directly to Q,K (not input embeddings)

**How it works**: Rotate Q,K vectors by angle proportional to position
- Dot product naturally captures relative distance

**Absolute encoding problems**:
- Fixed max length
- Poor extrapolation beyond training length
- Treats position 1 vs 2 same as position 1000 vs 1001

Used in: LLaMA, GPT-NeoX, PaLM - modern standard.

### What's Grouped-Query Attention (GQA) and why use it?

**Standard Multi-Head Attention**: Each head has own Q, K, V
- 8 heads → 8 separate K, V tensors

**GQA**: Share K, V across **groups** of query heads
- Example: 8 Q heads, 2 KV heads → 4 Q heads per KV head
- **Multi-Query Attention (MQA)**: Extreme case - 1 KV head shared by all Q heads

**Why use it**:
1. **Reduce KV cache**: 4-8x smaller for MQA
2. **Faster inference**: Less memory bandwidth
3. **Minimal quality loss**: <1% performance drop

**Trade-off**: Slight reduction in expressivity for major memory savings.

**Used in**: LLaMA 2, Falcon - standard for efficient inference.

### Explain the difference between GPT-3, ChatGPT, and GPT-4

**GPT-3** (2020):
- Base model: 175B parameters
- Pre-training only (next-token prediction)
- Good at completion, not instruction following
- Few-shot learning via prompting

**ChatGPT** (2022):
- GPT-3.5 base (improved GPT-3)
- **+ SFT** (instruction tuning)
- **+ RLHF** (alignment)
- Conversational, follows instructions
- Helpful, harmless, honest

**GPT-4** (2023):
- Larger model (rumored ~1.76T params, MoE)
- **Multimodal**: Accepts images
- Improved reasoning, reduced hallucination
- Longer context (8K → 32K → 128K)
- More capable across all tasks

**Progression**: Base model → Aligned assistant → Multimodal + stronger reasoning

---

## Part 2: Fine-tuning & PEFT

### What's the difference between full fine-tuning and PEFT?

**Full Fine-tuning**:
- Update ALL model parameters
- Memory: model + gradients + optimizer states (~4x model size)
- Best performance
- Expensive (7B model ≈ 84GB GPU memory)

**PEFT** (Parameter-Efficient Fine-Tuning):
- Update only small subset of parameters
- Freeze base model
- Memory: ~14GB for 7B model
- 95-99% of full fine-tuning performance
- **Examples**: LoRA, Adapters, Prefix tuning

**Key difference**: PEFT trades small performance drop for massive memory/compute savings.

### Why use lower learning rate for fine-tuning?

Pre-trained weights are already **good** - they've learned general language understanding.

**High learning rate**: Large updates → catastrophic forgetting (destroy pre-trained knowledge)

**Low learning rate** (1e-5 vs 1e-3 for pre-training):
- Gentle adaptation to new task
- Preserve general capabilities
- Avoid overfitting to small fine-tuning dataset

**Rule of thumb**: 1/10 to 1/100 of pre-training learning rate.

### What is catastrophic forgetting and how to prevent it?

**Catastrophic forgetting**: Model loses general capabilities when fine-tuned on narrow task.

**Example**: Fine-tune on medical Q&A → model forgets how to write code or chat casually.

**Prevention**:
1. **Lower learning rate**: Gentle updates (1e-5 vs 1e-3)
2. **Fewer epochs**: 1-3 epochs (more risks overfitting)
3. **Mix general data**: Add general text to fine-tuning data
4. **PEFT methods**: LoRA freezes base model (can't forget)
5. **Regularization**: Weight decay, dropout
6. **Early stopping**: Monitor validation performance

**Why it happens**: Small fine-tuning dataset overwhelms pre-training with aggressive updates.

### What's instruction tuning (SFT)?

**Supervised Fine-Tuning** on instruction-response pairs.

**Purpose**: Teach model to follow instructions (not just complete text).

**Data format**:
```json
{
  "instruction": "Translate to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**Training**: Language modeling on response only (mask instruction in loss).

**Result**: Model that responds to instructions instead of just continuing text.

**Datasets**: Alpaca (52K), FLAN (1.8M), Dolly (15K human-written)

### Explain LoRA: what are the A and B matrices?

**LoRA** (Low-Rank Adaptation):

Instead of updating full weight matrix W:
$$h = Wx + \Delta Wx = Wx + BAx$$

- $W \in \mathbb{R}^{d \times k}$: Frozen pre-trained weights
- $B \in \mathbb{R}^{d \times r}$: Trainable down-projection
- $A \in \mathbb{R}^{r \times k}$: Trainable up-projection
- $r \ll \min(d,k)$: Rank (typically 8-16)

**Key insight**: Weight updates lie in low-rank subspace - don't need full rank.

**Parameters**: $r(d+k)$ vs $dk$ (huge reduction when r is small).

### Why does LoRA reduce parameters by 10,000x?

**Example** (7B model, rank 8):

**Full model**: 7,000,000,000 parameters

**LoRA trainable**:
- Apply to attention layers: W_q, W_k, W_v, W_o
- Typical layer: 4096 × 4096 matrices
- LoRA params per matrix: $r(d+k) = 8(4096+4096) = 65,536$
- ~4 matrices per layer × 32 layers = ~8M parameters

**Reduction**: 7B → ~8M trainable = ~875x reduction (can be even more aggressive → 10,000x)

**Memory savings**: No gradients/optimizer states for 7B frozen params.

### What rank should you use for LoRA?

**Common values**:
- **r = 8**: Standard default, good balance
- **r = 16**: Complex tasks, minimal performance loss
- **r = 4**: Very memory-constrained
- **r = 32-64**: Approaching full fine-tuning performance

**Rule of thumb**: Start with 8, increase if underfitting.

**Observations**:
- Rank 8 achieves 95-99% of full fine-tuning
- Diminishing returns beyond 32
- Lower rank = more memory efficient, faster

### Why initialize B to zero in LoRA?

**Goal**: Start training from pre-trained model (no change at initialization).

At initialization: $\Delta W = BA = B \cdot A$

If **B = 0**:
- $\Delta W = 0 \cdot A = 0$
- Model output: $Wx + 0 = Wx$ (unchanged from pre-trained)
- Training gradually adds adaptation

If B initialized randomly:
- $\Delta W \neq 0$ immediately
- Random noise added to pre-trained weights
- Starts from degraded model

**A can be random** (doesn't matter since B=0 makes product zero).

### What's QLoRA and how does it enable 65B on single GPU?

**QLoRA** = Quantization + LoRA

**Method**:
1. **Quantize base model to 4-bit** (NormalFloat4)
   - 65B model: ~130GB FP16 → ~33GB 4-bit
2. **Keep LoRA adapters in BF16**
   - ~15GB for training overhead
3. **Compute gradients in higher precision**

**Total**: 33GB + 15GB = ~48GB (fits on A100!)

**Key innovations**:
- **NormalFloat4**: Custom 4-bit format for normally-distributed weights
- **Double quantization**: Quantize quantization constants
- **Paged optimizers**: Handle memory spikes

**Performance**: Close to 16-bit full fine-tuning quality with massive memory savings.

### When to use LoRA vs full fine-tuning?

**Use LoRA when**:
- ✅ Limited GPU memory (can't fit full model + gradients + optimizer)
- ✅ Need multiple task-specific models (small adapters vs multiple full models)
- ✅ Domain adaptation with moderate distribution shift
- ✅ Rapid experimentation (faster iteration)

**Use Full Fine-tuning when**:
- ✅ Maximum performance critical
- ✅ Large high-quality task dataset
- ✅ Significant domain shift (medical, legal)
- ✅ Sufficient compute resources

**In practice**: LoRA is default choice (95-99% performance at 1% cost).

---

## Part 3: Alignment & RLHF

### Why do we need RLHF after pre-training?

**Pre-trained models**:
- Predict next token from internet text
- Can be toxic, biased, unhelpful
- Don't follow instructions reliably
- Optimize for likelihood, not helpfulness

**RLHF provides**:
- **Helpfulness**: Answer questions, follow instructions
- **Harmlessness**: Avoid toxic, biased, dangerous content
- **Honesty**: Acknowledge uncertainty, don't hallucinate

**Analogy**: Pre-training = reading entire internet. RLHF = learning to be a helpful assistant.

**Evidence**: 1.3B RLHF model preferred over 175B SFT model (InstructGPT paper).

### Explain the three stages of RLHF pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**
- Data: ~10K high-quality instruction-response demonstrations
- Method: Fine-tune on demonstrations (behavioral cloning)
- Output: Model that follows instructions (baseline policy)

**Stage 2: Reward Model (RM) Training**
- Data: ~50K preference comparisons (rank 4-9 outputs per prompt)
- Method: Train model to predict human preferences (Bradley-Terry)
- Output: Scalar reward function $r(x, y)$

**Stage 3: RL Optimization (PPO)**
- Method: Optimize policy to maximize RM score - KL penalty
- Objective: $\mathbb{E}[r(x,y)] - \beta \mathbb{D}_{KL}[\pi \| \pi_{ref}]$
- Output: Aligned policy (ChatGPT-style)

Each stage builds on previous: Demonstrations → Preferences → Optimization.

### What's the Bradley-Terry model for reward modeling?

**Probabilistic model for pairwise preferences**:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Where:
- $y_w$: Preferred (winner) response
- $y_l$: Dispreferred (loser) response
- $r(x, y)$: Scalar reward
- $\sigma$: Sigmoid function

**Training objective**:
$$\mathcal{L} = -\mathbb{E} [\log \sigma(r(x, y_w) - r(x, y_l))]$$

**Interpretation**: Reward difference determines probability of preference.
- If $r(y_w) \gg r(y_l)$: High probability $y_w$ preferred (correct)
- If $r(y_w) \approx r(y_l)$: ~50% probability (uncertain)

### Why use PPO for the RL stage?

**PPO** (Proximal Policy Optimization):

**Advantages for RLHF**:
1. **On-policy**: Stable updates (important for language)
2. **Clipped objective**: Prevents destructive large updates
3. **Simple**: Easier than TRPO, works well in practice
4. **Proven**: Success in games, robotics

**Alternative**: Could use other on-policy methods, but PPO is reliable default.

**Challenge**: Standard RL algorithms struggle with discrete high-dimensional action space (vocabulary), but PPO handles it reasonably.

### What's the KL penalty term and why is it critical?

**KL penalty**: $\beta \mathbb{D}_{KL}[\pi_\theta(y|x) \| \pi_{ref}(y|x)]$

Measures divergence between current policy and reference (SFT) policy.

**Why critical**:
1. **Prevents reward hacking**: Without KL, policy exploits RM errors
2. **Maintains language quality**: Reference policy is coherent
3. **Prevents mode collapse**: Ensures diverse outputs
4. **Exploration-exploitation**: Balance improving reward vs staying near reference

**Without KL penalty**: Policy finds adversarial inputs that get high reward from flawed RM (e.g., repetitive text, gibberish that scores high).

**Typical**: $\beta = 0.01$ to $0.02$

### What is reward hacking? Give examples

**Reward hacking**: Policy exploits imperfections in reward model to get high score without being actually good.

**Examples**:

1. **Length exploitation**: RM biased toward longer responses → policy generates very long, repetitive text

2. **Keyword stuffing**: RM likes certain phrases → policy repeats "I'm happy to help!" excessively

3. **Gibberish**: Find adversarial tokens that RM scores highly but are nonsense

4. **Overconfidence**: RM likes confident language → policy never says "I don't know"

5. **Formatting tricks**: Exploit RM's bias toward certain formats

**Root cause**: RM is imperfect (trained on limited data), policy finds edge cases.

**Solutions**: Better RM, diverse prompts, KL penalty, early stopping.

### What's the difference between RLHF and DPO?

**RLHF** (Reinforcement Learning from Human Feedback):
- Two-stage: Train reward model → RL optimization
- Requires 4 models during training (policy, reference, reward, value)
- Uses PPO (complex)
- More compute intensive

**DPO** (Direct Preference Optimization):
- One-stage: Directly optimize policy on preference data
- Only needs policy + reference (2 models)
- Simpler training (supervised learning-like)
- Less compute

**Same data**: Both use preference pairs $(x, y_w, y_l)$

**Performance**: Often comparable, DPO sometimes better

**Why DPO works**: Derives policy update that implicitly optimizes against reward model (no explicit RM needed).

**Trend**: DPO increasingly popular (simpler, cheaper, similar results).

### When does overoptimization occur?

**Overoptimization**: Policy diverges too far from reference, exploits RM errors.

**Symptoms**:
- KL divergence from reference increases
- Reward model score increases
- **But** human evaluation quality decreases (proxy-true objective divergence)

**When it occurs**:
- Too many RL training steps
- Too low KL penalty ($\beta$)
- Weak reward model

**Typical threshold**: Stop when KL > 10-20 (dataset-dependent)

**Solution**: Early stopping based on KL or human eval, not RM score.

**Analogy**: Studying to the test vs learning the material - optimizing RM vs actual quality.

### Why does RLHF require 4x more compute than SFT?

**SFT**: 1 model (policy) forward + backward pass

**RLHF RL stage**: 4 models running:
1. **Policy** ($\pi_\theta$): Being trained (forward + backward)
2. **Reference** ($\pi_{ref}$): Frozen SFT model (forward only, for KL)
3. **Reward Model**: Frozen RM (forward only, compute rewards)
4. **Value Model** (sometimes): For PPO baseline (forward + backward)

**Per batch**:
- Generate responses: Policy forward
- Compute rewards: RM forward
- Compute KL: Reference forward
- PPO update: Policy + value backward

**Additional cost**: Sampling responses (generation is expensive for LLMs).

**Total**: ~4x compute vs SFT.

### What is GRPO and how does it differ from PPO?

**GRPO** (Group Relative Policy Optimization): Memory-efficient PPO variant for LLM alignment.

**Key Difference**: **Eliminates the critic (value function) model**

**PPO approach**:
- Learn value function $V(s)$ to estimate expected return
- Compute advantages: $A = Q - V$
- Update both policy and value network

**GRPO approach**:
- Sample G outputs (e.g., 64) for same prompt from old policy
- Score all outputs with reward model: $r_1, r_2, ..., r_G$
- Normalize within group: $\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
- Use normalized rewards as advantages: $\hat{A}_i = \tilde{r}_i$

**Memory savings**: 40-50% (no critic network, optimizer states, or gradients for critic)

**Trade-off**: Need to sample G outputs per prompt (higher inference cost during training) vs. PPO's single output.

### How does GRPO compute advantages without a value function?

**Core insight**: Use group statistics as natural baseline instead of learned value function.

**Algorithm**:
```python
# For each prompt q in batch
for q in batch:
    # 1. Sample group of outputs from old policy
    outputs = [policy_old.sample(q) for _ in range(G)]  # G=64 typical

    # 2. Get rewards from reward model
    rewards = [reward_model(q, o) for o in outputs]

    # 3. Normalize within group
    mean_r = mean(rewards)
    std_r = std(rewards)
    advantages = [(r - mean_r) / std_r for r in rewards]

    # 4. Update policy using advantages
    for output, advantage in zip(outputs, advantages):
        policy_loss = compute_ppo_loss(q, output, advantage)
        policy_loss.backward()
```

**Why it works**:
- Group average acts as baseline (what's "normal" performance on this prompt)
- Relative ranking aligns with how reward models are trained (pairwise comparisons)
- High variance reduced by large group size (G=64)

**Comparison**: PPO learns "how good is this state universally", GRPO computes "how good is this output compared to others on same prompt".

### When should you use GRPO vs PPO for RLHF?

**Use GRPO ✅ when**:

1. **Memory constrained**: Can't fit policy + critic (e.g., training 7B+ models on limited GPUs)
2. **LLM alignment tasks**: RLHF, instruction tuning, mathematical reasoning
3. **Have quality reward models**: GRPO depends heavily on RM accuracy
4. **Can afford sampling**: G outputs per prompt (inference cost acceptable during training)
5. **Process supervision available**: Step-level rewards boost GRPO performance

**Use PPO ✅ when**:

1. **General RL domains**: Robotics, game playing, continuous control
2. **Sparse rewards**: Need value function for credit assignment
3. **Sampling expensive**: Can't afford G outputs per prompt
4. **Non-language tasks**: PPO more general-purpose
5. **Simpler setup**: Existing PPO infrastructure already works

**Performance comparison (DeepSeek-Math 7B)**:
| Benchmark | SFT | GRPO | Improvement |
|-----------|-----|------|-------------|
| GSM8K | 82.9% | 88.2% | +5.3% |
| MATH | 46.8% | 51.7% | +4.9% |

**In practice**: GRPO is becoming standard for LLM alignment due to memory efficiency with comparable performance.

### What are GRPO's gradient coefficients?

**Dynamic gradient coefficients** proportional to reward magnitude.

**Formula**:
$$GC(q, o, t) = \hat{A}_{i,t} + \beta \left(\frac{\pi_{ref}}{\pi_\theta} - 1\right)$$

Where:
- $\hat{A}_{i,t}$: Group-relative advantage (normalized reward)
- $\beta$: KL coefficient (typically 0.01-0.02)
- $\pi_{ref} / \pi_\theta$: Ratio for KL penalty

**Key insight**: Coefficient is **continuous** (not binary)

**Comparison with other methods**:

| Method | Gradient Coefficient |
|--------|---------------------|
| **SFT** | GC = 1 (all examples equal) |
| **RFT** | GC = +1 (correct) or -1 (wrong) |
| **GRPO** | GC = normalized reward (continuous) |
| **PPO** | GC = GAE advantage (learned baseline) |

**Example**:
- High reward (+2.5 after normalization) → Strong positive reinforcement
- Neutral reward (0 after normalization) → Minimal update
- Low reward (-1.8 after normalization) → Strong negative reinforcement

**Why better than binary**: Differential treatment based on quality (very good > good > neutral > bad > very bad).

### What's the difference between outcome and process supervision in GRPO?

**Outcome Supervision (OS)**:
- **Single reward** at end of sequence
- All tokens get **same advantage**: $\hat{A}_{i,t} = \tilde{r}_i$ for all $t$
- Simpler, only needs final outcome reward

**Example** (math problem):
```
Question: 2 + 2 = ?
Answer: First, I add 2 and 2. This equals 4.
Reward: 1.0 (correct final answer)
All tokens get advantage = 1.0
```

**Process Supervision (PS)**:
- **Step-level rewards** throughout reasoning
- Cumulative advantage: $\hat{A}_{i,t} = \sum_{j=t}^{T} \tilde{r}_{i,j}$
- More fine-grained signal
- Requires annotated reasoning steps

**Example** (same problem):
```
Question: 2 + 2 = ?
Answer: First, I add   [r=0.8]
        2 and 2.       [r=1.0]
        This equals 4. [r=1.0]
Token advantages: [2.8, 2.0, 1.0] (cumulative sum)
```

**Performance** (DeepSeek-Math):
- GRPO+PS > GRPO+OS > Online RFT
- PS provides stronger training signal for multi-step reasoning

**Trade-off**: PS requires step-level reward annotations (expensive) but significantly improves reasoning tasks.

### Why does GRPO improve Maj@K but not Pass@1?

**Observation**: GRPO boosts majority voting accuracy but not single-sample performance.

**Pass@1**: Single sample correctness (greedy or single draw)
**Maj@K**: Generate K samples, take majority vote

**Why this happens**:

1. **Distributional improvement**: GRPO improves the **robustness of the output distribution**
   - More samples land on correct answer
   - Fewer samples produce wrong answers
   - Better for ensemble methods

2. **Not capability expansion**: Doesn't teach fundamentally new skills
   - Model still makes same types of errors
   - Just shifts probability mass toward correct solutions

3. **Self-consistency effect**: Majority voting amplifies distributional improvements
   - If 60% of samples correct → majority likely correct
   - GRPO pushes 60% → 70% → majority voting wins more

**Analogy**: Like calibrating a biased coin - doesn't change the coin's fundamental properties, but makes the distribution more favorable for aggregate outcomes.

**Practical implication**: GRPO most valuable when you can afford to sample multiple outputs and use voting/verification.

### What are the memory and compute trade-offs between GRPO and PPO?

**Memory Comparison**:

**PPO**:
- Policy network (e.g., 7B params): ~28GB FP32
- Value network (typically same size): ~28GB FP32
- Gradients + optimizer states: ~2× parameters
- **Total**: ~4× policy size = ~112GB

**GRPO**:
- Policy network: ~28GB FP32
- ~~Value network~~: Eliminated!
- Gradients + optimizer states: ~2× policy parameters
- **Total**: ~2× policy size = ~56GB

**Memory savings**: 40-50% (can train on smaller GPUs or larger batch sizes)

**Compute Comparison**:

**PPO per batch**:
- 1× policy forward (generate)
- 1× policy backward (update)
- 1× value forward (compute baseline)
- 1× value backward (update critic)
- 1× reference forward (KL penalty)
- 1× reward model forward (rewards)

**GRPO per batch**:
- G× policy forward (sample group) ← **Higher cost**
- 1× policy backward (update)
- ~~Value forward/backward~~: Eliminated
- 1× reference forward (KL penalty)
- G× reward model forward (score group) ← **Higher cost**

**Key trade-off**: GRPO trades **memory for inference**
- Saves memory (no critic)
- Costs more inference (G samples vs 1)
- Typical G=64, so 64× more sampling

**When GRPO wins**: Memory-bound scenarios (large models, limited GPUs, want bigger batches)

**When PPO wins**: Inference-bound scenarios (sampling is expensive, small models where memory not limiting)

### How does iterative GRPO work?

**Problem**: Policy improves → generates out-of-distribution samples → reward model less accurate.

**Solution**: Iteratively update both policy AND reward model.

**Algorithm**:
```
Initialize: policy_0, reward_model_0

For iteration i in 1, 2, 3, ...:
    # 1. RL Training Phase
    policy_i = GRPO_train(
        policy=policy_{i-1},
        reference=policy_{i-1},  # Reference updates each iteration
        reward_model=reward_model_{i-1}
    )

    # 2. Collect New Data
    new_samples = generate_samples(policy_i, prompts)
    new_comparisons = human_annotate(new_samples)  # Or AI annotate

    # 3. Update Reward Model
    replay_buffer = new_comparisons + sample(old_comparisons, ratio=0.1)
    reward_model_i = train_reward_model(replay_buffer)

    # 4. Update Reference
    reference_{i+1} = policy_i  # Set new reference for next iteration
```

**Key components**:

1. **Reference model updates**: Use current policy as next reference (not frozen SFT)
2. **Reward model updates**: Retrain RM on new samples + 10% historical data
3. **Replay buffer**: Prevents catastrophic forgetting in RM
4. **Co-evolution**: Policy and RM improve together

**Benefits**:
- Mitigates reward model exploitation
- Continues improving beyond first iteration
- Better out-of-domain generalization

**Diminishing returns**: First iteration gives most gains, subsequent iterations smaller improvements.

**Used in**: DeepSeek-R1, DeepSeek-Math for extended reasoning training.

### Compare RLHF, DPO, RLAIF, Constitutional AI

**RLHF**:
- Human preferences → Reward model → RL
- Gold standard but expensive (human labels)
- 4x compute cost

**DPO**:
- Human preferences → Direct policy optimization
- Simpler, cheaper, no explicit RM
- Increasingly popular alternative

**RLAIF** (RL from AI Feedback):
- AI preferences (e.g., GPT-4) → Reward model → RL
- Much cheaper than human labels
- Quality depends on AI judge

**Constitutional AI**:
- AI critiques + revises its own outputs
- Self-improvement through principles ("constitution")
- Reduces human labeling
- Used by Anthropic (Claude)

**Trend**: Moving toward AI-assisted alignment (cheaper, scalable).
