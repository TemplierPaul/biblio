# LLM Training & Fine-tuning - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

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
