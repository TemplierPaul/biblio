# Interview Questions & Answers

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---

## Part 1: LSTM

### What problem does LSTM solve compared to vanilla RNN?

**Vanishing gradients** in long sequences. Standard RNNs multiply gradients through time, causing exponential decay: $\frac{\partial h_t}{\partial h_{t-k}} \to 0$ as $k$ increases. This prevents learning dependencies beyond ~10 steps.

LSTM solves this with **additive cell state updates**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. Gradients flow through addition (not multiplication), creating a "gradient highway" that preserves information over hundreds of steps.

### Explain the three gates in LSTM and their purposes (forget, input, output)

1. **Forget gate** ($f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$): Controls what information to **remove** from cell state. Output 0 = forget completely, 1 = keep completely.

2. **Input gate** ($i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$): Controls how much **new information** to add to cell state. Works with candidate values $\tilde{C}_t$.

3. **Output gate** ($o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$): Controls what information from cell state to **expose** as hidden state output.

All three use sigmoid activation (0-1 range) to modulate information flow.

### Write the cell state update equation and explain why it prevents vanishing gradients

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Why it prevents vanishing gradients**:
- **Additive** update (not multiplicative like RNN)
- Gradient: $\frac{\partial C_t}{\partial C_{t-1}} = f_t$ (elementwise)
- If forget gate saturates near 1, gradient ≈ 1 (flows unchanged)
- Creates "highway" for gradients to travel back through time
- Contrast with RNN: $\frac{\partial h_t}{\partial h_{t-1}} = W^T \sigma'$ (repeated multiplication causes exponential decay)

### What's the difference between cell state and hidden state?

- **Cell state** ($C_t$): Long-term memory, flows through additive updates, not directly exposed to outside
- **Hidden state** ($h_t = o_t \odot \tanh(C_t)$): Short-term memory, filtered version of cell state, output to next layer

Think: Cell state is internal memory, hidden state is what you share with others.

### LSTM vs GRU: when to use each?

**GRU** (Gated Recurrent Unit):
- 2 gates (update, reset) vs LSTM's 3
- No separate cell state
- Fewer parameters: ~25% less than LSTM
- Faster training and inference

**When to use**:
- **GRU**: Smaller datasets, limited compute, similar performance in many tasks
- **LSTM**: Longer sequences, more complex dependencies, slightly better on some tasks

**In practice**: Try both, GRU is a good default for efficiency.

### Why do transformers dominate over LSTMs in NLP today?

1. **Parallelization**: LSTMs process sequentially (can't parallelize across time). Transformers process entire sequence in parallel → 10-100x faster training
2. **Long-range dependencies**: Direct connections via attention (O(1) path) vs O(n) for LSTM
3. **No vanishing gradients**: Even with LSTM's improvements, very long sequences still degrade
4. **Scalability**: Transformers scale better with data and compute
5. **Transfer learning**: Pre-trained transformers (BERT, GPT) transfer extremely well

### When would you still use LSTM over transformer?

1. **Streaming/online data**: Process one token at a time with O(1) memory (transformer needs full sequence)
2. **Very long sequences**: O(n) memory vs O(n²) for transformer attention
3. **Edge deployment**: Lower memory footprint
4. **Time series forecasting**: Inherently sequential data where recurrence is natural
5. **Small datasets**: Less prone to overfitting than transformers

---

## Part 2: Attention Mechanism

### What is the scaled dot-product attention formula?

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (queries): What we're looking for (n × d_k)
- $K$ (keys): What we're matching against (m × d_k)
- $V$ (values): What we retrieve (m × d_v)
- Output: n × d_v weighted values

### Why do we scale by √d_k?

Without scaling, dot products $QK^T$ grow large in magnitude as dimension $d_k$ increases (variance of dot product ∝ d_k).

Large dot products → softmax saturates → extremely small gradients → training difficulties.

Scaling by $\sqrt{d_k}$ keeps dot products in reasonable range (~N(0,1) if Q,K are normalized), ensuring softmax doesn't saturate.

**Example**: If $d_k = 512$, dot products could be ±200+. After dividing by √512 ≈ 22.6, they're ~±10 (good range for softmax).

### What are Q, K, V matrices and what do they represent?

**Database lookup analogy**:
- **Query (Q)**: What you're searching for (your search terms)
- **Key (K)**: Indexed fields to match against (database keys)
- **Value (V)**: Actual content to retrieve (database records)

**In attention**:
- Compute compatibility: $QK^T$ (how well each query matches each key)
- Get attention weights: softmax (which keys are most relevant)
- Retrieve weighted values: multiply by V (get relevant information)

All three are linear projections of input: $Q=XW^Q, K=XW^K, V=XW^V$

### Difference between self-attention and cross-attention?

**Self-attention**: Q, K, V all come from **same sequence**
- Each position attends to all positions in same sequence
- Used in: Transformer encoder, decoder (masked)
- Example: "The cat sat on the mat" - each word attends to all other words

**Cross-attention**: Q from **one sequence**, K and V from **another**
- Decoder attends to encoder outputs
- Used in: Encoder-decoder models (translation, captioning)
- Example: Translation - French query attends to English keys/values

### What is multi-head attention and why use it?

Instead of single attention, use **h parallel heads** with different learned projections:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, \ldots, head_h)W^O$$
$$head_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

**Why use it**:
- Each head can learn different relationships (syntax, semantics, coreference, etc.)
- Attend to different positions and representation subspaces
- More expressive than single attention
- Typical: 8-16 heads in transformers

**Analogy**: Like having multiple experts, each looking at the data from different perspectives.

### What's the computational complexity of attention? What's the bottleneck?

**Time complexity**: $O(n^2 d)$
- Compute $QK^T$: O(n² d_k)
- Softmax: O(n²)
- Multiply by V: O(n² d_v)
- Dominated by $n^2$ term

**Space complexity**: $O(n^2)$ for attention matrix

**Bottleneck**: Quadratic scaling with sequence length $n$
- 512 tokens: 262K attention scores
- 2048 tokens: 4.2M attention scores (16x more!)
- Limits: Can't easily do 100K+ token sequences

**Solutions**: Sparse attention, linear attention, Flash Attention (memory optimization)

### How does attention compare to RNN for long-range dependencies?

**Path length**:
- **RNN**: O(n) - information flows sequentially through hidden states
- **Attention**: O(1) - direct connection between any two positions

**Example**: Word at position 1 connecting to word at position 100:
- RNN: Goes through 99 intermediate steps (information degrades)
- Attention: Direct connection via attention weight

**Trade-offs**:
- Attention: Better at long-range, but O(n²) complexity
- RNN: O(n) complexity, but struggles with long dependencies

This is why transformers revolutionized NLP - direct modeling of long-range dependencies.

---

## Part 3: Transformer Architecture

### Draw transformer encoder and decoder architecture

**Encoder** (one layer):
```
Input Embeddings + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Norm (residual)
    ↓
Feed-Forward Network (2 linear layers)
    ↓
Add & Norm (residual)
    ↓
[Repeat 6 times]
```

**Decoder** (one layer):
```
Output Embeddings + Positional Encoding
    ↓
Masked Multi-Head Self-Attention (causal)
    ↓
Add & Norm
    ↓
Multi-Head Cross-Attention (to encoder)
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
[Repeat 6 times]
    ↓
Linear + Softmax
```

### What are the key components in each transformer layer?

**Encoder layer**:
1. Multi-head self-attention
2. Residual connection + layer norm
3. Position-wise FFN (2 linear layers with activation)
4. Residual connection + layer norm

**Decoder layer** (adds one more):
1. **Masked** multi-head self-attention (causal)
2. Add & Norm
3. **Cross-attention** to encoder
4. Add & Norm
5. Position-wise FFN
6. Add & Norm

**Key differences**: Decoder has masking and cross-attention.

### Why do we need positional encoding?

Attention is **permutation-equivariant**: If you shuffle input tokens, you get shuffled outputs (same values, different positions).

Without position info, "cat sat on mat" = "mat on sat cat" to the model.

**Solutions**:
- **Sinusoidal encoding**: $PE_{pos,2i} = \sin(pos/10000^{2i/d})$ - can extrapolate
- **Learned embeddings**: Train position embeddings
- **Relative positions**: T5, RoPE (modern approach)

Position encoding adds sequence order information to otherwise position-agnostic attention.

### What's the difference between pre-LN and post-LN?

**Post-LN** (original Transformer):
```
x = LayerNorm(x + Sublayer(x))
```
Normalize after residual addition

**Pre-LN** (modern):
```
x = x + Sublayer(LayerNorm(x))
```
Normalize before sublayer

**Why Pre-LN wins**:
- More stable training for deep models (>12 layers)
- Better gradient flow
- Can train without warmup
- Used in: GPT, LLaMA, modern transformers

**Post-LN**: Original paper, less stable, needs careful warmup.

### Explain causal masking and why it's needed for generation

**Causal mask**: Prevents attending to **future positions**

**Implementation**: Set attention scores to -∞ before softmax:
```
mask[i,j] = 0 if i ≥ j else -∞
```

**Why needed**:
1. **Training-inference mismatch**: During generation, future tokens don't exist yet
2. **Autoregressive property**: Model should predict $P(x_t | x_{<t})$, not $P(x_t | x_{all})$
3. **Prevent cheating**: Without mask, model sees answers during training

**Example** (4 tokens):
```
Token 0: can see token 0
Token 1: can see tokens 0,1
Token 2: can see tokens 0,1,2
Token 3: can see all tokens
```

### What's the difference between encoder-only, decoder-only, and encoder-decoder?

**Encoder-only** (BERT):
- Bidirectional attention (sees full sequence)
- Best for: Classification, NER, Q&A
- Can't generate text naturally
- Examples: BERT, RoBERTa

**Decoder-only** (GPT):
- Causal attention (only past)
- Best for: Generation, completion
- Simplest architecture
- Examples: GPT, LLaMA, Claude

**Encoder-Decoder** (Original Transformer):
- Encoder: bidirectional, Decoder: causal + cross-attention
- Best for: Translation, summarization
- Examples: T5, BART

### Why are decoder-only models dominant for LLMs?

1. **Simplicity**: One architecture, easier to scale
2. **Unified objective**: Single next-token prediction task
3. **In-context learning**: Emergent ability at scale
4. **Scalability**: Empirically scales better to 100B+ parameters
5. **Versatility**: Can do both understanding and generation
6. **Training efficiency**: No separate encoder-decoder coordination

GPT-3 showed decoder-only scales incredibly well, became dominant paradigm for LLMs.

### What is masked language modeling (MLM)?

**MLM** (BERT's pre-training objective):
1. Randomly select 15% of tokens
2. Replace selected tokens:
   - 80%: [MASK] token
   - 10%: Random token
   - 10%: Original token (unchanged)
3. Predict original tokens

**Objective**: $\mathcal{L} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\backslash i})$

**Why it works**: Forces bidirectional understanding - can use full context (left + right).

### Explain the 80/10/10 split in BERT's MLM

**80% → [MASK]**: Main training signal
- Model learns to predict masked tokens

**10% → Random**: Prevents over-reliance on [MASK]
- Model can't assume [MASK] always means "predict this"
- Handles noise/corruption

**10% → Unchanged**: Reduces pre-train/fine-tune mismatch
- [MASK] token only appears in pre-training
- Fine-tuning tasks don't have [MASK]
- This helps model work on real text

Without 10/10, model would only see [MASK] during pre-training but never during fine-tuning (distribution shift).

---

## Part 4: Large Language Models

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

## Part 5: Fine-tuning & PEFT

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

## Part 6: Alignment & RLHF

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

---

## Part 7: Graph Neural Networks

### What is message passing in GNNs?

**Core paradigm**: Update node representations by **aggregating information from neighbors**.

**Framework**:
$$h_v^{(k+1)} = \text{UPDATE}(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$$

**Three steps per layer**:
1. **Message**: Each neighbor sends its representation
2. **Aggregate**: Combine neighbor messages (sum, mean, max, attention)
3. **Update**: Combine aggregated message with own features

**Intuition**: "Tell me about your friends, I'll tell you who you are"

**Example** (social network):
- Layer 1: Learn from direct friends
- Layer 2: Learn from friends-of-friends
- Layer 3: Learn from 3-hop neighborhood

### Explain GCN, GraphSAGE, and GAT - key differences?

**GCN** (Graph Convolutional Network):
- **Aggregation**: Normalized mean (by degree)
- **Formula**: $h_v = \sigma(W \sum_{u \in \mathcal{N}(v)} \frac{h_u}{\sqrt{|N(u)||N(v)||}})$
- **Pros**: Simple, effective
- **Cons**: Fixed weighting (all neighbors equal)

**GraphSAGE**:
- **Aggregation**: Sample + aggregate (mean, max, LSTM)
- **Formula**: $h_v = \sigma(W[h_v \| \text{AGG}(\{h_u\})])$
- **Pros**: Inductive (works on unseen nodes), scalable (sampling)
- **Cons**: Sampling introduces variance

**GAT** (Graph Attention):
- **Aggregation**: Learned attention weights
- **Formula**: $h_v = \sigma(\sum_{u} \alpha_{vu} W h_u)$ where $\alpha_{vu}$ are attention weights
- **Pros**: Different neighbors have different importance
- **Cons**: More parameters, slower

**Key difference**: How neighbors are weighted - GCN (fixed), GraphSAGE (sampling), GAT (learned).

### What's the over-smoothing problem?

**Problem**: After many GNN layers, all node representations **converge to same value**.

**Why it happens**:
- Each layer mixes node with neighbors
- After k layers, node sees k-hop neighborhood
- In connected graph, all nodes eventually see entire graph
- Representations become indistinguishable

**Example**:
- Layer 1: Nodes are different
- Layer 10: All nodes ≈ same (graph-level average)

**Impact**: Can't distinguish nodes, lose local structure

**Solutions**:
1. **Fewer layers**: Use 2-4 layers (not 50+ like CNNs)
2. **Residual connections**: $h^{(k+1)} = h^{(k+1)} + h^{(k)}$
3. **Jumping knowledge**: Combine representations from all layers
4. **Regularization**: DropEdge, layer dropout

### Why do GNNs typically use only 2-4 layers?

**Over-smoothing**: Deep GNNs → all nodes converge to same representation

**Receptive field**:
- 2 layers: 2-hop neighborhood
- 4 layers: 4-hop neighborhood
- In many graphs: 4 hops covers most nodes (small-world property)

**Diminishing returns**: Beyond 4 layers, adding more layers often hurts performance

**Contrast with CNNs**:
- CNNs: 50-200 layers common (ResNet, etc.)
- GNNs: 2-4 layers typical

**Exceptions**: Special architectures (skip connections, normalization) can go deeper, but 2-4 is default.

### What's the difference between node-level, edge-level, and graph-level tasks?

**Node-level**: Predict property of each node
- Example: Classify user's interests, protein function
- Output: Use final node embeddings $h_v^{(K)}$
- Loss: Per-node (cross-entropy, MSE)

**Edge-level**: Predict property of edge or edge existence
- Example: Link prediction, relation classification
- Output: Combine node embeddings $f(h_u, h_v)$ (dot product, concat+MLP)
- Loss: Per-edge

**Graph-level**: Predict property of entire graph
- Example: Molecule toxicity, graph classification
- Output: Aggregate all nodes (sum, mean, attention pooling)
- Loss: Per-graph

### How to do graph-level prediction (readout functions)?

**Readout**: Aggregate all node embeddings into single graph embedding

**Methods**:

1. **Sum**: $h_G = \sum_{v \in V} h_v^{(K)}$
   - Permutation-invariant
   - Simple, works well

2. **Mean**: $h_G = \frac{1}{|V|} \sum_{v} h_v^{(K)}$
   - Normalizes by graph size
   - Better for varying sizes

3. **Max**: $h_G = \max_{v \in V} h_v^{(K)}$ (element-wise)
   - Focuses on most important features

4. **Attention**: $h_G = \sum_v \alpha_v h_v$ where $\alpha_v = \text{softmax}(a^T h_v)$
   - Learned weighting
   - Most expressive

5. **Hierarchical pooling**: Coarsen graph iteratively (DiffPool)

**Choice**: Mean/sum work well, attention for best performance.

### What's the difference between transductive and inductive learning?

**Transductive**:
- Train and test on **same graph**
- Graph structure fixed
- Can see test nodes during training (but not labels)
- Example: GCN on single citation network
- **Use**: Node classification on fixed graph

**Inductive**:
- Generalize to **unseen nodes/graphs**
- Graph structure can change
- Can't see test nodes during training
- Example: GraphSAGE (learns aggregation function)
- **Use**: New nodes added, multiple graphs

**Analogy**:
- Transductive: Closed-world (one graph)
- Inductive: Open-world (new data)

**Most real applications need inductive** (e.g., recommendation, new users).

### When to use GNN vs transformer vs RNN?

**GNN**:
- ✅ Graph-structured data (social, molecular, knowledge graphs)
- ✅ Relational information critical
- ✅ Irregular structure
- Example: Drug discovery, recommendation, traffic

**Transformer**:
- ✅ Sequences (text, time series)
- ✅ Long-range dependencies
- ✅ Parallelizable training
- Example: NLP, vision (as sequences of patches)

**RNN/LSTM**:
- ✅ Sequences with inherent order
- ✅ Streaming/online (process one step at a time)
- ✅ Memory-constrained (long sequences)
- Example: Real-time speech, sensor data

**Overlap**: Transformers can be viewed as fully-connected graph (all-to-all attention).

---

## Part 8: Diffusion Models

### Explain forward diffusion process (noising)

**Forward process**: Gradually add Gaussian noise over T steps

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Where:
- $x_0$: Original image
- $x_T$: Pure noise $\sim \mathcal{N}(0, I)$
- $\beta_t$: Noise schedule (increases: 0.0001 → 0.02)

**Closed form** (key property):
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

**Intuition**: After T=1000 steps, image becomes pure noise (destroys information).

### What does the model predict during training (noise, x_0, or score)?

**Standard (DDPM)**: Predict **noise** $\epsilon$

**Training**:
1. Sample $x_0$ from dataset
2. Sample timestep $t$ uniformly
3. Sample noise $\epsilon \sim \mathcal{N}(0,I)$
4. Create noisy image: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. Predict noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$

**Alternatives**:
- Predict $x_0$ directly
- Predict score $\nabla_{x_t} \log p(x_t)$

**Why predict noise**:
- More stable training
- Equivalent objectives (related by reparameterization)
- Empirically works best

### Write the training objective for DDPM

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

**Components**:
- $t \sim \text{Uniform}(1, T)$: Random timestep
- $x_0 \sim p_{data}$: Real image
- $\epsilon \sim \mathcal{N}(0, I)$: Random noise
- $\epsilon_\theta$: Noise prediction network

**Simplified**: Mean squared error between true noise and predicted noise.

**Key insight**: Train network to "denoise" by predicting what noise was added.

### What's the reparameterization trick for x_t?

Instead of sampling $x_t$ sequentially (apply noise T times):

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

Where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

**Advantages**:
1. **Direct sampling**: Jump to any timestep t in one step
2. **Efficient training**: No need to run forward process T times
3. **Parallelizable**: Sample different t's independently

**Derivation**: Apply Gaussian convolution repeatedly (product of Gaussians).

### DDPM vs DDIM sampling - differences?

**DDPM** (Denoising Diffusion Probabilistic Models):
- Stochastic sampling (adds noise at each step)
- Markovian process
- Requires ~1000 steps
- Slow (~50 seconds per image)

**DDIM** (Denoising Diffusion Implicit Models):
- **Deterministic** sampling (optional noise)
- **Non-Markovian** (can skip timesteps)
- Requires only 10-50 steps
- **10-100x faster**

**Trade-off**: DDIM slightly lower diversity, but quality nearly identical

**Why DDIM works**: Finds different (non-Markovian) process with same marginals $p(x_t)$

**Used in**: Stable Diffusion (50 DDIM steps by default)

### What's latent diffusion and why use it?

**Latent Diffusion**: Run diffusion in **compressed latent space** instead of pixel space

**Architecture**:
1. **VAE encoder**: $z = E(x)$ (512×512 → 64×64 latent)
2. **Diffusion**: Denoise in latent space
3. **VAE decoder**: $x = D(z)$ (64×64 → 512×512 image)

**Why use it**:
- **4-8x faster**: Smaller spatial dimensions ($64^2$ vs $512^2$)
- **Lower memory**: Less attention computation
- **Same quality**: VAE learned good compression

**Example** (Stable Diffusion):
- Pixel diffusion: 512×512×3 = 786K dimensions
- Latent diffusion: 64×64×4 = 16K dimensions (~50x reduction)

**Trade-off**: Slight VAE artifacts, but huge efficiency gain.

### Explain classifier-free guidance

**Goal**: Improve alignment with text prompt (make outputs more faithful to condition).

**Formula**:
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Where:
- $c$: Condition (text prompt)
- $\emptyset$: Unconditional (empty prompt)
- $w$: Guidance scale (typically 7-15)

**Interpretation**: Move prediction in direction away from unconditional, toward conditional

**Effect**:
- $w = 0$: Unconditional (ignore prompt)
- $w = 1$: Standard conditional
- $w > 1$: Amplify conditioning (sharper, more aligned)

**Trade-off**: Higher w → better alignment but less diversity

**Training requirement**: Randomly drop conditioning (10-20% of time) during training to learn $\epsilon_\theta(x_t, \emptyset)$

### Diffusion vs GAN vs VAE - when to use each?

**Diffusion**:
- ✅ Best sample quality (FID scores)
- ✅ Stable training (no mode collapse)
- ✅ Good mode coverage
- ❌ Slow sampling (10-1000 steps)
- **Use**: When quality matters most (text-to-image)

**GAN**:
- ✅ Fast sampling (1 step)
- ✅ Sharp images
- ❌ Training instability (mode collapse)
- ❌ Poor mode coverage
- **Use**: Real-time generation, video

**VAE**:
- ✅ Fast sampling
- ✅ Exact likelihood
- ❌ Blurry samples
- ❌ Lower quality
- **Use**: Compression, representation learning

**Trend**: Diffusion dominant for images (Stable Diffusion, DALL-E), GANs for video/real-time.

### Why do diffusion models achieve better sample quality than GANs?

1. **Stable training**: No adversarial game, just regression (predict noise)
2. **Better mode coverage**: No mode collapse (VAE-like coverage)
3. **Iterative refinement**: Multiple denoising steps improve quality
4. **Scalability**: Easier to scale to large models/datasets
5. **No GAN discriminator tricks**: Don't need careful balancing

**Evidence**: SOTA FID scores on ImageNet, realistic images (Stable Diffusion vs StyleGAN)

**Trade-off**: Slower sampling (but DDIM, distillation help)

---

## Part 9: Gaussian Processes

### What's a Gaussian Process?

**Definition**: Collection of random variables, any finite subset of which follows a **multivariate Gaussian** distribution.

**Distribution over functions**:
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

Where:
- $m(x)$: Mean function (often 0)
- $k(x, x')$: Covariance (kernel) function

**For any points** $X = \{x_1, \ldots, x_n\}$:
$$f(X) \sim \mathcal{N}(\mu, K)$$
Where $K_{ij} = k(x_i, x_j)$

**Intuition**: Instead of learning parameters, define distribution over entire functions.

### Explain mean and covariance (kernel) functions

**Mean function** $m(x)$:
- Expected value of $f(x)$
- Often set to 0 (assume data is centered)
- Can encode prior belief (e.g., linear trend)

**Kernel function** $k(x, x')$:
- **Covariance** between $f(x)$ and $f(x')$
- Encodes **similarity**: $k(x, x')$ high → $f(x) \approx f(x')$
- Determines function properties (smoothness, periodicity, etc.)

**Key insight**: "Similar inputs have similar outputs" - kernel defines similarity.

### Write the predictive mean and variance formulas

Given training data $(X, y)$, predict at test points $X_*$:

**Predictive mean**:
$$\mu_{f_*|y} = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$

**Predictive variance**:
$$\Sigma_{f_*|y} = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

Where:
- $K(X, X)$: Train-train covariance
- $K(X_*, X)$: Test-train covariance
- $K(X_*, X_*)$: Test-test covariance
- $\sigma_n^2$: Noise variance

**Key**: Exact Bayesian inference (condition Gaussian on observations).

### Why does GP provide uncertainty estimates?

**Bayesian framework**: Maintains full **posterior distribution** over functions.

**Predictive variance** tells us:
- **Near data**: Low variance (confident)
- **Far from data**: High variance (uncertain)

**Example**:
- Training points: x = [1, 2, 3], test point x = 1.5
- Variance at 1.5: Low (interpolating)
- Test point x = 10: High variance (extrapolating)

**Contrast with NN**:
- NN gives point prediction (no uncertainty)
- Need special techniques (ensembles, dropout) for uncertainty

**Why it works**: GP prediction is Gaussian → variance comes for free from conditioning.

### What's the RBF/squared exponential kernel?

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

**Parameters**:
- $\ell$: **Length scale** (how quickly correlation decays with distance)
- $\sigma_f^2$: Signal variance (vertical scale)

**Properties**:
- Smooth: Infinitely differentiable (smooth functions)
- Stationary: Depends only on $\|x - x'\|$
- Most common kernel

**Interpretation**:
- Small $\ell$: Functions vary quickly (wiggly)
- Large $\ell$: Functions vary slowly (smooth)

### How to tune hyperparameters (marginal likelihood)?

**Hyperparameters**: $\theta = \{\ell, \sigma_f^2, \sigma_n^2\}$ (kernel params + noise)

**Objective**: Maximize **marginal likelihood** $p(y | X, \theta)$

**Log marginal likelihood**:
$$\log p(y|X,\theta) = -\frac{1}{2} y^T K_y^{-1} y - \frac{1}{2} \log |K_y| - \frac{n}{2} \log 2\pi$$

Where $K_y = K(X,X) + \sigma_n^2 I$

**Three terms**:
1. Data fit: $y^T K_y^{-1} y$ (how well model explains data)
2. Complexity: $\log |K_y|$ (penalizes complex models)
3. Constant

**Optimization**: Gradient-based (L-BFGS, Adam) on $-\log p(y|X,\theta)$ (NLL)

**Automatic trade-off**: Balances fit and complexity (Bayesian Occam's razor).

### What's the computational complexity of GP?

**Training** (hyperparameter optimization):
- **Matrix inversion**: $O(n^3)$ for $K^{-1}$
- **Determinant**: $O(n^3)$ for $\log |K|$

**Prediction** (mean):
- $O(n^2)$ to compute $K(X_*, X) K^{-1}$
- $O(n)$ per test point

**Storage**: $O(n^2)$ for covariance matrix

**Bottleneck**: $O(n^3)$ scaling → **doesn't scale beyond ~10,000 points**

**Why**: Need to invert dense $n \times n$ matrix.

### How to scale GPs (sparse GP, inducing points)?

**Sparse GP** (inducing points):
1. Select $m \ll n$ inducing points (subset or pseudo-inputs)
2. Approximate full GP using only $m$ points
3. Complexity: $O(nm^2)$ training, $O(m^2)$ prediction

**Methods**:
- **FITC**: Fully Independent Training Conditional
- **SVGP**: Stochastic Variational GP (mini-batches)

**Trade-off**: $m$ = 100-1000 → approximate but scalable

**Other approaches**:
- **Local GPs**: Separate GPs per region
- **Deep GP**: Stack GPs (compositional)
- **Structured kernels**: Exploit Kronecker, Toeplitz structure

**Modern**: For very large n, use neural networks with uncertainty (ensembles, Bayesian NNs).

### When to use GP vs neural network?

**GP**:
- ✅ Small data (<10K points)
- ✅ Need uncertainty quantification
- ✅ Structured problem (can encode via kernel)
- ✅ Interpretability important
- ❌ Doesn't scale to large n or high dimensions
- **Use**: Bayesian optimization, robotics, active learning

**Neural Network**:
- ✅ Large data (millions of points)
- ✅ High-dimensional inputs
- ✅ Raw features (images, text)
- ❌ No natural uncertainty (need special techniques)
- **Use**: Deep learning, computer vision, NLP

**Middle ground**: Deep kernel learning (NN features + GP), neural tangent kernels.

### What's the connection between infinite-width NNs and GPs?

**Neural Tangent Kernel (NTK)** theorem:

**Infinite-width neural network** (during training) behaves like a **GP** with specific kernel.

**Key insight**:
- Finite width: Complex, non-linear dynamics
- **Infinite width**: Linearizes around initialization → GP with NTK kernel

**Implications**:
1. Connects deep learning and GPs
2. Explains why NNs generalize (GP prior)
3. Theoretical tool for understanding training dynamics

**Practical**: Finite-width NNs differ (non-linear feature learning), but infinite-width limit gives insights.

---

## Part 10: Vision-Language-Action Models

### What's the key innovation of VLA models?

**End-to-end learning** from (vision + language) → robot actions using transformers.

**Traditional robotics**:
```
Perception → State Estimation → Planning → Control
(separate hand-engineered modules)
```

**VLA**:
```
(Image, Text Instruction) → Transformer → Action
(single learned model)
```

**Key innovations**:
1. Treat robotics as **sequence modeling** (like language)
2. Leverage **pre-trained vision-language models**
3. **Action tokenization**: Discretize actions for transformers

### RT-1 vs RT-2 architecture differences?

**RT-1** (Robotics Transformer 1):
- Trained **from scratch** on robot data
- Vision: EfficientNet (ImageNet pre-trained)
- Language: Universal Sentence Encoder
- FiLM conditioning (vision + language)
- 130K robot episodes

**RT-2** (Robotics Transformer 2):
- **Co-fine-tune** pre-trained VLM (PaLI-X, PaLM-E)
- Vision: ViT (Vision Transformer)
- Language: PaLM (LLM)
- Add action tokens to vocabulary
- 100K robot episodes + web VLM data

**Key difference**: RT-2 leverages internet-scale vision-language knowledge.

### How are actions represented (discretization)?

**Continuous actions**: 7-DoF (x, y, z, roll, pitch, yaw, gripper) ∈ ℝ⁷

**Discretization** (RT-1):
- Bin each dimension into 256 values
- Action becomes sequence of 7 tokens (each in 0-255)
- Predict via **classification** (cross-entropy)

**Why discretize**:
- Transformers designed for discrete tokens
- Classification more stable than regression
- Enables multi-modal action distributions

**Alternatives**: Continuous actions (regression head), mixture of Gaussians, diffusion.

### What's FiLM conditioning?

**FiLM** (Feature-wise Linear Modulation):

Condition visual features on language via **affine transformation**:
$$\text{FiLM}(h) = \gamma \odot h + \beta$$

Where:
- $h$: Visual features
- $\gamma, \beta$: Learned from language embedding
- $\odot$: Element-wise multiplication

**In RT-1**:
1. Language → embedding
2. Embedding → $\gamma, \beta$ via MLP
3. Apply to vision features at multiple layers

**Advantage**: Efficient fusion, conditions low-level visual processing on language.

### How does RT-2 leverage pre-trained VLMs?

**Co-fine-tuning** strategy:

1. **Start**: Pre-trained VLM (PaLI-X) on billions of image-text pairs
   - Already understands objects, scenes, language

2. **Add**: Action tokens to vocabulary (like text tokens)

3. **Fine-tune**: On robot trajectories
   - Input: Image + "pick up the apple"
   - Output: Action tokens [0.1, 0.2, -0.3, ...]

4. **Result**: Model that does vision-language tasks AND robot control

**Key**: Internet knowledge transfers to robotics (symbol grounding, reasoning).

### What emergent capabilities does RT-2 show?

**Beyond basic control** (from web pre-training):

1. **Reasoning**: "Move banana to Taylor Swift album number" → moves to position 10

2. **Symbol grounding**: "Pick up extinct animal" → grasps toy dinosaur (understands "extinct")

3. **Math**: "Move to sum of 2+2" → position 4

4. **Chain-of-thought**: "I should pick X because..." → action

5. **Visual understanding**: "Pick the fruit" → recognizes apple is fruit

**Why surprising**: Never trained on these specific tasks - emerges from VLM pre-training.

**Evidence**: 62% success on novel tasks (vs 32% for RT-1).

### Why co-fine-tune on robot data?

**Alternative**: Freeze VLM, add action head → poor performance

**Co-fine-tuning**:
- Adapt **entire model** (vision + language + new action head)
- VLM features learn to support action prediction
- Better integration of vision-language-action

**Trade-off**: Risk catastrophic forgetting of VLM capabilities
- Solution: Mix robot data with VLM tasks during fine-tuning

**Result**: Model that retains VLM knowledge while learning robot control.

### What are the limitations of VLAs?

1. **Sample efficiency**: Still need 100K+ robot episodes (expensive to collect)

2. **Action precision**: Discretization limits fine-grained control

3. **Long-horizon tasks**: Struggle with multi-step planning (typically 1-3 steps)

4. **Safety**: No safety guarantees, can produce unexpected behaviors

5. **Sim-to-real gap**: Web images ≠ robot camera views

6. **Fixed morphology**: Doesn't generalize to different robot bodies

7. **Compute**: Large models, slow inference

**Open challenges**: Scaling robot data, sim-to-real transfer, safety.

---

## Part 11: Game Theory Foundations

### What's a normal form game? Components?

**Normal form** (strategic form): Model where players choose actions **simultaneously**.

**Components**:
1. **Players**: Finite set $N = \{1, 2, \ldots, n\}$
2. **Strategies**: Action sets $S_i$ for each player $i$
3. **Payoffs**: Utility functions $u_i: S \to \mathbb{R}$ where $S = S_1 \times \cdots \times S_n$

**Representation** (2-player): Payoff matrix
- Rows: Player 1's actions
- Columns: Player 2's actions
- Cells: $(u_1, u_2)$ payoffs

**Example** (Coordination):
```
         Left    Right
Up       (2,2)   (0,0)
Down     (0,0)   (1,1)
```

### What's a payoff matrix?

**Payoff matrix**: Tabular representation of normal form game.

**Format** (2-player):
- **Rows**: Player 1 (row player) actions
- **Columns**: Player 2 (column player) actions
- **Entries**: Tuple $(u_1, u_2)$ - payoffs for each player

**Example** (Prisoner's Dilemma):
```
         Cooperate  Defect
Coop     (-1,-1)    (-3,0)
Defect   (0,-3)     (-2,-2)
```

**Reading**: If P1 plays Cooperate, P2 plays Defect → P1 gets -3, P2 gets 0.

### Pure vs mixed strategy Nash equilibrium?

**Pure strategy NE**: Each player plays single action with probability 1
- Example: (Defect, Defect) in Prisoner's Dilemma
- Deterministic

**Mixed strategy NE**: Players randomize over actions
- Example: Rock-Paper-Scissors → (1/3, 1/3, 1/3) for each player
- Probability distribution over pure strategies

**Finding mixed NE**: Make opponent **indifferent** between their actions
- Set expected payoffs equal, solve for probabilities

### Does every finite game have a Nash equilibrium?

**Yes** - **Nash's Theorem** (1950):

Every finite game (finite players, finite actions) has **at least one Nash equilibrium in mixed strategies**.

**Key points**:
- May be only **mixed**, no pure NE (e.g., matching pennies)
- May have **multiple** NE (coordination games)
- Guarantees existence, not uniqueness

**Proof**: Uses Brouwer fixed-point theorem (beyond scope).

### What's an extensive form game?

**Extensive form**: Game tree representing **sequential** decision-making.

**Components**:
1. **Game tree**: Directed tree, nodes = game states
2. **Players**: Assigned to decision nodes
3. **Actions**: Edges from nodes
4. **Payoffs**: At terminal (leaf) nodes
5. **Information sets**: Group nodes player can't distinguish

**Example**: Chess, poker, tic-tac-toe

**Contrast normal form**: Sequential vs simultaneous moves.

### Difference between perfect and imperfect information?

**Perfect information**: Every player knows **complete history** when making decision
- Every information set is singleton (one node)
- Examples: Chess, Go, Tic-Tac-Toe
- Can solve via backward induction

**Imperfect information**: Some information is **hidden**
- Information sets contain multiple nodes (indistinguishable)
- Examples: Poker (hidden cards), Battleship
- Harder to solve (can't use backward induction directly)

**Incomplete information**: Players don't know payoffs/types (Bayesian games, not same as imperfect info).

### What are information sets?

**Information set**: Group of game tree nodes that player **cannot distinguish** when making a decision.

**Formal**: Set of nodes $I$ where:
- Same player acts at all nodes in $I$
- Player doesn't know which node in $I$ they're at
- Must choose same action at all nodes in $I$

**Example** (Poker):
- Two nodes: opponent has Ace, opponent has King
- You can't tell which → nodes in same information set
- Must choose same action (fold/call/raise) for both

**Perfect info**: Every information set has 1 node.

### How to solve perfect information games (backward induction)?

**Backward induction**: Work backwards from terminal nodes.

**Algorithm**:
1. Start at **terminal nodes** (leaves)
2. Assign payoffs from game definition
3. Move to **parent nodes**:
   - If parent is Player i's turn, choose action maximizing $u_i$
   - Assign that payoff to parent
4. Repeat until **root**

**Result**: Optimal strategy (subgame perfect equilibrium).

**Example** (simple game):
```
      P1
     /  \
    L    R
   /      \
  2,1    P2
        /  \
       l    r
      /      \
    3,0      0,2
```
- P2's turn: choose r (payoff 2 > 0)
- P1's turn: choose L (payoff 2 > 0)
- Solution: (L, r) with payoff (2,1)

**Limitation**: Only works for perfect information (no information sets with >1 node).

### What's a subgame perfect equilibrium?

**Subgame Perfect Equilibrium (SPE)**: Refinement of Nash equilibrium for extensive form games.

**Definition**: Strategy profile that is Nash equilibrium in **every subgame** (including the full game).

**Key property**: No player wants to deviate at **any** point in the game (not just at start).

**Why needed**: Rules out non-credible threats

**Example** (Entry game):
- Entrant: Enter or Stay Out
- Incumbent: Fight or Accommodate
- Payoffs: Fight (-1,-1), Accommodate (1,1), Stay Out (0,2)

**Nash**: (Enter, Fight) - but Fight is non-credible threat
**SPE**: (Enter, Accommodate) - only credible strategies

**Finding SPE**: Backward induction (in perfect info games).

### Nash equilibrium vs social optimum - are they the same?

**No** - Nash equilibrium ≠ Social optimum in general.

**Nash equilibrium**: No player wants to deviate unilaterally (individual rationality)

**Social optimum**: Maximizes sum of utilities (collective rationality)

**Counterexample** (Prisoner's Dilemma):
- Nash: (Defect, Defect) with payoff (-2, -2), sum = -4
- Social optimum: (Cooperate, Cooperate) with payoff (-1, -1), sum = -2

**Key insight**: Individual incentives don't always align with collective welfare.

**When they match**: Some games (e.g., coordination with unique optimum).

---

## Part 12: Social Dilemmas

### What's the Nash equilibrium of prisoner's dilemma?

**(Defect, Defect)** - both players defect.

**Why**:
- If opponent cooperates: Defect gives 0 > -1 (Cooperate)
- If opponent defects: Defect gives -2 > -3 (Cooperate)
- **Dominant strategy**: Defect regardless of opponent

**Verification**: Neither player can improve by deviating:
- P1: Can't improve from (D,D) by switching to C (-3 < -2)
- P2: Can't improve from (D,D) by switching to C (-3 < -2)

### Is Nash equilibrium Pareto optimal in PD?

**No** - Nash equilibrium is **Pareto dominated**.

**Pareto optimal**: Can't make any player better off without making another worse off.

**PD outcomes**:
- (D, D): (-2, -2) - **Nash equilibrium**
- (C, C): (-1, -1) - **Pareto optimal** (both better than Nash!)

**Pareto dominance**: (C, C) is better for **both** players than (D, D)

**The dilemma**: Individually rational to defect, but collectively better to cooperate.

### What's a dominant strategy?

**Dominant strategy**: Strategy that is **best response to all opponent strategies**.

**Formal**: Strategy $s_i^*$ is dominant for player $i$ if:
$$u_i(s_i^*, s_{-i}) \geq u_i(s_i, s_{-i}) \quad \forall s_i, s_{-i}$$

**Strictly dominant**: Strict inequality (always better).

**In Prisoner's Dilemma**: Defect is strictly dominant for both players
- Better than Cooperate whether opponent Cooperates or Defects

**Note**: Not all games have dominant strategies (e.g., coordination games).

### What makes it a "dilemma"?

**Dilemma**: **Individual rationality** (Nash) conflicts with **collective rationality** (Pareto optimum).

**Components**:
1. **Dominant strategy**: Defect (individual incentive)
2. **Nash equilibrium**: (D, D) with payoff (-2, -2)
3. **Pareto optimum**: (C, C) with payoff (-1, -1)
4. **Gap**: Nash is worse for everyone than cooperation

**Insight**: What's rational individually leads to collectively bad outcome.

**Broader implications**: Many real-world problems have this structure (climate change, arms race, etc.).

### How does iterated PD change the game?

**Iterated PD**: Play prisoner's dilemma **repeatedly** (finite or infinite rounds).

**Changes**:
1. **Future matters**: Today's actions affect tomorrow's outcomes (reputation)
2. **Cooperation emerges**: Conditional strategies like tit-for-tat can sustain cooperation
3. **Folk theorem**: Many outcomes (including cooperation) sustainable as equilibria

**Key difference**:
- **One-shot**: Defect dominant
- **Repeated**: Cooperate if you value future (discount factor δ high enough)

**Intuition**: "I'll cooperate if you do, otherwise I'll punish you" becomes credible threat.

### What's tit-for-tat strategy?

**Tit-for-tat**:
1. **Round 1**: Cooperate
2. **Round t**: Do whatever opponent did in round t-1 (copy opponent's last move)

**Properties**:
- **Nice**: Never defects first
- **Retaliatory**: Punishes defection immediately
- **Forgiving**: Returns to cooperation if opponent does
- **Simple**: Easy to understand and implement

**Success**: Won Axelrod's tournament (1980s) - very effective in iterated PD

**Vulnerability**: Noise can cause defection spirals (improved: generous tit-for-tat).

### Real-world examples of prisoner's dilemmas?

1. **Climate change**: Reduce emissions (C) vs pollute (D)
   - Individual: Better to pollute (save costs)
   - Collective: Everyone reducing emissions is best

2. **Arms race**: Disarm (C) vs arm (D)
   - Nash: Both arm (expensive, dangerous)
   - Optimal: Both disarm (peaceful, cheap)

3. **Overfishing**: Limit catch (C) vs overfish (D)
   - Individual: Catch more
   - Collective: Sustainable fishing

4. **Free riding on public goods**: Contribute (C) vs don't (D)
   - Individual: Don't contribute, enjoy benefits
   - Collective: Everyone contributes

5. **Code of silence** (original story): Stay silent (C) vs betray (D)

### How to achieve cooperation in PD?

1. **Repeated interaction**: Iterated PD with reputation
   - Conditional cooperation (tit-for-tat)
   - Requires high enough discount factor

2. **Communication**: Pre-play discussion
   - Limited without enforcement
   - But can establish norms

3. **Binding contracts**: Enforceable agreements
   - Changes payoff structure
   - Requires third-party enforcement

4. **Punishment mechanisms**: Sanctions for defectors
   - Social punishment, fines

5. **Changing preferences**: Altruism, social norms
   - Internalize externalities
   - Care about opponent's payoff

6. **Incomplete information**: Uncertainty about opponent's type
   - Some players might be "always cooperate"
   - Pooling equilibria possible

### What's the n-player version (public goods game)?

**Public Goods Game**: n-player extension of PD.

**Setup**:
- Each player $i$ contributes $c_i \in [0, C]$
- Total public good: $G = \sum_i c_i$
- Payoff: $u_i = \alpha G - c_i$
- Parameters: $\alpha < 1$ (individual return) but $n\alpha > 1$ (social return)

**Dilemma**:
- **Individual incentive**: Contribute 0 (free ride)
- **Collective optimum**: Everyone contributes C

**Nash equilibrium**: Everyone contributes 0 (tragedy of the commons)

**Examples**: Donations, volunteer work, team projects, environmental protection.

---

## Part 13: Classical Game Algorithms

### What's the minimax principle?

**Minimax**: Choose action that **maximizes** your **minimum** payoff (worst-case optimization).

**For maximizing player**:
$$\max_{a} \min_{opponent} value(a, opponent)$$

**For minimizing player**:
$$\min_{a} \max_{opponent} value(a, opponent)$$

**Assumption**: Opponent plays **optimally** to hurt you.

**Zero-sum games**: Your gain = opponent's loss, so:
$$\max \min = \min \max = v^*$$ (game value)

**Intuition**: Secure the best possible worst-case outcome.

### How does minimax algorithm work (recursive)?

**Recursive definition**:

```
function minimax(state, depth, isMaximizing):
    if terminal(state) or depth == 0:
        return evaluate(state)

    if isMaximizing:
        maxEval = -∞
        for child in children(state):
            eval = minimax(child, depth-1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = +∞
        for child in children(state):
            eval = minimax(child, depth-1, True)
            minEval = min(minEval, eval)
        return minEval
```

**Process**:
1. **Leaf nodes**: Return evaluation
2. **MAX nodes**: Choose max of children
3. **MIN nodes**: Choose min of children
4. **Propagate** values up tree

**Returns**: Best achievable value assuming optimal opponent.

### What's alpha-beta pruning and how much does it save?

**Alpha-beta pruning**: Optimization that prunes branches that can't affect final decision.

**Maintain**:
- $\alpha$: Best value for MAX found so far
- $\beta$: Best value for MIN found so far

**Prune when**: $\alpha \geq \beta$
- MAX won't choose this branch (MIN can force worse)

**Savings**:
- **Worst case**: No pruning, still $O(b^d)$
- **Best case**: $O(b^{d/2})$ with perfect move ordering
- **Practical**: Often 2-10x speedup

**Key**: Explore best moves first (move ordering critical).

### What's the complexity of minimax?

**Time**: $O(b^d)$
- $b$: Branching factor (average legal moves)
- $d$: Depth of search

**Space**: $O(bd)$
- DFS: Store path from root to leaf only

**Example** (Chess to depth 10):
- $b \approx 35$, $d = 10$
- $35^{10} \approx 2.8 \times 10^{15}$ nodes

**Problem**: Exponential explosion - can't search to game end for complex games.

**Solutions**: Alpha-beta, transposition tables, iterative deepening, evaluation functions.

### When is minimax optimal?

**Optimal when**:
1. **Two-player zero-sum game**
2. **Perfect information** (know complete state)
3. **Deterministic** (no randomness)
4. **Can search to terminal states** or have accurate evaluation

**Guarantees**: Finds game-theoretic optimal play (minimax value).

**Limitations**:
- Not optimal if can't search deep enough (relies on heuristic eval)
- Doesn't handle uncertainty (probabilistic games)
- Only two-player zero-sum

**Used successfully**: Chess (with alpha-beta), Checkers (solved!), Othello.

### Explain the four phases of MCTS (Selection, Expansion, Simulation, Backup)

**1. Selection**: Start at root, traverse tree using **UCT policy**
$$UCT(s,a) = \frac{W(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$$
- Choose child with highest UCT
- Balance exploitation (high $W/N$) and exploration (low $N$)
- Stop at leaf node

**2. Expansion**: If leaf is non-terminal, **add child node** to tree
- Expand one or more children
- Add to tree for future selection

**3. Simulation (Rollout)**: From new node, **play out** to terminal state
- Classic MCTS: Random moves
- AlphaZero: Use value network (no rollout)

**4. Backup**: **Propagate** result back to root
- Update visit counts: $N(s,a) \leftarrow N(s,a) + 1$
- Update values: $W(s,a) \leftarrow W(s,a) + result$
- Update all nodes on path

**Repeat** until budget exhausted → select most visited action.

### What's UCT formula?

**UCT** (Upper Confidence Bound for Trees):

$$UCT(s, a) = \underbrace{\frac{W(s,a)}{N(s,a)}}_{\text{exploitation}} + \underbrace{c \sqrt{\frac{\ln N(s)}{N(s,a)}}}_{\text{exploration}}$$

**Components**:
- $W(s,a)$: Total value (sum of rewards from action $a$)
- $N(s,a)$: Visit count for action $a$ from state $s$
- $N(s)$: Visit count for state $s$
- $c$: Exploration constant (typically $\sqrt{2}$)

**Intuition**:
- High $W/N$: Exploitation (action has high average value)
- High exploration term: Exploration (action rarely visited)
- Logarithm: Grows slowly (diminishing exploration bonus)

**Convergence**: UCT → optimal policy as visits → ∞ (proven).

### What's PUCT and how does it use neural networks?

**PUCT** (Predictor + UCT): UCT enhanced with neural network prior.

$$PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

**Components**:
- $Q(s,a)$: Mean action value (like $W/N$ in UCT)
- $P(s,a)$: **Prior probability** from policy network
- $c_{puct}$: Exploration constant

**Key difference from UCT**: Uses learned $P(s,a)$ to guide exploration

**Effect**:
- Network thinks move is good → explore more (high $P$)
- Network thinks move is bad → explore less (low $P$)
- **Dramatically reduces search space** vs random exploration

**Used in**: AlphaZero, MuZero

### Minimax vs MCTS - when to use each?

**Minimax**:
- ✅ Low branching factor ($b \approx 10$-50)
- ✅ Good evaluation function available
- ✅ Tactical games (Chess, Checkers)
- ❌ Struggles with $b > 100$

**MCTS**:
- ✅ High branching factor ($b > 100$)
- ✅ Hard to evaluate positions
- ✅ Strategic games (Go: $b \approx 250$)
- ✅ Anytime algorithm (improves with time)

**Key difference**:
- **Minimax**: Uniform exploration (full width to depth)
- **MCTS**: Asymmetric tree (focuses on promising variations)

**Modern**: AlphaZero uses MCTS + neural networks (best of both).

### Why is MCTS better for Go than minimax?

**Go characteristics**:
- **Huge branching factor**: ~250 legal moves (vs ~35 for chess)
- **Deep game**: ~150 moves
- **Hard to evaluate**: Positional evaluation very difficult

**Minimax problems**:
- Can't search deep: $250^{10}$ is astronomical
- Needs evaluation function: Go positions hard to evaluate

**MCTS advantages**:
1. **Selective search**: Focuses on promising moves (not all 250)
2. **Asymmetric tree**: Explores best variations deeply, ignores bad ones
3. **No evaluation needed**: Playouts to end (or neural network)
4. **Scales with time**: Always has best move, improves with more iterations

**Result**: MCTS enabled AlphaGo to beat world champion (minimax couldn't).

---

## Part 14: AlphaZero

### What are the two heads of AlphaZero's network?

**1. Policy head** ($\mathbf{p}$):
$$p(a|s) = \text{softmax}(\text{PolicyNet}(s))$$
- Outputs: Probability distribution over legal moves
- Used as: Prior in MCTS ($P(s,a)$)

**2. Value head** ($v$):
$$v(s) \in [-1, +1]$$
- Outputs: Scalar game outcome prediction
- -1 = loss, 0 = draw, +1 = win
- Used as: Evaluation in MCTS (replaces rollout)

**Shared**: Single ResNet backbone, two heads branch at end.

### How does PUCT use the policy network?

**PUCT selection** formula:
$$PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

**Policy network provides** $P(s,a)$:
- Network's prior belief about move quality
- High $P(s,a)$ → larger exploration bonus
- Guides MCTS toward promising moves

**Effect**:
- **Without prior**: MCTS explores randomly (slow)
- **With prior**: MCTS focuses on likely-good moves (fast)

**Intuition**: Network says "this move looks good" → MCTS explores it more.

### Why doesn't AlphaZero use rollouts (unlike classic MCTS)?

**Classic MCTS**: Random rollout to terminal state
- Fast but noisy
- Quality depends on rollout policy

**AlphaZero**: Value network evaluation $v(s)$
- **Learned** from self-play (accurate)
- **Direct** prediction (no randomness)
- **Fast** (single forward pass)

**Advantages**:
1. More accurate than random rollouts
2. Faster than smart rollouts
3. Single network for policy + value (efficient)

**Key insight**: Learned value function better than simulation.

### What's the training loop (self-play → train → repeat)?

```
Initialize network θ randomly
Initialize replay buffer D = ∅

repeat:
    // Self-play: Generate training data
    for N games in parallel:
        Play game using MCTS with π_θ
        Store (s, π, z) tuples in D
            s: board state
            π: MCTS visit distribution
            z: game outcome (+1/-1/0)

    // Training: Update network
    for M mini-batches:
        Sample batch from D
        Compute loss: L = (z - v)² - π^T log p + c||θ||²
        Update θ via SGD

until convergence
```

**Key**: Self-play provides training data, training improves network, better network → better self-play (curriculum).

### What's the loss function (three components)?

$$\mathcal{L} = \underbrace{(z - v)^2}_{\text{value loss}} - \underbrace{\pi^T \log \mathbf{p}}_{\text{policy loss}} + \underbrace{c \|\theta\|^2}_{\text{regularization}}$$

**1. Value loss** $(z - v)^2$:
- MSE between predicted value and game outcome
- Trains network to predict winner

**2. Policy loss** $-\pi^T \log \mathbf{p}$:
- Cross-entropy between MCTS policy and network policy
- Distills MCTS search into network

**3. Regularization** $c \|\theta\|^2$:
- L2 weight decay (prevents overfitting)
- $c = 10^{-4}$ typically

**Insight**: Network learns to match MCTS (which is stronger due to search).

### How does value network replace rollouts?

**Rollout** (classic MCTS):
1. From leaf node, play random moves
2. Reach terminal state
3. Get outcome (+1/-1/0)
4. Slow, noisy

**Value network** (AlphaZero):
1. From leaf node, evaluate $v_\theta(s)$
2. Single forward pass
3. Predict outcome directly
4. Fast, learned

**Why better**:
- **Learned**: Trained on millions of positions (better than random)
- **Fast**: No simulation needed
- **Accurate**: Approximates true value function

**Trade-off**: Requires good network (pre-training via self-play).

### What's the difference between AlphaGo and AlphaZero?

**AlphaGo** (2016):
- **Supervised pre-training**: Learn from human games
- **Separate networks**: Policy and value
- **Rollouts**: Fast rollout policy + value network
- **Handcrafted features**: Domain knowledge
- **Go-specific**

**AlphaZero** (2017):
- **Tabula rasa**: No human data, self-play from random
- **Single network**: Shared policy + value
- **No rollouts**: Value network only
- **Raw board**: No features, just position
- **General**: Chess, Go, Shogi (same algorithm)

**Performance**: AlphaZero beat AlphaGo 100-0 after 40 days of training.

### Why is AlphaZero considered "tabula rasa"?

**Tabula rasa** (blank slate):
- **No human knowledge**: Doesn't use human games, opening books, or expert features
- **Random initialization**: Starts with random weights
- **Learns from scratch**: Pure self-play discovers strategies
- **Only game rules**: Just needs legal moves definition

**Contrast**:
- Traditional engines: Heavily handcrafted (evaluation, openings)
- AlphaGo: Pre-trained on human games

**Significance**: Shows general learning can match/exceed human knowledge.

### What are limitations (perfect info, two-player, etc.)?

1. **Perfect information only**: Requires known game state (no hidden info)
   - Works: Chess, Go
   - Doesn't work: Poker

2. **Two-player zero-sum**: Designed for competitive games
   - Not multi-player
   - Not general-sum (cooperation)

3. **Discrete actions**: Finite move set
   - Not continuous control

4. **Massive compute**: 5000 TPUs for Go (44 million games)
   - Not sample-efficient

5. **Deterministic**: No randomness in game
   - Doesn't handle stochasticity well

6. **No transfer**: Trained separately for each game

### How does temperature affect move selection?

**After MCTS**, select action based on visit counts $N(s,a)$:

$$\pi(a|s) \propto N(s,a)^{1/\tau}$$

**Temperature $\tau$**:

- **$\tau = 1$**: Proportional to visits (stochastic, exploration)
  - Used during early game

- **$\tau \to 0$**: Greedy (always pick most visited)
  - $\pi(a^*) = 1$ where $a^* = \arg\max_a N(s,a)$
  - Used during later game

**Schedule** (AlphaZero):
- First 30 moves: $\tau = 1$ (exploration)
- After 30 moves: $\tau \to 0$ (exploitation)

**Why**: Exploration early for diversity, exploitation later for best play.

---

## Part 15: PSRO & Population-Based Methods

### What's the PSRO algorithm (4 steps)?

**PSRO loop**:

**1. Initialize**: Start with small population $\Pi_i^{(0)}$ for each player
   - Example: Random policies

**2. Meta-game**: Compute empirical payoff matrix
   - Evaluate all policy pairs via simulation
   - Build payoff table

**3. Meta-strategy**: Solve for Nash equilibrium $\sigma$ on empirical game
   - Linear programming or iterative solver

**4. Best Response**: Train BR policy to $\sigma$ using RL oracle
   - Each player trains $\pi_i^{new} = BR(\sigma_{-i})$

**5. Expand**: Add new BRs to population
   - $\Pi_i^{(k+1)} = \Pi_i^{(k)} \cup \{\pi_i^{new}\}$

**Repeat** until convergence (no profitable BR).

### What's an empirical game?

**Empirical game**: Finite game constructed from **simulations** of continuous game.

**Construction**:
- **Players**: Same as original
- **Actions**: Finite set of policies from population
- **Payoffs**: Average returns from playing policies against each other

**Example** (2-player, 3 policies each):
```
         π₂¹   π₂²   π₂³
π₁¹    (2,1) (1,3) (0,0)
π₁²    (3,2) (2,2) (1,1)
π₁³    (1,0) (2,1) (3,3)
```

**Purpose**: Reduce infinite strategy space to finite meta-game that can be solved.

### What's a best response oracle?

**Best Response (BR) oracle**: Algorithm that finds optimal policy against given opponent strategy.

**Formal**: For opponent strategy $\sigma_{-i}$, find:
$$\pi_i^{BR} = \arg\max_{\pi_i} \mathbb{E}_{\sigma_{-i}}[u_i(\pi_i, \sigma_{-i})]$$

**In PSRO**: Typically RL algorithm (PPO, DQN, SAC)
- Train against opponents sampled from $\sigma_{-i}$

**Exactness**:
- **Exact BR**: Truly optimal (hard to achieve)
- **Approximate BR**: Good enough in practice

**Convergence**: PSRO converges to Nash if BRs are exact (approximate BRs → approximate Nash).

### How does PSRO converge to Nash equilibrium?

**Convergence condition**: When no player can find profitable best response.

**Algorithm**:
1. Current meta-Nash: $\sigma^*$
2. Compute BR for each player: $\pi_i^{BR}$
3. If $u_i(\pi_i^{BR}, \sigma_{-i}^*) \leq u_i(\sigma_i^*, \sigma_{-i}^*)$ for all $i$:
   - **Converged**: $\sigma^*$ is approximate Nash
4. Else: Add $\pi_i^{BR}$ to population, repeat

**Guarantee**: Exploitability decreases monotonically (anytime PSRO variant)

**In practice**: Stop when exploitability below threshold.

**Theoretical**: Exact BRs → exact Nash. Approximate BRs → approximate Nash.

### Difference between PSRO and double oracle?

**Double Oracle** (classical):
- Game theory algorithm for matrix games
- Exact BR computation
- Tabular (small state/action spaces)

**PSRO** (modern):
- Extends double oracle to **deep RL**
- Approximate BR via RL (PPO, DQN)
- Handles **large/continuous** state/action spaces
- **Empirical game**: Estimate payoffs via simulation

**Relationship**: PSRO is double oracle + deep RL oracle.

**Key innovation**: Scalability to complex games (Starcraft, Dota).

### What's JPSRO and how does it differ from PSRO?

**JPSRO** (Joint PSRO):
- Handles **general-sum n-player** games
- Uses **joint strategy distributions** (not independent mixing)
- Computes **Correlated Equilibrium (CE)** or **CCE** (not Nash)
- Train BR against **joint marginal** $\nu_{-i}$

**PSRO**:
- Two-player **zero-sum** focus
- **Independent** strategy mixing (Nash)
- Train BR against **independent marginal** $\sigma_{-i}$

**Key difference**: Joint distribution captures **coordination** opportunities.

**Example** (Traffic Light):
- Nash: Mix independently → 2.5 utility
- CE: Coordinate (both go same direction) → 5 utility

### What's Correlated Equilibrium (CE)?

**Correlated Equilibrium**: Joint strategy distribution where no player wants to deviate **after** seeing their recommended action.

**Setup**: Mediator samples $(a_1, \ldots, a_n) \sim \mu$, recommends $a_i$ to each player privately.

**Condition**: No incentive to deviate:
$$\mathbb{E}_{a_{-i} | a_i}[u_i(a_i, a_{-i})] \geq \mathbb{E}_{a_{-i} | a_i}[u_i(a'_i, a_{-i})]$$
for all players $i$, actions $a_i$, deviations $a'_i$.

**Example** (Traffic):
```
         Left    Right
Up       (5,5)   (0,0)
Down     (0,0)   (5,5)
```
CE: Flip coin → both Up-Left or both Down-Right (EU = 5)
Nash: Mix 50-50 independently (EU = 2.5)

**Why better**: Coordination via mediator.

### What's Coarse Correlated Equilibrium (CCE)?

**CCE**: Relaxation of CE - players commit to follow or deviate **before** seeing recommendation.

**Condition**: No incentive to deviate from marginal:
$$\mathbb{E}_{\mu}[u_i(a)] \geq \mathbb{E}_{\mu}[u_i(a'_i, a_{-i})]$$
for all players $i$, deviations $a'_i$.

**Difference from CE**:
- **CE**: React after seeing recommendation
- **CCE**: Commit before seeing recommendation

**Computational**: CCE easier to compute (fewer constraints in LP)

**Used in**: JPSRO (tractable, general-sum)

### Hierarchy: Nash ⊆ CE ⊆ CCE - explain

**Inclusion**:
$$\text{Nash Equilibrium} \subseteq \text{Correlated Equilibrium} \subseteq \text{Coarse Correlated Equilibrium}$$

**Nash → CE**: Independent mixing is special case of joint distribution
- Nash: $\mu(a) = \prod_i \sigma_i(a_i)$ (product distribution)
- CE allows correlations

**CE → CCE**: Stronger incentive constraints
- CE: No profitable deviation **after** seeing recommendation
- CCE: No profitable deviation **before** seeing recommendation

**Why hierarchy matters**:
- Easier to compute: Nash (hardest) < CE < CCE (easiest)
- CCE largest set (most solutions)
- Nash smallest (most restrictive)

### Why use CCE for general-sum games?

1. **Enables coordination**: Beyond independent mixing (Nash)
   - Captures correlated strategies
   - Example: Traffic light coordination

2. **Efficient computation**: Linear program (polynomial time)
   - Nash can be PPAD-complete

3. **Existence**: Always exists (like Nash in mixed strategies)

4. **n-player**: Naturally extends to many players

5. **One BR per player**: CCE needs 1 BR/player (vs $|A_i|$ BRs for CE)

6. **Proven convergence**: JPSRO converges to CCE

**Trade-off**: Weaker than Nash (larger solution set), but more practical.

### How many BRs needed per iteration (PSRO vs JPSRO)?

**PSRO** (Nash):
- **1 BR per player** per iteration
- Total: $n$ BRs (for $n$ players)

**JPSRO with CCE**:
- **1 BR per player** per iteration
- Total: $n$ BRs
- Same as PSRO!

**JPSRO with CE** (full correlated equilibrium):
- **$|A_i|$ BRs per player** (one per possible recommendation)
- Much more expensive
- Rarely used in practice

**Conclusion**: CCE has same BR cost as Nash, but captures coordination.

---

## Part 16: Neural Population Learning

### What's the key innovation of NeuPL vs PSRO?

**PSRO**: $N$ separate policy networks for $N$ strategies
- Memory: $O(N)$ networks
- No transfer between strategies

**NeuPL**: **Single conditional network** for all strategies
$$\pi_\theta(a | s, i)$$
where $i$ is policy index.

**Memory**: $O(1)$ network (constant, regardless of $N$)

**Key innovation**:
- All policies share representation
- **Transfer learning** across strategies
- 10,000x parameter reduction

### How does conditional network work: π(a|s,i)?

**Conditioning on policy index $i$**:

**Methods**:
1. **Embedding**: $i \to e_i$ (learned embedding)
2. **Concatenation**: $[s, e_i] \to$ network
3. **FiLM**: Use $e_i$ to modulate features (affine transformation)
4. **Hypernetwork**: $e_i \to$ subset of network weights

**Example** (embedding + concat):
```
s: state (observation)
i: policy index (0, 1, 2, ...)
e_i = Embed(i)  # learnable embedding
input = concat(s, e_i)
output = Network(input)  # action distribution
```

**Effect**: Different $i$ → different policies, but shared network.

### What's the memory advantage (O(1) vs O(N))?

**PSRO**:
- Population size: $N$ policies
- Storage: $N \times$ (network parameters)
- Example: 100 policies × 10M params = 1B parameters

**NeuPL**:
- Population size: $N$ policies
- Storage: 1 network + $N$ embeddings
- Example: 10M params + 100 × 128 (embeddings) ≈ 10M parameters

**Reduction**: ~100x for population size 100

**Enables**: Large populations (1000+) that would be infeasible with separate networks.

### What's transfer learning in NeuPL?

**Transfer**: New strategies benefit from features learned by previous strategies.

**Mechanism**:
1. Train $\pi_\theta(a|s, 0)$ (first policy)
2. Network learns useful representations of states
3. Train $\pi_\theta(a|s, 1)$ (second policy)
4. Reuses learned features, only adapts via index 1
5. Faster learning, better performance

**Analogy**: Like transfer learning in vision (ImageNet → specific tasks), but for game strategies.

**Evidence**: NeuPL converges faster than PSRO (fewer samples needed).

### NeuPL vs NeuPL-JPSRO differences?

**NeuPL** (original):
- **Symmetric zero-sum** games only
- Nash equilibrium (implicit)
- Two-player focus

**NeuPL-JPSRO**:
- **General-sum n-player** games
- CCE equilibrium (explicit)
- Combines NeuPL efficiency + JPSRO generality
- Additional components: payoff estimator network

**Relationship**: NeuPL-JPSRO extends NeuPL to broader game class.

### What equilibrium does NeuPL converge to?

**Original NeuPL** (symmetric zero-sum):
- Empirically converges to **approximate Nash equilibrium**
- No formal convergence guarantee in original paper

**NeuPL-JPSRO** (general-sum):
- Proven convergence to **CCE** (under exact distillation assumption)
- Formal guarantee

**In practice**: Both converge to reasonable equilibria, NeuPL-JPSRO has theoretical backing.

### When to use NeuPL vs standard PSRO?

**Use NeuPL when**:
- ✅ Memory constrained (can't store many networks)
- ✅ Want large populations (100+ strategies)
- ✅ Transfer learning beneficial (similar strategies)
- ✅ Symmetric game (original NeuPL)

**Use PSRO when**:
- ✅ Small populations (< 20 strategies sufficient)
- ✅ Very heterogeneous strategies (little transfer)
- ✅ Simplicity preferred (separate networks easier to debug)

**Modern trend**: NeuPL-JPSRO increasingly default (efficiency + generality).

---

## Part 17: Fictitious Self-Play

### What's fictitious play (classical)?

**Fictitious Play**: Iterative algorithm where players best-respond to opponent's **empirical frequency**.

**Algorithm**:
1. Initialize: Play arbitrary action
2. Track: Opponent's action history
3. Compute: Empirical frequency $\hat{\sigma}_{-i}(t) = \frac{1}{t}\sum_{\tau=1}^t a_{-i}^\tau$
4. Best respond: $a_i^{t+1} = BR(\hat{\sigma}_{-i}(t))$

**Convergence**: Converges to Nash in:
- Two-player zero-sum games (guaranteed)
- Potential games
- Some other classes

**Does NOT converge**: Shapley's example (cycling).

### How does FSP use RL and SL?

**Fictitious Self-Play** (sample-based version):

**Two components**:
1. **RL**: Learn best response to current average strategy
   - Buffer: $M_{RL}$ (recent trajectories)
   - Trains: Best-response policy

2. **SL**: Learn average strategy
   - Buffer: $M_{SL}$ (all historical actions)
   - Trains: Average policy (supervised on own actions)

**Data flow**:
- Play using mixture of RL and SL policies
- RL data → $M_{RL}$ (for best response learning)
- Actions → $M_{SL}$ (for average strategy learning)

### What's NFSP (Neural FSP)?

**Neural Fictitious Self-Play**: Deep NN version of FSP.

**Two networks**:
1. **Q-network**: Best-response (DQN-style)
   - Trained from $M_{RL}$ (RL buffer)
   - $\epsilon$-greedy behavior

2. **$\Pi$-network**: Average strategy
   - Trained from $M_{SL}$ (SL buffer)
   - Supervised learning (predict own actions)

**Behavior policy**:
- With probability $\eta$: Use Q-network (explore)
- With probability $1-\eta$: Use $\Pi$-network (exploit average)

**Convergence**: To approximate Nash in large imperfect-info games.

### What are the two networks and two buffers?

**Networks**:
1. **Q-network** ($Q_\theta$): Best response
   - Input: State
   - Output: Q-values for actions
   - Training: RL (Q-learning / DQN)

2. **Average policy network** ($\Pi_\phi$):
   - Input: State
   - Output: Action probabilities
   - Training: SL (classification on historical actions)

**Buffers**:
1. **$M_{RL}$**: Reservoir for RL (recent data)
   - Stores: Transitions $(s, a, r, s')$
   - For: Training Q-network

2. **$M_{SL}$**: Reservoir for SL (all historical actions)
   - Stores: State-action pairs $(s, a)$
   - For: Training $\Pi$-network

**Why two buffers**: Different data requirements (RL needs recent, SL needs all history).

### Why does FSP converge to Nash in some games?

**Proof sketch** (two-player zero-sum):

1. **Average strategy**: $\bar{\sigma}_i^t = \frac{1}{t}\sum_{\tau=1}^t \sigma_i^\tau$

2. **Best response**: Each iteration plays BR to opponent's average

3. **Regret minimization**: Average regret goes to 0
   $$\frac{1}{T}\sum_{t=1}^T [u_i(BR_i^t, \bar{\sigma}_{-i}^t) - u_i(\bar{\sigma}_i^t, \bar{\sigma}_{-i}^t)] \to 0$$

4. **Folk theorem**: No-regret in two-player zero-sum → Nash

**Intuition**: Best-responding to average prevents cycling (vs best-responding to latest in vanilla self-play).

### FSP vs vanilla self-play - why more stable?

**Vanilla self-play**:
- Agent plays against latest version of itself
- Can cycle: Rock → Paper → Scissors → Rock → ...
- No convergence guarantee

**FSP**:
- Agent best-responds to **average** of all past opponents
- Average smooths out oscillations
- Provably converges (in some games)

**Example** (Rock-Paper-Scissors):
- Vanilla SP: Cycles through pure strategies
- FSP: Converges to (1/3, 1/3, 1/3) mixed Nash

**Why**: Averaging acts as regularization, prevents exploitation of single strategy.

### When to use FSP vs PSRO?

**FSP / NFSP**:
- ✅ Two-player zero-sum
- ✅ Imperfect information (poker)
- ✅ When Nash convergence critical
- ✅ Continuous learning (no population)

**PSRO**:
- ✅ Multi-player or general-sum
- ✅ Want diverse population
- ✅ Can afford sequential training
- ✅ Empirical game analysis important

**Modern**: PSRO/JPSRO more general, FSP for specific settings (poker-like).

---

## Part 18: Counterfactual Regret Minimization

### What games is CFR designed for (imperfect info, extensive form)?

**CFR** is designed for:
- **Extensive-form games** (game trees)
- **Imperfect information** (hidden information, information sets)
- **Two-player** (mainly, extensions exist for multi-player)
- **Zero-sum** (convergence guarantee)

**Examples**: Poker (Kuhn, Leduc, Texas Hold'em), Bridge, Liar's Dice

**Not for**: Perfect info (use minimax/MCTS instead), large continuous action spaces.

### What's an information set?

**Information set** $I$: Collection of game tree nodes that player **cannot distinguish**.

**Properties**:
- Same player acts at all nodes in $I$
- Player doesn't know which node they're at
- Must choose same action at all nodes in $I$

**Example** (Poker):
- Player sees own cards but not opponent's
- Nodes with same visible info → same information set
- E.g., "I have King, opponent has ???" is one infoset

**Perfect info**: Every infoset has 1 node (can distinguish all states).

### What's counterfactual value?

**Counterfactual value** $v_i(\sigma, I)$: Expected utility for player $i$ if they:
1. **Reach** information set $I$ (counterfactually, regardless of their own actions)
2. Then **follow** strategy $\sigma$ from $I$ onward

**Formula**:
$$v_i(\sigma, I) = \sum_{h \in I} \sum_{z \in Z} \pi_{-i}^\sigma(h) \pi^\sigma(h, z) u_i(z)$$

Where:
- $h$: History (node in $I$)
- $z$: Terminal node
- $\pi_{-i}^\sigma(h)$: Opponent's reach probability
- $u_i(z)$: Utility at terminal

**"Counterfactual"**: Assumes we reached $I$, even if our strategy wouldn't have.

### Explain regret matching strategy update

**Regret** for action $a$ at infoset $I$:
$$R^T(I, a) = \sum_{t=1}^T [v_i(\sigma^t_{I \to a}, I) - v_i(\sigma^t, I)]$$

Cumulative difference between playing $a$ and current strategy.

**Regret matching**: Convert regrets to strategy
$$\sigma^{T+1}(I, a) = \frac{R^{T,+}(I, a)}{\sum_{a'} R^{T,+}(I, a')}$$

Where $R^{+} = \max(R, 0)$ (positive part).

**If all regrets ≤ 0**: Play uniform random.

**Intuition**: Play actions proportional to how much we regret not playing them.

### Why use average strategy (not current strategy)?

**Current strategy** $\sigma^T$: Can oscillate (like vanilla self-play)

**Average strategy** $\bar{\sigma}^T$:
$$\bar{\sigma}^T(I, a) = \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I) \sigma^t(I, a)}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$

Weighted by reach probability.

**Convergence**: $\bar{\sigma}^T \to$ Nash equilibrium

**Why**: Regret matching guarantees **average** regret → 0, not instantaneous regret.

**Analogy**: Like FSP - averaging prevents cycling.

### What's CFR+ and how does it improve vanilla CFR?

**CFR+** improvements over vanilla CFR:

1. **Regret floor**: Cumulative regrets never go below 0
   $$R^{t+1}(I, a) = \max(R^t(I, a) + r^{t+1}(I, a), 0)$$
   (Reset negative regrets to 0)

2. **Alternating updates**: Update one player at a time (not simultaneous)

3. **Linear averaging**: Weight recent iterations more
   $$w_t = t$$ (instead of uniform weighting)

**Result**:
- **Faster convergence**: $O(1/T^2)$ vs $O(1/T)$ for vanilla CFR
- Used in Libratus (poker AI)

**Why it works**: Prevents negative regret accumulation (interference).

### What's Monte Carlo CFR (sampling)?

**Problem**: Full tree traversal is expensive for large games.

**Solution**: **Sample** parts of tree instead of traversing all.

**Variants**:

1. **External Sampling**:
   - Sample: Chance events + opponent actions
   - Traverse: All own actions at visited infosets
   - Lower variance, moderate speedup

2. **Outcome Sampling**:
   - Sample: Entire trajectory (single path)
   - Fastest, highest variance

**Trade-off**: Speed vs variance
- Full CFR: Slow, deterministic
- Outcome sampling: Fast, noisy

**Practical**: External sampling common compromise.

### What's Deep CFR?

**Deep CFR**: Use neural networks to approximate regrets/strategies.

**Problem**: Tabular CFR doesn't scale to huge games (too many infosets).

**Solution**:
1. **Value network**: Approximate counterfactual values $v(I)$
2. **Regret network**: Approximate cumulative regrets $R(I, a)$
3. **Strategy network**: Approximate average strategy

**Training**:
- Run MCCFR to generate samples
- Train networks on sampled regrets/values
- Use networks instead of tables

**Result**: Scales to games too large for tabular (e.g., no-limit poker).

### Complexity of CFR per iteration?

**Time per iteration**: $O(|I|)$
- Linear in number of information sets
- Must visit each infoset

**Full game tree**: $O(|H|)$ where $H$ is all histories (nodes)
- Larger than $|I|$

**Convergence**: $O(1/\sqrt{T})$ to Nash
- After $T$ iterations, exploitability $\propto 1/\sqrt{T}$

**Total cost**: $O(|I| \cdot T)$ for $\epsilon$-Nash where $T \propto 1/\epsilon^2$

**Practical**: Millions of iterations for poker, but feasible.

### Why does CFR converge to Nash in two-player zero-sum?

**Proof sketch**:

1. **Regret minimization**: CFR guarantees average regret → 0
   $$\frac{1}{T}\sum_{t=1}^T r_i^t \to 0$$

2. **Zero-sum + regret minimization**: Folk theorem
   - If both players have no-regret, their average strategies form Nash

3. **Formal**: Average exploitability bounded by average regret
   $$\epsilon = \max_{\pi_i} u_i(\pi_i, \bar{\sigma}_{-i}) - u_i(\bar{\sigma}) \leq \frac{1}{T}\sum_{t=1}^T R_i^t$$

**Key insight**: Regret matching is no-regret algorithm → convergence.

**Extension**: Works for broader class (two-player general-sum with modifications).

---

**End of answers. Good luck with your interviews!** 🚀
