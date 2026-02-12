# ANN Architecture - Interview Q&A

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
