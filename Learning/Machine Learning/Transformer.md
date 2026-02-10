# Transformer Architecture

## Definition
The Transformer is a neural architecture that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. Introduced in "Attention is All You Need" (Vaswani et al., 2017).

## Core Architecture

### Encoder-Decoder Structure
```
Input → Encoder (6 layers) → Decoder (6 layers) → Output
```

### Encoder Layer
Each of 6 identical layers contains:
1. **Multi-Head Self-Attention**
2. **Add & Norm** (residual connection + layer normalization)
3. **Feed-Forward Network** (2 linear layers with ReLU)
4. **Add & Norm**

### Decoder Layer
Each of 6 identical layers contains:
1. **Masked Multi-Head Self-Attention** (causal mask)
2. **Add & Norm**
3. **Multi-Head Cross-Attention** (attends to encoder output)
4. **Add & Norm**
5. **Feed-Forward Network**
6. **Add & Norm**

## Key Components

### 1. Positional Encoding
Since there's no recurrence, need to inject position information:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

- **Why sinusoidal?** Allows model to extrapolate to longer sequences
- **Alternatives**: Learned positional embeddings (BERT), relative position (T5), RoPE (LLaMA)

### 2. Feed-Forward Network
$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

- Applied identically to each position
- Hidden dimension typically $4d_{model}$ (2048 for $d_{model}=512$)
- Acts as position-wise transformation

### 3. Layer Normalization
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

- Normalizes across feature dimension (not batch)
- **Pre-LN vs Post-LN**: Modern models use Pre-LN (before attention/FFN) for training stability

### 4. Residual Connections
Every sub-layer has residual connection: $\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$

- Enables gradient flow in deep networks
- Critical for training stability

## Architecture Variants

### Encoder-Only (BERT-style)
- Bidirectional context (can see full sequence)
- Best for: Classification, NER, question answering
- Examples: BERT, RoBERTa, DeBERTa

### Decoder-Only (GPT-style)
- Unidirectional/causal (only sees past tokens)
- Best for: Generation, autoregressive tasks
- Examples: GPT, LLaMA, Mistral
- **Why dominant for LLMs?** Simpler, scales better, strong in-context learning

### Encoder-Decoder (Original Transformer)
- Encoder processes input, decoder generates output
- Best for: Translation, summarization
- Examples: T5, BART, mT5

## Training Details

### Original Transformer Hyperparameters
- Layers: 6 (base), 12 (big)
- $d_{model}$: 512 (base), 1024 (big)
- Heads: 8 (base), 16 (big)
- $d_{ff}$: 2048 (base), 4096 (big)
- Dropout: 0.1

### Optimization
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$)
- **Learning Rate Schedule**: Warmup + decay
  $$lrate = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})$$
- **Label Smoothing**: $\epsilon_{ls} = 0.1$

## Why Transformers Won

### Advantages
1. **Parallelization**: All positions processed simultaneously (vs sequential RNN)
2. **Long-range dependencies**: Direct connections between all positions ($O(1)$ path length vs $O(n)$ for RNN)
3. **Scalability**: Empirically scales well with data and compute
4. **Transfer learning**: Pre-train once, fine-tune for many tasks

### Disadvantages
1. **Quadratic complexity**: $O(n^2)$ attention limits long sequences
2. **No inherent sequence bias**: Needs positional encoding
3. **Data hungry**: Requires large datasets for pre-training

## Modern Improvements

### Efficiency
- **Flash Attention**: Fused kernel, reduces memory from $O(n^2)$ to $O(n)$
- **Linear Attention**: Approximations with $O(n)$ complexity
- **Sparse Attention**: Only attend to subset of positions

### Architecture
- **Pre-LN**: Layer norm before sub-layers (better training)
- **RMSNorm**: Simpler normalization (LLaMA)
- **SwiGLU**: Better activation function (PaLM, LLaMA)
- **Rotary Position Embeddings (RoPE)**: Better positional encoding (LLaMA)
- **Grouped-Query Attention (GQA)**: Reduce KV cache size (LLaMA 2)

## Interview Relevance

**Common Questions**:
1. **Why Transformer vs RNN?** Parallelizable, better long-range dependencies, no vanishing gradients
2. **What's the complexity?** $O(n^2 d)$ for attention dominates
3. **Why positional encoding?** Attention is permutation-equivariant; need position info
4. **Encoder vs Decoder?** Encoder: bidirectional (classification); Decoder: causal (generation)
5. **Why layer norm instead of batch norm?** Works better for sequences of varying length; normalizes per-example
6. **Pre-LN vs Post-LN?** Pre-LN is more stable for deep models

**Key Insight**: Transformer's success comes from parallelizable architecture + effective scaling + transfer learning, not just attention mechanism.
