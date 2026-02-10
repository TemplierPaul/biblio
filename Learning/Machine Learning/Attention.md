# Attention Mechanism

## Definition
Attention is a mechanism that allows neural networks to focus on different parts of the input when producing each element of the output. It computes a weighted combination of values based on the compatibility between a query and keys.

## Core Formula (Scaled Dot-Product Attention)
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- **Q (Query)**: What we're looking for (size: $n \times d_k$)
- **K (Key)**: What we're comparing against (size: $m \times d_k$)
- **V (Value)**: What we retrieve (size: $m \times d_v$)
- **$d_k$**: Dimension of keys/queries (scaling factor prevents gradient issues)
- Output: $n \times d_v$ weighted values

## Why Scale by $\sqrt{d_k}$?
- Without scaling, dot products grow large in magnitude as $d_k$ increases
- Large dot products push softmax into regions with extremely small gradients
- Scaling keeps dot products in a reasonable range for softmax

## Multi-Head Attention
Instead of single attention, use $h$ parallel attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

**Benefits**:
- Allows model to attend to different positions and representation subspaces
- Each head can learn different relationships (syntax, semantics, coreference, etc.)
- Typical: 8-16 heads in transformers

## Types of Attention

### 1. Self-Attention
- Q, K, V all come from the same sequence
- Each position attends to all positions in the same sequence
- Used in: Transformer encoder and decoder

### 2. Cross-Attention
- Q comes from one sequence, K and V from another
- Decoder attends to encoder outputs
- Used in: Encoder-decoder models (translation, image captioning)

### 3. Masked (Causal) Attention
- Prevents positions from attending to subsequent positions
- Ensures autoregressive property (only use past context)
- Used in: GPT, decoder-only models
- Implementation: Set future positions to $-\infty$ before softmax

## Computational Complexity
- Time: $O(n^2 d)$ where $n$ is sequence length, $d$ is dimension
- Memory: $O(n^2)$ for attention matrix
- **Problem**: Quadratic scaling limits long sequences
- **Solutions**: Sparse attention, linear attention, Flash Attention

## Attention vs Traditional Mechanisms

| Aspect | RNN/LSTM | Attention |
|--------|----------|-----------|
| Sequential | Yes (hard to parallelize) | No (fully parallel) |
| Long-range deps | Struggles (vanishing gradients) | Direct connections |
| Complexity | $O(n)$ per step | $O(n^2)$ total |
| Interpretability | Hidden states (opaque) | Attention weights (visualizable) |

## Interview Relevance

**Common Questions**:
1. **Why do we need attention?** Addresses limitations of fixed-length context vectors in seq2seq; allows direct access to all input positions
2. **What's the difference between self-attention and cross-attention?** Self: within same sequence; Cross: between two sequences
3. **Why multi-head?** Allows model to jointly attend to information from different representation subspaces
4. **Complexity issues?** $O(n^2)$ is prohibitive for long sequences; need efficient variants
5. **Why scale by $\sqrt{d_k}$?** Prevents softmax saturation and gradient issues

**Key Insight**: Attention is "database lookup with soft matching" - Query matches against Keys to retrieve weighted Values.
