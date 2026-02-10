# Masking in Transformers

## Definition
Masking is a technique to prevent attention from attending to certain positions, enabling controlled information flow in transformers. Critical for autoregressive modeling and pre-training strategies.

## Types of Masking

### 1. Causal (Autoregressive) Masking
**Purpose**: Prevent attention to future positions (for generation)

**Implementation**:
Set attention scores to $-\infty$ before softmax for future positions:
$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

**Result**: Position $i$ can only attend to positions $\leq i$

**Example** (4 positions):
```
[[0,   -∞,  -∞,  -∞],
 [0,    0,  -∞,  -∞],
 [0,    0,   0,  -∞],
 [0,    0,   0,   0]]
```

**Used in**: GPT, decoder-only models, language modeling

**Why?**:
- Ensures autoregressive property: $P(x_t | x_{<t})$
- Prevents information leakage during training
- Matches generation setting (can't see future)

### 2. Padding Masking
**Purpose**: Ignore padding tokens in variable-length sequences

**Problem**: Batches need uniform length → pad shorter sequences

**Solution**: Mask attention to padding positions

**Implementation**:
```python
# If token is padding (PAD=0):
attention_mask = (input_ids != PAD_TOKEN)  # [1, 1, 1, 0, 0] for sequence
# Set padded positions to -∞ in attention scores
```

**Used in**: All transformers with variable-length inputs

### 3. Attention Masking (General)
**Purpose**: Custom attention patterns

**Examples**:
- **Local attention**: Only attend to $k$ nearest neighbors
- **Blocked/Sparse attention**: Attend to fixed patterns (Longformer, BigBird)
- **Global tokens**: Special tokens that attend to everything

### 4. Masked Language Modeling (MLM) Masking
**Purpose**: Pre-training objective (BERT-style)

**Method**:
1. Randomly select 15% of tokens
2. For selected tokens:
   - 80%: Replace with [MASK]
   - 10%: Replace with random token
   - 10%: Keep original
3. Predict original tokens

**Objective**:
$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\backslash i})$$

**Example**:
```
Input:  The cat sat on the mat
Masked: The [MASK] sat on the mat
Predict: "cat"
```

**Why 80/10/10 split?**
- 80% [MASK]: Main objective
- 10% random: Prevent model from only learning [MASK]
- 10% unchanged: Prevent mismatch (no [MASK] in fine-tuning)

**Used in**: BERT, RoBERTa, ALBERT

## Implementation Details

### Attention Score Masking
**Before softmax**:
$$\text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

**Apply mask**:
$$\text{scores}_{ij}^{\text{masked}} = \text{scores}_{ij} + \text{mask}_{ij}$$

**After softmax** (masked positions have 0 weight):
$$\alpha_{ij} = \frac{\exp(\text{scores}_{ij}^{\text{masked}})}{\sum_k \exp(\text{scores}_{ik}^{\text{masked}})}$$

**Why $-\infty$?**
- $\exp(-\infty) = 0$
- Ensures masked positions contribute 0 to weighted sum

### Combining Masks
Multiple masks can be combined (logical OR):
```python
# Causal mask AND padding mask
combined_mask = causal_mask | padding_mask
```

**Example** (sequence with padding):
```
Input: [The, cat, sat, PAD, PAD]

Causal mask:
[[0, -∞, -∞, -∞, -∞],
 [0,  0, -∞, -∞, -∞],
 [0,  0,  0, -∞, -∞],
 [0,  0,  0,  0, -∞],
 [0,  0,  0,  0,  0]]

Padding mask (rows 3-4, all positions):
[-∞, -∞, -∞, -∞, -∞]
[-∞, -∞, -∞, -∞, -∞]

Combined: Causal + padding for positions 3-4 masked everywhere
```

## Pre-training Strategies with Masking

### BERT (Masked Language Model)
- **Bidirectional**: Can see full context (no causal mask)
- **MLM**: Predict masked tokens
- **NSP**: Next sentence prediction (deprecated in RoBERTa)

### GPT (Causal Language Model)
- **Unidirectional**: Causal mask only
- **Objective**: Predict next token
- **No masking** of input (all tokens visible to left)

### ELECTRA (Replaced Token Detection)
- **Generator**: Small MLM model generates replacements
- **Discriminator**: Detect which tokens were replaced
- More efficient than MLM

### SpanBERT
- Mask contiguous spans of tokens (not random)
- Predict entire span
- Better for tasks requiring phrase understanding

## Efficient Attention with Masking

### Sparse Attention Patterns
**Longformer**:
- Local attention: Window around each token
- Global attention: Few tokens attend to all
- Reduces $O(n^2)$ to $O(n \cdot w)$ where $w$ is window size

**BigBird**:
- Random + window + global attention
- Theoretically proven to approximate full attention

### Flash Attention
- Fused kernel that respects causal mask
- No memory for full attention matrix
- 2-4x speedup with masking

## Common Pitfalls

### 1. Forgetting Causal Mask in Decoder
**Problem**: Training allows seeing future, generation fails
**Solution**: Always use causal mask in decoder/decoder-only models

### 2. Incorrect Mask Shape
**Problem**: Mask must broadcast correctly to attention scores
**Solution**: Ensure mask shape is [batch, heads, seq_len, seq_len] or broadcastable

### 3. Not Masking [PAD] in Loss
**Problem**: Model learns to predict padding
**Solution**: Mask padding tokens in loss calculation

### 4. [MASK] in Generation
**Problem**: [MASK] token only in pre-training, not at inference
**Solution**: MLM models (BERT) not suitable for generation without adaptation

## Interview Relevance

**Common Questions**:
1. **Why causal masking?** Prevent seeing future during training (autoregressive property)
2. **How to implement?** Set future positions to $-\infty$ before softmax
3. **BERT vs GPT masking?** BERT: bidirectional (MLM); GPT: causal (next-token)
4. **Why $-\infty$?** $\exp(-\infty) = 0$, ensures 0 attention weight
5. **Padding masking?** Ignore PAD tokens in attention and loss
6. **MLM strategy?** 80% [MASK], 10% random, 10% unchanged
7. **Sparse attention masking?** Local windows, global tokens (Longformer, BigBird)
8. **Can BERT generate text?** Not directly - bidirectional context, trained with MLM not causal

**Key Formulas**:
- Causal mask: $\text{mask}_{ij} = 0$ if $i \geq j$, else $-\infty$
- Masked softmax: $\text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)$ where $M$ is mask
- MLM loss: Predict masked tokens only

**Key Insight**: Masking controls information flow in transformers - causal masking enables generation, padding masking handles variable lengths, and MLM masking enables bidirectional pre-training.
