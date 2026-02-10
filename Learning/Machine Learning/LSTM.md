# LSTM (Long Short-Term Memory)

## Definition
LSTM is a recurrent neural network architecture designed to address the vanishing gradient problem in standard RNNs, enabling learning of long-term dependencies through a gating mechanism.

## Motivation
**Standard RNN Problem**:
- Gradients vanish exponentially with sequence length
- Can't learn dependencies beyond ~10 steps
- $\frac{\partial h_t}{\partial h_{t-k}} \to 0$ as $k$ increases

**LSTM Solution**:
- Maintain separate cell state with additive updates
- Gates control information flow (learn what to remember/forget)

## Architecture

### Cell State & Gates
LSTM has 4 components at each timestep $t$:

1. **Forget Gate** ($f_t$): What to remove from cell state
2. **Input Gate** ($i_t$): What new information to add
3. **Cell Update** ($\tilde{C}_t$): Candidate values to add
4. **Output Gate** ($o_t$): What to output from cell state

### Equations

**Forget Gate**:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
Output: 0 (forget) to 1 (keep) for each cell state dimension

**Input Gate**:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
Output: How much of new information to add

**Candidate Cell State**:
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
Output: New candidate values (scaled by input gate)

**Cell State Update** (key step):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
- $\odot$: Element-wise multiplication
- Forget old information, add new information

**Output Gate**:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State**:
$$h_t = o_t \odot \tanh(C_t)$$

### Why This Works

**Gradient Flow**:
- Cell state $C_t$ has additive update (not multiplicative)
- Gradient: $\frac{\partial C_t}{\partial C_{t-1}} = f_t$ (near 1 if forget gate saturates high)
- Avoids vanishing/exploding gradients through "gradient highway"

**Selective Memory**:
- Forget gate: Remove irrelevant information
- Input gate: Add relevant new information
- Output gate: Control what's exposed to next layer

## LSTM vs Standard RNN

| Aspect | RNN | LSTM |
|--------|-----|------|
| Parameters | $O(d^2)$ | $O(4d^2)$ (4 gates) |
| Gradient flow | Multiplicative (vanishes) | Additive (highway) |
| Long-term deps | ~10 steps | 100s of steps |
| Training | Unstable | More stable |
| Speed | Fast | ~4x slower |

## Variants

### 1. GRU (Gated Recurrent Unit)
**Simplification** of LSTM:
- 2 gates instead of 3: update gate, reset gate
- No separate cell state
- Fewer parameters: $O(3d^2)$ vs $O(4d^2)$
- Often comparable performance to LSTM
- Faster training

**Equations**:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$ (update gate)
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$ (reset gate)
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 2. Bidirectional LSTM
- Process sequence forward and backward
- Concatenate hidden states: $h_t = [h_t^{forward}, h_t^{backward}]$
- Better for tasks where future context matters (NER, POS tagging)
- Cannot be used for generation (no access to future)

### 3. Peephole Connections
- Gates also look at cell state: $f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f)$
- Slightly better performance, more parameters

## Training Considerations

### Gradient Clipping
- LSTMs still can have exploding gradients
- Clip gradient norm to threshold (e.g., 1.0-5.0)
- Essential for stable training

### Initialization
- Forget gate bias: Initialize to 1 or 2 (default to remembering)
- Other biases: Initialize to 0
- Weights: Xavier/He initialization

### Sequence Length
- Truncated BPTT (Backprop Through Time)
- Process long sequences in chunks
- Typical chunk: 100-500 timesteps

## When LSTMs Are Still Used

### Advantages Over Transformers
1. **Sequential processing**: Process one token at a time (lower memory for long sequences)
2. **Constant memory**: Hidden state size fixed regardless of sequence length
3. **No positional encoding**: Inherently sequential
4. **Online learning**: Can process streams

### Modern Use Cases
- **Speech recognition**: Long audio sequences
- **Time series**: Stock prices, sensor data
- **Video processing**: Frame-by-frame analysis
- **Edge deployment**: Lower memory than Transformers for long sequences
- **State-space models**: Modern variants (S4, Mamba) revive recurrent approach

## LSTM vs Transformer

| Aspect | LSTM | Transformer |
|--------|------|-------------|
| Parallelization | Sequential (slow) | Parallel (fast) |
| Long-range deps | Limited (~100s) | Unlimited (quadratic cost) |
| Memory (training) | $O(nd)$ | $O(n^2)$ (attention) |
| Inference | $O(1)$ per step | $O(n)$ per step |
| Inductive bias | Sequential | Permutation-equivariant |

**Why Transformers won for NLP**:
- Parallelizable training (10-100x faster)
- Better long-range dependencies through attention
- Scale better with data and compute

**Where LSTMs still compete**:
- Long sequence streaming
- Memory-constrained deployment
- Inherently sequential data

## Interview Relevance

**Common Questions**:
1. **Why LSTM vs RNN?** Addresses vanishing gradients via additive cell state updates
2. **How do gates work?** Sigmoid outputs (0-1) control information flow: forget, input, output
3. **Why 3 gates?** Forget: remove old; Input: add new; Output: control exposure
4. **Cell state vs hidden state?** Cell: long-term memory; Hidden: short-term output to next layer
5. **LSTM vs GRU?** GRU: simpler (2 gates), faster, often comparable; LSTM: more capacity
6. **Bidirectional LSTM use?** When future context helps (tagging, classification), not generation
7. **Why Transformers replaced LSTMs?** Parallelizable, better long-range, scales better
8. **When still use LSTM?** Streaming, long sequences with memory constraints, time series

**Key Formulas to Remember**:
- Cell state update: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ (additive!)
- Hidden state: $h_t = o_t \odot \tanh(C_t)$
- 3 gates: forget, input, output (all sigmoid)

**Key Insight**: LSTM's additive cell state update creates a "gradient highway" that preserves gradients over long sequences, unlike multiplicative updates in standard RNNs.
