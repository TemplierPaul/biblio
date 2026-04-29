# GTrXL: Gated Transformer-XL for RL — Implementation Details

---

## Architecture

### Layer Block (full GTrXL forward pass)

For layer `l`, given current-segment embeddings `E^(l-1)` ∈ ℝ^(T×D) and memory `M^(l-1)` ∈ ℝ^(T_mem×D):

```python
# --- Multi-Head Attention submodule ---
normed_input = LayerNorm(concat([StopGrad(M[l-1]), E[l-1]]))  # [(T_mem+T) × D]
attn_out = RelativeMultiHeadAttention(normed_input)            # [T × D]  (queries only from E)
Y[l] = GRU_gate(x=E[l-1], y=ReLU(attn_out))

# --- MLP submodule ---
mlp_out = MLP(LayerNorm(Y[l]))                                 # [T × D]
E[l] = GRU_gate(x=Y[l], y=ReLU(mlp_out))
```

### GRU-type Gate

```python
def gru_gate(x, y, Wr, Ur, Wz, Uz, Wg, Ug, b_g):
    r = sigmoid(Wr @ y + Ur @ x)
    z = sigmoid(Wz @ y + Uz @ x - b_g)   # b_g > 0 → near-identity init
    h_hat = tanh(Wg @ y + Ug @ (r * x))
    return (1 - z) * x + z * h_hat
```

**Initialization**: set `b_g = 2.0` for GRU gating. This makes `z ≈ σ(-2) ≈ 0.12` at init, so the gate passes `x` almost unchanged (identity map).

### Relative Multi-Head Attention

```python
def relative_mha(E_q, E_kv, M_kv, W_Q, W_K, W_V, W_R, u, v, Phi, H, d):
    # E_q: queries from current segment [T × D]
    # E_kv: keys/values from concat(memory, current) [(T_mem+T) × D]
    E_kv_full = concat([M_kv, E_q], dim=0)  # [(T_mem+T) × D]

    Q = W_Q @ E_q         # [H × T × d]
    K = W_K @ E_kv_full   # [H × (T_mem+T) × d]
    V = W_V @ E_kv_full   # [H × (T_mem+T) × d]
    R = W_R @ Phi         # [H × (T_mem+T) × d]  sinusoid relative position

    # 4-term attention score (Transformer-XL decomposition):
    # α_{htm} = Q_{htd}K_{hmd} + Q_{htd}R_{hmd} + u_{hd}K_{hmd} + v_{hd}R_{hmd}
    alpha = (Q @ K.T) + (Q @ R.T) + (u @ K.T) + (v @ R.T)  # [H × T × (T_mem+T)]
    alpha = causal_mask(alpha)   # prevent attending to future positions within segment
    W = softmax(alpha / sqrt(d), dim=-1)
    Y_bar = W @ V                          # [H × T × d]
    Y_hat = E_q + Linear(reshape(Y_bar))   # residual (this is then replaced by gate in GTrXL)
    return Y_hat
```

Note: queries come only from the **current segment** `E_q`, while keys/values come from `[memory, current]`. The memory is stop-gradient'd before being concatenated.

### Memory Update

After each forward pass over a segment:

```python
# After computing all layer outputs for segment t
for l in range(L):
    M_new[l] = concat([M[l][-T_mem+T:], E_layer_outputs[l]])[-T_mem:]
    # slide the window: drop oldest, append current activations
```

Memory is **not differentiable** — it is treated as a fixed context from the previous forward pass.

---

## Full Agent Architecture

```
Observation (image 72×96×3 or proprioceptive)
       ↓
Image Encoder (ResNet: conv stride-1 → maxpool → 2× residual block)
       ↓
Input Embedding  E^(0) ∈ ℝ^(T × D)   # D=256 in paper
       ↓
GTrXL × 12 layers
  each layer: RelativeMHA + GRU gate → MLP + GRU gate
  memory size T_mem=512 per layer
       ↓
Final embedding E^(L) ∈ ℝ^(T × D)
       ↓  (take per-timestep slice)
Policy head: Linear(256) → logits / Gaussian params
Value head:  Linear(256) → scalar
```

---

## Training with PPO + GTrXL: The Time Dimension

### The Core Challenge

Standard PPO collects rollouts from actors and trains on them. For a recurrent/attention policy, each forward pass needs **context from the past** (the memory `M^(l)`). The challenge is:

1. Memory must be computed during collection and stored alongside the trajectory
2. During training updates, you need to re-compute log-probs, but the stored memories are **stale** (from the old policy)
3. Gradients cannot flow through the memory (stop-gradient), so this staleness is partially tolerable

### Segment-Based Rollout Collection

```python
# Per actor: maintain a rolling memory buffer
memory = zeros(L, T_mem, D)  # one per layer

for each step in episode:
    obs_embedding = encoder(obs)
    
    # Every `unroll_len` steps, package as a segment
    if buffer is full (unroll_len steps collected):
        store_segment(
            obs_embeddings=segment_embeddings,   # [unroll_len × D]
            actions=segment_actions,
            rewards=segment_rewards,
            log_probs_old=segment_log_probs,
            values_old=segment_values,
            init_memory=memory.copy()            # M^(l) at segment start — CRITICAL
        )
        # Update memory for next segment
        memory = forward_pass_memory(segment_embeddings, memory)
```

**Key**: store `init_memory` at the **start** of each segment. This is what the learner will use during training.

### PPO Training Update

```python
for epoch in range(K_epochs):
    for segment_batch in dataloader(rollout_buffer):
        obs_emb     = segment_batch.obs_embeddings   # [B × T × D]
        init_mem    = segment_batch.init_memory      # [B × L × T_mem × D]  (stale, no grad)
        actions     = segment_batch.actions
        log_probs_old = segment_batch.log_probs_old
        advantages  = compute_gae(segment_batch.rewards, segment_batch.values)

        # Forward pass with stale memory (stop-gradient'd)
        policy_out, values = gtrxl_forward(
            obs_emb,
            memory=StopGrad(init_mem)   # stale memory treated as fixed context
        )
        log_probs_new = policy_out.log_prob(actions)

        # PPO clipped objective
        ratio = exp(log_probs_new - log_probs_old)
        clip_ratio = clip(ratio, 1-ε, 1+ε)
        policy_loss = -mean(min(ratio * advantages, clip_ratio * advantages))

        # Value loss
        value_loss = MSE(values, returns)

        loss = policy_loss + c_v * value_loss - c_ent * entropy
        loss.backward()
        optimizer.step()
```

### Handling Memory Staleness

The stored `init_memory` is from the old policy — this is the **stale memory problem**:

- **Why it's acceptable**: the stop-gradient on memory means the gradient computation is decoupled from the memory contents. The network treats memory as a read-only context, so even stale memory provides useful (if slightly off-policy) context.
- **Why it causes issues**: the attention weights computed over stale memory are based on activations from an old policy, introducing bias in the log-prob estimates that goes beyond what the PPO clip corrects for.
- **Practical mitigation**: keep `K_epochs` small (1-2) and the segment length short enough that the memory doesn't go too stale between collection and update.

### Alternative: Re-computing Memory Online During Training

More expensive but more correct:

```python
# During training, replay the episode from scratch to get fresh memory
memory = zeros(L, T_mem, D)
for segment_idx in range(episode_length // unroll_len):
    segment = episode[segment_idx]
    # Forward pass with freshly-computed memory (not stale)
    policy_out, values, memory = gtrxl_forward_and_update_memory(
        segment.obs_embeddings, memory
    )
    # Now compute PPO loss on this segment with fresh memory
    ...
```

This is used in some R2D2/recurrent PPO implementations. It's O(episode_length) in compute and memory per training example.

### Burn-in

A practical middle ground: collect longer segments than you train on, and **burn in** the first `B` steps to warm up the memory before computing the loss:

```python
# Segment has length T_burn + T_train
# Only compute loss on T_train steps, use T_burn to get memory into good state
policy_out, values = gtrxl_forward(
    concat([burn_in_obs, train_obs]),
    memory=StopGrad(stored_init_memory)
)
# Compute PPO loss only on the last T_train timesteps
loss = ppo_loss(policy_out[T_burn:], values[T_burn:], ...)
```

---

## Practical Implementation Details

### Architecture Config (paper defaults)

```python
config = dict(
    n_layers=12,
    d_model=256,      # embedding dim = n_heads * head_dim
    n_heads=8,
    d_head=64,        # d_model / n_heads
    memory_size=512,  # T_mem: steps of memory per layer
    unroll_length=95, # segment length for training
    batch_size=128,   # number of actors
    b_g=2.0,          # GRU gate identity init bias
)
```

### MLP Submodule

```python
def mlp_submodule(x, d_model, d_ff=None):
    d_ff = d_ff or 4 * d_model
    return Linear(d_ff→d_model, Linear(d_model→d_ff, x))
    # no activation after output (as in Transformer-XL)
```

### Causal Masking in RL

Since the agent processes a **causal** time sequence (can't see future observations), the attention mask prevents queries at time `t` from attending to keys at time `t' > t`. However, queries CAN attend to all memory positions (which are all in the past):

```python
def causal_mask(alpha):
    # alpha: [H × T_current × (T_mem + T_current)]
    T_mem, T = ...
    # Positions 0..T_mem-1 are memory (always visible)
    # Positions T_mem..T_mem+T-1 are current (causally masked)
    mask = tril(ones(T, T))                      # lower-triangular for current
    full_mask = concat([ones(T, T_mem), mask])   # all memory accessible
    alpha.masked_fill_(full_mask == 0, -inf)
    return alpha
```

### Encoder and Observation Embedding

For image observations (DMLab):
- ResNet: (conv 3×3 stride-1) → (maxpool 3×3 stride-2) → 2× residual blocks with ReLU
- Produces per-timestep embedding of shape `D`
- These per-step embeddings form the sequence input `E^(0) ∈ ℝ^(T×D)`

For proprioceptive (Numpad):
- Simple 2-layer MLP with tanh activations

---

## Gating Variants Summary

| Gate | Formula | Init behavior | Notes |
|---|---|---|---|
| Input | σ(Wg·x) ⊙ x + y | Blocks input | Worse than residual |
| Output | x + σ(Wg·x - b) ⊙ y | Near-identity (b>0) | Decent, 12% diverge |
| Highway | σ(Wg·x+b)⊙x + (1-σ)⊙y | Near-identity | Unstable |
| SigTanh | x + σ(Wg·y - b)⊙tanh(Ug·y) | Near-identity | Sensitive to HP |
| **GRU** | (1-z)⊙x + z⊙ĥ | Near-identity (b>0) | **Best overall** |

GRU wins because it gates **both** the input and output streams jointly, with a reset gate `r` giving it full expressivity to decide how much of the current input to use.

---

## Comparison to LSTM Recurrence

| | LSTM | GTrXL |
|---|---|---|
| State type | Hidden vector h + cell c | Cached layer activations M^(l) |
| Horizon | Fixed size hidden state | L × T_mem steps |
| Gradient through time | BPTT, can vanish | Stop-grad on memory |
| Parallelism over time | Sequential | Parallel within segment |
| Memory update | At every step | At segment boundaries |
| PPO integration | Store (h,c) at segment start | Store M^(l) at segment start |
