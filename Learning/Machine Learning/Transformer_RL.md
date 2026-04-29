# Stabilizing Transformers for Reinforcement Learning (GTrXL)

**Paper**: Parisotto et al., "Stabilizing Transformers for Reinforcement Learning" (DeepMind, ICLR 2020)

---

## The Problem

Transformers handle long temporal horizons better than LSTMs by design: they don't compress history into a fixed hidden state, and they avoid vanishing/exploding gradients over long sequences. This makes them appealing for partially observable RL, where an agent's critical observations can span thousands of steps.

However, naively applying the standard transformer in RL consistently fails — the training is unstable and often produces random-policy-level performance. This instability is worse in RL than in supervised learning because the RL loss signal is noisier and the optimization landscape is harder.

The paper identifies the root cause and fixes it with two targeted changes.

---

## The Fix: GTrXL (Gated Transformer-XL)

Two changes to the canonical Transformer-XL architecture:

### 1. Identity Map Reordering (LayerNorm placement)

Move LayerNorm **before** each submodule (attention, MLP), instead of after the residual connection.

**Why it helps**: At initialization, the submodule outputs are near zero, so the representation passes through the network almost untouched. The agent effectively starts as a Markovian policy — it can learn reactive behaviors first, then gradually learn to exploit memory. This is crucial because reactive behaviors must be learned before memory-based ones.

```
Canonical:    x → submodule → x + out → LayerNorm
TrXL-I:       x → LayerNorm → submodule → x + ReLU(out)
```

TrXL-I alone already beats LSTM substantially.

### 2. GRU-type Gating (replacing residual connections)

Replace the residual addition `x + y` with a learnable GRU-style gate:

```
r = σ(Wr·y + Ur·x)
z = σ(Wz·y + Uz·x - b)
ĥ = tanh(Wg·y + Ug·(r ⊙ x))
output = (1 − z) ⊙ x + z ⊙ ĥ
```

The bias `b > 0` (initialized to 2 for GRU gating) makes the gate start near identity, reinforcing the Markovian initialization.

Other gating variants were tested (Input, Output, Highway, SigTanh) — GRU-type wins across all metrics: final performance, stability, and hyperparameter robustness.

---

## Memory: Transformer-XL Scheme

The GTrXL uses Transformer-XL's **segment-level recurrence**:

- At each layer `l`, maintain a memory tensor `M^(l)` of shape `[T_mem × D]` holding the previous segment's layer activations
- The current segment attends over `[M^(l), E^(l)]` (memory + current)
- Memory is **stop-gradient'd** — no backpropagation through it
- Relative position encodings allow the model to distinguish memory vs. current positions

This gives a receptive field of `L × T_mem` steps (12 layers × 512 memory = 6144 steps in the paper's main config).

---

## Results

| Model | Mean Human-Norm. Score (DMLab-30) |
|---|---|
| LSTM (3-layer) | 99.3 |
| TrXL (canonical) | 5.0 |
| TrXL-I (reordering only) | 107.0 |
| MERLIN (external memory, 100B steps) | 115.2 |
| **GTrXL (GRU)** | **117.6** |

- Large improvement on memory tasks, no regression on reactive tasks
- 0% training divergence across 25 hyperparameter seeds (vs. 12-16% for other variants)
- Scales better with memory horizon than LSTM (shown via Numpad task)

---

## Training Algorithm: V-MPO

The paper trains with V-MPO (on-policy, uses estimated advantages to build a target policy distribution under KL constraint). The same losses can be used with standard PPO — see `Transformer_RL_detailed.md` for how to handle the time dimension with PPO.

---

## Key Takeaways

1. **Canonical transformers fail in RL** due to optimization instability — not a fundamental architectural mismatch
2. **Two changes fix it**: Identity Map Reordering + GRU gating
3. **Gated identity init** (positive bias b) is critical for fast learning
4. **GTrXL is a drop-in replacement for LSTM** in actor-critic RL, with better performance and similar training stability
5. The memory mechanism is segment-based (not step-by-step recurrence), which matters for how you collect rollouts

---

## Related Concepts

- [[Transformer]]: base architecture
- [[LSTM]]: the architecture GTrXL replaces
- [[Attention]]: the core mechanism (here used over time, not just input space)
- [[RLHF]]: transformer policies in RL from human feedback use the same memory challenge
