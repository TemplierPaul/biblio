# LoRA (Low-Rank Adaptation)

## Definition
LoRA is a parameter-efficient fine-tuning technique that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into each layer, reducing trainable parameters by 10,000x while maintaining comparable performance.

## Core Idea

### Standard Fine-tuning
Update all parameters $W \in \mathbb{R}^{d \times k}$:
$$W' = W + \Delta W$$

**Problem**:
- For 7B model: need to store/update 7 billion parameters
- Memory: model + gradients + optimizer states ≈ 84GB for full precision

### LoRA Approach
Freeze original weights $W$, add trainable low-rank decomposition:
$$h = Wx + \Delta W x = Wx + BAx$$

Where:
- $W \in \mathbb{R}^{d \times k}$: Frozen pre-trained weights
- $B \in \mathbb{R}^{d \times r}$: Trainable "down-projection"
- $A \in \mathbb{R}^{r \times k}$: Trainable "up-projection"
- $r \ll \min(d, k)$: Rank (typically 4-64)

**Trainable Parameters**: $r(d + k)$ instead of $dk$

## Key Properties

### 1. Extreme Parameter Efficiency
**Example**: For a 7B parameter model with rank $r=8$:
- Original parameters: 7B
- LoRA trainable: ~4.7M (0.067%)
- **10,000x reduction**

### 2. No Additional Inference Latency
Can merge adapter weights into original model:
$$W' = W + BA$$

After training, compute merged weights once and use like normal model.

### 3. Task Switching
Keep base model frozen, swap different LoRA adapters:
- Adapter 1: Summarization
- Adapter 2: Translation
- Adapter 3: Code generation

Single base model + small adapters (MBs each) vs multiple full models (GBs each)

### 4. Composable
Can combine multiple LoRA adapters:
$$W' = W + B_1A_1 + B_2A_2 + \ldots$$

## Implementation Details

### Which Layers to Adapt?
**Original LoRA (GPT-3)**:
- Only apply to query and value projection matrices in attention
- Leave MLP and other projection matrices frozen
- Still achieves strong performance

**Common Practice (Modern)**:
- Apply to all linear layers: $W_q, W_k, W_v, W_o$ (attention)
- Optionally: MLP layers ($W_{up}, W_{down}$)
- More layers = better performance but more parameters

### Rank Selection
| Rank | Use Case | Performance | Parameters |
|------|----------|-------------|------------|
| 1-4 | Memory-constrained | Lower | Minimal |
| 8-16 | **Recommended** | Good | Low |
| 32-64 | Complex tasks | Better | Medium |
| 128+ | Diminishing returns | Marginal gain | Higher |

**Rule of thumb**: Start with $r=8$, increase if underfitting

### Initialization
- $A$: Random Gaussian (small std, e.g., 0.01)
- $B$: **Zero initialization**
- Ensures $\Delta W = BA = 0$ at start (model starts from pre-trained state)

### Scaling
Apply scaling factor $\alpha / r$ to $\Delta W$:
$$h = Wx + \frac{\alpha}{r}BAx$$

- $\alpha$: Constant (often 16 or 32)
- Makes learning rate less sensitive to rank choice
- Typical: $\alpha = 2r$ to $4r$

## Training

### Memory Savings
**Standard Fine-tuning** (7B model, AdamW):
- Model: 14GB (FP16)
- Gradients: 14GB
- Optimizer states: 56GB (2x params for Adam)
- **Total**: ~84GB

**LoRA** (rank 8):
- Frozen model: 14GB (no gradients needed)
- LoRA params: ~10MB
- LoRA gradients: ~10MB
- LoRA optimizer: ~40MB
- **Total**: ~14GB (6x reduction!)

### Hyperparameters
- **Learning rate**: Higher than full fine-tuning (1e-4 to 5e-4)
  - Fewer parameters to update, can be more aggressive
- **Rank**: 8-16 (start small)
- **Alpha**: 16-32
- **Dropout**: 0.05-0.1 on LoRA layers (optional)
- **Batch size**: Can be larger due to memory savings

## QLoRA (Quantized LoRA)

### Idea
Combine LoRA with quantization for even more efficiency

**Method**:
1. Quantize base model to 4-bit (NormalFloat4)
2. Keep LoRA adapters in BF16/FP16
3. Compute gradients in higher precision

**Result**:
- **65B model on single 48GB GPU**
- Base model: ~33GB (4-bit)
- LoRA + training: ~15GB
- Performance close to 16-bit full fine-tuning

**Techniques**:
- **4-bit NormalFloat**: Custom data type for normally-distributed weights
- **Double quantization**: Quantize quantization constants
- **Paged optimizers**: Handle memory spikes

## Variants and Extensions

### 1. AdaLoRA
- **Adaptive rank**: Different rank for different layers
- Prune less important adapters during training
- Better parameter budget allocation

### 2. LoRA+
- Use different learning rates for A and B matrices
- Typically: $\eta_B = 16 \times \eta_A$
- Faster convergence, better performance

### 3. DoRA (Weight-Decomposed LoRA)
- Decompose into magnitude and direction
- $W' = m \frac{W + BA}{||W + BA||}$
- Better matches full fine-tuning performance

### 4. LoRA-FA (LoRA with Frozen-A)
- Freeze $A$ to random values, only train $B$
- 2x fewer parameters
- Slightly lower performance

## When to Use LoRA

### Use LoRA When:
- ✅ Limited GPU memory
- ✅ Need to fine-tune multiple tasks (adapter switching)
- ✅ Domain adaptation with limited data
- ✅ Rapid experimentation
- ✅ Deployment constraints (small adapters)

### Use Full Fine-tuning When:
- ✅ Maximum performance critical
- ✅ Sufficient compute resources
- ✅ Large amount of high-quality task data
- ✅ Significant domain shift from pre-training

## Empirical Results

### Performance (from original paper)
On GPT-3 175B:
- LoRA (rank 4-16): Matches or exceeds full fine-tuning on various tasks
- Parameters: 0.01% of full model
- Tasks: Natural language understanding, generation, reasoning

### Typical Performance Gap
- LoRA (rank 8): ~95-99% of full fine-tuning performance
- LoRA (rank 16): ~98-100% of full fine-tuning performance
- Task-dependent: simpler tasks need lower rank

## Interview Relevance

**Common Questions**:
1. **How does LoRA work?** Freeze base weights, add trainable low-rank matrices $BA$
2. **Why low-rank?** Most fine-tuning updates lie in low-rank subspace (hypothesis)
3. **Typical rank?** 8-16 for most tasks
4. **Inference cost?** None - merge adapters into base weights: $W' = W + BA$
5. **LoRA vs full fine-tuning?** LoRA: 10000x fewer params, ~95-99% performance; Full: best performance, 10000x more memory
6. **QLoRA innovation?** 4-bit quantized base + BF16 adapters = 65B on single GPU
7. **Why initialize B to zero?** Start from pre-trained weights (no change at initialization)
8. **Can you combine LoRA adapters?** Yes, $W' = W + B_1A_1 + B_2A_2$
9. **Where to apply LoRA?** Attention projections minimum; all linear layers for best performance
10. **Memory savings?** ~6x for training (no gradients/optimizer for frozen weights)

**Key Insight**: LoRA exploits the intrinsic low-rank structure of weight updates during fine-tuning, enabling practical fine-tuning of massive models on consumer hardware.

## Code Example (Conceptual)

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Frozen pre-trained weight
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        # Base model output
        base_out = x @ self.weight.T

        # LoRA adaptation: (x @ A^T) @ B^T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T

        # Combine with scaling
        return base_out + (self.alpha / self.rank) * lora_out
```
