# Quality Diversity through Human Feedback (QDHF)

## Core Idea

**QDHF** learns diversity metrics from human feedback rather than requiring manually crafted behavioral descriptors. It combines:
- **Quality-Diversity (QD)** optimization framework
- **Human feedback** on solution similarity (via contrastive learning)
- **Progressive online learning** that refines diversity metrics as optimization discovers better solutions

**Key Innovation**: QD algorithms excel at finding diverse, high-quality solutions but rely on hand-designed diversity metrics. QDHF removes this limitation by learning what humans find "interestingly different" and using this to drive exploration.

## Why QDHF?

### Problems with Existing Approaches
- **Manual diversity metrics** (standard QD): Requires domain expertise, limits applicability to open-ended tasks
- **Unsupervised methods** (AURORA, RUDA): Capture variance in current solutions but may not align with useful diversity for discovering better solutions
- **RLHF** (standard): Optimizes for average human preference, leading to mode collapse in generative tasks

### QDHF Advantages
- **Flexibility**: No need to manually design behavioral descriptors
- **Alignment**: Diversity metrics match human intuition of "interesting differences"
- **Open-ended**: Progressively adapts as the algorithm discovers novel solutions
- **Scalable**: Can use preference models trained on fixed human data instead of humans in the loop

## How It Works

### 1. Diversity Characterization via Latent Projection
Transform solutions into a compact latent space where each dimension represents a diversity metric:
```
solution x → feature extractor f(x) → latent projection D_r(f(x), θ) → z
```
- Linear projection (simple, general)
- Latent dimensions = diversity metrics for QD archive

### 2. Aligning to Human Feedback
Use **contrastive learning** with human similarity judgments:
- **Two Alternative Forced Choice (2AFC)**: Given triplet (x₁, x₂, x₃), which is more similar to x₁: x₂ or x₃?
- **Triplet loss**: Minimize distance(z₁, z₂) and maximize distance(z₁, z₃)
- **Result**: Latent space reflects human notions of similarity/diversity

### 3. Progressive Online Learning Loop
```
1. Initialize: Learn initial diversity metrics from random solutions
2. Run QD: Optimize using current diversity metrics
3. Update: Collect human feedback on newly discovered solutions
4. Fine-tune: Refine diversity metrics with new feedback
5. Repeat: Re-run QD with updated metrics (keep archive)
```

**Training Schedule**: Decreasing frequency (iterations 1, 10%, 25%, 50%) as metrics stabilize

## Algorithms

### QDHF (Progressive)
Online method that iteratively refines diversity metrics during optimization:
- Better aligns to high-quality solution space
- Adapts as exploration discovers novel behaviors
- Outperforms offline baseline

### QDHF-Base (Baseline)
Offline method following standard RLHF pipeline:
1. Collect human feedback on random solutions
2. Learn diversity metrics (like RLHF reward model)
3. Run QD optimization with fixed metrics

## Theoretical Foundation

### Information Bottleneck Perspective
Low-dimensional latent projection forces learning of **minimal sufficient statistics**:
- Compress solution X → latent Z (limited mutual information I(X;Z) ≤ I_c)
- Preserve predictive power for human judgments Y (maximize I(Z;Y))
- Result: Compact, discriminative diversity metrics

### Active Learning via QD
QD serves as **active sampling strategy** for online learning:
- Discovers novel, under-explored regions
- Provides informative samples for refining diversity metrics
- Analogous to OASIS (online similarity learning) but with adaptive exploration

## Results

### Structured Tasks (Robotics, RL)
**Robotic Arm & Maze Navigation**:
- Ground truth diversity: (x, y) position
- Simulated human feedback: L2 distance similarity
- **QDHF** significantly outperforms AURORA (unsupervised) and QDHF-Base
- Matches performance of QD with ground truth metrics (oracle)
- Better scale alignment than unsupervised methods

### Open-Ended Tasks (Text-to-Image)
**Stable Diffusion Latent Space Illumination**:
- Optimize latent vectors for diverse, high-CLIP-score images
- Human feedback: DreamSim preference model trained on NIGHTS dataset

**Singular Prompts** (e.g., "bear in national park"):
- Similar CLIP scores to baseline
- 28% higher pairwise diversity (mean distance)
- **User Study**: 58% prefer QDHF, 67% find it more diverse

**Compositional Prompts** (e.g., "red apple and yellow banana"):
- Mitigates attribute leakage through diversity
- **User Study**: 53% judge QDHF more correct vs 41% baseline

## Key Insights

### Scalability
- Performance correlates with judgment prediction accuracy (validation set)
- Can estimate needed feedback by monitoring online validation
- Preference models (DreamSim) can replace human labelers
- **Sample Efficiency**: 1,000 judgments (robotic arm), 10,000 (text-to-image)

### Robustness
- 5% label noise: Minimal impact
- 20% label noise: Still outperforms AURORA and QDHF-Base
- Validates robustness to human annotator errors

### Quality of Learned Metrics
- Better captures scale of ground truth diversity space than AURORA
- Fills under-explored regions more effectively
- Enables discovery of novel, high-quality solutions

## When to Use QDHF

**Best for**:
- Open-ended generative tasks (text-to-image, level generation, design)
- Complex domains where diversity is qualitative and hard to specify
- Applications needing personalization (diverse outputs for varied preferences)
- Tasks where manual diversity metrics are unclear or unavailable

**Consider alternatives if**:
- Clear, simple diversity metrics already exist (use standard QD)
- Budget for human feedback is extremely limited (use unsupervised QD)
- Single best solution is needed (use standard optimization or RLHF)

## Comparison to Related Methods

| Method | Diversity Source | Alignment | Adaptation |
|--------|-----------------|-----------|------------|
| **Standard QD** | Manual metrics | Expert knowledge | Fixed |
| **AURORA** | Unsupervised (PCA/AE) | Data variance | Incremental |
| **RUDA** | Unsupervised + relevance | Task performance | Adaptive |
| **QDHF** | Human feedback | Human intuition | Progressive online |
| **RLHF** | Human preferences | Average preference | Offline |

## Implementation Notes

- **Feature Extractor**: Domain-specific (forward kinematics, CLIP embeddings, etc.)
- **Projection**: Linear for simplicity, can use multi-layer
- **Distance Metric**: L2 in latent space
- **Margin**: Typically m=1 in triplet loss
- **QD Algorithm**: MAP-Elites (any QD algorithm compatible)

## References

**Paper**: Ding et al., "Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization" (ICML 2024)

**Code**: https://liding.info/qdhf

## Related

- [[QDHF_detailed]] — Technical implementation details
- [[Quality_Diversity]] — QD paradigm overview
- [[MAP_Elites]] — QD algorithm used in experiments
- [[DNS]] — Another QD variant with automatic diversity
