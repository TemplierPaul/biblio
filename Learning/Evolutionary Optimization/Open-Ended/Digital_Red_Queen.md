# Digital Red Queen (DRQ)

**Quick reference**: [[Digital_Red_Queen_detailed]]

---

## Overview

**Digital Red Queen (DRQ)** uses LLMs to evolve assembly programs (warriors) through adversarial self-play in Core War, embracing Red Queen dynamics where each new warrior must defeat all previous ones.

**Authors**: Akarsh Kumar, Ryan Bahlous-Boldi, Prafull Sharma, Phillip Isola, Sebastian Risi, Yujin Tang, David Ha  
**Affiliations**: MIT, Sakana AI  
**Key Innovation**: LLM-guided multi-round self-play produces increasingly general warriors through convergent evolution

---

## The Problem

### Static vs. Dynamic Evolution

**Traditional LLM evolution**: Optimize for fixed objective
- Problem: Overfitting to static target
- Result: Brittle specialists

**Red Queen dynamics** (nature): Continual adaptation to changing environment
- "It takes all the running you can do, to keep in the same place"
- Examples: Virus-drug arms races, predator-prey coevolution

**Need**: Study adversarial LLM evolution in controlled sandbox

---

## Core War: The Testbed

### What is Core War?

**Game**: Assembly-like programs (warriors) compete for control of virtual machine

**Environment**:
- Circular memory (8,000 cells)
- Code = data (self-modifying programs)
- Each warrior gets process(es), executed round-robin
- Win condition: Last warrior running (crash opponents)

**Why Core War**:
- ✅ **Turing-complete** (rich enough for open-ended arms race)
- ✅ **Sandboxed** (safe, controlled)
- ✅ **Studied in ALife & cybersecurity**
- ✅ **Chaotic** (small code changes → drastic outcome changes)

**Classic strategies**:
- **Bombing**: Scatter DAT instructions (crash opponents)
- **Replication**: Copy self to multiple locations
- **Scanning**: Probe memory to locate enemies

---

## DRQ Algorithm

### High-Level

```
Initialize: w₀ (base warrior)

For round t = 1 to T:
  1. Evolve new warrior wₜ to defeat {w₀, w₁, ..., wₜ₋₁}
  2. Use MAP-Elites with LLM mutations
  3. Add wₜ to history (don't update old warriors)

Output: Lineage of warriors {w₀, w₁, ..., wₜ}
```

**Key design choices**:
- **Historical self-play**: Train against all previous (not just latest) → avoid cycles
- **MAP-Elites**: Diversity preservation prevents local minima
- **LLM mutations**: Domain-aware code modifications

---

## Key Results

### 1. Static Optimization (Single Round)

**Against 294 human warriors**:

| Method | Specialist Coverage | Generalist Performance |
|--------|---------------------|------------------------|
| LLM zero-shot | 1.7% | ~1.7% |
| Best-of-8 | 22.1% | ~22.1% |
| **DRQ single round** | **96.3%** | **27.9%** |

**Interpretation**:
- **Specialist**: ≥1 evolved warrior defeats each human (collective)
- **Generalist**: Single warrior defeats many humans (individual)
- **Gap**: Evolution produces brittle specialists, not robust generalists

### 2. Multi-Round DRQ (Red Queen Dynamics)

**Observations across 96 independent runs**:

1. **Increasing generality**: Warriors defeat more diverse human opponents (p < 0.05)
2. **Decreasing phenotype variance**: Behavior converges across runs (p < 0.001)
3. **Stable genotype variance**: Source code diversity remains constant
4. **Slowing phenotype change**: Convergence rate decreases logarithmically

**Interpretation**: **Convergent evolution** toward single general-purpose behavior
- Similar to mammalian/insect eyes evolving independently
- Different code (genotype) → similar behavior (phenotype)

---

## When to Use DRQ

✅ **Use when**:
- Studying **adversarial dynamics** in controlled environment
- Need **robust generalists** (not brittle specialists)
- Have **simulatable competition** (e.g., games, security, biology)
- Want to leverage **LLM priors** for code/program evolution

❌ **Consider alternatives when**:
- Static objective sufficient → Standard evolution
- No adversarial interaction → POET, OMNI-EPIC (environment diversity)
- Real-world deployment → Safety concerns (sandbox first!)

---

## Applications

### 1. Core War (Current)

**Scientific study**:
- Understand Red Queen dynamics in LLMs
- Benchmark LLM evolution capabilities
- Safe testbed for adversarial adaptation

### 2. Cybersecurity (Potential)

**Exploit/defense evolution**:
- Evolve exploits against hardening defenses
- Evolve defenses against emerging exploits
- Controlled study before real deployment

### 3. Drug Discovery (Potential)

**Virus/drug arms race**:
- Evolve drug candidates against resistant viruses
- Model resistance mechanisms
- Guide actual drug development

### 4. Multi-Agent RL (Potential)

**General adversarial domains**:
- Self-driving car safety
- Game AI
- Robotic soccer
- Any competitive multi-agent setting

---

## Comparison to Related Work

### vs POET/Enhanced POET
- **POET**: Environment-agent coevolution (diversity of environments)
- **DRQ**: Adversarial self-play (Red Queen dynamics)
- **Both**: Open-ended, but different pressures

### vs Fictitious Self-Play (FSP) / PSRO
- **FSP/PSRO**: Game-theoretic frameworks, construct meta-strategies
- **DRQ**: Direct optimization against all previous (no meta-game)
- **DRQ**: Evolutionary inner loop (not action-based RL)

### vs Foundation Model Self-Play (FMSP)
- **FMSP**: LLM self-play in 2D evader-pursuer, red-teaming
- **DRQ**: Core War (Turing-complete, richer domain)
- **DRQ**: Emphasis on scientific study, not just method

---

## Key Insights

1. **Red Queen > static**: Multi-round produces generalists, single-round produces specialists
2. **Convergent evolution**: Independent runs converge in behavior (not code)
3. **Diversity preservation matters**: MAP-Elites prevents local minima in deceptive landscapes
4. **LLM priors essential**: Zero-shot ~2%, evolved ~96% (orders of magnitude speedup)
5. **Predictability**: Linear probe on embeddings achieves R²=0.461 for generality
6. **Logarithmic convergence**: Full convergence requires exponential rounds

---

## Components

### MAP-Elites (Intra-Round Optimization)

**Behavioral descriptors**:
- Total spawned processes (via SPL opcodes)
- Total memory coverage

**Archive**: Grid of cells (discretized behavior space)
- Each cell stores best warrior with that behavior
- Localized selection pressure, global diversity

**Mutation**: LLM-guided
- System prompt: Core War environment, Redcode manual
- Mutation prompt: Modify existing warrior to improve

### Fitness Function

$$\text{Fitness}(w_i; \{w_j\}_{j \neq i}) = \sum_{\tau=1}^{\mathcal{T}} \frac{N}{\mathcal{T}} \frac{A^i_\tau}{\sum_j A^j_\tau}$$

- $A^i_\tau$ = 1 if warrior $i$ alive at time $\tau$, else 0
- Incentivizes: Survive long + eliminate others (increase share)
- Context-dependent (depends on other warriors)

---

## Limitations

1. **Computational cost**: Multi-round expensive (96 runs, many iterations)
2. **Domain-specific**: Core War may not transfer to all adversarial domains
3. **Convergence rate**: Logarithmic (exponential rounds for full convergence)
4. **LLM dependence**: Requires capable code-generation model
5. **Safety**: Real-world deployment needs careful sandboxing

---

## References

- **Paper**: Kumar et al., "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs" (MIT, Sakana AI, 2025)
- **Website**: https://pub.sakana.ai/drq
- **Code**: https://github.com/SakanaAI/drq
- **Core War**: Dewdney (1984), Jones (1984)
- **Related**: MAP-Elites (Mouret & Clune, 2015), PSRO (Lanctot et al., 2017), FMSP (Dharna et al., 2025)

---

**See `Digital_Red_Queen_detailed.md` for complete algorithm, Core War details, prompts, and experimental analysis.**
