# OMNI-EPIC: Foundation Model-Driven Open-Ended Learning

## Overview

**OMNI-EPIC** (Open-endedness via Models of human Notions of Interestingness + Environments Programmed in Code) leverages foundation models (LLMs and VLMs) to autonomously generate learnable and interesting tasks in perpetuity. Unlike POET which modifies predefined environment parameters, OMNI-EPIC generates complete environment code from natural language descriptions, moving toward **Darwin Completeness**: the ability to generate any possible learning environment.

**Core Innovation**: Uses LLMs to generate both task descriptions AND executable Python code for environments, reward functions, and success detectors, enabling theoretically unbounded task diversity.

---

## Key Problem Solved

### Limitations of Prior Open-Ended Algorithms

**POET and Enhanced POET**:
- ✅ Demonstrated open-ended coevolution
- ❌ Limited to predefined environment templates (obstacle courses)
- ❌ Fixed parameter spaces exhaustible after sufficient iterations
- ❌ Required human domain expertise to design environments

**OMNI-EPIC Solution**:
- Generates arbitrary environment code (not just parameters)
- No predefined templates or task distributions
- Leverages foundation models' understanding of "interestingness" from internet-scale data
- Moves toward Darwin Completeness

---

## Key Concepts

### 1. Darwin Completeness

**Definition**: The ability to generate **any possible learning environment**.

**Why It Matters**:
- Biological evolution required vast possibility space (billions of years)
- Only known process to produce general intelligence
- Prior algorithms artificially constrained to narrow domains
- Darwin Completeness removes these constraints

**Current Status**:
- ✅ Can generate arbitrary code within PyBullet simulator
- ❌ Not yet Darwin Complete (constrained to single simulator)
- 🎯 Path forward: Allow LLMs to install/modify/write new simulators

**Vision**: Once Darwin Complete, could generate:
- Logic puzzles
- Math problems
- Minecraft quests
- Novel game mechanics
- Scientific simulation tasks
- Any computable environment

### 2. Model of Interestingness (MoI)

**Problem**: How to define "interesting" without human-specified objectives?

**Solution**: Dual-stage MoI inspired by human cultural dynamics.

**Stage 1: Pre-Generation MoI**
- Embedded in LLM's training on internet-scale human data
- Foundation models have "distilled sense of what is interesting"
- Guides initial task proposals
- Considers tasks that are "worthwhile and sufficiently novel"

**Stage 2: Post-Generation MoI**
- Separate evaluation stage using retrieval-augmented generation
- Compares new task against 10 most similar archived tasks
- Asks: "Is this novel, surprising, diverse, and worthwhile?"
- Discards uninteresting tasks

**Analogy to Science**:
1. Scientist studies subset of prior work (not all papers)
2. Gets inspired, proposes new research
3. Verifies novelty against literature
4. If novel, publishes and adds to knowledge base

### 3. LLM-Generated Environments

**Pipeline**:
```
Natural Language Description → LLM → Python Code → Executable Environment
```

**Components Generated**:
1. **World simulation**: Object placements, physics setup
2. **Reward function**: Learning signals for RL agents
3. **Success detector**: Task completion verification
4. **Termination conditions**: Episode ending logic

**Example**:
- Task: "Cross a rainbow bridge with moving segments"
- Generated code:
  - Initializes bridge with colorful segments
  - Implements segment motion (horizontal oscillation)
  - Rewards forward progress + balance maintenance
  - Detects falling as terminal state
  - Success when agent reaches end

---

## Algorithm Overview

### Main Loop

```
1. Initialize task archive with 3 seed descriptions
2. FOR each iteration:
    a. Task Generator:
       - Retrieve 5 similar successful + 5 failed tasks from archive
       - Generate new natural language task description (Claude 3 Opus)

    b. Environment Generator:
       - Convert task description to Python/PyBullet code (Claude 3 Opus)
       - Include error correction loop (max 5 attempts)

    c. Post-Generation MoI:
       - Retrieve 10 most similar archived tasks
       - Evaluate interestingness (GPT-4o)
       - If uninteresting: discard and regenerate

    d. RL Training:
       - Initialize policy from most similar successful task (transfer)
       - Train agent with DreamerV3 (2M timesteps, ~1 hour)

    e. Success Detection:
       - Evaluate task completion with LLM-generated get_success() function

    f. Archive Update:
       - If successful: add to archive as learned
       - If failed: add to archive as failed (informs future generation)
       - If unlearnable after max attempts: mark as failed
```

### Key Components

**Task Archive**:
- Stores: natural language + code + success status
- Grows unbounded over time
- Used for: retrieval-augmented generation, stepping stones

**Stepping Stone Strategy**:
- Retrieve **similar** tasks (not maximally different)
- Build on partial knowledge
- Progressive complexity increase
- Example: "cross bridge" → "cross bridge with gaps" → "cross bridge with moving segments"

**Adaptive Curriculum**:
- Learns from both successes AND failures
- Failed tasks inform avoidance patterns
- Example: "push box on dynamic platform" fails → future tasks avoid pushing on platforms

---

## Experimental Results

### Long Run (200 Tasks, Simulated Learning)

**Task Diversity**:
- Generated 200 diverse tasks
- Self-organized into meaningful clusters:
  - Ball-kicking tasks (kick ball into goal)
  - Dynamic ball tasks (moving obstacles + ball)
  - Object manipulation (pushing/delivering objects)
  - Navigation challenges (platforms, varied terrain)

**Emergent Complexity**:
- Generated elements absent from seeds:
  - Moving platforms (horizontal/vertical)
  - Buttons and levers
  - Moving obstacles
  - Multi-level environments

**Task Variations**:
- Different simulated worlds (outdoor/indoor/multi-level)
- Varying object counts
- Time limits
- Combinations of learned skills

### Short Runs (5 Runs with Actual RL Training)

**Performance**:
- Example run: 16 successful, 6 failed, 1 uninteresting
- Successfully trained agents on progressively complex tasks
- Demonstrated skill composition (combining learned abilities)

**Curriculum Adaptation**:
- Failed tasks inform future generation
- Example: "navigate terrain with obstacles" fails → generate easier version with fewer obstacles

**Quantitative Metrics**:

**1. Cell Coverage (Diversity)**
- Measured via embedding space discretization (2D PCA)
- **Result**: OMNI-EPIC significantly higher coverage (p < 0.05)
- **Ablation**: Both archive AND MoI contribute significantly

**2. ANNECS-OMNI (Sustained Innovation)**
- Extends ANNECS with interestingness criterion
- **Result**: Consistent increase throughout run (no stagnation)
- **vs. Controls**: Significantly outperforms (p < 0.05)
  - Without archive: redundant tasks
  - Without MoI: learnable but uninteresting tasks

---

## When to Use OMNI-EPIC

### ✅ Good Fit

**1. Unlimited Task Generation**
- Need perpetual curriculum
- No predefined task distribution
- Example: Open-ended game content generation

**2. Novel Environment Types**
- Tasks don't fit existing templates
- Require custom reward functions
- Example: Research exploring new RL domains

**3. Human-Interpretable Tasks**
- Natural language descriptions valuable
- Need explainability
- Example: Educational applications with adaptive difficulty

**4. Leveraging Foundation Models**
- Have access to capable LLMs (Claude Opus, GPT-4)
- Can afford inference costs
- Example: Commercial applications with API budgets

### ❌ Poor Fit

**1. Fixed Task Distribution**
- Known set of tasks to master
- Better suited for curriculum learning
- Example: Standard RL benchmarks

**2. Real-Time Generation**
- Need instant environment creation
- LLM latency unacceptable
- Better suited for template-based generation

**3. Limited Compute Budget**
- Each task requires ~1 GPU-hour training
- LLM API costs
- Better suited for lightweight methods

**4. Safety-Critical Domains**
- Generated code may have bugs
- Unverified environments risky
- Better suited for human-designed environments

---

## Key Innovations

### 1. Code Generation Beyond Parameters

**Problem**: POET limited to predefined parameter spaces

**Solution**: Generate complete Python code including:
- Environment initialization
- Physics simulation
- Reward shaping
- Termination logic

**Benefit**: Theoretically unbounded environment diversity

### 2. Universal Success Detector

**Problem**: Hand-coding success conditions for each task

**Solution**: LLM generates `get_success()` function

**Validation**: 72.7% human alignment

**Benefit**: Works across any task type without domain expertise

### 3. Dual-Stage MoI

**Problem**: Defining "interesting" without human objectives

**Solution**:
- Pre-generation: LLM's inherent sense from training
- Post-generation: Explicit comparison to archive

**Benefit**: Prevents redundancy while maintaining open-endedness

### 4. Stepping Stones from Failures

**Problem**: Failed tasks waste compute

**Solution**: Archive failures to inform future generation

**Example**: Failed "push on dynamic platform" → avoid similar patterns

**Benefit**: Negative curriculum learning accelerates finding learnable tasks

### 5. ANNECS-OMNI Metric

**Problem**: Measuring open-ended progress

**Solution**: Extend ANNECS with interestingness criterion

**Result**: Demonstrates sustained innovation (no plateau)

---

## Comparison to Related Methods

| Method | Environment Generation | Task Diversity | Interestingness Model |
|--------|----------------------|----------------|---------------------|
| **POET** | Parameter mutation | Limited (template-based) | Learning progress only |
| **Enhanced POET** | CPPN evolution | Higher (but template-based) | Learning progress only |
| **OMNI** | FM ideation + parameters | Medium (parameter-constrained) | FM-based MoI |
| **OMNI-EPIC** | FM code generation | Theoretically unbounded | Dual-stage FM-based MoI |

**OMNI-EPIC's Unique Position**: First to generate executable environment code via LLMs, enabling arbitrary task types beyond predefined templates.

---

## Interview Relevance

### For Research Scientists

**Common Questions**:
- "What is Darwin Completeness and why does it matter?"
  - Ability to generate any computable environment; mirrors biological evolution's vast possibility space
- "How do foundation models encode 'interestingness'?"
  - Distilled from internet-scale human data; reflects human notions of worthwhile/novel challenges
- "Why stepping stones from similar (not maximally different) tasks?"
  - Mimics human creativity; partial knowledge more useful than overwhelming context

**Discussion Topics**:
- Open-endedness as path to AGI
- Foundation models for scientific discovery
- Computational constraints vs. biological evolution timescales
- Interestingness as emergent property of language models

### For ML Engineers

**Practical Considerations**:
- LLM APIs: Claude 3 Opus for generation, GPT-4o for evaluation
- Compute: ~1 GPU-hour per task (DreamerV3 training)
- Framework: PyBullet for physics, Gymnasium API compliance
- Error handling: Code generation failures require iteration

**Applications**:
- Game content generation with adaptive difficulty
- Automatic RL benchmark creation
- Educational curriculum generation
- Research domain exploration

---

## Connection to Other Topics

- **[[POET]]**: Foundation algorithm OMNI-EPIC extends beyond parameter spaces
- **[[Enhanced_POET]]**: CPPN evolution for richer environments (still template-based)
- **[[Quality_Diversity]]**: OMNI-EPIC maintains diverse archive like QD algorithms
- **[[Foundation_Models]]**: Core enabling technology for code generation
- **[[Curriculum_Learning]]**: OMNI-EPIC automatically generates progressive curricula
- **[[DreamerV3]]**: World model-based RL used for agent training
- **[[Open_Endedness]]**: OMNI-EPIC as approach to sustained innovation

---

## Limitations

**Current Constraints**:
1. **Simulator-Bound**: Limited to PyBullet (not Darwin Complete)
2. **Specialist Agents**: One policy per task (not generalist)
3. **Computational Cost**: 1 GPU-hour per task
4. **Code Quality**: Iterative error correction needed (max 5 attempts)
5. **Success Detection**: 72.7% human alignment (not perfect)

**Future Directions**:
1. Allow arbitrary simulator installation/modification
2. Train generalist policies across task distributions
3. More efficient RL algorithms
4. Better code generation (fewer errors)
5. VLM-based success detection (visual understanding)

---

## Quick Summary

**What**: LLM-driven open-ended learning system that generates complete environment code from natural language

**Why**: Moves toward Darwin Completeness (any computable environment) beyond predefined templates

**How**:
1. Generate task descriptions with archive context (similar tasks)
2. Convert to executable Python code (Gymnasium API)
3. Filter by interestingness (compare to archive)
4. Train RL agents (transfer from similar tasks)
5. Archive results (successes and failures)

**Key Result**: Sustained innovation over 200+ tasks with emergent complexity and self-organized diversity

---

**See [[OMNI_EPIC_detailed]] for implementation details, prompts, hyperparameters, and code examples.**
