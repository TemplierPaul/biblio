# Voyager: LLM-Powered Embodied Lifelong Learning Agent

**Paper**: Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (NVIDIA, Caltech, UT Austin, Stanford, 2023)

**Problem**: Building embodied agents that continuously explore, plan, and develop new skills in open-ended worlds without human intervention.

**Solution**: First LLM-powered embodied lifelong learning agent that uses GPT-4 to autonomously explore Minecraft, acquire diverse skills, and make novel discoveries.

---

## Core Idea

Voyager leverages GPT-4 as a blackbox (no fine-tuning) to drive open-ended exploration in Minecraft through three key components:

1. **Automatic Curriculum**: Proposes suitable tasks based on current skill level and world state
2. **Skill Library**: Stores and retrieves executable code representing complex behaviors
3. **Iterative Prompting**: Generates and refines code using environment feedback, execution errors, and self-verification

**Key insight**: Use **code as action space** instead of low-level motor commands → enables temporally extended, compositional, and interpretable actions.

---

## Three Key Components

### 1. Automatic Curriculum

**Purpose**: Maximize exploration by proposing achievable tasks

**How it works**:
- Prompts GPT-4 with goal: "Discover as many diverse things as possible"
- Provides agent's current state (inventory, nearby blocks, biome, health, position)
- Lists previously completed/failed tasks
- GPT-4 proposes next task based on exploration frontier

**Example progression**:
- Desert biome → "Harvest sand and cactus" (not "mine iron")
- Forest biome → "Chop wood and craft tools"

**Benefits**:
- Bottom-up curriculum (adapts to agent's state)
- Curiosity-driven exploration (novelty search in-context)
- No predefined goals

### 2. Skill Library

**Purpose**: Store and reuse complex behaviors to compound capabilities

**Representation**: Executable JavaScript code (Mineflayer APIs)

**Storage**:
- Key: Embedding of skill description (GPT-3.5)
- Value: Code program itself
- Vector database for similarity-based retrieval

**Retrieval**:
- Query: Embedding of task + environment feedback
- Returns: Top-5 relevant skills

**Example skills**:
```javascript
craftStoneShovel()
combatZombieWithSword()
fillBucketWithWater()
```

**Benefits**:
- **Compositional**: Complex skills built from simpler ones
- **Interpretable**: Human-readable code
- **Reusable**: Transfer to similar situations
- **Alleviates catastrophic forgetting**: Skills persist in library

### 3. Iterative Prompting Mechanism

**Problem**: LLMs struggle to generate correct code in one shot

**Solution**: Self-improvement loop with three types of feedback

**Three feedback types**:

1. **Environment feedback**: Intermediate execution progress
   - Example: "I cannot make iron chestplate because I need: 7 more iron ingots"
   - Shows why task failed

2. **Execution errors**: Invalid operations or syntax errors
   - Example: "craftAcaciaAxe is not a function" → craft wooden axe instead
   - Enables bug fixing

3. **Self-verification**: GPT-4 as critic checks task success
   - Input: Agent's current state + task
   - Output: Success/failure + critique if failed
   - More comprehensive than self-reflection (check + critique)

**Iterative loop**:
```
1. GPT-4 generates code for task
2. Execute code → get environment feedback + execution errors
3. Incorporate feedback into prompt
4. Refine code (up to 4 rounds)
5. Self-verification checks success
   - If success → add to skill library, query curriculum for next task
   - If failure after 4 rounds → query curriculum for different task
```

---

## Results

**Experimental setup**: MineDojo (Minecraft), GPT-4 API, Mineflayer JavaScript APIs

**Baselines**: ReAct, Reflexion, AutoGPT (re-implemented for embodied setting)

### Exploration Performance

- **3.3× more unique items**: 63 items vs 19 (AutoGPT), ~5 (ReAct/Reflexion)
- **Sustained discovery**: Continuous progress over 160 iterations

### Tech Tree Mastery

Minecraft tech tree: wooden → stone → iron → diamond tools

- **Wooden**: 15.3× faster (6 vs 92 iterations)
- **Stone**: 8.5× faster (11 vs 94 iterations)
- **Iron**: 6.4× faster (21 vs 135 iterations)
- **Diamond**: Only Voyager unlocks (102 iterations, 1/3 trials)

### Map Coverage

- **2.3× longer distances** than baselines
- Traverses diverse terrains (8-12 biomes vs 2-4 for baselines)

### Zero-Shot Generalization

**Setup**: New world, cleared inventory, unseen tasks

**Tasks**: Craft diamond pickaxe, golden sword, lava bucket, compass

**Results**:
- Voyager: Solves all tasks (100% success in 18-21 iterations)
- Baselines: 0% success (cannot solve any task in 50 iterations)
- AutoGPT w/ Voyager's skill library: Partial success (helps other methods!)

---

## Ablation Studies

**Critical components** (% drop if removed):

1. **Self-verification**: -73% items (most important feedback)
2. **Automatic curriculum**: -93% with random curriculum
3. **Skill library**: Plateaus in later stages without it
4. **GPT-4 vs GPT-3.5**: 5.7× more items (GPT-4 essential for code quality)

**Manual curriculum**: Requires domain expertise, not scalable, worse than automatic

---

## Key Innovations

### 1. Lifelong Learning Without Gradient Updates

- No model fine-tuning
- No reinforcement learning training
- Pure in-context learning via prompting

### 2. Code as Compositional Action Space

**Why code?**
- Temporally extended actions (multi-step sequences)
- Compositional (build complex from simple)
- Interpretable (human-readable)
- Reusable (store in library)

**Contrast with**:
- Low-level actions: Hard to explore, not interpretable
- Natural language plans: Not executable

### 3. Self-Driven Curriculum

**In-context novelty search**: GPT-4 proposes diverse tasks without explicit diversity objective

**Adaptive**: Considers agent's current capabilities and world state

**Warm-up schedule**: Gradually incorporates more information as agent progresses

### 4. Comprehensive Feedback Loop

**Three complementary feedback types**:
- Environment: Why task failed
- Execution: How to fix bugs
- Self-verification: Whether task succeeded + how to improve

---

## Limitations

**Cost**: GPT-4 API expensive (15× more than GPT-3.5), but necessary for code quality

**Inaccuracies**: Agent sometimes gets stuck, automatic curriculum retries later

**Hallucinations**:
- Proposes non-existent items ("copper sword")
- Uses invalid operations (cobblestone as fuel)
- Calls non-existent functions

**Perception**: Text-only (no vision), but can integrate human feedback for 3D structures

---

## Comparison with Prior Minecraft Agents

| Method | Demos | Rewards | Actions | Curriculum | Skill Library | Gradient-Free |
|--------|-------|---------|---------|------------|---------------|---------------|
| VPT | Videos | Sparse | Keyboard/Mouse | ✗ | ✗ | ✗ |
| DreamerV3 | None | Dense | Discrete | ✗ | ✗ | ✗ |
| DECKARD | Videos | Sparse | Keyboard/Mouse | ✓ | ✗ | ✗ |
| DEPS | None | None | Keyboard/Mouse | ✗ | ✗ | ✗ |
| Plan4MC | None | Dense | Discrete | ✗ | ✓ (predefined) | ✗ |
| **Voyager** | **None** | **None** | **Code** | **✓ (GPT-4)** | **✓ (self-generated)** | **✓** |

---

## Broader Applications

**Current domain**: Minecraft (safe, harmless environment)

**Potential domains**:
- Robotics (with safety constraints)
- Other open-world games
- Procedurally generated environments
- Any domain with executable APIs

**Requirements**:
- Environment with programmatic control APIs
- LLM with strong code generation capabilities
- Verifiable task completion

---

## Connection to Open-Endedness

**Voyager as open-ended system** (Hughes et al., 2024 framework):

- **Novel**: Discovers items/skills unpredictable from initial state
- **Learnable**: Skills build on previous skills (tech tree progression)
- **Stepping stones**: Automatic curriculum creates path to hard tasks
- **Self-driven**: No predefined endpoint

**Open-ended foundation model** (one of four paths):
- Uses LLM (foundation model) for task generation + code generation
- Combines self-improvement + task generation paths
- Example of FM-augmented open-endedness

---

## Key Takeaways

1. **LLMs as lifelong learners**: GPT-4 can drive autonomous exploration without fine-tuning
2. **Code > actions**: Programs compound capabilities faster than low-level controls
3. **Curriculum matters**: Automatic curriculum 93% better than random
4. **Feedback is essential**: Self-verification most critical (73% drop without)
5. **Skill library enables transfer**: Generalizes to new worlds and novel tasks
6. **Interpretability for free**: Code skills are human-readable and debuggable

**The vision**: Embodied agents that continuously learn, explore, and improve in open-ended worlds through self-driven discovery.
