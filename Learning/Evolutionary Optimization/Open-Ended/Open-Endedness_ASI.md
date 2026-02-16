# Open-Endedness is Essential for Artificial Superhuman Intelligence

**Quick reference**: [[Open-Endedness_ASI_detailed]]

---

## Overview

This is a **position paper** from Google DeepMind arguing that **open-endedness is essential for ASI** and providing the first rigorous formal definition of open-endedness through the lens of novelty and learnability.

**Authors**: Hughes, Dennis, Parker-Holder, Behbahani, Mavalankar, Shi, Schaul, Rocktäschel (Google DeepMind)
**Venue**: ICML 2024
**Type**: Position paper / Conceptual framework
**Key Contribution**: Formal definition + roadmap for open-ended foundation models

---

## Core Definition

### What is Open-Endedness?

**Position**: *From the perspective of an observer, a system is open-ended if and only if the sequence of artifacts it produces is both novel and learnable.*

**Formal Definition**:

A **system** S produces artifacts X_t at time t. An **observer** O has a statistical model X̂_t that predicts future artifacts based on history X_{1:t}. Quality measured by loss ℓ(X̂_t, X_T).

**1. Novelty**: Artifacts become increasingly unpredictable
```
∀t, ∀T>t, ∃T'>T: E[ℓ(t,T')] > E[ℓ(t,T)]
```
*There's always a less predictable artifact coming in the future*

**2. Learnability**: Longer history makes artifacts more predictable
```
∀T, ∀t<T, ∀t'>t: E[ℓ(t',T)] < E[ℓ(t,T)]
```
*Conditioning on more history improves predictions*

**Open-ended** = Novel AND Learnable

---

## Why This Definition Matters

### The Noisy TV Test

**Noisy TV**: Produces uniform random noise
- ✅ Learnable (can learn the uniform distribution)
- ❌ Novel (once you've learned uniform, no more novelty)
- **Not open-ended**

**Channel-Switching TV**: Random channels with arbitrary distributions
- ✅ Novel (every channel switch brings new patterns)
- ❌ Learnable (past channels don't predict future ones)
- **Not open-ended**

**Research lab publishing papers**:
- ✅ Novel (quantum mechanics is more surprising than Newtonian)
- ✅ Learnable (citations help you understand current papers)
- **Open-ended!**

### Observer-Dependent

Open-endedness depends on WHO is observing:

**Example: Aircraft design system**

| Observer | Novel? | Learnable? | Open-ended? |
|----------|--------|------------|-------------|
| Mouse | ✅ Yes | ❌ No | ❌ No |
| Aerospace student | ✅ Yes | ✅ Yes | ✅ Yes |
| Superintelligent alien | ❌ No | ✅ Yes | ❌ No |

**Insight**: Open-endedness is relative to observer's knowledge and capabilities.

---

## Why Open-Endedness is Essential for ASI

### The Argument

**ASI Definition**: Artificial Superhuman Intelligence that accomplishes tasks beyond any human capability.

**Why open-endedness is essential**:
1. **Superhuman by definition** = produces novel solutions humans can't predict
2. **Useful to humanity** = humans must be able to learn from these solutions
3. **Self-improving** = must create, refute, refine its own knowledge

**Key insight**: "We'll be surprised but we'll be surprised in a way that makes sense in retrospect" (Lisa B. Soros)

### Current Limitations

**Foundation models are NOT open-ended**:
- Trained on **fixed datasets**
- Once you model the epistemic uncertainty → no more novelty
- Like a noisy TV: learnable but eventually not novel

**Existing open-ended systems are NOT general**:
- **AlphaGo**: Open-ended in Go, but narrow domain
- **AdA**: Open-ended in XLand2, but plateaus (finite open-endedness)
- **POET**: Open-ended evolution, but limited by environment parameterization

**The gap**: Need to combine open-endedness (for novelty) with foundation models (for generality)

---

## Open-Ended Foundation Models: The Path to ASI

### Why This Combination is Powerful

**Foundation models provide**:
1. **Human knowledge**: Trained on internet-scale data
2. **Human interestingness**: Capture what humans find meaningful
3. **General mutation operators**: Can generate variations from examples
4. **Semantic understanding**: Guide search toward human-relevant artifacts

**Open-ended algorithms provide**:
1. **Experience generation**: Active exploration, not passive data collection
2. **Self-improvement**: Create own explanatory knowledge
3. **Perpetual novelty**: Don't plateau like fixed datasets

**Synergy**: Open-ended FM can both vary data AND assess novelty/interestingness to decide what to explore next.

### Four Paths to Open-Ended Foundation Models

#### 1. Reinforcement Learning

**Key idea**: Agents shape their experience for reward + learning

**Examples**:
- **Voyager** (LLM-powered Minecraft): Curriculum + iterative prompting + skill library
- **AdA** (XLand2): Automatic curriculum over 25B task variants
- **POET**: Coevolving agent-environment pairs

**Challenge**: How to guide exploration toward novel AND learnable behaviors in high-D spaces?

**Solution**: Use FMs as **proxy observers** to:
- Provide rewards from text (MOTIF)
- Compile curriculum based on interestingness (OMNI)
- Guide exploration toward human-relevant artifacts

**Multi-agent**: Self-play creates non-stationarity → open-ended strategy evolution (AlphaGo, StarCraft, Diplomacy)

#### 2. Self-Improvement

**Key idea**: Model generates new knowledge, not just consumes pre-collected feedback

**Requirements**:
- Scalable self-evaluation mechanism
- Identify areas for improvement
- Adapt learning process

**Examples**:
- Constitutional AI: Self-critique and revision
- Self-instruction: Generate training data for instruction following
- Self-debugging: Improve code generation
- Self-rewarding: LLMs as reward models

**Challenge**: Move beyond amplifying human data to discovering truly new knowledge

#### 3. Task Generation

**Key idea**: Adapt task difficulty to agent capability → forever challenging yet learnable

**Examples**:
- **Setter-solvers**: One model generates tasks, another solves them
- **UED** (Unsupervised Environment Design): POET, PAIRED
- **Web as environment**: Internet provides rich, ever-growing task domain (WebArena)
- **Learned world models**: Genie, Sora as simulators

**Challenge**: Generate tasks that remain human-relevant at scale

**Vision**: FM world models + learned reward models → open-ended curriculum generation

#### 4. Evolutionary Algorithms

**Key idea**: LLMs as selection and mutation operators

**Advantages**:
- Trained on vast human knowledge → semantically meaningful mutations
- Can evaluate quality and diversity of candidates

**Examples**:

**Prompt evolution**:
- **PromptBreeder**: Evolve prompts far beyond human-designed
- Iterative improvement via LLM feedback

**Code evolution** (Genetic Programming):
- **Eureka**: Evolve reward functions for control
- **FunSearch**: Evolve programs discovering new math (extremal combinatorics)

**Quality Diversity**:
- LLMs generate variation AND evaluate quality/diversity of text
- Guide search for creative, novel outputs

**Challenge**: Scale code evolution to general setting beyond specific domains

---

## Existing Systems: Open-Ended but Not General, or General but Not Open-Ended

### AlphaGo: Open-Ended but Narrow

**Artifacts**: Sequence of policies produced during training

**Novel?** ✅ Plays moves low-probability for humans but winning
**Learnable?** ✅ Humans improve win-rate by learning from AlphaGo

**Limitation**: Narrow superhuman intelligence. Can't help discover new science or technology across fields.

### AdA: Open-Ended but Finite

**Artifacts**: Agent checkpoints across training in XLand2 (25B task variants)

**Novel?** ✅ Continually shows new zero-shot/few-shot capabilities
**Learnable?** ✅ Task prioritization provides interpretable skill ordering

**Limitation**: Novelty plateaus after ~1 month. Needs richer environment + larger agent for longer timescales.

### POET: Open-Ended but Domain-Limited

**Artifacts**: Paired agent-environment populations

**Novel?** ✅ QD hunts for challenging problems → diverging performance
**Learnable?** ✅ Small mutations → past lineage predicts current features

**Limitation**: Plateaus once agent solves all possible terrains. Environment parameterization bounds open-endedness.

### Foundation Models: General but Not Open-Ended

**Artifacts**: Model outputs

**Novel?** ❌ Fixed dataset → epistemic uncertainty eventually modeled
**Learnable?** ✅ (That's what training did!)

**Limitation**: May appear open-ended to humans (broad domain + memory limits) but narrow focus exposes limitations (e.g., planning tasks).

**Key insight**: FMs need distributional shift (currently seen as "model collapse" threat) to become open-ended. Flip the script: augment FMs with open-endedness!

---

## Time Horizons & Observer Types

### Finite vs Infinite Open-Endedness

**Infinitely open-ended**: Remains open-ended for any timescale τ → ∞

**Finitely open-ended** (time horizon τ): Open-ended for t,T < τ

**Example**: AdA is finitely open-ended with τ ≈ 1 month (then novelty plateaus)

### Observer Constraints

**Memory limitations**: Wikipedia reading curriculum
- Novel until human memory saturates
- Then learnability fails (forget definitions needed for later articles)

**Cognitive breadth**: Narrow domain (elliptic curve cryptography)
- Finite Wikipedia articles → novelty violated once all understood
- But humans can still make new discoveries via experimentation

**Human vs ASI observers**:
- ASI may have less stringent memory constraints
- Judges itself open-ended beyond point humans do
- **Safety implication**: Human observers must remain pre-eminent

---

## Safety: Achieving ASI Responsibly

### The Challenge

**Power of open-endedness** comes with **safety risks** beyond existing FM concerns.

**Critical principle**: Safety and open-endedness must be pursued **in tandem**.

### Knowledge Creation & Transfer Framework

Five processes in human-AI open-ended systems:
1. **AI builds on AI knowledge**
2. **Humans understand AI creations**
3. **AI understands human knowledge**
4. **Humans build on human knowledge**
5. **Emergent knowledge from the process**

Each process offers opportunity to embed safety methods.

### Key Safety Challenges

#### 1. AI Creation and Agency

**Risks**:
- Dual-use dangers (powerful new affordances without direction)
- Goal misgeneralization
- Specification gaming
- Safe exploration in real-world deployment

**Mitigations**:
- Safe exploration techniques from RL
- Impact regularization
- Constrain agency scope initially (narrow simulations → broader → real world)

#### 2. Humans Understanding AI Creations

**Problem**: As artifact complexity grows → can't give informed oversight → no longer learnable → no longer open-ended!

**Implication**: Making systems understandable is not just safety—it's **requirement for usefulness**.

**Approaches**:

**Interpretability**:
- Automated interpretability (scale with system complexity)
- Challenge: Would require universal explainer

**Design for interpretability**:
- Systems that directly inform users of implicit knowledge
- Maintain informed oversight by design
- Facilitate understanding and control

#### 3. Humans Guiding AI Creation

**Challenge**: How to meaningfully guide increasingly unpredictable system?

**Problem beyond RL**: Open-ended systems lack well-defined objectives AND are unpredictable by design.

**Approaches**:
- **Open-endedness from human feedback** (like PicBreeder, OMNI)
- Objectives that preserve controllability
- Mechanisms to surface unexpected important artifacts

**Directability paradox**: Must direct toward safe/useful artifacts while maintaining open-endedness (avoid rabbit-holing into obscure but uninteresting areas).

#### 4. Human Society Adapting

**Non-technical concerns**:
- Society understanding new technological capabilities
- Preparing for and reacting to novel artifacts
- Impact on social structures (communities, organizations, markets, nations)

**Challenges**:
- Avoid tipping points (feedback loops, flash crashes)
- Balance information gathering vs avoiding entrenchment
- Rapid, retrospective governance adaptation (Collingridge dilemma)

#### 5. Emergent Risks

**Even if all components are safe**, aggregate system may have unforeseen problems:
- Two open-ended systems negatively interact → neither remains open-ended
- Novel ASI safety failures emerge

**Anti-fragile safety**: System adapts to emerging risks and gets stronger
- Monitor emerging risks
- Rapidly coordinate responses
- Design for adaptation to unpredictable problems

---

## Alternative Definition: Compression-Based

### Compression Formulation

**System** S produces artifacts X_t. **Observer** O has compression map C_{h_t} that encodes X_T into binary string.

**Novelty**: Information content increases
```
∀t, ∀T>t, ∃T'>T: |C_{h_t}(X_{T'})| > |C_{h_t}(X_T)|
```

**Learnability**: Longer history increases compressibility
```
∀T, ∀t'>t: |C_{h_{t'}}(X_T)| < |C_{h_t}(X_T)|
```

**Open-ended** = Novel (complexity grows) AND Learnable (extracting patterns helps compress future)

**Connection**: Compression and statistical learning are formally related (Hutter, David et al.)

**Rate-distortion extension**: Plot minimum compression rate vs distortion threshold
- **Broad novelty**: Curves get "fatter" across columns (T increasing)
- **Broad learnability**: Curves get "flatter" down rows (t increasing)

---

## Practical Implications

### How to Measure Open-Endedness

**1. Human feedback**: Direct elicitation (like RLHF, PicBreeder)

**2. LLM judges**: Use LLMs to assess novelty and learnability (OMNI)

**3. Explicit learning**: Learn model with online learning (Follow-the-Regularized-Leader)

### When is a System Open-Ended?

**YES (Open-ended)**:
- ✅ AlphaGo (novel strategies, humans learn from them)
- ✅ AdA (novel capabilities, interpretable ordering)
- ✅ POET (QD + coevolution)
- ✅ Research labs publishing papers

**NO (Not open-ended)**:
- ❌ Noisy TV (learnable but not novel once converged)
- ❌ Channel-switching TV (novel but not learnable)
- ❌ Current foundation models (fixed datasets → epistemic uncertainty eventually modeled)
- ❌ Wikipedia to amnesiac (violates learnability once memory saturates)

**DEPENDS on observer**:
- Aircraft design: Open-ended for student, not for alien expert
- AlphaGo: Open-ended for human, not for Nash oracle

---

## Connection to Related Concepts

### Open-Endedness vs Other Concepts

**vs Curiosity/Intrinsic Motivation**:
- Novelty = generalization of curiosity (prediction error)
- Learnability ensures capturing epistemic (not just aleatoric) uncertainty
- Avoids "stochastic traps" (random noise)

**vs AI-Generating Algorithms (AIGA)**:
- AIGA = meta-learning to build general AI
- AIGA need not be open-ended (stops after passing Turing test)
- Open-ended need not be AIGA (narrow-scope systems like AlphaGo)
- **Open-ended FMs** = intersection

**vs Continual RL**:
- Continual RL = agent never stops learning
- But may cycle among fixed strategies (no novelty accumulation)
- Open-ended continual RL = policies accumulate novelty

**vs Novelty Search**:
- Novelty search = local search for novel/interesting (avoid deceptive objectives)
- Our definition formalizes novelty (unpredictability) and interestingness (learnability)

**vs Potential Surprise (Economics)**:
- "How surprised would I be if this occurred, looking at world as I do now?"
- Open-ended system = ever-increasing Shackle surprise in learning observer

**vs Knightian Uncertainty**:
- Knightian = lack of quantifiable knowledge (vs quantifiable risk)
- Open-ended system = induces Knightian uncertainty in learning observer

---

## Key Takeaways

### The Core Argument

1. **Foundation models** scaling on passive data will plateau → won't reach ASI alone
2. **Open-endedness** (novelty + learnability) is essential property of any ASI
3. **Open-ended foundation models** = path to ASI by combining:
   - FMs: General knowledge, human interestingness, semantic understanding
   - Open-ended algorithms: Experience generation, self-improvement, perpetual novelty
4. **Safety** must be co-developed with open-endedness (not afterthought)

### Why Now?

**Ingredients are in place**:
- Foundation models capture human knowledge and interestingness
- Can serve as general mutation operators
- Can guide search toward human-relevant artifacts
- Just need to add open-ended algorithms (RL, self-improvement, task generation, evolution)

**The scientific method as blueprint**: Hypothesize → Falsify with experiments → Codify new knowledge → Repeat

### Implementation Paths

**Four overlapping approaches**:
1. **RL**: Proxy observers (FMs) guide exploration toward novel, learnable behaviors
2. **Self-improvement**: Generate hypotheses, evaluate with tools/environments, refine
3. **Task generation**: Learned world models + reward models → open-ended curricula
4. **Evolution**: LLMs as mutation/selection for prompts/code

### Safety Principles

**Five critical areas**:
1. AI creation (dual-use, goal misgeneralization, safe exploration)
2. Human understanding (interpretability, informed oversight)
3. Human guidance (controllability while maintaining open-endedness)
4. Society adaptation (governance, avoiding tipping points)
5. Emergent risks (anti-fragile design, monitor + adapt)

**Central tension**: System must be learnable to humans (for safety) while producing superhuman novelty (for ASI).

**Design philosophy**: Open-endedness isn't just about capability—it's about **building systems that can teach us** as they grow beyond us.

---

## Philosophical Implications

### On Interestingness

**Not explicitly mentioned** in definition. Instead:
- Interestingness = what observer chooses to learn about (via loss function ℓ)
- Different observers find different features interesting
- Captures human values via choice of what to predict

### On Objectivity

**Is open-endedness objective?**
- Objective given fixed observer (measurable, provable)
- Subjective across observers (depends on prior knowledge, capabilities, timescales)
- **Convergence hypothesis**: All "reasonable" observers (approximating Solomonoff induction) eventually agree

### On Self-Observation

**Can a system observe itself?**
- Yes! AlphaGo observes own policy in self-play
- Humans have "Eureka moments" building on previous insights
- When feedback improves system → **proxy observer** (internal, not external)

### On Completeness

**Open-endedness ≠ Coverage**:
- AlphaGo is open-ended but doesn't explore Go fully
- Adversarial search found simple policies beating AlphaZero
- Novelty + learnability ≠ guarantee of exhaustive search

---

## Future Directions

### Theoretical

- Prove open-endedness theorems for specific systems
- Compare statistical learning vs compression definitions
- Study variants (probabilistic novelty/learnability)
- Investigate broad open-endedness via rate-distortion curves

### Practical

- Optimize directly for open-endedness (not just discover it)
- Scale code evolution to general domains
- Build universal explainers keeping pace with system complexity
- Develop anti-fragile safety mechanisms

### Experimental

- Measure open-endedness with human studies
- Use LLMs as oracle observers
- Test convergence hypothesis (do reasonable observers agree?)
- Evaluate existing systems (which are open-ended to humans?)

---

## One-Sentence Summary

**Open-endedness—the property that a system produces artifacts that are both increasingly unpredictable yet learnable to an observer—is essential for artificial superhuman intelligence, and combining open-ended algorithms with foundation models provides a concrete path toward achieving ASI responsibly.**
