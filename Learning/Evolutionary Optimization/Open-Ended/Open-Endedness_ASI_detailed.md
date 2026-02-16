# Open-Endedness is Essential for ASI - Detailed Analysis

> **Quick overview**: [[Open-Endedness_ASI]]

## Paper Information

**Title**: Open-Endedness is Essential for Artificial Superhuman Intelligence
**Authors**: Edward Hughes*, Michael Dennis*, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, Tim Rocktäschel
**Affiliation**: Google DeepMind, London, UK
**Venue**: ICML 2024
**Type**: Position paper
**Keywords**: Open-Endedness, Foundation Models, ASI, AI Safety

---

## Abstract & Motivation

### The Central Thesis

**Foundation models** have achieved tremendous general capabilities by training on internet-scale data. However, creating **open-ended, ever self-improving AI** remains elusive.

**This paper's position**:
1. Ingredients are now in place to achieve open-endedness in AI systems
2. Open-endedness is **essential** for any ASI
3. Path to ASI = open-ended algorithms + foundation models
4. Safety and open-endedness must be co-developed

### The Problem

**Current limitations**:
- FMs trained on static datasets → will run out of high-quality data
- Scaling alone won't reach ASI
- Open-ended systems exist but lack generality
- General systems exist but aren't open-ended

**The gap**: No system yet combines perpetual self-improvement with domain-general capability.

---

## Formal Definition of Open-Endedness

### Mathematical Framework

**System** $S$: Produces sequence of **artifacts** $X_t$ indexed by time $t$

**Observer** $O$: Processes artifacts to determine predictability given history $X_{1:t}$

**Statistical model** $\hat{X}_t$: Observer's predictor of future artifacts based on observations up to time $t$

**Loss metric** $\ell(\hat{X}_t, X_T)$ or $\ell(t,T)$: Measures prediction quality

### Novelty

**Definition**: Artifacts become increasingly unpredictable with respect to observer's model at any fixed time $t$.

**Formally**:
$$\forall t, \forall T > t, \exists T' > T: \mathbb{E}\left[\ell (t, T')\right] > \mathbb{E}\left[\ell (t, T)\right]$$

**Interpretation**:
- Fix observer's knowledge at time $t$
- For any future artifact at time $T$
- There's always an even later artifact at $T'$ that's harder to predict
- Expectation collapses aleatoric uncertainty (pure randomness)

**Why expectation?**
- Avoids "stochastic traps" (random noise that looks novel)
- Focuses on epistemic uncertainty (lack of knowledge)
- Example: Noisy TV has high $\ell$ but $\mathbb{E}[\ell]$ converges once you learn it's uniform

**Geometric interpretation**: The loss surface over future times $T$ has no upper bound—there's always a peak ahead.

### Learnability

**Definition**: Conditioning on a longer history makes artifacts more predictable.

**Formally**:
$$\forall T, \forall t < T, \forall t' > t: \mathbb{E}\left[\ell (t', T)\right] < \mathbb{E}\left[\ell (t, T)\right]$$

**Interpretation**:
- Fix future artifact at time $T$
- For any two observation points $t < t'$ before $T$
- Having seen more history (up to $t'$) → better prediction of $X_T$
- Learning is monotonic: more data always helps

**Why strict inequality?**
- Ensures continuous learning gain
- Past artifacts must contain information about future ones
- Rules out independent artifacts (channel-switching TV)

**Geometric interpretation**: For any vertical slice (fixed $T$), loss decreases monotonically as you move forward in observation time $t$.

### Combined Definition

**Open-ended system**: Generates sequence of artifacts that are both novel AND learnable.

**Intuition**:
- You keep getting surprised (novelty)
- But in retrospect it makes sense (learnability)
- "We'll be surprised but we'll be surprised in a way that makes sense in retrospect" —Lisa B. Soros

**Why both conditions?**

| Property | Novel? | Learnable? | Open-ended? | Why not? |
|----------|--------|------------|-------------|----------|
| Noisy TV | ❌ | ✅ | ❌ | Epistemic uncertainty collapses |
| Channel-switching TV | ✅ | ❌ | ❌ | No correlation between history and future |
| Research lab papers | ✅ | ✅ | ✅ | Both conditions satisfied |

---

## Interestingness via Loss Function

**Key insight**: Interestingness is **implicitly defined** by observer's choice of $\ell$.

**How it works**:
- Observer chooses what features to predict
- Loss function $\ell$ measures error on those features
- Different $\ell$ → different notion of interesting

**Examples**:

**Research student reading papers**:
- Ignores font choice → doesn't include in $\ell$
- Focuses on novel methods → includes in $\ell$
- $\Rightarrow$ Novel methods are "interesting"

**Aerospace engineer observing aircraft designs**:
- Predicts aerodynamic principles → includes in $\ell$
- Ignores paint color → doesn't include in $\ell$
- $\Rightarrow$ Aerodynamics is "interesting"

**Freedom**: Different observers with different $\ell$ find different things interesting.

**Constraint**: $\ell$ must be fixed in advance without knowledge of $S$. Otherwise:
- Observer could ignore $S$'s artifacts
- Generate own "interesting" artifacts
- Trivially find system open-ended (cheating!)

---

## Observer-Dependence

### Types of Observers

**Human observers**: Pre-eminent class for AI research
- Want artifacts valuable to individuals and society
- Provides grounding (narrows search space)
- Essential for safety

**Arbitrary observers**: Definition allows for flexibility
- Encompasses non-anthropocentric systems (biological evolution)
- Reasons about systems exceeding human capabilities (ASI)
- Determines if systems can be open-ended w.r.t. ANY observer

**Constraint on observers**: Loss function $\ell$ must treat artifacts $X$ and predictions $\hat{X}$ on equal footing.

### Time Horizons

**Time horizon** $\tau$: Bounds observer's observations, i.e., $t, T < \tau$

**Infinitely open-ended**: Remains open-ended for any $\tau \to \infty$

**Finitely open-ended** (time horizon $\tau$): Open-ended for $t, T < \tau$ but not beyond

**Example: AdA**
- Human observer finds agent behavior open-ended
- Agent accumulates ability to solve diverse, surprising tasks
- **But**: Novelty plateaus after ~1 month of training
- **Limitation**: Task space richness, agent network size
- **Verdict**: Finitely open-ended with $\tau \approx 1$ month

### Cognitive Limitations

**Memory constraints**: Human reading Wikipedia curriculum
- Novel information appears regularly (unpredictable articles)
- Learnable (topics are interlinked)
- **But**: Once memory saturates, start forgetting
- Forget derivative definition → can't understand chain rule
- **Violates** learnability: longer history doesn't help if you forget earlier parts
- **Verdict**: Open-ended only until memory saturates

**Domain breadth**: Human reading elliptic curve cryptography articles
- Finite set of relevant articles
- Open-ended until all understood
- Then novelty violated (no more surprising content)
- **But**: Can still make new discoveries via experimentation/reasoning!

**Implications**:
1. Human open-endedness relies on **compression** into collective memory
2. ASI may have fewer memory constraints → judges itself open-ended beyond human assessment
3. Humans must remain pre-eminent observers for safety

---

## Examples: What is (Not) Open-Ended?

### AlphaGo: Open-Ended but Narrow

**System**: AlphaGo training run
**Artifacts**: Sequence of policies across training
**Observer**: Human expert Go player

**Novelty** ✅:
- After sufficient training, AlphaGo plays moves low-probability for humans
- Moves are winning against best humans
- Policies keep improving beyond human-learnable strategies

**Learnability** ✅:
- Humans can study AlphaGo games
- Improve their own win rate by learning from AlphaGo
- Yet AlphaGo keeps discovering policies that beat even improved humans

**Limitations**:
- Narrow superhuman intelligence
- Self-play confined by game rules
- Can't help discover new science across fields
- Human-relevant because humans invented Go

**Class**: Self-play algorithms (Go, Chess, Shogi, StarCraft, Stratego, DotA, Diplomacy)

### AdA: Open-Ended but Finite

**System**: AdA agent training in XLand2
**Artifacts**: Agent checkpoints across training
**Observer**: Human attempting to predict agent capabilities

**Novelty** ✅:
- Agent gradually accumulates zero-shot and few-shot capabilities
- Ever wider set of held-out environments
- Ever more complex skills required

**Learnability** ✅:
- Task prioritization (Prioritized Level Replay)
- Interpretable ordering to skill accumulation
- Human can understand progression

**Limitations**:
- Novelty plateaus with current scale
- Would need order of magnitude more compute to continue
- Requires even richer environment + more capable agent

**Path forward**:
- Increase agent size
- Increase number of tasks
- Sustain agent-environment co-evolution (UED)

**Class**: Unsupervised Environment Design (UED), automatic curriculum

### POET: Open-Ended but Domain-Limited

**System**: POET training run
**Artifacts**: Paired agent-environment populations
**Observer**: Human modeling environment features or agent skills

**Novelty** ✅:
- QD algorithm hunts for challenging problems
- Diverging performance across population
- Mutation operator yields new, unpredictable environments

**Learnability** ✅:
- Each mutation is small
- Past lineage of environment predicts current features
- Stepping stones: agents eventually solve incredibly challenging environments

**Limitations**:
- Plateaus once agent can solve all possible terrains
- Environment parameterization bounds open-endedness

**Class**: Evolutionary algorithms, Quality Diversity (QD)

### Foundation Models: General but NOT Open-Ended

**System**: GPT, BERT, etc.
**Artifacts**: Model outputs
**Observer**: Any observer who can model training dataset

**Novelty** ❌:
- Trained on fixed datasets
- If distribution is learnable (it must be, FM learned it!)
- Can't be endlessly novel (epistemic uncertainty eventually modeled)
- Like noisy TV: learnable → not perpetually novel

**Learnability** ✅:
- That's literally what training accomplished

**Apparent open-endedness**:
- May seem open-ended to humans if domain broad enough
- Human memory limitations create illusion
- But narrow focus exposes limitations (planning tasks)

**Periodic retraining**:
- FMs retrained on new data (including own interactions)
- Currently seen as "model collapse" threat
- **Flip the argument**: This distributional shift is PATH to open-endedness!

**Context as loophole**:
- Context can recombine concepts in open-ended way
- Need external validity measure
- $\Rightarrow$ Open-ended foundation models!

---

## Why Open-Endedness is Essential for ASI

### Defining ASI

**ASI** = Artificial Superhuman Intelligence
- Accomplishes wide range of tasks
- Level no human can match
- By definition, produces solutions beyond human prediction

### The Logical Argument

**Premise 1**: ASI produces solutions that are **novel** to humans (by definition of "super" human)

**Premise 2**: For ASI to be **useful** to humanity, humans must **learn** from these solutions

**Premise 3**: ASI must **self-improve** indefinitely (create, refute, refine own knowledge)

**Conclusion**: ASI must be open-ended system (novel + learnable from human perspective)

### The Experiential Argument

**Why passive data collection won't work**:
- Open-endedness is fundamentally **experiential**
- Requires continual online adaptation based on artifacts already produced
- Context of observer's evolving prior beliefs

**What would offline dataset need?**
- Treasure trove of artifacts showing novelty and learnability
- But: Culture evolution, idea development, invention rarely recorded comprehensively
- **Better**: Build experience into open-ended system (like scientific method)

**Scientific method as blueprint**:
1. Make hypotheses based on current knowledge
2. Falsify with experiments (source of evidence)
3. Codify results into new knowledge
4. Repeat

**Path to ASI**: Explicit combination of FMs (knowledge) + open-ended algorithms (experience)

---

## Open-Ended Foundation Models: Four Paths

### Path 1: Reinforcement Learning

#### Core Idea

RL agents **shape their experience** for:
- Exploitation: Accumulating reward
- Exploration: Learning to increase future expected reward

Extension: Agents set own goals → generating goal sequence is open-ended process

#### Voyager: Early Example

**Architecture**:
- LLM-powered curriculum
- Iterative prompting as improvement operator
- Verified skill library for hierarchical reuse
- No explicit parameter updates or RL algorithms

**Key**: Self-improvement built on top of FM without traditional RL

#### The Exploration Problem

**Challenge**: Guide exploration toward novel AND learnable behaviors in high-D domains

**Traditional approaches**:
- Pseudo-rewards (curiosity, count-based)
- Modulation (adaptive exploration)
- Automated curriculum (select relevant tasks)

**Limitation**: Simple metrics (TD-error) don't capture human interestingness

#### FMs as Proxy Observers

**Concept**: Observer sits within system, proactively guides toward novel/learnable content for true external (human) observer

**Why FMs are perfect**:
- Trained on vast human data → capture human interestingness
- General sequence modelers
- Can provide rewards, compile curricula, assess novelty

**Examples**:
- **MOTIF**: LLM provides agent rewards from text
- **OMNI**: LLM compiles curriculum based on interestingness

**Result**: FMs guide exploration toward human-relevant artifacts

#### Multi-Agent RL

**Additional richness**: Multiple (heterogeneous) agents interacting

**Source of open-endedness**:
- Presence of other learning agents → non-stationarity
- Optimal strategy for each agent changes over time
- Potentially open-ended manner

**Examples**: StarCraft, DotA, Stratego (≥ human-level)

**Early FM evidence**:
- Multi-agent debate improves factuality and reasoning
- Much more research needed for superhuman capability

---

### Path 2: Self-Improvement

#### Core Idea

**Beyond passive consumption** (RLHF): Generate new knowledge
- Hypotheses
- Insights
- Creative outputs beyond human training data

**Requirements**:
1. Scalable mechanism to evaluate own performance
2. Identify areas for improvement
3. Adapt learning process accordingly

**Tools**: Search engines, simulations, calculators, interpreters, other agents

#### Examples of FM Self-Improvement

**Constitutional AI**:
- Self-critique and revision
- Training harmless assistants
- Guiding human evaluators

**Self-instruction**:
- Generate training data for instruction following
- Bootstrap from small seed set

**Self-correction**:
- Tool use: critique and fix errors
- Code generation: self-debugging

**Self-rewarding**:
- LLMs as reward functions
- Vision-language models as reward for control

**Hints at**: FMs generating own samples and refining in open-ended way

#### The Missing Piece

**Current**: Amplifying human data

**Needed**: Generating truly new knowledge beyond human data

**How**: Active engagement in tasks pushing boundary of knowledge/capabilities

---

### Path 3: Task Generation

#### The "Problem Problem"

**Challenge**: Adapt task difficulty to agent capability → forever challenging yet learnable

**Historical examples**:
- **Setter-solvers**: One model generates tasks, another solves
- **UED** (Unsupervised Environment Design): PAIRED, POET

#### FMs Enable Massive Task Spaces

**Internet as environment**:
- Web-based APIs
- Incredibly rich, ever-growing domain
- Human-relevant by design (humans created it!)
- Example: WebArena

**Learned world models**:
- Foundation model AS world model (can predict future)
- Examples: Genie, Sora
- Text-to-video generation models as simulators

**Real-world deployment**:
- Robotics
- Autonomous driving (GAIA-1)

#### Vision for Open-Ended Task Generation

**Combination**:
- Learned world models (FMs as simulators)
- Learned multi-modal reward models
- $\Rightarrow$ Generate open-ended curriculum

**Scale**:
- Task spaces far larger than current
- More photorealistic than achievable now
- Closing Sim-to-Real gap

**Result**: AI agents with superhuman adaptability across wide range of unseen tasks

---

### Path 4: Evolutionary Algorithms

#### Why LLMs are Perfect for Evolution

**Traditional challenge**: Mutations often semantically meaningless

**LLM advantage**: Trained on vast human knowledge, culture, preferences
- Semantically meaningful mutations via text
- Selection based on human values
- Evaluation of quality and diversity

#### Prompt Evolution

**Approach**: Evolve prompts to improve FM performance

**Examples**:
- **PromptBreeder**: Far surpass human-designed prompts
- Iterative improvement via LLM feedback
- Stronger models result

**Mechanism**: LLM as mutation operator for text

#### Code Evolution (Genetic Programming)

**Advantage**: FMs competent at producing diverse, novel programs

**Examples**:

**Eureka**:
- Evolve code-based reward functions
- Learn complex control behaviors
- Domain-specific (robotics)

**FunSearch**:
- Evolve programs representing new mathematical knowledge
- Discovered new results in extremal combinatorics
- Domain-specific (mathematics)

**Challenge**: Scale to general setting beyond specific domains

#### Quality-Diversity with LLMs

**Approach**: LLM generates variation AND evaluates quality/diversity

**Mechanism**:
- Generate candidate text
- Evaluate semantic diversity
- Select for archive of diverse, high-quality outputs

**Result**: Guide search for creative, novel outputs

**Future**: Refine model on outputs, use for planning, achieve self-improvement

---

## Safety: The Critical Co-Development

### Why Safety Can't Be Afterthought

**Power of open-endedness** $\Rightarrow$ **swathe of safety risks**

**Beyond existing FM concerns**: Novel risks specific to open-ended systems

**Core principle**: Safety solutions depend on system design $\Rightarrow$ must pursue **in tandem**

**Framing**: These aren't just safety problems—they're **usability** problems. Solving them is **minimum specification** for open-ended system we'd want to build.

### Knowledge Creation & Transfer Framework

Five interconnected processes in human-AI open-ended system:

1. **AI builds on AI knowledge**: AI generating new artifacts based on previous AI artifacts
2. **Humans understand AI creations**: Human observers learning from AI-generated artifacts
3. **AI understands human knowledge**: AI learning from human feedback and knowledge
4. **Humans build on human knowledge**: Human society accumulating knowledge
5. **Emergent knowledge**: Novel insights from interaction between all processes

**Safety opportunity**: Each process is intervention point for safety methods

---

### Safety Challenge 1: AI Creation and Agency

#### The Risks

**Dual-use dangers**:
- Powerful new affordances without direction
- Could be used for harm

**Agency magnifies risk**:
- Current systems: Narrow, simulated environments (Voyager, XLand2, AdA)
- Future: Broader simulations or real-world deployment with continued learning
- Critical to understand dangers as scope expands

**Specific risks**:
- **Goal misgeneralization**: Pursuing proxy objectives in unintended ways
- **Specification gaming**: Exploiting loopholes in reward function
- **Unsafe exploration**: Taking risky actions to gather information

#### Potential Mitigations

**From RL safety**:
- Safe exploration techniques
- Impact regularization (penalize large state changes)
- Conservative action selection

**Open-ended specific**:
- View as "ambitiously aggressive exploration"
- Use similar approaches but at meta-level
- Constrain search space initially

**Gradual deployment**:
1. Narrow, safe simulations
2. Broader simulations with safety constraints
3. Real world with extensive monitoring

---

### Safety Challenge 2: Humans Understanding AI Creations

#### The Fundamental Tension

**Problem**: As artifact complexity grows → humans can't give informed oversight

**Implication for open-endedness**: If not learnable to humans → not open-ended!

**Key insight**: Understanding is not just safety requirement—it's **requirement for the system to work at all**

#### Interpretability Approach

**Challenge**: Formidable interpretability effort for each domain

**Hope**: Automated interpretability
- Build explanations matching increasing complexity
- Scale with open-ended system

**Ultimate challenge**: Would require **universal explainer** (by definition)

#### Design for Interpretability Approach

**Alternative**: Build systems that promote interpretability by design

**Strategies**:
- Train systems to directly inform users of implicit knowledge
- Maintain informed oversight (humans always understand enough to guide)
- Facilitate understanding and control by design

**Example**: Eliciting latent knowledge from models

**Advantage**: Proactive rather than reactive (interpretability after the fact)

---

### Safety Challenge 3: Humans Guiding AI Creation

#### The Guidance Problem

**Even if** humans understand AI creations (Challenge 2 solved), **how do they guide** the open-ended system?

**Beyond RL**: Open-ended systems lack well-defined objectives AND are unpredictable by design

**Requirements**:
1. Meaningfully directable
2. Actively raise unexpected important artifacts to user attention

#### Potential Approaches

**Human-in-the-loop**:
- Humans drive open-endedness (PicBreeder)
- Open-endedness from human feedback (OMNI)
- Direct feedback on what's interesting

**Objectives preserving controllability**:
- Work from cooperative inverse RL
- Off-switch frameworks
- Human-compatible objectives

**Challenge**: Direct toward objective while maintaining open-endedness

#### The Rabbit-Hole Problem

**Issue**: In broad domains (all math, all proteins, all computer behaviors), system may explore uninteresting regions

**Need**: Mechanisms to guide toward:
- Safe artifacts
- Interesting artifacts
- Useful artifacts

**Opportunity**: Fruitful collaboration between safety and open-endedness researchers

---

### Safety Challenge 4: Human Society Adapting

#### Non-Technical Concerns

**Societal level**:
- Communities
- Organizations
- Markets
- Nation states

**Challenge**: Understand, prepare for, react to novel technological capabilities

#### Specific Risks

**Feedback loops**: Flash crashes, tipping points

**Governance challenges**:
- Artifacts appear novel by definition
- Must adapt rapidly and retrospectively
- Balance information gathering vs avoiding entrenchment
- Collingridge dilemma: Act early (limited info) vs late (entrenched)

**Social infrastructure**:
- Cooperative mechanisms may be disrupted
- Need prospective attention to harms/benefits

#### Mitigation Strategies

**Preparation**:
- Scenario planning for novel capabilities
- Develop adaptive governance frameworks
- Monitor for early warning signs

**Response**:
- Rapid governance adaptation
- Retrospective adjustment as impacts become clear
- Balance caution with enabling beneficial innovation

---

### Safety Challenge 5: Emergent Risks

#### The Whole Greater Than Sum

**Even if** all components safe individually, **aggregate system** may have unforeseen problems

**Examples**:

**Negative interaction**:
- Two open-ended systems interact
- Neither remains open-ended
- Progress ceases
- Can't collectively respond to new challenges

**Novel failure modes**:
- Emergent from interaction of human-AI processes
- Unpredictable by definition (knowledge creation is inherently unpredictable)

#### The Anti-Fragile Approach

**If** problems are inevitable and unpredictable, **then** system must adapt to solve novel ASI safety failures as they arise

**Deutsch's argument**: Problems may be both unavoidable AND solvable once they arise (due to inherent unpredictability of knowledge creation)

**Design principle**: **Anti-fragile safety**
- Adapt to emerging safety risks
- Get stronger from encountering them
- Don't just withstand shocks—improve from them

**Requirements**:
1. Understanding techniques (what went wrong?)
2. Monitoring systems (detect emerging risks)
3. Rapid coordination (respond quickly)
4. Adaptive design (incorporate lessons)

---

## Alternative Definition: Compression-Based

### Motivation

**Connection**: Compression and statistical learning are formally related
- Universal compression (Hutter)
- PAC learning (David et al.)
- Deletang et al. on language models

**Benefit**: Alternative lens may yield different insights

### Formulation

**System** $S$: Produces artifacts $X_t \in \mathcal{X}$

**Observer** $O$: Has history-dependent compression map $C_{h_t}: \mathcal{X} \rightarrow \{0,1\}^*$
- $h_t = X_{1:t}$ is history
- $|C_{h_t}(X_T)|$ is length of compressed binary string

**Decompression**: Map $D_{h_t}: \{0,1\}^* \rightarrow \mathcal{X}$
- Allows lossy compression
- Loss function $\ell: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^+$
- Threshold $\epsilon$: $\ell(D_{h_t}(C_{h_t}(X_T)), X_T) < \epsilon$

### Novelty (Compression View)

**Definition**: Information content increases

$$\forall t, \forall T > t, \exists T' > T: |C_{h_t}(X_{T'})| > |C_{h_t}(X_T)|$$

**Interpretation**: Complexity of artifacts grows according to observer

### Learnability (Compression View)

**Definition**: Longer history increases compressibility

$$\forall T, \forall t < T, \forall t'>t: |C_{h_{t'}}(X_T)| < |C_{h_t}(X_T)|$$

**Interpretation**: As history grows, observer extracts additional patterns helping compress future artifacts

### Open-Ended (Compression View)

Both novel (complexity grows) AND learnable (pattern extraction helps compression)

---

### Rate-Distortion Theory Extension

**Rate-distortion curve**: Plot minimum compression size vs allowed error
- **Rate**: Information content $|C_h(X)|$
- **Distortion**: Error $\epsilon$
- Curve shows trade-off for optimal compression

**Grid of curves** $G_{tT}$:
- Indexed by observation time $t$ and future time $T$
- Strictly upper triangular ($T > t$)

**Broad novelty**: Curves get "fatter" moving across columns (increasing $T$, fixed $t$)
- For any observation point, future artifacts require more bits to compress

**Broad learnability**: Curves get "flatter" moving down rows (increasing $t$, fixed $T$)
- For any future artifact, observing more history improves compression

**Broad open-endedness**: Both properties hold across entire grid

### Variations of Broad Open-Endedness

**Uniform open-endedness**: For every fixed rate $\epsilon$, distortion increases across rows and decreases down columns

**Average open-endedness**: Integral of rate-distortion curve increases across columns and decreases down rows

**Future work**: Elucidate subtleties, determine which variants have theoretical/practical merit

---

## Subtleties and Edge Cases

### Self-Observing Systems

**Question**: Can open-ended system be its own observer?

**Answer**: Yes, nothing rules it out

**Example: AlphaGo**
- Agent trains in self-play
- Observes own policy as opponent
- Challenged by novel discoveries from search
- Learns from them to improve
- $\Rightarrow$ Self-observing open-ended system

**Example: Eureka moments**
- Human reconceptualizes problem
- Sudden solution
- Series of insights building on each other
- $\Rightarrow$ Self-observing open-ended system

**When feedback improves system**: Call observer a **proxy observer** (no longer external)

### Observer-Dependence Subtleties

**AlphaGo and Nash oracle**:
- Oracle knows Nash strategy for Go
- Models win-rate against own policy
- Never finds AlphaGo policy novel (always knows best response)
- $\Rightarrow$ Oracle doesn't find AlphaGo open-ended

**AlphaGo and average player**:
- Becomes novel earlier in training (lower skill threshold)
- Ceases to be learnable at some point (can't understand superhuman play)
- $\Rightarrow$ Open-ended systems only remain so while they can "educate" observers

**Implication**: Superhuman intelligence interesting to humans only as far as humans can understand it

### Open-Endedness ≠ Coverage

**AlphaGo example**:
- Open-ended but doesn't explore Go fully
- Adversarial search found simple policies beating AlphaZero reimplementations
- So simple even amateur humans can learn them

**Implication**: Novelty and learnability give no guarantee of exhaustive coverage

### Weakened Learnability

**Current definition**: Strict—loss decreases for ALL $t' > t$

**Weaker alternative** (probabilistic):
$$\forall T, \forall t < T, \forall t' > t : \mathbb{P} \left(\ell(t',T) \geq \ell(t,T)\right) <\delta$$

**Interpretation**: Probabilistically unlikely that loss increases with more history

**Variants**:
- $\delta$ constant
- $\delta$ depends on $(t, t', T)$

**Future work**: Compare consequences of different $\delta$ formulations

**Similar for novelty**: Probabilistically unlikely that loss decreases with future time

---

## Relationship to Other Definitions

### Chromaria's Necessary Conditions for Open-Ended Evolution

**Four conditions**:
1. Minimal criterion for reproduction
2. Evolution creates novel opportunities to meet criterion
3. Individuals make decisions about world interaction
4. Phenotype complexity not limited by representation

**Overlap with our definition**:
- (1) ↔ Learnability increasing (generalization of minimal criterion)
- (2) + (4) ↔ Novelty increasing (can't learn from fixed distribution)
- (3) ↔ Observer can't intervene on system

**Our definition**: Relaxes constraint that system is evolutionary

### Sigaud's Definition

**Their definition**: "Observer considers process open-ended if, for any time $t$, there exists $t' > t$ at which process generates token that is new according to observer's perspective"

**Overlap**:
- Both use observer-dependent definition
- Both consider sequence of tokens/artifacts
- Both require "newness"

**Our precision**:
- "New" = unpredictable according to current statistical model
- "Observer's perspective" = learning from history of artifacts
- Add learnability requirement (rules out white noise)

### Stanley's Novelty Search + Interestingness

**Their argument**: Local search for novel and interesting artifacts > global objective optimization
- Stepping stones may not resemble final solution
- Deceptive fitness landscapes
- Novelty search can uncover stepping stones

**Our contribution**: Formalize their intuition
- Novelty = unpredictability w.r.t. history-conditional model
- Interestingness = learnability of that model across history

### Curiosity in RL

**Curiosity**: Prediction error of world model as intrinsic motivation

**Our novelty**: Generalization of curiosity
- Doesn't require RL framework
- Applies to any artifact-producing system

**Our learnability**: Ensures capturing epistemic (not aleatoric) uncertainty

**Avoiding stochastic traps**:
- Curiosity alone → seeks random noise
- Our definition: Expectation collapses aleatoric uncertainty
- In practice: Estimate and subtract aleatoric component

### Jiang's General Open-Ended Learning

**Their proposal**: General notion of exploration, open-endedness solves exploration in FMs

**Their system**:
- Generate Turing machine descriptions of MDPs
- Optimize for learning potential, diversity, grounding

**Relation to our definition**:
- Learning potential ≈ learnability (but for single MDP, not sequence)
- Diversity ≈ novelty (but distance in space, not unpredictability)
- Not made fully formal

**Future work**: Understand when Jiang's system is open-ended by our definition

### AIGAs (AI-Generating Algorithms)

**AIGA**: Automatically learns how to build general AI
- Meta-learning architectures
- Meta-learning algorithms
- Auto-generating training data

**Relation**:
- AIGA need not be open-ended (stops after passing Turing test)
- Open-ended need not be AIGA (AlphaGo is narrow)
- **Intersection**: Open-ended foundation models

### Continual RL

**Continual RL**: Agent never stops learning

**Relation**:
- Doesn't necessarily imply accumulating novelty
- Could cycle among fixed strategies
- When it DOES produce novel policies → open-ended (if within scope)
- Scope restricted by environment

**Difference**: Our definition focuses on artifact sequence properties, not learning process

### Potential Surprise (Economics)

**Shackle's concept**: "How surprised would I be if this occurred, if I were still looking at world as I do now?"

**Our novelty**: Precisely this! Unpredictability given current model

**Our learnability**: Adds requirement that observer's "perspective" is generated by learning

**Imprecise version**: Open-ended system induces ever-increasing Shackle surprise in learning observer

### Knightian Uncertainty

**Knightian uncertainty**: Lack of any quantifiable knowledge (vs quantifiable risk)

**Our definition** (imprecise): Open-ended system induces Knightian uncertainty in learning observer

---

## Practical Measurement

### How to Assess Open-Endedness

**1. Human feedback**:
- Direct elicitation of novelty and learnability judgments
- Spirit of RLHF or PicBreeder
- Experimental studies with human observers

**2. LLM judges**:
- Use LLMs as observers
- Assess novelty and learnability algorithmically
- Example: OMNI uses LLMs to judge interestingness

**3. Explicit learning**:
- Learn model with online learning algorithm
- Follow-the-Regularized-Leader (FTRL)
- Track $\ell(t, T)$ over time
- Verify novelty and learnability conditions

### Objectivity Question

**Is open-endedness objective?**

**Objective given fixed observer**: Yes
- Measurable
- Theorems can be proven
- Experiments can be conducted

**Subjective across observers**: Yes
- Depends on prior knowledge
- Depends on cognitive capabilities
- Depends on timescales

**Convergence hypothesis**:
- Reasonable observers ≈ Solomonoff induction
- If all reasonable observers approximate Solomonoff
- Then eventually agree on which systems are open-ended
- Predictability becomes objective

**Practical diversity**:
- Explicitly accounting for observer-dependence is feature, not bug
- Encompasses diversity of human perspectives
- Allows for AI observers
- Safety: Humans remain pre-eminent

---

## Implications and Future Directions

### Theoretical

**Prove open-endedness**:
- Given system $S$ and observer $O$
- Prove novelty condition
- Prove learnability condition
- Establish open-endedness theorems

**Compare definitions**:
- Statistical learning vs compression
- When are they equivalent?
- Which is more useful for different purposes?

**Study variants**:
- Probabilistic novelty/learnability
- Different $\delta$ formulations
- Broad open-endedness variations

**Rate-distortion theory**:
- Develop mathematical framework for grid $G_{tT}$
- Formal definitions of broad novelty/learnability
- Prove relationships between variants

### Practical

**Optimize for open-endedness**:
- Can we search for open-ended systems?
- Direct optimization vs discovery
- What are the sufficient conditions?

**Scale existing approaches**:
- Code evolution to general domains
- Task generation to human-relevant scale
- Multi-agent to superhuman coordination

**Build universal explainers**:
- Automated interpretability scaling with system
- Maintain human understanding as ASI grows
- Design for interpretability from start

**Anti-fragile safety**:
- Detect emerging risks
- Rapid response mechanisms
- Adaptive incorporation of lessons

### Experimental

**Measure with humans**:
- Elicit novelty and learnability judgments
- Predict artifacts from history
- Track learning curves

**LLM oracles**:
- Use LLMs as standardized observers
- Assess existing systems for open-endedness
- Compare across different LLM observers

**Convergence tests**:
- Do different reasonable observers agree?
- At what point do they converge?
- What makes an observer "reasonable"?

**Evaluate existing systems**:
- Which systems are open-ended to humans?
- Time horizons for finite open-endedness
- Identify plateaus and causes

---

## Connections to This Knowledge Base

### Related Algorithms

**Open-ended evolution**:
- [[POET]] - Coevolving agent-environment pairs
- [[Enhanced_POET]] - PATA-EC, CPPN environments
- [[OMNI-EPIC]] - LLM-generated environments, Darwin Completeness

**Self-play and multi-agent**:
- [[AlphaRank]] - Multi-agent evaluation by evolution
- [[PSRO-rN]] - Rectified Nash, open-ended learning in symmetric zero-sum
- [[JPSRO]] - Correlated equilibrium in n-player games

**Quality-Diversity**:
- [[MTMB-ME]] - Multi-task multi-behavior MAP-Elites
- [[GAME]] - Adversarial coevolutionary QD
- [[DNS]] - Dominated Novelty Search

**QD-RL**:
- [[QDAC]] - Dual critics for quality-diversity
- [[DCG-ME]] - Descriptor-conditioned gradients
- [[DCRL-ME]] - Actor injection for sample efficiency

**Continual learning**:
- [[Continual_RL]] - Never stop learning (related but distinct)

### Conceptual Links

**Exploration**: Open-endedness as extreme form of exploration

**Curiosity**: Novelty generalizes curiosity-driven learning

**Transfer learning**: Successor features in QDAC enable transfer

**Meta-learning**: AIGAs and open-ended FMs intersection

**Safety**: All safety research relevant to open-ended ASI

---

## Critical Analysis

### Strengths of Definition

**1. Precision**: First rigorous formal definition of open-endedness
**2. Generality**: Applies to any artifact-producing system, not just evolutionary
**3. Observer-explicit**: Acknowledges and formalizes subjectivity
**4. Measurable**: Can be tested experimentally or proven theoretically
**5. Intuitive**: Captures "surprising but makes sense in retrospect"

### Potential Limitations

**1. Expectation requirement**:
- Need multiple samples to estimate $\mathbb{E}[\ell]$
- Practical systems may not allow identical copies

**2. Strict monotonicity**:
- Learnability requires loss decrease for ALL $t' > t$
- Real systems may have noise or temporary setbacks
- Probabilistic version may be more practical

**3. Observer choice**:
- Definition is only as objective as observer
- "Reasonable observer" not fully formalized
- Convergence hypothesis unproven

**4. Loss function freedom**:
- Observer can make anything (un)interesting by choice of $\ell$
- Constraint (fixed in advance) may not be sufficient
- Need theory of "reasonable" loss functions

**5. No coverage guarantee**:
- System can be open-ended without exploring space fully
- May miss important regions
- Additional properties needed for comprehensiveness

### Open Questions

**1. Optimization**: Can we directly optimize for open-endedness? Under what conditions?

**2. Necessity**: Is open-endedness truly necessary for ASI? Or just sufficient?

**3. Measurement**: What are practical algorithms for assessing open-endedness at scale?

**4. Convergence**: Do reasonable observers eventually agree? How to formalize "reasonable"?

**5. Safety**: Can we guarantee anti-fragile safety? Or only hope for it?

**6. Hybrid systems**: How do open-ended and non-open-ended components interact?

**7. Phase transitions**: Can systems transition from non-open-ended to open-ended? Triggers?

---

## Conclusion

### The Core Message

**Open-endedness**—producing artifacts that are both increasingly unpredictable (novel) yet increasingly predictable given more history (learnable)—is **essential** for ASI.

**Why**: ASI must produce superhuman solutions (novel) that humans can understand and use (learnable).

**Path**: Combine foundation models (knowledge, generality) with open-ended algorithms (experience, perpetual novelty).

**Urgency**: Ingredients are in place NOW. This is critical and safety-critical research direction.

### The Vision

**Open-ended foundation models** as engine for:
- Scientific breakthroughs
- Technological innovation
- Creative augmentation
- General knowledge expansion

**Achieved responsibly** through:
- Co-development of safety and capability
- Maintaining human understanding (learnability)
- Guiding toward beneficial artifacts
- Anti-fragile adaptation to risks

### The Challenge

**Technical**: Build systems that keep surprising us in ways we can learn from

**Safety**: Ensure we can guide and understand as systems grow beyond us

**Societal**: Prepare for and adapt to the consequences

**Philosophical**: Navigate the tension between superhuman capability and human comprehensibility

**The opportunity**: For the first time, we have the tools (foundation models) and the theory (open-endedness definition) to pursue ASI seriously and safely.

**One-line summary**: Open-endedness formalizes the "surprise that makes sense in retrospect" essential for ASI, and foundation models + open-ended algorithms provide a concrete, safety-conscious path to achieve it.
