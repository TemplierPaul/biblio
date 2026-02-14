# Open-Ended Learning Algorithms

Overview of algorithms that enable continuous, unbounded generation of challenges and solutions.

---

## What is Open-Ended Learning?

**Goal**: Create systems that continuously discover novel and increasingly complex behaviors without predefined endpoints or curricula.

**Key principles**:
- **No fixed objective**: The target changes and expands during evolution
- **Automatic curriculum**: System generates its own sequence of challenges
- **Sustained innovation**: Complexity/capability increases over extended runs
- **Stepping stones**: Intermediate challenges enable eventual solutions to hard problems

---

## Algorithms in this Folder

### POET (Paired Open-Ended Trailblazer)
**Type**: Environment-Agent Coevolution  
**Innovation**: Parallel exploration of environment-agent pairs with bidirectional transfer  
**Domain**: BipedalWalker-like locomotion tasks  
**Files**: `POET.md`, `POET_detailed.md`

**When to use**: Need automatic curriculum, can parameterize environments, want stepping stones to emerge naturally

---

### Enhanced POET
**Type**: Domain-General Coevolution  
**Innovation**: PATA-EC (performance-based novelty), CPPN-generated environments, unbounded complexity  
**Domain**: Any environment encodable as CPPN  
**Files**: `Enhanced_POET.md`, `Enhanced_POET_detailed.md`

**When to use**: Want domain-independence, need theoretically unbounded generation, sustained innovation over 60k+ iterations

**Improvement over POET**: No hand-designed features, richer environments, improved transfer mechanisms

---

### OMNI-EPIC
**Type**: LLM-Driven Task Generation  
**Innovation**: Foundation models generate executable Python code for arbitrary tasks  
**Domain**: Any computable environment (Darwin Completeness goal)  
**Files**: `OMNI-EPIC.md`, `OMNI_EPIC_detailed.md`

**When to use**: Need unlimited task diversity, want interpretable natural language descriptions, can leverage LLM priors

**Improvement over Enhanced POET**: Replaces CPPNs with LLM code synthesis, generates any computable task (not just parameterized), dual Model of Interestingness

---

### Digital Red Queen (DRQ)
**Type**: Adversarial Self-Play with LLMs  
**Innovation**: Historical self-play + MAP-Elites produces convergent evolution toward general-purpose behavior  
**Domain**: Core War (Turing-complete assembly game)  
**Files**: `Digital_Red_Queen.md`, `Digital_Red_Queen_detailed.md`

**When to use**: Adversarial domains (cybersecurity, competitive games), need robust generalists (not brittle specialists), LLM-guided code evolution

**Difference from POET family**: Adversarial self-play (opponents, not environments), Red Queen dynamics, convergent evolution in behavior space

---

## Comparison

| Algorithm | What Evolves | Generator | Typical Runtime | Key Metric |
|-----------|--------------|-----------|-----------------|------------|
| **POET** | Env + Agent | Mutations on params | ~2k iterations | Max env difficulty solved |
| **Enhanced POET** | Env + Agent | CPPNs | ~60k iterations | Sustained innovation |
| **OMNI-EPIC** | Tasks + Agents | LLMs (code synthesis) | ~100 tasks | Darwin Completeness |
| **Digital Red Queen** | Adversaries | LLMs (assembly mutations) | ~10 rounds, 96 runs | Generalist coverage (%) |

---

## Choosing an Algorithm

**Use POET if:**
- You have a locomotion-like domain with parameterizable environments
- You want a proven, well-studied baseline
- Short-to-medium runs (thousands of iterations)

**Use Enhanced POET if:**
- You need domain-general open-endedness
- You want sustained innovation over very long runs
- You can encode environments as CPPNs

**Use OMNI-EPIC if:**
- You want unlimited task diversity beyond what CPPNs can express
- You can leverage LLM priors for your domain
- You want human-interpretable task descriptions

**Use Digital Red Queen if:**
- Your domain is adversarial (competitive, security)
- You need robust generalists that defeat diverse opponents
- You're evolving code/programs (not just parameters)
- You want to study Red Queen dynamics and convergent evolution

---

## Key Insights Across Algorithms

1. **Stepping stones are critical**: All methods rely on intermediate challenges to reach hard problems
2. **Diversity preservation matters**: Transfer/coevolution require maintaining population diversity
3. **Generator matters**: CPPNs → bounded, LLMs → unbounded (Darwin Complete)
4. **Evaluation strategy matters**: POET (minimal criterion), Enhanced POET (PATA-EC), DRQ (historical self-play)
5. **Convergent evolution**: Independent runs can discover similar solutions (especially DRQ)

---

## Related Work

**Other open-ended methods not in this folder:**
- **GAME** (Generational Adversarial MAP-Elites): Adversarial QD with tournament selection, in main `Evolutionary Optimization/` folder
- **Novelty Search**: Pure novelty without fitness, foundational technique
- **NEAT**: Topology + weight evolution, enables complexification

**Connections:**
- POET family shares lineage: POET → Enhanced POET → OMNI-EPIC
- DRQ uses MAP-Elites internally for intra-round optimization
- All benefit from diversity preservation mechanisms (niching, archives, MAP-Elites)

---

**See individual files for detailed algorithms, mathematical formulations, pseudocode, and implementation notes.**
