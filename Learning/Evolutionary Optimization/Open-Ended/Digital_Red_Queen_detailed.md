# Digital Red Queen - Detailed Implementation

**Paper**: Kumar et al., "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs" (MIT, Sakana AI, 2025)

---

## Complete Algorithm

```python
import numpy as np
from typing import List, Dict
from llm import LLMClient  # LLM API wrapper
from corewar import CoreWarSimulator  # Core War battle simulator

# ============================================================================
# DRQ: Digital Red Queen
# ============================================================================

def digital_red_queen(
    w0: str,  # Initial warrior (Redcode source)
    T: int,  # Number of rounds
    K: int,  # History length (how many previous warriors to include)
    iterations_per_round: int = 1000,
    llm_client: LLMClient = None,
    **mapelites_kwargs
) -> List[str]:
    """
    Run Digital Red Queen algorithm.
    
    Args:
        w0: Initial warrior source code (Redcode)
        T: Number of rounds
        K: History length (1 = previous only, ∞ = all previous)
        iterations_per_round: MAP-Elites iterations per round
        llm_client: LLM for mutations
        **mapelites_kwargs: Additional MAP-Elites parameters
    
    Returns:
        lineage: List of champion warriors [w0, w1, ..., wT]
    """
    lineage = [w0]
    
    for t in range(1, T + 1):
        print(f"\n=== Round {t}/{T} ===")
        
        # Select opponents: last K warriors from lineage
        opponents = lineage[-K:] if K > 0 else lineage
        
        # Evolve new warrior to defeat opponents
        wt = evolve_warrior_mapelites(
            opponents=opponents,
            iterations=iterations_per_round,
            llm_client=llm_client,
            bootstrap_archive=lineage,  # Initialize with previous champions
            **mapelites_kwargs
        )
        
        # Add to lineage (don't update old warriors)
        lineage.append(wt)
        
        print(f"Round {t} champion: {len(wt)} chars, "
              f"avg fitness vs opponents: {eval_fitness(wt, opponents):.3f}")
    
    return lineage


# ============================================================================
# MAP-Elites (Intra-Round Optimization)
# ============================================================================

def evolve_warrior_mapelites(
    opponents: List[str],
    iterations: int,
    llm_client: LLMClient,
    grid_size: tuple = (10, 10),  # (processes, memory_coverage)
    bootstrap_archive: List[str] = None,
    **kwargs
) -> str:
    """
    Evolve warrior using MAP-Elites to defeat opponents.
    
    Args:
        opponents: List of opponent warriors (Redcode source)
        iterations: Number of MAP-Elites iterations
        llm_client: LLM for generating/mutating warriors
        grid_size: Behavioral descriptor grid discretization
        bootstrap_archive: Optional previous warriors to seed archive
    
    Returns:
        champion: Best warrior from archive
    """
    # Initialize archive: behavior cell -> (warrior, fitness)
    archive = {}
    
    # Bootstrap with previous champions
    if bootstrap_archive:
        for w in bootstrap_archive:
            bd = compute_behavior_descriptor(w, opponents)
            cell = discretize_bd(bd, grid_size)
            fitness = eval_fitness(w, opponents)
            archive[cell] = (w, fitness)
    
    # Random initialization (fill archive with random warriors)
    for _ in range(100):
        w = llm_client.generate_warrior()  # Zero-shot generation
        bd = compute_behavior_descriptor(w, opponents)
        cell = discretize_bd(bd, grid_size)
        fitness = eval_fitness(w, opponents)
        
        if cell not in archive or fitness > archive[cell][1]:
            archive[cell] = (w, fitness)
    
    # Main MAP-Elites loop
    for iteration in range(iterations):
        # 1. Sample parent from archive
        parent_cell = np.random.choice(list(archive.keys()))
        parent_w, parent_f = archive[parent_cell]
        
        # 2. Mutate via LLM
        offspring_w = llm_client.mutate_warrior(parent_w)
        
        # 3. Evaluate offspring
        bd = compute_behavior_descriptor(offspring_w, opponents)
        cell = discretize_bd(bd, grid_size)
        fitness = eval_fitness(offspring_w, opponents)
        
        # 4. Update archive
        if cell not in archive or fitness > archive[cell][1]:
            archive[cell] = (offspring_w, fitness)
        
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1}/{iterations}, "
                  f"archive size: {len(archive)}, "
                  f"best fitness: {max(f for w, f in archive.values()):.3f}")
    
    # Return best warrior from archive
    champion, champion_fitness = max(archive.values(), key=lambda x: x[1])
    return champion


# ============================================================================
# Core War Simulation & Evaluation
# ============================================================================

def eval_fitness(warrior: str, opponents: List[str]) -> float:
    """
    Evaluate warrior fitness against opponents.
    
    Fitness function:
        Fitness(w_i; {w_j}) = Σ_τ (N/T) * (A^i_τ / Σ_j A^j_τ)
    
    where:
        N = total warriors
        T = total timesteps
        A^i_τ = 1 if warrior i alive at timestep τ, else 0
    
    Returns:
        Average fitness over multiple battle seeds
    """
    all_warriors = [warrior] + opponents
    N = len(all_warriors)
    
    fitnesses = []
    for seed in range(10):  # Multiple evaluation seeds
        # Run Core War battle
        sim = CoreWarSimulator(
            warriors=all_warriors,
            core_size=8000,
            max_cycles=80000,
            seed=seed
        )
        
        # Track alive status over time
        alive_history = sim.run()  # Returns [timestep][warrior_idx] -> alive?
        T = len(alive_history)
        
        # Compute fitness
        fitness = 0.0
        for tau in range(T):
            alive_warriors = sum(alive_history[tau])
            if alive_history[tau][0]:  # warrior is index 0
                fitness += (N / T) * (1.0 / alive_warriors)
        
        fitnesses.append(fitness)
    
    return np.mean(fitnesses)


def compute_behavior_descriptor(warrior: str, opponents: List[str]) -> tuple:
    """
    Compute behavioral descriptor: (total_processes, memory_coverage).
    
    Captures high-level strategic behavior during simulation.
    """
    all_warriors = [warrior] + opponents
    
    # Run simulation
    sim = CoreWarSimulator(warriors=all_warriors, core_size=8000, max_cycles=80000)
    stats = sim.run_and_collect_stats()
    
    # Extract warrior 0's statistics
    total_processes = stats[0]['total_spawned_processes']
    memory_coverage = stats[0]['total_memory_coverage']  # Fraction of core touched
    
    return (total_processes, memory_coverage)


def discretize_bd(bd: tuple, grid_size: tuple) -> tuple:
    """
    Discretize behavioral descriptor to grid cell.
    
    Uses log-space discretization for processes and linear for memory.
    """
    processes, memory = bd
    grid_processes, grid_memory = grid_size
    
    # Log-space discretization for processes
    if processes == 0:
        cell_processes = 0
    else:
        log_processes = np.log10(processes + 1)
        max_log = np.log10(1000)  # Assume max ~1000 processes
        cell_processes = int((log_processes / max_log) * grid_processes)
        cell_processes = min(cell_processes, grid_processes - 1)
    
    # Linear discretization for memory coverage
    cell_memory = int(memory * grid_memory)
    cell_memory = min(cell_memory, grid_memory - 1)
    
    return (cell_processes, cell_memory)


# ============================================================================
# LLM Integration
# ============================================================================

class LLMClient:
    """Wrapper for LLM API calls."""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """
        System prompt describing Core War and Redcode.
        
        Includes:
        - Core War environment description
        - Redcode assembly language manual
        - Addressing modes
        - Example warriors
        """
        return """
You are an expert at writing Redcode programs for the game Core War.

Core War is a programming game where assembly-like programs called "warriors"
compete for control of a virtual computer. Warriors are loaded into a circular
memory array (the Core) and execute in round-robin fashion. The goal is to be
the last warrior running by causing opponents to crash.

Redcode Instructions:
- MOV A, B: Copy A to B
- ADD A, B: Add A to B
- SUB A, B: Subtract A from B
- MUL A, B: Multiply A by B
- DIV A, B: Divide B by A
- MOD A, B: Modulo B by A
- JMP A: Jump to A
- JMZ A, B: Jump to A if B is zero
- JMN A, B: Jump to A if B is non-zero
- DJN A, B: Decrement B, jump to A if non-zero
- CMP A, B: Skip next instruction if A equals B
- SEQ A, B: Same as CMP
- SNE A, B: Skip next instruction if A not equals B
- SLT A, B: Skip next instruction if A < B
- SPL A: Split execution (spawn process at A)
- DAT A: Data (terminates process when executed)

Addressing Modes:
- # (immediate): Direct value
- $ (direct): Absolute address
- * (A-indirect): Indirect via A field
- @ (B-indirect): Indirect via B field
- < (pre-decrement): Decrement before use
- > (post-increment): Increment after use

Example Warrior (Dwarf):
; Classic bombing strategy
ADD #4, 3
MOV 2, @2
JMP -2
DAT #0, #0

Your task is to write creative and effective Redcode warriors.
"""
    
    def generate_warrior(self) -> str:
        """Generate new warrior from scratch (zero-shot)."""
        prompt = "Generate a novel and effective Redcode warrior."
        response = self._call_llm(prompt)
        return self._extract_code(response)
    
    def mutate_warrior(self, parent: str) -> str:
        """Mutate existing warrior."""
        prompt = f"""
Here is a Redcode warrior:

{parent}

Generate a modified version that could perform better in Core War battles.
You can:
- Change opcodes or addressing modes
- Adjust numeric parameters
- Add, remove, or reorder instructions
- Combine strategies in new ways

Output only the modified Redcode program.
"""
        response = self._call_llm(prompt)
        return self._extract_code(response)
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call LLM API (pseudo-code)."""
        import openai
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _extract_code(self, response: str) -> str:
        """Extract Redcode from LLM response."""
        # Remove markdown code blocks if present
        if "```" in response:
            lines = response.split('\n')
            in_code = False
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                elif in_code:
                    code_lines.append(line)
            return '\n'.join(code_lines)
        
        return response.strip()
```

---

## Experimental Details

### Setup

**Core War Simulator**:
- Core size: 8,000 cells (standard)
- Max cycles: 80,000
- Circular memory (address wraps around)
- Round-robin execution

**Human Warrior Datasets**:
- **Main**: 294 diverse human warriors
- **Eval**: 317 human warriors (for generality measurement)
- **Small**: 96 diverse warriors (for multi-round experiments)

**MAP-Elites Configuration**:
- Grid: 10×10 (processes × memory coverage)
- Iterations per round: 1,000
- Bootstrap: Previous champions
- Random init: 100 warriors

**LLM**:
- Model: GPT-4o mini (`gpt-4o-mini-2025-04-14`)
- Temperature: 0.7 (default)
- Note: Larger models showed minimal improvement

---

## Experimental Results (Detailed)

### Experiment 1: Static Optimization Baseline

**Setup**: Single round of DRQ against each of 294 human warriors

**Results**:

| Method | Specialist (%) | Generalist (%) | Details |
|--------|----------------|----------------|---------|
| **Zero-shot** | 1.7 | ~1.7 | Single LLM-generated warrior |
| **Best-of-8** | 22.1 | ~22.1 | 8 zero-shot samples, pick best |
| **Evolved (DRQ)** | **96.3** | **27.9** | 1000 iterations MAP-Elites |

**Analysis**:

**Specialist coverage** (collective):
- ≥1 evolved warrior defeats/ties each human warrior
- Demonstrates: Evolution discovers diverse specialists

**Generalist performance** (individual):
- Average single warrior defeats/ties only 27.9% of humans
- Demonstrates: Overfitting to training opponent (brittleness)

**Implication**: Need multi-round DRQ for robust generalists

### Experiment 2: Multi-Round DRQ Dynamics

**Setup**: 96 independent DRQ runs (different initial warriors), 10 rounds each

**Metrics**:
- **Generality**: Fraction of 317 held-out human warriors defeated/tied
- **Phenotype**: Vector of fitness vs each held-out human
- **Genotype**: Text embedding of source code (OpenAI `text-embedding-3-small`)

**Results** (statistical trends over rounds):

| Metric | Trend | p-value | Model |
|--------|-------|---------|-------|
| **Generality** (mean) | ↑ Increasing | < 0.05 | Linear/Log |
| **Phenotype variance** | ↓ Decreasing | < 0.001 | Log |
| **Phenotype change rate** | ↓ Decreasing | < 0.05 | Log |
| **Genotype variance** | → Stable | > 0.05 | — |

**Detailed Findings**:

**1. Increasing Generality**:
```
Round 1: ~30% generality
Round 10: ~40% generality
Trend: Logarithmic growth (diminishing returns)
```

**2. Decreasing Phenotype Variance**:
```
Round 1: High variance (diverse behaviors)
Round 10: Low variance (convergent behaviors)
Convergence: Phenotype space collapses toward general strategy
```

**3. Stable Genotype Variance**:
```
All rounds: Similar code diversity
Implication: Different code → same behavior (many-to-one mapping)
```

**4. History Length Ablation** (K):
```
K=1 (previous only): Slower generality improvement
K=3 (last 3): Moderate improvement
K=∞ (all previous): Best generality improvement
```

**Interpretation**: **Convergent Evolution**
- Like mammalian/insect eyes evolving independently
- Pressure: Defeat diverse opponents → converge to general-purpose strategy
- Mechanism: Red Queen dynamics select for robustness

---

## Predictability Study

**Question**: Can we predict warrior performance from source code?

**Method**:
1. Embed warrior source code: `embed(w) ∈ ℝ^1536` (OpenAI embedding model)
2. Train linear probe: `generality = w^T · embed(warrior) + b`
3. Evaluate on held-out warriors

**Results**:
- **Test R² = 0.461** (moderate predictive power)
- **Interpretation**: Embeddings capture some performance-relevant structure

**Implications**:
- **Surrogate modeling**: Could bypass expensive simulation (partially)
- **Interpretability**: Analyze what code features correlate with generality
- **Future work**: Improve predictor → use for guided search

---

## Core War Primer

### Redcode Example: Imp

```redcode
; Simplest warrior: continuously copy self forward
MOV 0, 1
```

**Behavior**:
1. Copy instruction at address 0 (self) to address 1
2. Execute next instruction (now at address 1)
3. Copy instruction at address 1 to address 2
4. Infinite loop: spreads through memory

**Strategy**: Naive, but demonstrates self-propagation

### Redcode Example: Dwarf (Classic Bomber)

```redcode
       ADD #4, 3       ; Increment bombing target
       MOV 2, @2       ; Bomb target (write DAT)
       JMP -2          ; Loop back
       DAT #0, #0      ; Data (crashes on execution)
```

**Behavior**:
1. Increment target address by 4
2. Write DAT to target (indirect addressing)
3. Jump back to line 1
4. Repeatedly bombs memory every 4 cells

**Strategy**: Bombing (scatter DAT instructions to crash opponents)

### Redcode Example: Scanner

```redcode
start  MOV  #8,      <ptr    ; Set scan increment
       ADD  #5,      ptr     ; Move pointer
       JMZ  start,   @ptr    ; If zero, continue scanning
       SPL  @ptr             ; Found! Split to attack
       JMP  start            ; Keep scanning
ptr    DAT  #0,      #1000   ; Scan pointer
```

**Behavior**:
1. Scan memory for non-zero instructions (enemies)
2. When found, spawn attack process
3. Continue scanning

**Strategy**: Scanning + replication

---

## Worked Example: Single DRQ Round

### Setup

**Opponent**: Dwarf (classic bomber)
**Goal**: Evolve warrior to defeat Dwarf
**Iterations**: 1000 (MAP-Elites)

### Evolution Trace

**Iteration 0** (Random init):
```redcode
; Random warrior (LLM zero-shot)
SPL 1
MOV 0, 1
DAT #0, #0
```
**Fitness**: 0.3 (poor, Dwarf dominates)

**Iteration 100** (Early evolution):
```redcode
; Starting to counter bombing
SPL 1
SPL 2
MOV #0, <5
JMP -1
```
**Fitness**: 0.6 (some resistance via replication)
**Behavior**: (processes=4, memory=15%)

**Iteration 500** (Mid evolution):
```redcode
; Scanning + evasion
MOV #10, <ptr
ADD #8, ptr
JMZ start, @ptr
SPL @ptr
JMP start
ptr DAT #0, #500
```
**Fitness**: 0.8 (defeats Dwarf 80% of the time)
**Behavior**: (processes=8, memory=40%)

**Iteration 1000** (Converged):
```redcode
; Optimized scanner-bomber
start  MOV  #12,    <ptr
       ADD  #6,    ptr
       JMZ  start, @ptr
       SPL  @ptr
       MOV  bomb,  @ptr
       JMP  start
bomb   DAT  #0,    #0
ptr    DAT  #0,    #1000
```
**Fitness**: 0.95 (consistent defeat of Dwarf)
**Behavior**: (processes=16, memory=60%)

**Archive size**: 73 cells filled (out of 100)

**Champion**: Final warrior (fitness 0.95)

---

## Multi-Round Example: Red Queen Dynamics

### Round 1

**Opponents**: {w₀ = Imp}
**Evolved**: w₁ = Scanner-bomber
**Generality**: 32% (defeats 101/317 humans)

### Round 2

**Opponents**: {w₀ = Imp, w₁ = Scanner-bomber}
**Pressure**: Must defeat both naive Imp and sophisticated scanner
**Evolved**: w₂ = Adaptive scanner with anti-scan evasion
**Generality**: 35% (defeats 111/317 humans)

### Round 5

**Opponents**: {w₀, w₁, w₂, w₃, w₄}
**Pressure**: Diverse strategies in environment
**Evolved**: w₅ = Hybrid strategy (scan + bomb + replicate)
**Generality**: 38% (defeats 120/317 humans)

### Round 10

**Opponents**: {w₀, ..., w₉}
**Pressure**: Highly diverse, covering many strategic niches
**Evolved**: w₁₀ = General-purpose robust warrior
**Generality**: 41% (defeats 130/317 humans)
**Convergence**: Phenotype variance = 0.3 (down from 0.8 at round 1)

**Observation**: Later warriors more general but less diverse (convergent evolution)

---

## Design Rationale

### Why MAP-Elites?

**Problem**: Deceptive fitness landscape
- Small code changes → drastic outcome changes
- Local optima everywhere
- Greedy search fails

**Solution**: Quality-Diversity
- Maintain diverse stepping stones
- Exploration + exploitation
- Grid prevents diversity collapse

**Empirical evidence**: Ablation shows MAP-Elites > simple EA

### Why Historical Self-Play (All Previous)?

**Alternatives**:
- **Latest only** (K=1): Faster but cycles
- **Fixed opponent**: Overfitting, no generality

**Chosen** (K=∞):
- Stability: Avoid cycles (rock-paper-scissors dynamics)
- Generality: Pressure to handle diverse strategies
- Consistent with best practices from RL self-play

### Why Don't Update Old Warriors?

**Reason**: Stability
- Co-adapting populations can cycle
- Historical snapshots provide stable curriculum
- Similar to fictitious self-play (FSP)

---

## Limitations & Future Work

### Limitations

1. **Computational cost**: Multi-round expensive (~1000 simulations/iteration)
2. **Domain-specific**: Core War may not transfer to all domains
3. **Convergence rate**: Logarithmic (slow for full convergence)
4. **Predictor accuracy**: R²=0.461 (moderate, not production-ready)
5. **LLM capability**: Requires strong code-generation model

### Future Directions

1. **Better LLM usage**:
   - Diff-based mutations (not full rewrites)
   - Simulation-conditioned feedback (Reflexion-style)
   - Multi-turn refinement

2. **Improved predictors**:
   - Nonlinear models (neural networks)
   - Pretrain on larger warrior datasets
   - Use as surrogate to bypass simulation

3. **Real-world applications**:
   - Cybersecurity (exploit/defense evolution)
   - Drug discovery (virus/drug arms race)
   - Multi-agent RL (self-driving, robotics)

4. **Theoretical analysis**:
   - Convergence guarantees
   - Nash equilibrium connection
   - Diversity-generality trade-off

---

## Key Takeaways

1. **Red Queen > Static**: Multi-round self-play produces robust generalists
2. **Convergent evolution emerges**: Behavior converges, code doesn't
3. **MAP-Elites essential**: Diversity preservation in deceptive landscapes
4. **LLM priors crucial**: 2% zero-shot → 96% evolved (orders of magnitude)
5. **Predictability exists**: Linear probe R²=0.461 (promising for surrogates)
6. **Core War is rich**: Turing-complete, safe sandbox for adversarial AI
7. **Simplicity works**: Minimal self-play algorithm achieves strong results

---

## References

- **Paper**: Kumar et al., "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs", MIT & Sakana AI, 2025
- **Website**: https://pub.sakana.ai/drq
- **Code**: https://github.com/SakanaAI/drq
- **Core War**: Dewdney (1984), Jones (1984), Rasmussen (1990)
- **MAP-Elites**: Mouret & Clune (2015)
- **Self-play**: FSP (Heinrich et al., 2015), PSRO (Lanctot et al., 2017)
- **LLM evolution**: Lehman et al. (2023), AlphaEvolve (Novikov et al., 2025)
