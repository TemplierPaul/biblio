# QDAIF (Quality-Diversity through AI Feedback)

## Definition
QDAIF extends Quality-Diversity search to **subjective creative domains** by using Large Language Models (LMs) for generation, mutation, and evaluation. It enables QD algorithms to discover diverse, high-quality solutions in domains where hand-crafted fitness functions are infeasible (e.g., creative writing, art).

## Core Idea
Traditional QD requires:
1. Hand-designed quality metrics
2. Hand-designed diversity metrics (behavior descriptors)
3. Domain-specific mutation operators

QDAIF replaces all three with **AI Feedback via natural language prompts**, making QD applicable to qualitative domains like creative writing, poetry, and storytelling.

## Three Roles of Language Models

### 1. Generation (Mutation)
**LMX (Language Model Crossover)**: Use few-shot prompting to evolve text-based solutions
- Parent solution forms part of in-context examples
- LM generates variation through prompted completion
- Example: "Here is a random opinion about vegetables: ..."

### 2. Quality Evaluation
**AI Feedback for Quality**: Prompt LM to assess solution quality
- Natural language query: "Is this a high-quality story? yes/no"
- Quality score = log-probability ratio of "yes" vs "no"
- Replaces hand-crafted fitness functions

### 3. Diversity Evaluation
**AI Feedback for Diversity**: Prompt LM to identify diversity attributes
- Example: "Is this opinion positive or negative?"
- Diversity measure = log-probability ratio
- Maps continuous values to archive bins

## Key Innovations

### AI Feedback for Diversity
First QD algorithm to use LM-based diversity evaluation:
- **Traditional QD**: Behavior descriptors = robot (x,y) position, robot morphology dimensions
- **QDAIF**: Behavior descriptors = sentiment (positive/negative), genre (horror/romance), tone, etc.
- Enables subjective diversity measures through natural language

### Non-Uniform Binning
- Standard MAP-Elites: Uniform grid discretization
- **QDAIF**: Custom non-uniform bins denser at range ends
- **Why?** LM calibration is non-linear; log-probabilities don't uniformly map to qualitative changes
- Example bins: [0, 0.005, 0.01, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995, 1]

### LMX Variants
**LMX-Near** (default):
- Mutate all few-shot examples simultaneously
- Faster iteration, higher exploration

**LMX-Replace**:
- Mutate one few-shot example at a time
- Uses archive depth to maintain prompt pool
- Better quality with zero-shot initialization

## Algorithm Overview
```
1. Initialize archive (grid indexed by diversity measures)
2. Seed with few-shot examples OR zero-shot generations
3. For each iteration:
   a. Select parent solution from archive
   b. Mutate via LMX (few-shot prompting with LM)
   c. Evaluate quality via AI feedback (log-prob ratio)
   d. Evaluate diversity via AI feedback (log-prob ratio)
   e. Discretize diversity → bin index (non-uniform bins)
   f. If new solution beats current bin occupant → replace
```

## Results

### Domains Tested
1. **Opinions**: Sentiment diversity (negative → positive)
2. **Stories (Genre)**: Genre diversity (horror → romance)
3. **Stories (Ending)**: Ending type (tragic → happy)
4. **Stories (2D)**: Genre × Ending
5. **Poetry**: Genre × Tone (5×5 categories)

### Performance
- **QD Score**: QDAIF >> all baselines across domains
- **Coverage**: QDAIF fills more bins with high-quality solutions
- **Human Evaluation**:
  - AI-human agreement on diversity labels: ~73-80%
  - AI-human quality correlation: Strong (except at extreme high fitness → reward hacking)
  - Human QD Score: QDAIF = 0.772 vs best baseline = 0.696

### Key Findings
- AI feedback aligns well with human judgment
- Seeded initialization >> zero-shot (0.772 vs 0.383 human QD score)
- LMX-Replace better with zero-shot init
- Non-uniform binning improves qualitative diversity alignment

## When to Use QDAIF

### Ideal Use Cases
- **Creative text generation**: Diverse stories, poems, opinions
- **Subjective quality domains**: Where hand-crafted metrics fail
- **Exploration-heavy tasks**: Need diverse solutions, not just optimal one
- **Prompt engineering**: Discover diverse high-quality prompts

### Not Ideal For
- Well-defined objective functions (use standard QD/EA)
- Single-solution optimization (use standard LLM prompting)
- Real-time applications (LLM calls are expensive)
- Domains where LLM feedback is unreliable

## Limitations
1. **Reward Hacking**: Very high AI fitness scores (>0.995) often misalign with human preferences
2. **Cost**: Requires many LLM API calls (2000+ iterations typical)
3. **Diversity Specification**: Still requires defining diversity axes manually
4. **LLM Calibration**: Non-uniform binning needed due to model calibration issues

## Connections
- **Evolution through Large Models (ELM)**: QDAIF is QD + ELM
- **Constitutional AI / RLAIF**: Both use AI feedback, but QDAIF focuses on diversity
- **Novelty Search**: QDAIF seeks quality AND diversity (NS only diversity)
- **MAP-Elites**: QDAIF = MAP-Elites with LM-based operators and evaluation

> Detailed implementation: [[QDAIF_detailed]]

## Related
- [[MAP_Elites]] — Base QD algorithm
- [[Quality_Diversity]] — QD paradigm overview
- [[Evolutionary Optimization]] — Parent topic
