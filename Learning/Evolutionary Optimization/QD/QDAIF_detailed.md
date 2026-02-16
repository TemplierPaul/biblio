# QDAIF — Detailed Implementation Notes

> **Quick overview**: [[QDAIF]]

## Paper

**Title**: Quality-Diversity through AI Feedback

**Authors**: Herbie Bradley, Andrew Dai, Hannah Teufel, Jenny Zhang, Koen Oostermeijer, Marco Bellagente, Jeff Clune, Kenneth Stanley, Grégory Schott, Joel Lehman

**Year**: 2023

**Project Page**: https://qdaif.github.io/

## Full Algorithm Pseudocode

### Core QDAIF Algorithm

```python
class QDAIF:
    def __init__(self, quality_prompt, diversity_prompts, bin_ticks,
                 lm_generator, lm_evaluator):
        """
        quality_prompt: Natural language prompt for quality evaluation
        diversity_prompts: List of prompts for diversity dimensions
        bin_ticks: Non-uniform bin boundaries per diversity dimension
        lm_generator: LM for generating solutions (e.g., GPT-3.5)
        lm_evaluator: LM for evaluating solutions (may be same or different)
        """
        self.archive = {}  # bin_index → (solution, quality, diversity)
        self.quality_prompt = quality_prompt
        self.diversity_prompts = diversity_prompts
        self.bin_ticks = bin_ticks
        self.lm_gen = lm_generator
        self.lm_eval = lm_evaluator

    def run(self, iterations, init_method='seeded', seed_texts=None):
        # Initialize archive
        if init_method == 'seeded':
            prompt_pool = seed_texts  # Hand-written examples
        else:  # zero-shot
            prompt_pool = self._generate_zero_shot_init()

        # Seed archive with initial solutions
        for text in prompt_pool:
            quality = self._evaluate_quality(text)
            diversity = self._evaluate_diversity(text)
            bin_idx = self._discretize(diversity)
            self._try_add(bin_idx, text, quality, diversity)

        # Main evolution loop
        for iteration in range(iterations):
            # Selection: Random parent from archive
            parent = random.choice(list(self.archive.values()))

            # Mutation: LMX (Language Model Crossover)
            offspring = self._mutate_lmx(parent, prompt_pool)

            # Evaluation
            quality = self._evaluate_quality(offspring)
            diversity = self._evaluate_diversity(offspring)
            bin_idx = self._discretize(diversity)

            # Update archive
            added = self._try_add(bin_idx, offspring, quality, diversity)

            # Update prompt pool (for LMX-Replace variant)
            if added:
                self._update_prompt_pool(prompt_pool, offspring, quality)

        return self.archive

    def _evaluate_quality(self, text):
        """Use AI feedback to evaluate quality"""
        prompt = f"{self.quality_prompt}\n\n{text}\n\nAnswer: "

        # Get log probabilities for "yes" vs "no"
        logprobs = self.lm_eval.get_logprobs(prompt, tokens=["yes", "no"])

        # Quality = log(P(yes) / P(no))
        # Normalize to [0, 1] via sigmoid
        quality = sigmoid(logprobs["yes"] - logprobs["no"])
        return quality

    def _evaluate_diversity(self, text):
        """Use AI feedback to evaluate diversity attributes"""
        diversity_values = []

        for dim_prompt in self.diversity_prompts:
            prompt = f"{dim_prompt}\n\n{text}\n\nAnswer: "

            # Get log probabilities for diversity labels
            # E.g., "positive" vs "negative" for sentiment
            logprobs = self.lm_eval.get_logprobs(prompt)

            # Extract diversity measure from log-prob ratio
            # Specific to domain (e.g., positive vs negative labels)
            diversity_val = self._extract_diversity_from_logprobs(logprobs)
            diversity_values.append(diversity_val)

        return np.array(diversity_values)

    def _discretize(self, diversity):
        """Map continuous diversity values to bin indices"""
        bin_indices = []
        for dim_idx, val in enumerate(diversity):
            # Find which bin this value falls into
            bins = self.bin_ticks[dim_idx]
            bin_idx = np.searchsorted(bins, val) - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            bin_indices.append(bin_idx)

        return tuple(bin_indices)

    def _try_add(self, bin_idx, solution, quality, diversity):
        """Add solution to archive if it improves the bin"""
        if (bin_idx not in self.archive or
            quality > self.archive[bin_idx]['quality']):
            self.archive[bin_idx] = {
                'solution': solution,
                'quality': quality,
                'diversity': diversity
            }
            return True
        return False

    def _mutate_lmx(self, parent, prompt_pool):
        """Language Model Crossover mutation"""
        # Sample few-shot examples from prompt pool
        examples = random.sample(prompt_pool, k=min(3, len(prompt_pool)))

        # Create few-shot prompt
        prompt = "Here is a random example:\n\n"
        for ex in examples:
            prompt += f"{ex}\n\n"

        # Generate offspring
        offspring = self.lm_gen.generate(prompt, max_tokens=256)
        return offspring
```

### LMX-Replace Variant

```python
class QDAIF_LMXReplace(QDAIF):
    """
    LMX-Replace: Slower mutation, maintains larger prompt pool
    Uses archive depth to track multiple solutions per bin
    """
    def __init__(self, *args, archive_depth=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive_depth = archive_depth
        self.prompt_pool_depth = {}  # bin_idx → list of solutions

    def _update_prompt_pool(self, prompt_pool, offspring, quality):
        """Maintain prompt pool with archive depth"""
        # Add to depth archive
        bin_idx = self._discretize(self._evaluate_diversity(offspring))

        if bin_idx not in self.prompt_pool_depth:
            self.prompt_pool_depth[bin_idx] = []

        self.prompt_pool_depth[bin_idx].append({
            'solution': offspring,
            'quality': quality
        })

        # Sort by quality, keep top-k per bin
        self.prompt_pool_depth[bin_idx].sort(
            key=lambda x: x['quality'], reverse=True
        )
        self.prompt_pool_depth[bin_idx] = \
            self.prompt_pool_depth[bin_idx][:self.archive_depth]

        # Rebuild flat prompt pool from depth archive
        prompt_pool.clear()
        for bin_solutions in self.prompt_pool_depth.values():
            for entry in bin_solutions:
                prompt_pool.append(entry['solution'])

    def _mutate_lmx(self, parent, prompt_pool):
        """LMX-Replace: Mutate only one example at a time"""
        # Sample few-shot examples
        examples = random.sample(prompt_pool, k=min(3, len(prompt_pool)))

        # Replace one example with parent
        replace_idx = random.randint(0, len(examples) - 1)
        examples[replace_idx] = parent['solution']

        # Create prompt and generate
        prompt = "Here is a random example:\n\n"
        for ex in examples:
            prompt += f"{ex}\n\n"

        offspring = self.lm_gen.generate(prompt, max_tokens=256)
        return offspring
```

## Prompt Engineering for Evaluation

### Quality Evaluation Prompts

**Opinions Domain:**
```
Does the following text contain a relevant, well-written opinion about
eating vegetables and plant-based foods? Answer 'yes' or 'no'.

[TEXT]

Answer:
```

**Stories Domain:**
```
Is the following story high-quality, coherent, and relevant to a
spy and politician? Answer 'yes' or 'no'.

[TEXT]

Answer:
```

**Poetry Domain (Rating Scale):**
```
[POEM]

Rate the quality of the above poem on a scale from 1 to 10.
Answer in JSON with the key 'quality'.
```

### Diversity Evaluation Prompts

**Sentiment (Opinions):**
```
Is the sentiment of the following opinion positive or negative?

[TEXT]

Answer:
```
- Diversity = log(P("positive")) - log(P("negative"))
- Normalized to [0, 1]

**Genre (Stories):**
```
What is the genre of this story? Reply with 'horror' or 'romance'.

[STORY]

Genre:
```

**Ending Type (Stories):**
```
Does this story have a happy ending or tragic ending?
Reply with 'happy' or 'tragic'.

[STORY]

Ending:
```

**Poetry (Categorical):**
```
[POEM]

What genre is this poem closest to from the following list:
["haiku", "sonnet", "ballad", "limerick", "hymn"]

What tone is this poem closest to from the following list:
["happy", "dark", "mysterious", "romantic", "reflective"]

Respond in JSON with the keys "genre" and "tone".
```
- Use raw predicted category (not log-probs) for binning

## Non-Uniform Binning Strategy

### Why Non-Uniform?
LM calibration is non-linear when predicting labels:
- Token probabilities don't uniformly correlate with sentiment intensity
- More bins needed at extremes to capture qualitative differences
- Example: Stories rated 0.99 vs 0.995 in "romance" may be qualitatively similar

### Bin Configuration

**1D Archive (20 bins):**
```python
bin_ticks = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
             0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97,
             0.98, 0.985, 0.99, 0.995, 1]
```

**2D Archive (10×10 bins):**
```python
bin_ticks_2d = [0, 0.005, 0.02, 0.05, 0.20, 0.50,
                0.80, 0.95, 0.98, 0.995, 1]
```

### Binning Ablation Study Results
- **Opinions**: Non-uniform >> uniform (better sentiment progression)
- **Stories**: Uniform ≈ non-uniform (LM better calibrated for genre)
- **Recommendation**: Test both; depends on domain and LM calibration

## Initialization Methods

### Seeded Initialization (Default)
Use hand-written examples spanning diversity space:

**Opinions (Sentiment: Negative → Neutral → Positive):**
1. "Vegetables taste quite bad, and I don't like eating them. I would much prefer eating meat and ice cream."
2. "I do not have an opinion on eating vegetables and other plant-based foods."
3. "Plant-based foods are a great source of healthy micronutrients... I would highly recommend including many different foods such as vegetables and pulses in your regular diet."

**Stories (Spy + Politician):**
- 3 hand-written seed stories
- Cover different genres/endings
- Introduce named characters

### Zero-Shot Initialization
Generate initial population without examples:
```python
def generate_zero_shot_init(n=10):
    prompt = "Write a random opinion about eating vegetables."
    return [lm.generate(prompt) for _ in range(n)]
```

**Performance:**
- Seeded Init: Human QD Score = 0.772
- Zero-Shot Init: Human QD Score = 0.383
- Zero-shot enables more exploration but risks reward hacking
- Use LMX-Replace with zero-shot for better quality

## Experimental Results

### QD Score Performance

| Method | Opinions | Stories (Genre) | Stories (Ending) |
|--------|----------|-----------------|------------------|
| QDAIF (Seeded) | **18.5** | **18.3** | **18.9** |
| LMX Quality-Only | 14.2 | 15.1 | 16.2 |
| Fixed Few-Shot | 12.8 | 14.6 | 15.8 |
| Random Search | 11.5 | 13.2 | 14.1 |

Maximum possible QD score = 20 (20 bins × max quality 1.0)

### Coverage

| Method | Opinions | Stories (Genre) | Stories (Ending) |
|--------|----------|-----------------|------------------|
| QDAIF | **100%** | **95%** | **100%** |
| LMX Quality-Only | 85% | 78% | 92% |
| Fixed Few-Shot | 70% | 75% | 88% |

### Human Evaluation

**Quality Scores (1-5 Likert scale):**
- QDAIF (Seeded): 3.90
- Fixed Few-Shot: 4.13 (higher but less diverse)
- LMX Quality-Only: 3.53
- Random Search: 3.30

**Human-AI Agreement:**
- Diversity labels: 73% overall, 82% when humans agree
- Quality correlation: Strong up to AI fitness ~0.95
- Reward hacking: AI fitness >0.995 often misaligns with human preference

## Implementation Tips

### 1. Model Selection
**Generator LM:**
- Use instruction-tuned models (GPT-3.5, GPT-4, Luminous)
- Temperature: 0.8-1.2 for diversity
- Top-p: 0.9-0.95

**Evaluator LM:**
- Can be same as generator or separate
- Instruction-tuned models work best
- Consider finetuning on domain-specific quality/diversity examples

### 2. Hyperparameters

```python
config = {
    # Search
    'num_iterations': 2000,
    'batch_size': 1,  # Evaluate 1 offspring per iteration

    # Archive
    'num_bins_1d': 20,
    'num_bins_2d': 10,  # Per dimension
    'archive_depth': 100,  # For LMX-Replace

    # LM Generation
    'temperature': 1.0,
    'top_p': 0.95,
    'max_tokens': 256,

    # Initialization
    'init_method': 'seeded',  # or 'zero_shot'
    'num_seed_examples': 3,
}
```

### 3. Prompt Engineering
- **Be specific**: "Is this opinion positive or negative?" not "Evaluate sentiment"
- **Use binary choices**: Easier to extract log-probs
- **Test calibration**: Check if LM probabilities align with human judgment
- **Few-shot for evaluation**: Can improve consistency (but tested zero-shot by default)

### 4. Computational Budget
- ~2000 iterations typical
- Each iteration: 1 generation + 1-3 evaluation calls
- Total LM calls: 4000-8000
- Cost estimate (GPT-3.5): $5-20 per run
- Cost estimate (GPT-4): $50-200 per run

### 5. Handling Reward Hacking
**Symptoms:**
- Very high AI quality scores (>0.995)
- Repetitive phrases that game the evaluator
- Outputs don't match human preferences

**Mitigations:**
1. Use seeded initialization with high-quality examples
2. Add quality threshold filters (e.g., reject if quality >0.99)
3. Use ensemble of evaluator LMs
4. Manual inspection of high-fitness solutions
5. RLHF-tuned evaluator models

### 6. Expanding Diversity Dimensions
Can automatically discover new diversity axes:
```python
# Ask LM to suggest diversity dimensions
prompt = """
For creative short stories about a spy and politician,
what are 3 interesting diversity dimensions to explore?

Examples: genre, narrative perspective, ending type, tone

Suggest 3 new dimensions:
"""

# Use GPT-4 to generate suggestions
suggestions = lm.generate(prompt)

# Manually review and add to diversity_prompts
```

**Expanding During Search:**
- Start with 1D archive (e.g., genre)
- At iteration 1000/2000: Add 2nd dimension (e.g., ending)
- Results: Approaches 2D performance, better quality than 1D-only

## Advanced Variants

### 1. Few-Shot AI Feedback
Default is zero-shot evaluation. Can add examples:
```python
eval_prompt = """
### Instruction:
What is the genre of this story? Reply with 'horror' or 'romance'

### Input: [HORROR EXAMPLE STORY]
### Genre: horror

### Input: [ROMANCE EXAMPLE STORY]
### Genre: romance

### Input: [NEW STORY]
### Genre:
"""
```
- 8-shot > 4-shot > 2-shot > zero-shot for quality
- But lower human-AI agreement (80% → 57%)

### 2. Model Scaling
Tested 13B, 30B, 70B parameter models:
- Quality: 70B > 30B > 13B (4.03 vs 3.60 vs 3.43)
- QD Score: Not monotonic (30B ≈ 13B > 70B in some domains)
- Larger models may need different binning strategies

### 3. Finetuning the Generator
- Collect high-quality solutions from archive
- Finetune LM on these examples
- **Warning**: Easy to overfit → hurts diversity
- Adapter finetuning > full-model finetuning
- Generally not recommended (default pre-trained works well)

## Code Resources

### Minimal Working Example

```python
import openai
import numpy as np

class SimplifiedQDAIF:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.archive = {}

    def run(self, iterations=100):
        # Seed archive
        seeds = [
            "Vegetables are disgusting.",
            "Vegetables are okay.",
            "Vegetables are delicious!"
        ]

        for seed in seeds:
            self.evaluate_and_add(seed)

        # Evolution
        for i in range(iterations):
            parent = self.select_parent()
            offspring = self.mutate(parent)
            self.evaluate_and_add(offspring)

            if i % 20 == 0:
                print(f"Iteration {i}: QD Score = {self.qd_score():.2f}")

        return self.archive

    def mutate(self, parent):
        prompt = f"Here is a random opinion about vegetables:\n\n{parent}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0
        )
        return response.choices[0].message.content

    def evaluate_quality(self, text):
        prompt = f"Is this a well-written opinion? yes/no\n\n{text}"
        # Simplified: Use completion probability
        # Real implementation would use logprobs API
        return np.random.random()  # Placeholder

    def evaluate_diversity(self, text):
        prompt = f"Is this opinion positive or negative?\n\n{text}"
        # Simplified: Random sentiment
        return np.random.random()  # Placeholder

    def evaluate_and_add(self, text):
        quality = self.evaluate_quality(text)
        diversity = self.evaluate_diversity(text)
        bin_idx = int(diversity * 10)  # 10 bins

        if bin_idx not in self.archive or quality > self.archive[bin_idx]['quality']:
            self.archive[bin_idx] = {'text': text, 'quality': quality}

    def select_parent(self):
        return np.random.choice([v['text'] for v in self.archive.values()])

    def qd_score(self):
        return sum(v['quality'] for v in self.archive.values())

# Run
qdaif = SimplifiedQDAIF()
archive = qdaif.run(iterations=200)
```

### Using OpenAI API for Logprobs

```python
import openai

def get_logprobs_for_labels(prompt, labels=["yes", "no"]):
    """Get log probabilities for specific completion tokens"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1,
        logprobs=5,  # Return top-5 logprobs
        temperature=0
    )

    # Extract logprobs for our labels
    top_logprobs = response.choices[0].logprobs.top_logprobs[0]

    logprobs = {}
    for label in labels:
        # Handle case variations
        for key in top_logprobs.keys():
            if key.strip().lower() == label.lower():
                logprobs[label] = top_logprobs[key]
                break
        if label not in logprobs:
            logprobs[label] = -float('inf')  # Very low prob

    return logprobs

# Example usage
prompt = "Is this opinion positive? 'I love vegetables!'\n\nAnswer:"
logprobs = get_logprobs_for_labels(prompt, labels=["yes", "no"])
quality = 1 / (1 + np.exp(-(logprobs["yes"] - logprobs["no"])))  # Sigmoid
```

## Limitations and Future Directions

### Current Limitations
1. **Reward Hacking**: High AI scores don't always mean high human scores
2. **Cost**: Expensive for large-scale searches (1000s of LM calls)
3. **Manual Diversity Axes**: Still need to specify what diversity means
4. **LM-Dependent**: Performance tied to specific LM capabilities

### Future Research Directions
1. **Automatic Diversity Discovery**: Use LMs to suggest diversity axes
2. **Multi-Modal QDAIF**: Extend to images, video, code
3. **Interactive QDAIF**: Human-in-the-loop for real-time refinement
4. **Hierarchical QD**: Nested diversity spaces (genre → subgenre → style)
5. **Meta-Learning**: Learn better evaluation prompts over time
6. **Efficient Search**: Reduce LM calls via surrogate models

## References

- [QDAIF Paper (2023)](https://qdaif.github.io/)
- [MAP-Elites (Mouret & Clune, 2015)](https://arxiv.org/abs/1504.04909)
- [Evolution through Large Models (Lehman et al., 2022)](https://arxiv.org/abs/2206.08896)
- [Language Model Crossover (Meyerson et al., 2023)](https://arxiv.org/abs/2302.09236)

## Related

- [[QDAIF]] — Quick overview
- [[MAP_Elites]] / [[MAP_Elites_detailed]] — Base QD algorithm
- [[Quality_Diversity]] — QD paradigm
- [[Evolutionary Optimization]] — Parent topic
