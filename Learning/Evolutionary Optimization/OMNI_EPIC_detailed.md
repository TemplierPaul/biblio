# OMNI-EPIC: Detailed Implementation Guide

## System Architecture

OMNI-EPIC consists of six integrated components:
1. **Task Archive**: Storage for task descriptions, code, and outcomes
2. **Task Generator**: LLM-based natural language task creation
3. **Environment Generator**: Code synthesis from task descriptions
4. **Post-Generation MoI**: Interestingness filtering
5. **RL Training**: Agent learning with DreamerV3
6. **Success Detection**: LLM-generated task completion evaluation

---

## Task Archive Implementation

### Data Structure

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Task:
    """Single task entry in archive."""
    task_id: str
    description: str  # Natural language
    environment_code: str  # Python code for env
    success_code: str  # get_success() function code
    status: str  # 'learned', 'failed', 'uninteresting'
    embedding: np.ndarray  # OpenAI embedding
    training_metrics: Optional[dict] = None


class TaskArchive:
    """
    Archive of all generated tasks.
    """
    def __init__(self):
        self.tasks: List[Task] = []
        self.embedding_model = "text-embedding-3-small"

    def add_task(self, task: Task):
        """Add task to archive."""
        # Compute embedding for task description + code
        task.embedding = self.compute_embedding(
            task.description + "\n" + task.environment_code
        )
        self.tasks.append(task)

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute OpenAI embedding for text."""
        import openai
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def get_similar_tasks(self, query_embedding: np.ndarray,
                         k: int, status_filter: Optional[str] = None):
        """
        Retrieve k most similar tasks by embedding distance.

        Args:
            query_embedding: Query vector
            k: Number of tasks to retrieve
            status_filter: Optional filter by status ('learned', 'failed')

        Returns:
            similar_tasks: List of k most similar tasks
        """
        # Filter by status if specified
        if status_filter:
            candidates = [t for t in self.tasks if t.status == status_filter]
        else:
            candidates = self.tasks

        # Compute cosine similarities
        similarities = []
        for task in candidates:
            sim = np.dot(query_embedding, task.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(task.embedding)
            )
            similarities.append((task, sim))

        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [task for task, sim in similarities[:k]]
```

### Initialization

```python
def initialize_archive():
    """
    Initialize archive with 3 seed task descriptions.
    """
    archive = TaskArchive()

    # Seed tasks (examples - actual seeds from paper appendix)
    seed_descriptions = [
        "Run forward as fast as possible across flat terrain",
        "Push a red box to a blue receptacle",
        "Cross a bridge with gaps by jumping over them"
    ]

    for desc in seed_descriptions:
        # Generate environment code for seeds
        env_code = generate_environment_code(desc)
        success_code = generate_success_code(desc, env_code)

        task = Task(
            task_id=str(uuid.uuid4()),
            description=desc,
            environment_code=env_code,
            success_code=success_code,
            status='learned',  # Seeds assumed learnable
            embedding=None  # Will be computed when added
        )
        archive.add_task(task)

    return archive
```

---

## Task Generator (LLM-Based)

### Prompt Template

```python
TASK_GENERATOR_PROMPT = """
You are an expert at generating creative and learnable robotics tasks.

Your goal: Generate a new task that is:
1. Novel and interesting compared to previous tasks
2. Learnable by an RL agent (not too easy, not impossibly hard)
3. Inspired by the successful and failed tasks below

# Previously Successful Tasks
{successful_tasks}

# Previously Failed Tasks
{failed_tasks}

# Instructions
- Build upon patterns from successful tasks
- Avoid patterns from failed tasks
- Introduce creative variations or combinations
- Keep the task learnable and interesting

Generate a new task description (one sentence, clear and specific).

New Task:"""


def generate_task_description(archive: TaskArchive) -> str:
    """
    Generate new task description using Claude 3 Opus.

    Args:
        archive: Task archive for context retrieval

    Returns:
        task_description: Natural language description
    """
    import anthropic

    # Compute embedding for retrieval (use recent task as query)
    if len(archive.tasks) > 0:
        query_embedding = archive.tasks[-1].embedding
    else:
        # First iteration: use random embedding
        query_embedding = np.random.randn(1536)  # OpenAI embedding dim

    # Retrieve 5 successful + 5 failed tasks
    successful_tasks = archive.get_similar_tasks(
        query_embedding, k=5, status_filter='learned'
    )
    failed_tasks = archive.get_similar_tasks(
        query_embedding, k=5, status_filter='failed'
    )

    # Format for prompt
    successful_str = "\n".join([
        f"- {task.description}" for task in successful_tasks
    ])
    failed_str = "\n".join([
        f"- {task.description}" for task in failed_tasks
    ])

    # Fill prompt template
    prompt = TASK_GENERATOR_PROMPT.format(
        successful_tasks=successful_str or "None yet",
        failed_tasks=failed_str or "None yet"
    )

    # Call Claude 3 Opus
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        temperature=0,  # Deterministic
        messages=[{"role": "user", "content": prompt}]
    )

    task_description = message.content[0].text.strip()
    return task_description
```

---

## Environment Generator (Code Synthesis)

### Prompt Template

```python
ENVIRONMENT_GENERATOR_PROMPT = """
You are an expert Python programmer specializing in PyBullet robotics simulation.

Task Description: {task_description}

Generate a complete Gymnasium-compliant environment for this task using PyBullet.

# Requirements
1. Must implement these functions:
   - reset(): Initialize environment, return observation
   - step(action): Update state, return (observation, reward, terminated, truncated, info)
   - reward(): Compute reward signal
   - terminated(): Check episode termination

2. Use PyBullet for physics simulation
3. Use R2D2 robot (provided): 6 discrete actions (nothing, forward, backward, rotate CW, rotate CCW, jump)
4. Observation: 64x64x3 RGB image + proprioceptive data
5. Reward function: Shape learning (not just sparse terminal reward)
6. Episode limit: 1000 timesteps

# Example Structure (simplified)
```python
import pybullet as p
import gymnasium as gym

class CustomEnv(gym.Env):
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        # Initialize objects, robot, etc.

    def reset(self):
        # Reset positions, counters
        return observation

    def step(self, action):
        # Apply action, step physics
        reward = self.reward()
        terminated = self.terminated()
        return obs, reward, terminated, False, {{}}

    def reward(self):
        # Compute reward (encourage task completion)
        return reward_value

    def terminated(self):
        # Check termination conditions
        return done
```

Generate complete environment code for: {task_description}

Code:"""


def generate_environment_code(task_description: str,
                              max_attempts: int = 5) -> str:
    """
    Generate environment code with error correction loop.

    Args:
        task_description: Natural language task description
        max_attempts: Maximum compilation attempts

    Returns:
        environment_code: Executable Python code
    """
    import anthropic

    client = anthropic.Anthropic()
    prompt = ENVIRONMENT_GENERATOR_PROMPT.format(
        task_description=task_description
    )

    for attempt in range(max_attempts):
        # Generate code
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        code = message.content[0].text.strip()

        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()

        # Try to compile
        try:
            compile(code, '<string>', 'exec')
            return code  # Success!
        except SyntaxError as e:
            # Add error feedback to prompt
            error_msg = f"\n\nCompilation error: {str(e)}\nPlease fix the code."
            prompt += error_msg

    # Failed after max attempts
    raise RuntimeError(f"Could not generate valid code after {max_attempts} attempts")
```

### Few-Shot Examples

Include 5 exemplar task-code pairs in the prompt to improve quality:

```python
FEW_SHOT_EXAMPLES = [
    {
        "task": "Run forward across flat terrain",
        "code": "# ... complete implementation ..."
    },
    {
        "task": "Push box to receptacle",
        "code": "# ... complete implementation ..."
    },
    # ... 3 more examples
]

# Add to prompt
prompt += "\n\n# Examples\n"
for ex in FEW_SHOT_EXAMPLES:
    prompt += f"\nTask: {ex['task']}\n{ex['code']}\n"
```

---

## Post-Generation Model of Interestingness

### Prompt Template

```python
MoI_PROMPT = """
You are evaluating whether a newly generated robotics task is interesting.

# New Task
{new_task_description}

# Similar Previously Completed Tasks
{similar_tasks}

# Evaluation Criteria
A task is interesting if it is:
1. Novel compared to previous tasks (not redundant)
2. Surprising or creative (introduces new elements)
3. Diverse (explores different aspects of robotics)
4. Worthwhile (meaningful challenge, not trivial)

Is this new task interesting compared to the previous tasks?

Answer (Yes/No):"""


def evaluate_interestingness(new_task_description: str,
                             archive: TaskArchive) -> bool:
    """
    Evaluate if task is interesting using GPT-4o.

    Args:
        new_task_description: New task to evaluate
        archive: Task archive for context

    Returns:
        is_interesting: Whether task passes MoI filter
    """
    import openai

    # Compute embedding for new task
    new_embedding = archive.compute_embedding(new_task_description)

    # Retrieve 10 most similar tasks
    similar_tasks = archive.get_similar_tasks(new_embedding, k=10)

    # Format for prompt
    similar_str = "\n".join([
        f"- {task.description}" for task in similar_tasks
    ])

    prompt = MoI_PROMPT.format(
        new_task_description=new_task_description,
        similar_tasks=similar_str or "None yet"
    )

    # Call GPT-4o
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer
```

---

## Success Detection (LLM-Generated)

### Prompt Template

```python
SUCCESS_DETECTOR_PROMPT = """
Generate a Python function `get_success(self)` that returns True if the task is completed, False otherwise.

Task: {task_description}

Environment Code:
{environment_code}

# Requirements
- Function signature: `def get_success(self) -> bool:`
- Use environment state variables (self.robot_pos, self.object_positions, etc.)
- Return True only when task objective is clearly accomplished
- Different from reward function: success is binary completion, reward shapes learning

Example:
```python
def get_success(self) -> bool:
    # Task: "Run forward 10 meters"
    return self.robot_pos[0] >= 10.0
```

Generate get_success() for: {task_description}

Code:"""


def generate_success_code(task_description: str,
                         environment_code: str) -> str:
    """
    Generate success detection function using GPT-4o.

    Args:
        task_description: Task description
        environment_code: Generated environment code

    Returns:
        success_code: get_success() function code
    """
    import openai

    prompt = SUCCESS_DETECTOR_PROMPT.format(
        task_description=task_description,
        environment_code=environment_code
    )

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        temperature=0,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    code = response.choices[0].message.content.strip()

    # Extract code from markdown
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()

    return code
```

---

## RL Training (DreamerV3)

### Configuration

```python
DREAMERV3_CONFIG = {
    # Training
    'total_timesteps': 2_000_000,
    'batch_size': 16,
    'batch_length': 64,
    'learning_rate': 3e-4,

    # Replay buffer
    'replay_buffer_size': 1_000_000,

    # Discount
    'discount': 0.997,

    # World model
    'model_hidden_size': 512,
    'model_rnn_size': 512,

    # Policy
    'policy_hidden_size': 512,
    'actor_grad': 'dynamics',  # Backprop through world model

    # Value
    'value_hidden_size': 512,

    # Encoder/Decoder
    'encoder_depth': 32,
    'decoder_depth': 32,
}


def train_agent(environment_code: str,
               initial_policy: Optional[str] = None) -> Tuple[Policy, dict]:
    """
    Train RL agent on environment using DreamerV3.

    Args:
        environment_code: Python code for environment
        initial_policy: Optional policy to initialize from (transfer learning)

    Returns:
        trained_policy: Trained policy
        metrics: Training metrics (rewards, success rate, etc.)
    """
    import dreamerv3

    # Create environment from code
    exec_globals = {}
    exec(environment_code, exec_globals)
    env_class = exec_globals['CustomEnv']
    env = env_class()

    # Initialize DreamerV3
    agent = dreamerv3.Agent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        config=DREAMERV3_CONFIG
    )

    # Load initial policy if provided (transfer learning)
    if initial_policy is not None:
        agent.load_state(initial_policy)

    # Training loop
    metrics = {
        'episode_rewards': [],
        'success_rate': [],
        'training_time': 0
    }

    start_time = time.time()

    for timestep in range(DREAMERV3_CONFIG['total_timesteps']):
        # Collect experience
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.policy(obs)
            obs, reward, done, _ = env.step(action)
            agent.observe(obs, reward, done)
            episode_reward += reward

        # Update agent
        if timestep % 100 == 0:
            agent.train()

        # Record metrics
        metrics['episode_rewards'].append(episode_reward)

        # Check success
        if hasattr(env, 'get_success'):
            success = env.get_success()
            metrics['success_rate'].append(1.0 if success else 0.0)

    metrics['training_time'] = time.time() - start_time

    return agent.get_policy(), metrics
```

### Transfer Learning

```python
def get_initial_policy(new_task_embedding: np.ndarray,
                      archive: TaskArchive) -> Optional[str]:
    """
    Get initial policy from most similar successful task.

    Args:
        new_task_embedding: Embedding of new task
        archive: Task archive

    Returns:
        policy_path: Path to policy checkpoint, or None if no similar tasks
    """
    # Retrieve most similar successful task
    similar_tasks = archive.get_similar_tasks(
        new_task_embedding, k=1, status_filter='learned'
    )

    if len(similar_tasks) == 0:
        return None  # Train from scratch

    # Return policy from most similar task
    most_similar = similar_tasks[0]
    return most_similar.policy_path
```

---

## Complete OMNI-EPIC Algorithm

```python
def omni_epic(num_iterations: int = 200,
             max_task_attempts: int = 5):
    """
    Complete OMNI-EPIC algorithm.

    Args:
        num_iterations: Number of tasks to generate
        max_task_attempts: Max attempts to learn each task

    Returns:
        archive: Final task archive
    """
    # Initialize
    archive = initialize_archive()

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Step 1: Generate task description
        task_description = generate_task_description(archive)

        # Step 2: Generate environment code
        try:
            environment_code = generate_environment_code(task_description)
        except RuntimeError as e:
            print(f"Failed to generate environment: {e}")
            continue

        # Step 3: Generate success detector
        success_code = generate_success_code(task_description, environment_code)

        # Merge success code into environment
        environment_code += "\n\n" + success_code

        # Step 4: Post-generation MoI
        is_interesting = evaluate_interestingness(task_description, archive)

        if not is_interesting:
            print(f"Task deemed uninteresting, regenerating")
            task = Task(
                task_id=str(uuid.uuid4()),
                description=task_description,
                environment_code=environment_code,
                success_code=success_code,
                status='uninteresting',
                embedding=None
            )
            archive.add_task(task)
            continue

        # Step 5: RL Training
        # Get initial policy from similar task
        task_embedding = archive.compute_embedding(
            task_description + "\n" + environment_code
        )
        initial_policy = get_initial_policy(task_embedding, archive)

        # Train agent
        for attempt in range(max_task_attempts):
            try:
                policy, metrics = train_agent(
                    environment_code,
                    initial_policy=initial_policy
                )

                # Step 6: Evaluate success
                success_rate = np.mean(metrics['success_rate'][-10:])

                if success_rate >= 0.7:  # 70% success threshold
                    status = 'learned'
                    break
                else:
                    status = 'failed'
            except Exception as e:
                print(f"Training failed: {e}")
                status = 'failed'
                break

        # Step 7: Add to archive
        task = Task(
            task_id=str(uuid.uuid4()),
            description=task_description,
            environment_code=environment_code,
            success_code=success_code,
            status=status,
            embedding=None,
            training_metrics=metrics
        )
        archive.add_task(task)

        print(f"Task {status}: {task_description}")

    return archive
```

---

## Hyperparameters Summary

### LLM Configuration

| Component | Model | Temperature | Max Tokens |
|-----------|-------|-------------|------------|
| Task Generator | Claude 3 Opus | 0 | 200 |
| Environment Generator | Claude 3 Opus | 0 | 4096 |
| Post-Generation MoI | GPT-4o | 0 | 200 |
| Success Detector | GPT-4o | 0 | 500 |

### Retrieval Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Successful tasks retrieved | 5 | Context for task generation |
| Failed tasks retrieved | 5 | Patterns to avoid |
| Similar tasks for MoI | 10 | Interestingness comparison |
| Embedding model | text-embedding-3-small | OpenAI embedding |

### RL Training (DreamerV3)

| Parameter | Value |
|-----------|-------|
| Total timesteps | 2,000,000 |
| Batch size | 16 |
| Batch length | 64 |
| Learning rate | 3 × 10⁻⁴ |
| Discount (γ) | 0.997 |
| Replay buffer size | 1,000,000 |

### Task Management

| Parameter | Value |
|-----------|-------|
| Max code generation attempts | 5 |
| Max task learning attempts | 5 |
| Success threshold | 70% success rate |
| Episode limit | 1000 timesteps |

### Compute Resources

| Resource | Specification |
|----------|--------------|
| GPUs | 2× NVIDIA RTX 6000 Ada |
| CPU cores | 32 |
| Training time per task | ~1 hour |
| LLM API costs | ~$0.50 per task (estimated) |

---

## R2D2 Robot Specification

```python
R2D2_SPEC = {
    # Action space (discrete)
    'actions': [
        'do_nothing',      # 0
        'move_forward',    # 1
        'move_backward',   # 2
        'rotate_cw',       # 3
        'rotate_ccw',      # 4
        'jump'            # 5
    ],

    # Observation space
    'visual': {
        'type': 'RGB image',
        'shape': (64, 64, 3),
        'range': [0, 255]
    },
    'proprioceptive': {
        'joint_positions': 'robot.getJointStates()',
        'joint_velocities': 'robot.getJointStates()',
        'base_position': 'robot.getBasePositionAndOrientation()',
        'base_velocity': 'robot.getBaseVelocity()'
    },

    # Physical properties
    'urdf_file': 'r2d2.urdf',
    'mass': '50 kg',
    'height': '1.09 m',
    'max_force': 100,  # Newton
    'max_velocity': 10  # m/s
}
```

---

## Implementation Tips

### 1. Caching LLM Responses

```python
import hashlib
import pickle

def cached_llm_call(prompt: str, model: str, cache_dir: str = './cache'):
    """Cache LLM responses to avoid redundant API calls."""
    # Create hash of prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_path = f"{cache_dir}/{model}_{prompt_hash}.pkl"

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Call LLM (implementation depends on model)
    response = call_llm(prompt, model)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(response, f)

    return response
```

### 2. Parallel Task Generation

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_generate_tasks(archive: TaskArchive, num_tasks: int = 10):
    """Generate multiple tasks in parallel."""
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [
            executor.submit(generate_task_description, archive)
            for _ in range(num_tasks)
        ]
        tasks = [f.result() for f in futures]
    return tasks
```

### 3. Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_archive(archive: TaskArchive):
    """Visualize task diversity in 2D embedding space."""
    # Extract embeddings
    embeddings = np.array([task.embedding for task in archive.tasks])

    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Color by status
    colors = {
        'learned': 'green',
        'failed': 'red',
        'uninteresting': 'gray'
    }

    plt.figure(figsize=(10, 8))
    for task, (x, y) in zip(archive.tasks, embeddings_2d):
        plt.scatter(x, y, c=colors[task.status], alpha=0.6)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Task Archive Diversity')
    plt.legend(handles=[
        plt.scatter([], [], c='green', label='Learned'),
        plt.scatter([], [], c='red', label='Failed'),
        plt.scatter([], [], c='gray', label='Uninteresting')
    ])
    plt.show()
```

---

## Common Pitfalls

### 1. Code Generation Failures

**Problem**: LLM generates invalid Python code

**Solutions**:
- Increase max_attempts (currently 5)
- Provide clearer few-shot examples
- Add syntax checking before execution
- Use code linting tools (pylint) in loop

### 2. Uninteresting Task Proliferation

**Problem**: MoI filter passes redundant tasks

**Solutions**:
- Increase similarity threshold (retrieve more than 10 tasks)
- Add explicit diversity metric in MoI prompt
- Tune GPT-4o temperature (currently 0)

### 3. Training Failures

**Problem**: Agent fails to learn even simple tasks

**Solutions**:
- Check reward function (should be dense, not sparse)
- Increase training timesteps (currently 2M)
- Verify success detector aligns with reward
- Transfer from more similar initial policies

### 4. LLM API Costs

**Problem**: Expensive at scale ($0.50/task × 200 tasks = $100/run)

**Solutions**:
- Cache responses aggressively
- Use smaller models for non-critical components
- Batch API calls where possible
- Consider self-hosted models (Llama, Mixtral)

---

## Extensions

### 1. Multi-Agent Tasks

```python
# Modify task generator prompt
TASK_GENERATOR_PROMPT += """
You can generate multi-agent collaborative or competitive tasks.

Examples:
- "Two robots cooperate to push a heavy box"
- "Two robots race to reach a goal first"
"""
```

### 2. Hierarchical Task Decomposition

```python
def decompose_task(complex_task: str) -> List[str]:
    """
    Decompose complex task into subtasks using LLM.
    """
    prompt = f"""
    Decompose this complex task into 3-5 simpler subtasks:

    Task: {complex_task}

    Subtasks:
    """
    # Call LLM to get subtasks
    subtasks = call_llm(prompt, "gpt-4o")
    return subtasks
```

### 3. Human-in-the-Loop Feedback

```python
def get_human_feedback(task: Task) -> str:
    """
    Request human feedback on generated task.
    """
    print(f"Task: {task.description}")
    print("Is this task interesting? (yes/no/modify)")
    response = input("> ")

    if response == "modify":
        print("Suggested modification:")
        modification = input("> ")
        return modification

    return response
```

---

## Evaluation Metrics

### ANNECS-OMNI Implementation

```python
def compute_annecs_omni(archive: TaskArchive,
                       novelty_threshold: float = 0.1) -> int:
    """
    Compute ANNECS-OMNI: Accumulated Number of Novel, Learnable,
    and Interesting Environments Created and Solved.

    Args:
        archive: Task archive
        novelty_threshold: Minimum distance for novelty

    Returns:
        annecs_omni: Metric value
    """
    annecs_omni = 0
    seen_embeddings = []

    for task in archive.tasks:
        # Must be learned
        if task.status != 'learned':
            continue

        # Must be novel
        is_novel = True
        for seen_emb in seen_embeddings:
            distance = np.linalg.norm(task.embedding - seen_emb)
            if distance < novelty_threshold:
                is_novel = False
                break

        # Must be interesting (passed MoI filter)
        # (Implicitly true if in archive and not marked 'uninteresting')

        if is_novel:
            annecs_omni += 1
            seen_embeddings.append(task.embedding)

    return annecs_omni
```

### Cell Coverage

```python
def compute_cell_coverage(archive: TaskArchive,
                         num_bins: int = 50) -> float:
    """
    Compute cell coverage in 2D embedding space.

    Args:
        archive: Task archive
        num_bins: Discretization resolution

    Returns:
        coverage: Fraction of cells occupied
    """
    # PCA to 2D
    embeddings = np.array([task.embedding for task in archive.tasks])
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Discretize
    hist, _, _ = np.histogram2d(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        bins=num_bins
    )

    # Count occupied cells
    occupied_cells = np.sum(hist > 0)
    total_cells = num_bins ** 2

    coverage = occupied_cells / total_cells
    return coverage
```

---

## Summary

OMNI-EPIC's implementation combines:
1. **LLM-based task generation** with retrieval-augmented context
2. **Code synthesis** with iterative error correction
3. **Dual-stage MoI** for interestingness filtering
4. **Transfer learning** from similar tasks
5. **LLM-generated success detection** for universal task completion evaluation

**Key Takeaway**: The system's power comes from foundation models' ability to generate executable code and evaluate interestingness without explicit mathematical definitions.

**Practical Recommendation**: Start with small-scale experiments (10-20 tasks) to validate pipeline, then scale to full runs (200+ tasks). Cache aggressively to manage API costs.

---

**See [[OMNI-EPIC]] for high-level overview and motivation.**
