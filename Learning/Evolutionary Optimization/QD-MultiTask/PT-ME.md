# Parametric-Task MAP-Elites (PT-ME)

## Overview

**Parametric-Task MAP-Elites (PT-ME)** is a black-box algorithm for **continuous multi-task optimization** (parametric-task optimization). It solves a new task at each iteration, asymptotically covering the entire continuous task space, and can distill results into a neural network that maps any task parameter to its optimal solution.

**Authors**: Timothée Anne, Jean-Baptiste Mouret (Université de Lorraine, CNRS, Inria)
**Paper**: "Parametric-Task MAP-Elites"
**Venue**: Preprint, 2024
**ArXiv**: 2402.01275

---

## Core Problem: Parametric-Task Optimization

### Problem Formulation

Find a function $G: \Theta \to \mathcal{X}$ that maps any task parameter to its optimal solution:

$$\forall \theta \in \Theta, \quad G(\theta) = x^*_{\theta} = \underset{x \in \mathcal{X}}{\operatorname{argmax}} f(x, \theta)$$

where:
- $\mathcal{X} = [0,1]^{d_x}$ — Solution space (dimension $d_x$)
- $\Theta = [0,1]^{d_{\theta}}$ — **Continuous** task parameter space (dimension $d_{\theta}$)
- $f: \mathcal{X} \times \Theta \to \mathbb{R}$ — Fitness function (black-box)
- $G: \Theta \to \mathcal{X}$ — Mapping from task to optimal solution

**Key difference from multi-task optimization**: Task space $\Theta$ is **continuous**, not a finite set.

### Motivating Examples

**Robotics**: Walking gaits for various morphologies
- **Tasks**: Different leg lengths, joint bounds (continuous morphology space)
- **Solutions**: Controller parameters
- **Problem**: Discretization is arbitrary (1000 tasks = only 10 steps/dimension in 3D!)

**Machine Learning**: Hyperparameter tuning
- **Tasks**: Different datasets (continuous feature distributions)
- **Solutions**: Hyperparameter settings
- **Problem**: Need solution for any dataset, not just sampled ones

**Industrial Design**: Ergonomic workstations
- **Tasks**: Different worker morphologies (continuous height, arm length, etc.)
- **Solutions**: Workstation configurations
- **Problem**: Cover full human morphology range

---

## Key Innovation

### Limitation of Existing Approaches

**Multi-Task MAP-Elites (MT-ME)**:
- Discretizes continuous task space into finite set (e.g., 2000 tasks)
- Only evaluates on cell centroids
- **Problems**:
  1. Can only solve finite tasks (even if original space continuous)
  2. Discretization doesn't scale with dimensions
  3. Selection pressure ∝ 1/n_cells (more tasks → slower convergence)

**PT-ME Solution**:
1. ✅ **Samples new task at each iteration** (continuous coverage)
2. ✅ **De-correlates selection pressure from #tasks** (efficiency doesn't degrade)
3. ✅ **New variation operator** (local linear regression exploits task structure)
4. ✅ **Distillation** (neural network for any task)

---

## Algorithm

### Core Components

**1. Archive Structure** (like MT-ME, but different usage):
- Discretize $\Theta$ into $N$ cells using CVT (e.g., $N=200$)
- Each cell has centroid $\theta_c$ and stores elite $(x, f)$ for best solution in that region
- **Key difference**: Evaluate on **sampled tasks** $\theta$, not just centroids $\theta_c$

**2. Variation Operators** (50/50 split):

**Operator 1: SBX with Tournament** (50% of iterations)
- Select 2 parents from archive
- Sample $s$ candidate tasks, pick closest to parent 1's task (tournament)
- Apply SBX crossover + mutation
- **Bandit** (UCB1) adapts tournament size $s \in \{1, 5, 10, 50, 100, 500\}$

**Operator 2: Local Linear Regression** (50% of iterations) ✨
- Sample task $\theta$ uniformly from $\Theta$
- Find adjacent cells using Delaunay triangulation
- Extract task parameters $\boldsymbol{\theta}$ and solutions $\boldsymbol{x}$ from adjacent cells
- Perform linear least squares: $M = (\boldsymbol{\theta}^T \boldsymbol{\theta})^{-1} \boldsymbol{\theta}^T \boldsymbol{x}$
- Generate candidate: $x = M \cdot \theta + \sigma_{reg} \cdot \mathcal{N}(0, \text{var}(\boldsymbol{x}))$

**3. Evaluation & Update**:
- Evaluate candidate $x$ on sampled task $\theta$: $f = \text{fitness}(x, \theta)$
- Assign to nearest cell centroid $\theta_c$
- Update if $f \geq A[\theta_c].f$
- **Store all evaluations** for distillation

### Complete Algorithm (Implementable)

```python
import numpy as np
from scipy.spatial import KDTree, Delaunay, distance
from scipy.spatial import cKDTree as fast_cvt  # For CVT initialization

# ============================================================================
# INITIALIZATION
# ============================================================================

budget = 100_000  # Total evaluations
N_cells = 200  # Number of archive cells
d_x = 10  # Solution space dimension
d_theta = 2  # Task parameter space dimension

# CVT: Create N centroids in task space [0,1]^d_theta
# Simple CVT: k-means on random samples
samples = np.random.uniform(0, 1, (10000, d_theta))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=N_cells, random_state=42).fit(samples)
centroids = kmeans.cluster_centers_  # Shape: (N_cells, d_theta)

# Create spatial data structures
kdtree = KDTree(centroids)
delaunay = Delaunay(centroids)

# Initialize archive
# A[i] = {'theta': task_param, 'x': solution, 'f': fitness}
A = []
for i, theta_c in enumerate(centroids):
    x = np.random.uniform(0, 1, d_x)
    f = fitness_function(x, theta_c)
    A.append({
        'theta': theta_c,
        'x': x,
        'f': f,
        'adj': set(delaunay.neighbors[i][delaunay.neighbors[i] >= 0])
    })

# Store ALL evaluations for distillation
E = [(A[i]['theta'].copy(), A[i]['x'].copy(), A[i]['f'])
     for i in range(N_cells)]

# Tournament size bandit
S = [1, 5, 10, 50, 100, 500]
selected = np.zeros(len(S))
successes = np.zeros(len(S))
s_idx = 0  # Current tournament size index

# ============================================================================
# MAIN LOOP
# ============================================================================

sigma_SBX = 10.0  # SBX mutation factor
sigma_reg = 1.0  # Regression noise factor

for iteration in range(budget - N_cells):

    # ========================================================================
    # VARIATION OPERATOR SELECTION (50/50)
    # ========================================================================

    if np.random.random() < 0.5:
        # ====================================================================
        # OPERATOR 1: SBX WITH TOURNAMENT
        # ====================================================================

        # 1. Select 2 random parents from archive
        p1_idx = np.random.randint(N_cells)
        p2_idx = np.random.randint(N_cells)
        p1 = A[p1_idx]
        p2 = A[p2_idx]

        # 2. Sample s candidate tasks uniformly from [0,1]^d_theta
        s = S[s_idx]
        theta_candidates = np.random.uniform(0, 1, (s, d_theta))

        # 3. Tournament: pick task closest to p1's task
        distances = distance.cdist([p1['theta']], theta_candidates)[0]
        theta = theta_candidates[np.argmin(distances)]

        # 4. SBX crossover + polynomial mutation
        x = simulated_binary_crossover(p1['x'], p2['x'],
                                       eta_c=20.0)  # Crossover
        x = polynomial_mutation(x, eta_m=sigma_SBX)  # Mutation

        # Track bandit
        selected[s_idx] += 1
        used_tournament = True

    else:
        # ====================================================================
        # OPERATOR 2: LOCAL LINEAR REGRESSION
        # ====================================================================

        # 1. Sample task uniformly from [0,1]^d_theta
        theta = np.random.uniform(0, 1, d_theta)

        # 2. Find nearest centroid
        cell_idx = kdtree.query(theta)[1]

        # 3. Get adjacent cells
        adj_indices = list(A[cell_idx]['adj'])
        if len(adj_indices) == 0:
            # Fallback: use current cell only
            adj_indices = [cell_idx]

        # 4. Extract task parameters and solutions from adjacent cells
        Theta_adj = np.array([A[i]['theta'] for i in adj_indices])
        X_adj = np.array([A[i]['x'] for i in adj_indices])

        # 5. Linear least squares: M = (Θ^T Θ)^{-1} Θ^T X
        M = np.linalg.lstsq(Theta_adj, X_adj, rcond=None)[0]
        # M shape: (d_theta, d_x)

        # 6. Predict + noise: x = M · θ + σ_reg · N(0, var(X))
        x_pred = M.T @ theta  # Shape: (d_x,)
        noise_std = sigma_reg * np.std(X_adj, axis=0)
        x = x_pred + np.random.normal(0, noise_std)

        # Clip to [0, 1]
        x = np.clip(x, 0, 1)

        used_tournament = False

    # ========================================================================
    # EVALUATION
    # ========================================================================

    # Find cell for this task
    cell_idx = kdtree.query(theta)[1]

    # Evaluate
    f = fitness_function(x, theta)

    # Store evaluation
    E.append((theta.copy(), x.copy(), f))

    # ========================================================================
    # ARCHIVE UPDATE
    # ========================================================================

    if f >= A[cell_idx]['f']:
        A[cell_idx]['theta'] = theta
        A[cell_idx]['x'] = x
        A[cell_idx]['f'] = f

        # Update bandit if tournament was used
        if used_tournament:
            successes[s_idx] += 1

    # ========================================================================
    # BANDIT UPDATE (UCB1)
    # ========================================================================

    if used_tournament:
        # UCB1 formula
        ucb_scores = (successes / (selected + 1e-10) +
                     np.sqrt(2 * np.log(selected.sum() + 1) /
                            (selected + 1e-10)))
        s_idx = np.argmax(ucb_scores)

# ============================================================================
# DISTILLATION
# ============================================================================

G_NN = distill_to_neural_network(E, d_theta, d_x)

return G_NN, E, A
```

### Detailed Operators

**1. Simulated Binary Crossover (SBX)**:
```python
def simulated_binary_crossover(parent1, parent2, eta_c=20.0):
    """
    SBX crossover operator.

    Args:
        parent1, parent2: Parent solutions (arrays)
        eta_c: Distribution index (higher = more similar to parents)

    Returns:
        Offspring solution
    """
    offspring = np.zeros_like(parent1)

    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            # Perform crossover
            u = np.random.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

            offspring[i] = 0.5 * ((1.0 + beta) * parent1[i] +
                                  (1.0 - beta) * parent2[i])
        else:
            # Don't crossover, use parent 1
            offspring[i] = parent1[i]

    return np.clip(offspring, 0, 1)


def polynomial_mutation(solution, eta_m=20.0, mutation_prob=0.1):
    """
    Polynomial mutation operator.

    Args:
        solution: Solution to mutate
        eta_m: Distribution index (higher = smaller mutations)
        mutation_prob: Probability per dimension

    Returns:
        Mutated solution
    """
    mutated = solution.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_prob:
            u = np.random.random()
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

            mutated[i] += delta
            mutated[i] = np.clip(mutated[i], 0, 1)

    return mutated
```

**2. Linear Least Squares**:
```python
def linear_least_squares(Theta, X):
    """
    Solve M such that X ≈ Θ · M via least squares.

    Args:
        Theta: (n_samples, d_theta) - Task parameters
        X: (n_samples, d_x) - Solutions

    Returns:
        M: (d_theta, d_x) - Linear mapping matrix
    """
    # Solution: M = (Θ^T Θ)^{-1} Θ^T X
    M, residuals, rank, s = np.linalg.lstsq(Theta, X, rcond=None)
    return M  # Shape: (d_theta, d_x)
```

---

## Local Linear Regression (New Operator)

### Motivation

**SBX doesn't exploit multi-task structure**: It doesn't use the known task $\theta$ when generating offspring.

**Idea**: Build local model $G: \theta \mapsto x^*_{\theta}$ from archive, use it to "guess" solution for new task.

### Why Linear Model?

**Advantages**:
- Simple and fast (closed-form solution)
- Exploits local smoothness in $G$
- Adapts to archive content (uses adjacent cells)

**Why not global model?**
- Function $G$ may be complex globally but locally smooth
- Linear model sufficient for small neighborhoods

### Why Adjacency (Delaunay)?

**Alternatives**:
- k-nearest neighbors: Need to tune $k$ (dimension-dependent)
- ε-ball: Need to tune $\varepsilon$ (scale-dependent)

**Delaunay triangulation**:
- ✅ Automatically adapts number of neighbors to dimension
- ✅ No hyperparameter to tune
- ✅ Precomputed once (fast lookup)

### Noise Coefficient $\sigma_{reg}$

**Role**: Balance exploration vs exploitation
- High $\sigma_{reg}$: More exploration (noisier predictions)
- Low $\sigma_{reg}$: More exploitation (trust regression)

**Default**: $\sigma_{reg} = 1.0$ (robust across problems, though tuning could improve)

---

## Distillation: From Discrete Archive to Continuous Function

### Problem

PT-ME stores all evaluations $E = \{(\theta_i, x_i, f_i)\}$, but:
- Need solution for **any** task $\theta \in \Theta$
- Archive has finite resolution

### Complete Distillation Procedure

**Step 1: Re-archive at desired resolution**:

```python
def re_archive(E, n_cells):
    """
    Create archive at resolution n_cells from evaluation history.

    Args:
        E: List of (theta, x, f) tuples (all evaluations)
        n_cells: Desired archive resolution

    Returns:
        Dataset of (theta, x) pairs for neural network training
    """
    # 1. Create CVT centroids at new resolution
    samples = np.random.uniform(0, 1, (10000, d_theta))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_cells, random_state=42).fit(samples)
    centroids = kmeans.cluster_centers_

    # 2. Create kdtree for fast nearest-neighbor lookup
    kdtree = KDTree(centroids)

    # 3. Initialize empty archive
    archive = {}
    for i, theta_c in enumerate(centroids):
        archive[i] = {
            'theta': theta_c,
            'x': None,
            'f': 0.0  # Assume fitness >= 0 (normalized)
        }

    # 4. Fill archive with best solutions per cell
    for (theta, x, f) in E:
        # Find nearest centroid
        cell_idx = kdtree.query(theta)[1]

        # Update if better fitness
        if f >= archive[cell_idx]['f']:
            archive[cell_idx]['theta'] = theta
            archive[cell_idx]['x'] = x
            archive[cell_idx]['f'] = f

    # 5. Extract training data (only cells with solutions)
    training_data = []
    for cell_idx in range(n_cells):
        if archive[cell_idx]['x'] is not None:
            training_data.append((
                archive[cell_idx]['theta'],
                archive[cell_idx]['x']
            ))

    return training_data


def distill_to_neural_network(E, d_theta, d_x,
                               n_cells=10000,  # High resolution
                               n_epochs=1000,
                               batch_size=256,
                               learning_rate=0.001):
    """
    Train neural network to map task parameter to solution.

    Args:
        E: Evaluation history
        d_theta: Task parameter dimension
        d_x: Solution dimension
        n_cells: Re-archive resolution (more = better coverage)
        n_epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate

    Returns:
        Trained neural network G: Θ → X
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # ========================================================================
    # STEP 1: RE-ARCHIVE AT HIGH RESOLUTION
    # ========================================================================

    training_data = re_archive(E, n_cells)

    # Extract inputs and targets
    inputs = np.array([theta for (theta, x) in training_data])
    targets = np.array([x for (theta, x) in training_data])

    print(f"Training dataset: {len(training_data)} samples")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(inputs)
    Y_train = torch.FloatTensor(targets)

    # ========================================================================
    # STEP 2: CREATE NEURAL NETWORK ARCHITECTURE
    # ========================================================================

    class PolicyNetwork(nn.Module):
        """
        2-layer MLP (same architecture as PPO's stable-baselines3).
        """
        def __init__(self, input_dim, output_dim, hidden_dim=64):
            super(PolicyNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()  # Output in [0, 1]
            )

        def forward(self, theta):
            return self.network(theta)

    model = PolicyNetwork(d_theta, d_x)

    # ========================================================================
    # STEP 3: TRAIN WITH MSE LOSS
    # ========================================================================

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            # Forward pass
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # ========================================================================
    # STEP 4: RETURN TRAINED MODEL
    # ========================================================================

    model.eval()
    return model


# ============================================================================
# USAGE: QUERY TRAINED MODEL FOR ANY TASK
# ============================================================================

def query_solution(model, theta):
    """
    Get solution for any task parameter.

    Args:
        model: Trained neural network
        theta: Task parameter (numpy array or list)

    Returns:
        Optimal solution x* for task theta
    """
    import torch

    with torch.no_grad():
        theta_tensor = torch.FloatTensor(theta).unsqueeze(0)
        x_pred = model(theta_tensor).squeeze(0).numpy()

    return x_pred


# Example usage:
# G_NN = distill_to_neural_network(E, d_theta=2, d_x=10)
# x_star = query_solution(G_NN, theta=[0.5, 0.3])
```

### Re-archiving at Multiple Resolutions

**Purpose**: Evaluate generalization across different discretizations

```python
def evaluate_multi_resolution_QD_score(model, E, resolutions):
    """
    Compute QD-Score at multiple resolutions.

    Args:
        model: Trained neural network
        E: Evaluation history
        resolutions: List of cell counts (e.g., [100, 500, 1000, 5000])

    Returns:
        QD-Scores at each resolution
    """
    qd_scores = []

    for n_cells in resolutions:
        # Re-archive at this resolution
        training_data = re_archive(E, n_cells)

        # Compute QD-Score using neural network predictions
        score = 0.0
        for (theta, x_true) in training_data:
            x_pred = query_solution(model, theta)
            f_pred = fitness_function(x_pred, theta)
            score += f_pred

        qd_scores.append(score)
        print(f"Resolution {n_cells}: QD-Score = {score:.2f}")

    return qd_scores
```

### Why Re-archive Before Training?

**Direct approach** (train on all evaluations):
- Many evaluations in same region
- Imbalanced training data
- Overfits to frequently-sampled regions

**Re-archive approach** (PT-ME's method):
- 1 elite per cell (balanced)
- Even coverage of task space
- Better generalization

**Trade-off**: Resolution vs. sample efficiency
- Higher resolution → more training data → better NN
- But: need enough evaluations to fill cells
- Typical: 10,000 cells for 100,000 evaluations

---

## Experimental Validation

### Domains

**1. 10-DoF Arm** (toy problem)
- **Task**: Target end-effector position $(x, y) \in [0, 1]^2$
- **Solution**: 10 joint angles
- **Fitness**: Negative distance to target
- **Used in**: MT-ME, many prior works

**2. Archery** (challenging toy problem)
- **Task**: Target position and wind
- **Solution**: Arrow launch parameters
- **Fitness**: Hit target (sparse reward)
- **Challenge**: Requires precise trajectories

**3. Door-Pulling** (realistic robotics)
- **Task**: Door position, handle height, door stiffness
- **Solution**: Humanoid robot arm trajectory
- **Fitness**: Door opening angle
- **Complexity**: Whole-body control, contact physics

### Baselines

1. **Random Search**: Uniform sampling from $\mathcal{X}$
2. **CMA-ES**: Run independently on each task (discrete set)
3. **MT-ME**: Multi-Task MAP-Elites (finite discretization)
4. **PPO**: Proximal Policy Optimization (deep RL, 1-step horizon)

### Results Summary

**10-DoF Arm** (100k evaluations):
- **PT-ME**: Best on all resolutions
- **MT-ME**: Struggles at high resolution (selection pressure drops)
- **PPO**: Worse than PT-ME, similar to MT-ME
- **Random/CMA-ES**: Far behind

**Archery** (sparse rewards):
- **PT-ME**: Only method that reliably solves tasks
- **Others**: Struggle with sparse rewards

**Door-Pulling** (realistic):
- **PT-ME**: Best performance
- **Demonstrates**: Works on high-dimensional, complex domains

### Metrics

**QD-Score** (at resolution $n$):
$$\text{QD-Score}_n = \sum_{\theta_c \in C_n} f(\hat{x}_{\theta_c}, \theta_c)$$

where $\hat{x}_{\theta_c}$ is distilled NN prediction for cell centroid.

**Multi-Resolution QD-Score**:
Average QD-Score over multiple resolutions (tests generalization).

---

## Ablation Studies

### Impact of Linear Regression Operator

**Comparison**: PT-ME vs PT-ME without linear regression (only SBX)

**Result**: Linear regression provides **significant improvement** (~2× better)

**Interpretation**: Exploiting task structure via local model >> random crossover

### Impact of Sampling Strategy

**Comparison**: Sample new task each iteration vs only evaluate on centroids

**Result**: Continuous sampling >> fixed centroids

**Interpretation**: More budget → more tasks covered, better generalization

---

## Comparison to Related Approaches

### vs Multi-Task MAP-Elites (MT-ME)

| Aspect | MT-ME | PT-ME |
|--------|-------|-------|
| **Task sampling** | Fixed centroids only | New task each iteration |
| **Coverage** | Finite (2000 tasks) | Asymptotically complete |
| **Selection pressure** | ∝ 1/n_cells | Independent of n_cells |
| **Variation** | SBX + tournament | SBX + **linear regression** |
| **Output** | Discrete archive | Archive + **continuous NN** |

### vs Parametric Programming

| Aspect | Parametric Programming | PT-ME |
|--------|----------------------|-------|
| **Function** | Known, differentiable | Black-box |
| **Method** | Critical regions + KKT | Evolutionary + regression |
| **Guarantees** | Exact (if assumptions hold) | Approximate |
| **Applicability** | Linear/Quadratic/Simple NLP | Any black-box function |

### vs Deep Reinforcement Learning (PPO)

| Aspect | PPO (1-step horizon) | PT-ME |
|--------|---------------------|-------|
| **Problem type** | Sequential decision | Direct optimization |
| **Credit assignment** | Not needed (1-step) | Not needed |
| **Sample efficiency** | Lower (neural network overhead) | Higher (evolutionary) |
| **Performance** | Comparable | Better (on tested problems) |

**Note**: PPO with 1-step horizon effectively solves parametric-task optimization, but PT-ME outperforms it.

---

## Hyperparameters

**Robust defaults** (work well across problems):
- $B = 100{,}000$ — Evaluation budget
- $N = 200$ — Number of cells
- $S = \{1, 5, 10, 50, 100, 500\}$ — Tournament sizes for bandit
- $\sigma_{SBX} = 10.0$ — SBX mutation factor
- $\sigma_{reg} = 1.0$ — Linear regression noise factor

**Notes**:
- $N=200$ chosen empirically (robust across dimensions)
- Bandit avoids tuning tournament size manually
- $\sigma_{reg}$ could benefit from per-problem tuning but default works reasonably

---

## When to Use PT-ME

**✅ Use PT-ME when**:
- Task space is **continuous** (not just a few discrete tasks)
- Need solution for **any** task parameter (not just sampled ones)
- Black-box fitness function (no gradients available)
- Want to **distill** knowledge into continuous function (NN)
- Tasks have smooth structure (local similarity)

**❌ Consider alternatives when**:
- Only have finite set of tasks → MT-ME
- Function is differentiable → Parametric programming
- Need sequential decision-making → Deep RL
- Tasks completely unrelated → Independent optimization

---

## Key Insights

1. **Continuous coverage**: Sampling new task each iteration >> discretization
2. **Local linear regression**: Exploiting local task structure >> blind crossover
3. **De-correlated pressure**: Efficiency independent of task count >> scalability
4. **Distillation**: Neural network enables querying any task
5. **Black-box power**: Works where parametric programming can't (non-differentiable, complex)

---

## Limitations & Future Work

**Limitations**:
1. Linear regression assumes local smoothness (may not hold for all problems)
2. Delaunay triangulation cost grows with dimension (but still feasible)
3. Neural network distillation quality depends on dataset density
4. Number of cells $N$ impacts both selection pressure and regression quality

**Future directions** (from authors):
- Adaptive $\sigma_{reg}$ (like bandit for tournament size)
- Higher-order regression (quadratic, GP) for complex local structure
- Curriculum sampling (focus on hard-to-solve regions)
- Multi-fidelity evaluation (cheap approximations for exploration)

---

## Implementation Notes

### KDTree for Fast Lookup

```python
from scipy.spatial import KDTree

kdtree = KDTree(centroids)
θ_c = kdtree.query(θ)[1]  # Index of nearest centroid
```

### Delaunay Triangulation for Adjacency

```python
from scipy.spatial import Delaunay

delaunay = Delaunay(centroids)
adjacency = delaunay.neighbors  # adjacency[i] = neighbors of cell i
```

### Linear Least Squares

```python
import numpy as np

M = np.linalg.lstsq(Θ_adj, X_adj, rcond=None)[0]
x = M @ θ + σ_reg * np.random.normal(0, np.var(X_adj, axis=0))
```

### UCB1 Bandit

```python
def select_tournament_size(selected, successes, S):
    ucb_scores = successes / selected + np.sqrt(2 * np.log(selected.sum()) / selected)
    return S[np.argmax(ucb_scores)]
```

---

## Comparison Summary Table

| Algorithm | Tasks | Continuous | Variation | Output | Black-Box |
|-----------|-------|------------|-----------|--------|-----------|
| **MAP-Elites** | 1 | N/A | Gaussian | Archive | ✅ |
| **MT-ME** | Finite | ❌ | SBX + tournament | Archive | ✅ |
| **MTMB-ME** | Finite | ❌ | Crossover between tasks | Archive × tasks | ✅ |
| **PT-ME** | ∞ | ✅ | SBX + **linear regression** | Archive + **NN** | ✅ |
| **Parametric Programming** | ∞ | ✅ | Critical regions | Exact function | ❌ |
| **PPO** | ∞ | ✅ | Policy gradient | NN policy | ✅ |

---

## Worked Example: Linear Regression Operator

### Setup
- Task space: Θ = [0,1]² (2D)
- Solution space: X = [0,1]³ (3D)
- 3 archive cells with Delaunay triangulation

**Archive state**:
```python
# Cell 0: θ₀ = [0.2, 0.3], x₀ = [0.5, 0.6, 0.7]
# Cell 1: θ₁ = [0.5, 0.4], x₁ = [0.7, 0.8, 0.9]
# Cell 2: θ₂ = [0.3, 0.6], x₂ = [0.6, 0.7, 0.8]

# Delaunay adjacency:
# Cell 0 adjacent to: {1, 2}
# Cell 1 adjacent to: {0, 2}
# Cell 2 adjacent to: {0, 1}
```

### Iteration Using Linear Regression

**Step 1**: Sample new task
```python
θ_new = [0.35, 0.45]  # Sampled uniformly
```

**Step 2**: Find nearest cell
```python
# Distances: ||θ_new - θ₀|| = 0.21, ||θ_new - θ₁|| = 0.15, ||θ_new - θ₂|| = 0.15
# Nearest: Cell 1 (tie-breaking)
cell_idx = 1
```

**Step 3**: Get adjacent cells
```python
adj_cells = {0, 2}  # From Delaunay
```

**Step 4**: Extract data
```python
Θ_adj = [[0.2, 0.3],   # From cell 0
         [0.3, 0.6]]   # From cell 2

X_adj = [[0.5, 0.6, 0.7],   # From cell 0
         [0.6, 0.7, 0.8]]   # From cell 2
```

**Step 5**: Linear least squares
```python
# Solve: X_adj ≈ Θ_adj · M
# M = (Θ_adj^T Θ_adj)^{-1} Θ_adj^T X_adj

Θ^T Θ = [[0.13, 0.24],
         [0.24, 0.45]]

(Θ^T Θ)^{-1} = [[8.66, -4.62],
                 [-4.62, 2.50]]

Θ^T X = [[0.28, 0.33, 0.38],
         [0.51, 0.60, 0.69]]

M = [[0.33, 0.33, 0.33],
     [0.50, 0.67, 0.83]]  # (2×3 matrix)
```

**Step 6**: Predict solution
```python
x_pred = M^T · θ_new
       = [[0.33, 0.50],     ^T    [[0.35],
          [0.33, 0.67],      ·     [0.45]]
          [0.33, 0.83]]

       = [0.34, 0.42, 0.49]
```

**Step 7**: Add noise
```python
var(X_adj) = [0.005, 0.005, 0.005]
noise = σ_reg * N(0, [0.005, 0.005, 0.005])
      = 1.0 * [0.01, -0.02, 0.03]  # Example sample

x_final = [0.35, 0.40, 0.52]  # Clipped to [0,1]
```

**Step 8**: Evaluate
```python
f = fitness(x_final, θ_new) = 7.5  # Example

# Compare to cell 1: f > 6.0 → update!
archive[1] = (θ_new, x_final, 7.5)
```

**Key insight**: Linear regression interpolates between nearby solutions, then adds exploration noise!

## Key Takeaways

1. **First black-box algorithm** for continuous multi-task optimization (parametric-task)
2. **Local linear regression** exploits task structure more effectively than blind crossover
3. **Asymptotic coverage** via sampling new task each iteration
4. **Distillation** enables continuous function approximation from discrete evaluations
5. **Outperforms** PPO, MT-ME, CMA-ES on tested problems
6. **Practical**: Robust hyperparameters, scalable to realistic robotics problems
7. **Delaunay triangulation**: Automatic neighbor selection, no hyperparameter tuning
8. **50/50 split**: SBX for global exploration, regression for local exploitation

---

## Related

- [[PT-ME_detailed]] — Implementation details (to be created)
- [[MAP-Elites]] — Base single-task QD algorithm
- [[Multi-Task_MAP-Elites]] — Finite multi-task variant
- [[MTMB-ME]] — Multi-task multi-behavior variant

## References

- **Paper**: Anne & Mouret, "Parametric-Task MAP-Elites", arXiv:2402.01275, 2024
- **Code**: https://zenodo.org/doi/10.5281/zenodo.10926438
- **Related**: MT-ME (Pierrot et al., 2022), MAP-Elites (Mouret & Clune, 2015)
- **Comparison**: Parametric programming (Pistikopoulos et al., 2007), PPO (Schulman et al., 2017)
