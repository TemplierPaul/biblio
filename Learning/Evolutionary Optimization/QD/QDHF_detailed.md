# QDHF — Detailed Implementation Notes

> **Quick overview**: [[QDHF]]

## Paper

**Title**: Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization

**Authors**: Li Ding, Jenny Zhang, Jeff Clune, Lee Spector, Joel Lehman

**Year**: 2024 (ICML)

**Code**: https://liding.info/qdhf

## Core Algorithms

### QDHF (Progressive Online Learning)

```python
class QDHF:
    def __init__(self, feature_extractor, latent_dim, qd_config):
        self.f = feature_extractor  # f: X → Y (feature space)
        self.D_r = LinearProjection(input_dim, latent_dim)  # D_r: Y → Z
        self.qd = MAPElites(qd_config)
        self.feedback_data = []

    def run(self, total_iterations):
        # Update schedule: iterations 1, 10%, 25%, 50%
        update_iters = [1, int(0.1*total_iterations),
                       int(0.25*total_iterations),
                       int(0.5*total_iterations)]

        # Initial diversity learning from random solutions
        random_solutions = generate_random_solutions(n_init)
        self.collect_feedback(random_solutions)
        self.train_projection()

        for i in range(total_iterations):
            # QD optimization with current diversity metrics
            self.qd.iterate(diversity_fn=self.get_diversity_metrics)

            # Progressive fine-tuning
            if i in update_iters:
                new_solutions = self.qd.archive.sample()
                self.collect_feedback(new_solutions)
                self.train_projection()  # Fine-tune, not reset
                # Re-initialize QD with updated metrics but keep archive
                self.qd.reset_emitter(keep_archive=True)

        return self.qd.archive

    def get_diversity_metrics(self, x):
        """Compute diversity metrics from solution"""
        y = self.f(x)  # Extract features
        z = self.D_r(y)  # Project to latent space
        return z  # Each dimension is a diversity metric

    def collect_feedback(self, solutions):
        """Collect 2AFC human judgments on solution triplets"""
        triplets = sample_triplets(solutions, n_triplets)
        for (x1, x2, x3) in triplets:
            # Query: Is x2 or x3 more similar to x1?
            label = human_2afc_query(x1, x2, x3)
            self.feedback_data.append((x1, x2, x3, label))

    def train_projection(self):
        """Train/fine-tune latent projection via contrastive learning"""
        optimizer = Adam(self.D_r.parameters(), lr=1e-4)

        for epoch in range(n_epochs):
            for (x1, x2, x3, label) in self.feedback_data:
                # Extract features
                y1, y2, y3 = self.f(x1), self.f(x2), self.f(x3)

                # Project to latent space
                z1, z2, z3 = self.D_r(y1), self.D_r(y2), self.D_r(y3)

                # Triplet loss based on 2AFC label
                if label == "x2_more_similar":
                    loss = triplet_loss(z1, z2, z3)  # Pull z1-z2, push z1-z3
                else:
                    loss = triplet_loss(z1, z3, z2)  # Pull z1-z3, push z1-z2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

def triplet_loss(z_anchor, z_positive, z_negative, margin=1.0):
    """Contrastive triplet loss"""
    d_pos = torch.norm(z_anchor - z_positive, p=2)
    d_neg = torch.norm(z_anchor - z_negative, p=2)
    return torch.relu(margin + d_pos - d_neg)
```

### QDHF-Base (Offline Baseline)

```python
class QDHFBase:
    def __init__(self, feature_extractor, latent_dim, qd_config):
        self.f = feature_extractor
        self.D_r = LinearProjection(input_dim, latent_dim)
        self.qd = MAPElites(qd_config)

    def run(self, total_iterations):
        # Phase 1: Offline diversity learning (like RLHF reward model)
        random_solutions = generate_random_solutions(n_init)
        feedback_data = self.collect_feedback(random_solutions)
        self.train_projection(feedback_data)

        # Phase 2: QD optimization with fixed diversity metrics
        for i in range(total_iterations):
            self.qd.iterate(diversity_fn=self.get_diversity_metrics)

        return self.qd.archive

    # get_diversity_metrics, collect_feedback, train_projection same as QDHF
```

### Latent Projection Module

```python
class LinearProjection(nn.Module):
    """Linear projection from feature space to latent space"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, latent_dim, bias=False)
        # Can use multi-layer for nonlinear projection:
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, latent_dim)
        # )

    def forward(self, y):
        """y: feature vector, returns: z latent vector"""
        return self.W(y)  # Or self.layers(y) for nonlinear
```

## Human Feedback Collection

### Two Alternative Forced Choice (2AFC)

```python
def human_2afc_query(x1, x2, x3):
    """
    Present triplet to human evaluator

    Query: "Given reference solution x1, which is more similar: x2 or x3?"

    Returns: "x2_more_similar" or "x3_more_similar"
    """
    # Display solutions (domain-specific visualization)
    display_reference(x1)
    display_options(x2, x3)

    # Collect binary choice
    response = wait_for_user_input()

    return response

def sample_triplets(solutions, n_triplets):
    """Sample triplets for 2AFC queries"""
    triplets = []
    for _ in range(n_triplets):
        # Random sampling (can use active learning strategies)
        x1 = random.choice(solutions)  # Anchor
        x2 = random.choice(solutions)  # Candidate 1
        x3 = random.choice(solutions)  # Candidate 2
        triplets.append((x1, x2, x3))
    return triplets
```

### Using Preference Models (Scalable Alternative)

```python
class PreferenceModel:
    """Trained model for predicting human similarity judgments"""
    def __init__(self, model_path):
        # E.g., DreamSim for image similarity
        self.model = load_pretrained_model(model_path)

    def predict_similarity(self, x1, x2):
        """Predict similarity score between x1 and x2"""
        return self.model(x1, x2)

    def simulated_2afc_query(self, x1, x2, x3):
        """Simulate human judgment using preference model"""
        sim_x2 = self.predict_similarity(x1, x2)
        sim_x3 = self.predict_similarity(x1, x3)

        return "x2_more_similar" if sim_x2 > sim_x3 else "x3_more_similar"

# Replace human_2afc_query with preference_model.simulated_2afc_query
```

## QD Integration

### MAP-Elites with Learned Diversity Metrics

```python
def integrate_with_map_elites(qdhf, objective_fn):
    """Run MAP-Elites with QDHF diversity metrics"""

    # Archive grid based on latent dimensions
    archive = Grid(dims=qdhf.latent_dim, bins_per_dim=50)

    # Initialize population
    population = generate_random_solutions(n_init)

    for solution in population:
        fitness = objective_fn(solution)
        behavior = qdhf.get_diversity_metrics(solution)  # z vector
        archive.add(solution, fitness, behavior)

    # MAP-Elites loop
    for iteration in range(n_iterations):
        # Sample parent from archive
        parent = archive.random_sample()

        # Generate offspring via mutation
        offspring = mutate(parent.solution)

        # Evaluate
        fitness = objective_fn(offspring)
        behavior = qdhf.get_diversity_metrics(offspring)

        # Add to archive
        archive.add(offspring, fitness, behavior)

    return archive
```

## Domain-Specific Implementations

### Robotic Arm

```python
# Problem: Find inverse kinematics solutions for planar robotic arm
# Solution: Vector of 10 joint angles

class RoboticArmQDHF:
    def __init__(self, n_joints=10):
        self.n_joints = n_joints
        self.qdhf = QDHF(
            feature_extractor=self.extract_features,
            latent_dim=2,  # (x, y) endpoint position
            qd_config={'objective': 'minimize_joint_variance'}
        )

    def extract_features(self, joint_angles):
        """Extract features from solution"""
        # Apply cumulative sum and trig transforms
        cumsum = np.cumsum(joint_angles)
        features = []
        for angle in cumsum:
            features.extend([np.sin(angle), np.cos(angle)])
        return np.array(features)  # 20-dim feature vector

    def objective_fn(self, joint_angles):
        """Minimize variance of joint angles"""
        return -np.var(joint_angles)

    def ground_truth_diversity(self, joint_angles):
        """For evaluation: ground truth is endpoint (x, y)"""
        return forward_kinematics(joint_angles)  # (x, y)
```

**Hyperparameters**:
- Feature dimension: 20 (10 joints × sin/cos)
- Latent dimension: 2
- Human feedback: 1,000 judgments
- MAP-Elites: 1,000 iterations, batch size 100
- Archive: 50×50 grid
- Mutation: Gaussian N(0, 0.1²)

### Maze Navigation (RL)

```python
# Problem: Discover diverse navigation policies for maze
# Solution: Neural network policy parameters

class MazeNavigationQDHF:
    def __init__(self):
        self.qdhf = QDHF(
            feature_extractor=self.extract_trajectory_features,
            latent_dim=2,  # (x, y) final position
            qd_config={'objective': 'maximize_reward'}
        )

    def extract_trajectory_features(self, policy_params):
        """Extract features from policy trajectory"""
        # Rollout policy in environment
        trajectory = rollout_policy(policy_params, env="maze")

        # Extract (x, y) positions at each timestep
        positions = [(state.x, state.y) for state in trajectory]
        return np.array(positions).flatten()  # Feature vector

    def objective_fn(self, policy_params):
        """Accumulated reward"""
        trajectory = rollout_policy(policy_params, env="maze")
        return sum(trajectory.rewards)
```

**Hyperparameters**:
- Policy: MLP with hidden size 8
- Feature dimension: Variable (trajectory positions)
- Latent dimension: 2
- Human feedback: 200 judgments
- MAP-Elites: 1,000 iterations, batch size 200
- Archive: 50×50 grid
- Mutation: Gaussian N(0, 0.2²)
- Episode length: 250 steps

### Latent Space Illumination (Text-to-Image)

```python
# Problem: Generate diverse, high-quality images from Stable Diffusion
# Solution: Latent vectors for diffusion model

class StableDiffusionQDHF:
    def __init__(self, text_prompt):
        self.prompt = text_prompt
        self.sd_model = load_stable_diffusion("v2.1-base")  # 512×512 images

        # Use CLIP for features and objective
        self.clip = load_clip("ViT-B/16")  # 512-dim features

        # Use DreamSim for human feedback simulation
        self.preference_model = load_dreamsim("DINO-ViT-B/16")

        self.qdhf = QDHF(
            feature_extractor=self.clip_features,
            latent_dim=2,  # Learned diversity metrics
            qd_config={'objective': 'clip_score'}
        )

    def clip_features(self, latent_vector):
        """Extract CLIP features from generated image"""
        image = self.sd_model.generate(latent_vector, prompt=self.prompt)
        features = self.clip.encode_image(image)  # 512-dim
        return features

    def objective_fn(self, latent_vector):
        """CLIP score: image-text alignment"""
        image = self.sd_model.generate(latent_vector, prompt=self.prompt)
        image_emb = self.clip.encode_image(image)
        text_emb = self.clip.encode_text(self.prompt)
        return cosine_similarity(image_emb, text_emb)

    def collect_feedback(self, solutions):
        """Use DreamSim to simulate human feedback"""
        triplets = sample_triplets(solutions, n_triplets=10000)
        for (x1, x2, x3) in triplets:
            # Generate images
            img1 = self.sd_model.generate(x1, prompt=self.prompt)
            img2 = self.sd_model.generate(x2, prompt=self.prompt)
            img3 = self.sd_model.generate(x3, prompt=self.prompt)

            # Query preference model
            label = self.preference_model.simulated_2afc_query(
                img1, img2, img3
            )
            self.feedback_data.append((x1, x2, x3, label))
```

**Hyperparameters**:
- Latent shape: (4, 64, 64) flattened
- Feature dimension: 512 (CLIP ViT-B/16)
- Latent dimension: 2
- Human feedback: 10,000 judgments (via DreamSim)
- MAP-Elites: 200 iterations, batch size 5
- Archive: 20×20 grid
- Mutation: Gaussian N(0, 0.1²)
- Preference model: DreamSim trained on NIGHTS dataset (20k triplets)

## Training Details

### Update Schedule

Progressive fine-tuning frequency (decreases as metrics stabilize):

| Update | Iteration | Feedback Budget |
|--------|-----------|----------------|
| 1 | 1 (start) | 25% |
| 2 | 10% × n | 25% |
| 3 | 25% × n | 25% |
| 4 | 50% × n | 25% |

Total feedback = sum of all updates

### Contrastive Learning Hyperparameters

```python
# Triplet loss
margin = 1.0  # Hinge margin
distance = "L2"  # Euclidean distance in latent space

# Optimization
optimizer = Adam(lr=1e-4)
n_epochs = 50  # Per update
batch_size = 32
```

### Validation & Early Stopping

```python
def validate_projection(qdhf, validation_triplets):
    """Measure accuracy of predicting human judgments"""
    correct = 0
    total = len(validation_triplets)

    for (x1, x2, x3, true_label) in validation_triplets:
        # Get latent representations
        z1 = qdhf.get_diversity_metrics(x1)
        z2 = qdhf.get_diversity_metrics(x2)
        z3 = qdhf.get_diversity_metrics(x3)

        # Predict which is more similar
        d2 = np.linalg.norm(z1 - z2)
        d3 = np.linalg.norm(z1 - z3)
        pred_label = "x2_more_similar" if d2 < d3 else "x3_more_similar"

        if pred_label == true_label:
            correct += 1

    accuracy = correct / total
    return accuracy

# Correlation between validation accuracy and QD score
# High accuracy → good diversity metrics → high QD score
```

## Evaluation Metrics

### QD Metrics (Standard)

```python
def qd_score(archive, ground_truth_grid=None):
    """Sum of objective values in archive"""
    if ground_truth_grid is None:
        # Evaluate on learned metrics
        return sum(entry.fitness for entry in archive.values())
    else:
        # Evaluate on ground truth metrics (for validation)
        gt_archive = remap_to_ground_truth(archive, ground_truth_grid)
        return sum(entry.fitness for entry in gt_archive.values())

def coverage(archive, total_bins):
    """Fraction of filled cells"""
    return len(archive) / total_bins
```

### Diversity Metrics (Open-Ended Tasks)

```python
def pairwise_diversity(solutions, similarity_fn):
    """Mean and std of pairwise distances"""
    distances = []
    for i, x1 in enumerate(solutions):
        for x2 in solutions[i+1:]:
            dist = 1 - similarity_fn(x1, x2)  # Convert similarity to distance
            distances.append(dist)

    return np.mean(distances), np.std(distances)

# For images: Use DreamSim similarity
# Higher mean → more diverse, higher std → varied distribution
```

### Human Evaluation (User Studies)

```python
def user_study_2afc(method_a_solutions, method_b_solutions, n_users=50):
    """Compare two methods via blind user study"""

    questions = {
        "preference": "Which set do you prefer?",
        "diversity": "Which set is more diverse?",
        "correctness": "Which set is more correct?" (compositional prompts)
    }

    results = {q: {"method_a": 0, "method_b": 0, "cannot_decide": 0}
               for q in questions}

    for user in range(n_users):
        # Randomize order
        if random.random() < 0.5:
            shown_first, shown_second = method_a_solutions, method_b_solutions
            first_is_a = True
        else:
            shown_first, shown_second = method_b_solutions, method_a_solutions
            first_is_a = False

        for question, prompt in questions.items():
            response = ask_user(shown_first, shown_second, prompt)

            # Map response back to method
            if response == "first":
                winner = "method_a" if first_is_a else "method_b"
            elif response == "second":
                winner = "method_b" if first_is_a else "method_a"
            else:
                winner = "cannot_decide"

            results[question][winner] += 1

    # Convert to percentages
    for question in results:
        for option in results[question]:
            results[question][option] = results[question][option] / n_users * 100

    return results
```

## Active Learning Extensions

### Uncertainty-Based Sampling

```python
def uncertainty_based_triplet_sampling(solutions, qdhf, n_triplets):
    """Sample triplets where model is most uncertain"""
    triplets = []

    for _ in range(n_triplets):
        x1 = random.choice(solutions)
        candidates = random.sample(solutions, k=20)

        # Find pair with most ambiguous similarity
        z1 = qdhf.get_diversity_metrics(x1)
        uncertainties = []
        for x2 in candidates:
            for x3 in candidates:
                if x2 == x3:
                    continue
                z2 = qdhf.get_diversity_metrics(x2)
                z3 = qdhf.get_diversity_metrics(x3)

                # Uncertainty = how close the distances are
                d2 = np.linalg.norm(z1 - z2)
                d3 = np.linalg.norm(z1 - z3)
                uncertainty = -abs(d2 - d3)  # More negative = more certain
                uncertainties.append((uncertainty, (x1, x2, x3)))

        # Select most uncertain triplet
        uncertainties.sort()
        triplets.append(uncertainties[0][1])

    return triplets
```

## Theoretical Analysis

### Information Bottleneck

Minimize:
```
L_IB = I(X; Z) - β·I(Z; Y)
```

Where:
- X = solutions
- Z = latent diversity metrics
- Y = human similarity judgments
- β = trade-off parameter

**Constraint**: I(X; Z) ≤ I_c (via low-dimensional projection)

**Result**: Z learns minimal sufficient statistic of X for predicting Y

### Online Learning (OASIS Connection)

**OASIS Update** (passive-aggressive):
```
W_i = argmin_W (0.5·||W - W_{i-1}||_F^2 + C·ξ)
s.t. l_W(triplet_i) ≤ ξ, ξ ≥ 0
```

**QDHF Update** (combines old + new data):
```
L_W = Σ l_W(old triplets) + Σ l_W(new triplets)
```

- First term: Retain learned diversity (like ||W - W_{i-1}||_F^2)
- Second term: Adapt to new discoveries
- QD provides active sampling strategy for new triplets

## Ablation Studies

### Effect of Feedback Sample Size

```python
def sample_size_ablation(task, sample_sizes=[100, 200, 500, 1000, 2000]):
    results = []

    for n_samples in sample_sizes:
        qdhf = QDHF(feature_extractor, latent_dim=2, qd_config)

        # Limit feedback to n_samples
        qdhf.n_triplets_per_update = n_samples // 4

        archive = qdhf.run(total_iterations=1000)

        qd_score_val = qd_score(archive)
        val_acc = validate_projection(qdhf, validation_set)

        results.append({
            'n_samples': n_samples,
            'qd_score': qd_score_val,
            'validation_accuracy': val_acc
        })

    # Finding: Strong correlation between QD score and sample size
    # Validation accuracy predicts QD performance
    return results
```

### Noise Robustness

```python
def noise_robustness_ablation(task, noise_levels=[0.0, 0.05, 0.10, 0.20]):
    results = []

    for noise_level in noise_levels:
        qdhf = QDHF(feature_extractor, latent_dim=2, qd_config)

        # Inject noise: flip random % of labels
        def noisy_feedback_collection(solutions):
            feedback = qdhf.collect_feedback(solutions)
            for i in range(len(feedback)):
                if random.random() < noise_level:
                    # Flip label
                    x1, x2, x3, label = feedback[i]
                    new_label = "x3_more_similar" if label == "x2_more_similar" else "x2_more_similar"
                    feedback[i] = (x1, x2, x3, new_label)
            return feedback

        qdhf.collect_feedback = noisy_feedback_collection
        archive = qdhf.run(total_iterations=1000)

        results.append({
            'noise_level': noise_level,
            'qd_score': qd_score(archive)
        })

    # Finding: 5% noise = minimal impact, 20% noise still beats baselines
    return results
```

## Implementation Tips

### Feature Extraction
- **Images**: Use pretrained vision models (CLIP, DINO, ResNet)
- **Trajectories**: Concatenate state features across timesteps
- **Policies**: Use behavioral traces, not network parameters directly
- **Structured data**: Domain-specific transformations (forward kinematics, etc.)

### Latent Dimension Selection
- Start with 2D for visualization and interpretability
- Increase to 3-6D for complex domains
- Trade-off: Higher dim = more expressive, lower dim = better information bottleneck
- Validate with judgment prediction accuracy

### QD Algorithm Choice
- **MAP-Elites**: Simple, fast, works well
- **CMA-ME**: Better for high-dimensional solutions
- **PGA-MAP-Elites**: Use for RL tasks with gradient info
- Any QD algorithm compatible with learned metrics

### Computational Efficiency
- Cache feature extractions (f(x) is expensive for images/rollouts)
- Use batch processing for projection training
- Parallelize QD evaluation across cores/GPUs
- Preference models >> humans in the loop for scalability

## Code Resources

**Official Implementation**: https://liding.info/qdhf

**Dependencies**:
- QD frameworks: QDax (JAX), pyribs (Python)
- Preference models: DreamSim (image similarity)
- Generative models: Stable Diffusion, CLIP
- RL environments: Kheperax (JAX), Gym

## References

- [QDHF Paper (Ding et al., ICML 2024)](https://liding.info/qdhf)
- [DreamSim (Fu et al., 2023)](https://github.com/ssundaram21/dreamsim)
- [OASIS (Chechik et al., 2010)](https://proceedings.neurips.cc/paper/2010/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html)
- [Information Bottleneck (Tishby et al., 2000)](https://arxiv.org/abs/physics/0004057)

## Related

- [[QDHF]] — Quick overview
- [[MAP_Elites]] / [[MAP_Elites_detailed]] — QD algorithm
- [[Quality_Diversity]] / [[Quality_Diversity_detailed]] — QD paradigm
- [[DNS]] — Another automatic diversity QD variant
