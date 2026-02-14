# Classical Machine Learning - Interview Q&A

Comprehensive coverage of classical ML algorithms: SVM, k-NN, Gaussian Processes, and ensemble methods.

---


## Table of Contents

- [[#Part 1: Classification & Regression Models]]
  - [[#What is the difference between generative and non-generative (discriminative) models?]]
  - [[#Explain Support Vector Machines (SVM). What are the fundamentals?]]
  - [[#Explain k-Nearest Neighbors (k-NN). What is the concept?]]
- [[#Part 2: Gaussian Processes]]
  - [[#What's a Gaussian Process?]]
  - [[#Explain mean and covariance (kernel) functions]]
  - [[#Write the predictive mean and variance formulas]]
  - [[#Why does GP provide uncertainty estimates?]]
  - [[#What's the RBF/squared exponential kernel?]]
  - [[#How to tune hyperparameters (marginal likelihood)?]]
  - [[#What's the computational complexity of GP?]]
  - [[#How to scale GPs (sparse GP, inducing points)?]]
  - [[#When to use GP vs neural network?]]
  - [[#What's the connection between infinite-width NNs and GPs?]]

---

## Part 1: Classification & Regression Models

### What is the difference between generative and non-generative (discriminative) models?

**Discriminative Models** learn the decision boundary directly by modeling P(y|x) - the probability of the label given the features.
- **Goal**: Separate classes or predict outputs directly
- **Examples**: Logistic Regression, SVM, Neural Networks, BERT
- **Advantages**: Usually better performance for classification tasks, require less data
- **Question answered**: "What is y given x?"

**Generative Models** learn the joint distribution P(x, y) = P(y)P(x|y), modeling how the data is generated.
- **Goal**: Understand the underlying data distribution
- **Examples**: Naive Bayes, Gaussian Mixture Models, VAEs, GANs, GPT
- **Advantages**: Can generate new samples, handle missing data, provide probability estimates
- **Question answered**: "How is the data generated?" and "What is the probability of this data?"

**Why BERT is not generative:**
BERT is trained with masked language modeling (predicting masked tokens given context) but uses bidirectional attention, making it unsuitable for sequential generation. It learns P(word|context) but cannot naturally generate sequences left-to-right like GPT.

**Code Example - Discriminative vs Generative:**
```python
# Discriminative: Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # Direct classification

# Generative: Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
# Can also compute P(X|y) and generate samples
```

---

### Explain Support Vector Machines (SVM). What are the fundamentals?

**Core Concept:**
SVM finds the optimal hyperplane that maximizes the margin between classes. The margin is the distance between the hyperplane and the nearest data points (support vectors).

**Key Components:**

1. **Hyperplane**: Decision boundary defined by w·x + b = 0
2. **Support Vectors**: Data points closest to the hyperplane that define the margin
3. **Margin**: Distance from hyperplane to nearest points (2/||w||)
4. **Kernel Trick**: Map data to higher dimensions for non-linear separation

**Mathematical Formulation:**

Minimize: (1/2)||w||² + C∑ξᵢ

Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

where:
- w: weight vector (defines hyperplane)
- b: bias
- C: regularization parameter (tradeoff between margin and misclassification)
- ξᵢ: slack variables (allow some misclassification)

**Common Kernels:**

```python
# Linear kernel: K(x, x') = x·x'
# Good for linearly separable data

# RBF (Gaussian) kernel: K(x, x') = exp(-γ||x - x'||²)
# Most popular, works well for non-linear data
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_rbf.fit(X_train, y_train)

# Polynomial kernel: K(x, x') = (x·x' + c)^d
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train, y_train)
```

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient (only stores support vectors)
- Versatile (different kernel functions)
- Works well with clear margin of separation

**Disadvantages:**
- Slow for large datasets (O(n²) to O(n³) complexity)
- Sensitive to feature scaling
- Requires careful kernel selection and parameter tuning
- Doesn't provide probability estimates directly

**When to use SVM:**
- Small to medium datasets
- High-dimensional data (text classification, bioinformatics)
- Clear separation between classes
- Need for robust classifier with good generalization

---

### Explain k-Nearest Neighbors (k-NN). What is the concept?

**Core Concept:**
k-NN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority vote of its k nearest neighbors in the feature space.

**Algorithm Steps:**

1. Choose k (number of neighbors)
2. Calculate distance between query point and all training points
3. Sort distances and select k nearest neighbors
4. Classification: majority vote; Regression: average of k neighbors

**Distance Metrics:**

```python
import numpy as np

# Euclidean distance (most common)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Minkowski distance (generalization)
def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

# Cosine similarity (for text/high-dim)
def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
```

**Implementation Example:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features (important for k-NN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train k-NN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # 'uniform' or 'distance'
    metric='euclidean',
    algorithm='auto'  # 'ball_tree', 'kd_tree', 'brute'
)
knn.fit(X_train_scaled, y_train)

# Predict
predictions = knn.predict(X_test_scaled)
probabilities = knn.predict_proba(X_test_scaled)
```

**Choosing k:**

```python
# Cross-validation to find optimal k
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    scores.append(score.mean())

# Plot k vs accuracy
import matplotlib.pyplot as plt
plt.plot(k_values, scores)
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.show()

# Rule of thumb: k = sqrt(n), where n is number of samples
# Use odd k for binary classification to avoid ties
```

**Advantages:**
- Simple and intuitive
- No training phase (lazy learning)
- Naturally handles multi-class problems
- Non-parametric (no assumptions about data distribution)
- Can learn complex decision boundaries

**Disadvantages:**
- Slow prediction for large datasets (O(nd) per query)
- Memory intensive (stores all training data)
- Sensitive to irrelevant features and feature scaling
- Curse of dimensionality (performance degrades in high dimensions)
- Need to choose k and distance metric

**When to use k-NN:**
- Small to medium datasets
- Low-dimensional data
- Non-linear decision boundaries
- No need for model interpretability
- Baseline model for comparison

**Optimization Techniques:**

```python
# Use efficient data structures
knn_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',  # Fast for low-dim
    leaf_size=30
)

knn_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',  # Better for high-dim
    leaf_size=30
)

# Dimensionality reduction before k-NN
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn.fit(X_train_pca, y_train)
```

---

---



## Table of Contents

- [Part 1: Gaussian Processes](#part-1-gaussian-processes)
  - [What's a Gaussian Process?](#whats-a-gaussian-process)
  - [Explain mean and covariance (kernel) functions](#explain-mean-and-covariance-kernel-functions)
  - [Write the predictive mean and variance formulas](#write-the-predictive-mean-and-variance-formulas)
  - [Why does GP provide uncertainty estimates?](#why-does-gp-provide-uncertainty-estimates)
  - [What's the RBF/squared exponential kernel?](#whats-the-rbfsquared-exponential-kernel)
  - [How to tune hyperparameters (marginal likelihood)?](#how-to-tune-hyperparameters-marginal-likelihood)
  - [What's the computational complexity of GP?](#whats-the-computational-complexity-of-gp)
  - [How to scale GPs (sparse GP, inducing points)?](#how-to-scale-gps-sparse-gp-inducing-points)
  - [When to use GP vs neural network?](#when-to-use-gp-vs-neural-network)
  - [What's the connection between infinite-width NNs and GPs?](#whats-the-connection-between-infinite-width-nns-and-gps)

---


## Part 2: Gaussian Processes

### What's a Gaussian Process?

**Definition**: Collection of random variables, any finite subset of which follows a **multivariate Gaussian** distribution.

**Distribution over functions**:
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

Where:
- $m(x)$: Mean function (often 0)
- $k(x, x')$: Covariance (kernel) function

**For any points** $X = \{x_1, \ldots, x_n\}$:
$$f(X) \sim \mathcal{N}(\mu, K)$$
Where $K_{ij} = k(x_i, x_j)$

**Intuition**: Instead of learning parameters, define distribution over entire functions.

### Explain mean and covariance (kernel) functions

**Mean function** $m(x)$:
- Expected value of $f(x)$
- Often set to 0 (assume data is centered)
- Can encode prior belief (e.g., linear trend)

**Kernel function** $k(x, x')$:
- **Covariance** between $f(x)$ and $f(x')$
- Encodes **similarity**: $k(x, x')$ high → $f(x) \approx f(x')$
- Determines function properties (smoothness, periodicity, etc.)

**Key insight**: "Similar inputs have similar outputs" - kernel defines similarity.

### Write the predictive mean and variance formulas

Given training data $(X, y)$, predict at test points $X_*$:

**Predictive mean**:
$$\mu_{f_*|y} = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$

**Predictive variance**:
$$\Sigma_{f_*|y} = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

Where:
- $K(X, X)$: Train-train covariance
- $K(X_*, X)$: Test-train covariance
- $K(X_*, X_*)$: Test-test covariance
- $\sigma_n^2$: Noise variance

**Key**: Exact Bayesian inference (condition Gaussian on observations).

### Why does GP provide uncertainty estimates?

**Bayesian framework**: Maintains full **posterior distribution** over functions.

**Predictive variance** tells us:
- **Near data**: Low variance (confident)
- **Far from data**: High variance (uncertain)

**Example**:
- Training points: x = [1, 2, 3], test point x = 1.5
- Variance at 1.5: Low (interpolating)
- Test point x = 10: High variance (extrapolating)

**Contrast with NN**:
- NN gives point prediction (no uncertainty)
- Need special techniques (ensembles, dropout) for uncertainty

**Why it works**: GP prediction is Gaussian → variance comes for free from conditioning.

### What's the RBF/squared exponential kernel?

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

**Parameters**:
- $\ell$: **Length scale** (how quickly correlation decays with distance)
- $\sigma_f^2$: Signal variance (vertical scale)

**Properties**:
- Smooth: Infinitely differentiable (smooth functions)
- Stationary: Depends only on $\|x - x'\|$
- Most common kernel

**Interpretation**:
- Small $\ell$: Functions vary quickly (wiggly)
- Large $\ell$: Functions vary slowly (smooth)

### How to tune hyperparameters (marginal likelihood)?

**Hyperparameters**: $\theta = \{\ell, \sigma_f^2, \sigma_n^2\}$ (kernel params + noise)

**Objective**: Maximize **marginal likelihood** $p(y | X, \theta)$

**Log marginal likelihood**:
$$\log p(y|X,\theta) = -\frac{1}{2} y^T K_y^{-1} y - \frac{1}{2} \log |K_y| - \frac{n}{2} \log 2\pi$$

Where $K_y = K(X,X) + \sigma_n^2 I$

**Three terms**:
1. Data fit: $y^T K_y^{-1} y$ (how well model explains data)
2. Complexity: $\log |K_y|$ (penalizes complex models)
3. Constant

**Optimization**: Gradient-based (L-BFGS, Adam) on $-\log p(y|X,\theta)$ (NLL)

**Automatic trade-off**: Balances fit and complexity (Bayesian Occam's razor).

### What's the computational complexity of GP?

**Training** (hyperparameter optimization):
- **Matrix inversion**: $O(n^3)$ for $K^{-1}$
- **Determinant**: $O(n^3)$ for $\log |K|$

**Prediction** (mean):
- $O(n^2)$ to compute $K(X_*, X) K^{-1}$
- $O(n)$ per test point

**Storage**: $O(n^2)$ for covariance matrix

**Bottleneck**: $O(n^3)$ scaling → **doesn't scale beyond ~10,000 points**

**Why**: Need to invert dense $n \times n$ matrix.

### How to scale GPs (sparse GP, inducing points)?

**Sparse GP** (inducing points):
1. Select $m \ll n$ inducing points (subset or pseudo-inputs)
2. Approximate full GP using only $m$ points
3. Complexity: $O(nm^2)$ training, $O(m^2)$ prediction

**Methods**:
- **FITC**: Fully Independent Training Conditional
- **SVGP**: Stochastic Variational GP (mini-batches)

**Trade-off**: $m$ = 100-1000 → approximate but scalable

**Other approaches**:
- **Local GPs**: Separate GPs per region
- **Deep GP**: Stack GPs (compositional)
- **Structured kernels**: Exploit Kronecker, Toeplitz structure

**Modern**: For very large n, use neural networks with uncertainty (ensembles, Bayesian NNs).

### When to use GP vs neural network?

**GP**:
- ✅ Small data (<10K points)
- ✅ Need uncertainty quantification
- ✅ Structured problem (can encode via kernel)
- ✅ Interpretability important
- ❌ Doesn't scale to large n or high dimensions
- **Use**: Bayesian optimization, robotics, active learning

**Neural Network**:
- ✅ Large data (millions of points)
- ✅ High-dimensional inputs
- ✅ Raw features (images, text)
- ❌ No natural uncertainty (need special techniques)
- **Use**: Deep learning, computer vision, NLP

**Middle ground**: Deep kernel learning (NN features + GP), neural tangent kernels.

### What's the connection between infinite-width NNs and GPs?

**Neural Tangent Kernel (NTK)** theorem:

**Infinite-width neural network** (during training) behaves like a **GP** with specific kernel.

**Key insight**:
- Finite width: Complex, non-linear dynamics
- **Infinite width**: Linearizes around initialization → GP with NTK kernel

**Implications**:
1. Connects deep learning and GPs
2. Explains why NNs generalize (GP prior)
3. Theoretical tool for understanding training dynamics

**Practical**: Finite-width NNs differ (non-linear feature learning), but infinite-width limit gives insights.
