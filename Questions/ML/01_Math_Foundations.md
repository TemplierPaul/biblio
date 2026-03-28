# Mathematical Foundations - Interview Q&A

Comprehensive coverage of mathematical concepts essential for machine learning: optimization theory, linear algebra, calculus, and their applications.

---


## Table of Contents

- [[#Part 1: Optimization Theory]]
  - [[#Explain gradient descent. What are second-order optimization algorithms?]]
  - [[#What are second-order optimization algorithms? How can the 2nd derivative be used?]]
  - [[#Describe Newton's algorithm. Where does it come from? How can it be adapted to find a minimum?]]
- [[#Part 2: Linear Algebra]]
  - [[#What is a Jacobian matrix?]]
- [[#Part 3: Tensors and Linear Algebra]]
  - [[#Explain tensors, eigenvalues, and matrix decomposition]]
  - [[#Eigenvalues and Eigenvectors]]
  - [[#Matrix Decompositions]]
- [[#Part 4: Backpropagation & Derivatives]]
  - [[#What is backpropagation?]]
  - [[#Explain the chain rule. How does it apply to neural networks?]]
  - [[#What are common derivatives every deep learning practitioner must know?]]
  - [[#How do we backpropagate through a single linear layer?]]
  - [[#What is a computational graph and how does backprop traverse it?]]
  - [[#Walk through a full backprop example on a 2-layer network]]
  - [[#What is the vanishing/exploding gradient problem and how does it relate to backprop?]]
- [[#Summary: Key Takeaways]]

---

## Part 1: Optimization Theory

### Explain gradient descent. What are second-order optimization algorithms?

**Gradient Descent - The Foundation:**

**Core Idea:**
Move in the direction of steepest descent (negative gradient) to find a local minimum of a function.

**Update Rule:**

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

where:
- $\theta$: parameters to optimize
- $\eta$: learning rate (step size)
- $\nabla L(\theta)$: gradient of loss function

**Intuition:**
Imagine you're on a mountain in fog and want to reach the valley. You can't see far, but you can feel the slope under your feet. Always step downhill (in the direction of steepest descent).

**Types of Gradient Descent:**

**1. Batch Gradient Descent**
- Compute gradient using ALL training data
- Update formula: $\theta = \theta - \eta \times \frac{1}{n} \times \sum_i \nabla L(x_i, y_i; \theta)$
- **Pros**: Stable updates, converges to true gradient
- **Cons**: Slow for large datasets, memory intensive

**2. Stochastic Gradient Descent (SGD)**
- Compute gradient using ONE random sample at a time
- Update formula: $\theta = \theta - \eta \times \nabla L(x_i, y_i; \theta)$
- **Pros**: Fast, can escape local minima (due to noise)
- **Cons**: Noisy updates, may not converge exactly

**3. Mini-batch Gradient Descent**
- Compute gradient using small batch of samples (e.g., 32, 64, 256)
- Best of both worlds: efficiency + stability
- Standard in deep learning

**Learning Rate Challenges:**
- Too large: Overshoots minimum, diverges
- Too small: Slow convergence, gets stuck
- Solution: Learning rate schedules or adaptive methods

---

### What are second-order optimization algorithms? How can the 2nd derivative be used?

**First-Order vs Second-Order:**

**First-Order Methods (use gradient only):**
- Gradient descent, SGD, Adam
- Information: Direction of steepest descent
- Computational cost: $O(n)$ per update

**Second-Order Methods (use gradient + curvature):**
- Newton's method, L-BFGS, Natural gradient
- Information: Direction AND curvature (how fast function curves)
- Computational cost: $O(n^2)$ or $O(n^3)$ per update

**Why Use Second Derivatives?**

The gradient tells you which direction to go, but the **Hessian** (matrix of second derivatives) tells you:
1. How much to step (curvature information)
2. Whether you're in a valley or on a ridge
3. Which directions are "steep" vs "flat"

**Newton's Method:**

**Update Rule:**

$$\theta_{t+1} = \theta_t - H^{-1} \nabla L(\theta_t)$$

where $H$ is the Hessian matrix:

$$H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}$$

**Intuition:**
Newton's method approximates the loss function as a quadratic (parabola) near the current point, then jumps directly to the minimum of that parabola.

**Advantages:**
- Converges much faster than gradient descent (quadratic convergence)
- Automatically adapts step size based on curvature
- Near the optimum, takes larger steps in flat directions, smaller in steep directions

**Disadvantages:**
- Computing and inverting Hessian is expensive: $O(n^3)$ for $n$ parameters
- Not practical for deep learning (millions of parameters)
- Requires Hessian to be positive definite

**Practical Second-Order Methods:**

**1. L-BFGS (Limited-memory BFGS)**
- Approximates inverse Hessian using recent gradients
- Only stores last ~10-20 gradient updates
- Popular for small-to-medium ML problems (thousands of parameters)
- Used in: scikit-learn's LogisticRegression, SVM

**2. Natural Gradient**
- Uses Fisher Information Matrix instead of Hessian
- Better suited for probability distributions
- Used in: Trust Region Policy Optimization (TRPO)

**3. Gauss-Newton**
- For non-linear least squares problems
- Approximates Hessian using only first derivatives
- Faster and more stable than full Newton's method

**Comparison Table:**

| Method | Per-iteration Cost | Convergence Rate | Use Case |
|--------|-------------------|------------------|----------|
| Gradient Descent | $O(n)$ | Linear | Simple problems |
| SGD | $O(1)$ | Sublinear | Large-scale ML |
| Adam | $O(n)$ | Linear | Deep learning (default) |
| Newton | $O(n^3)$ | Quadratic | Small-scale, smooth problems |
| L-BFGS | $O(n \times m)$ | Superlinear | Medium-scale ML |

**When to Use Each:**

- **Deep Learning**: Adam or SGD (first-order only, due to parameter count)
- **Convex Optimization**: Newton or L-BFGS (if you can afford it)
- **Large-Scale**: Stochastic methods only (L-BFGS doesn't scale)
- **Online Learning**: SGD or adaptive methods

---

### Describe Newton's algorithm. Where does it come from? How can it be adapted to find a minimum?

**Newton's Method - Origins:**

Originally designed for **finding roots** of equations (where $f(x) = 0$), later adapted for optimization.

**Root Finding Version:**

Problem: Find $x$ such that $f(x) = 0$

**Taylor approximation:**

$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x$$

Set this to zero and solve for $\Delta x$:

$$f(x) + f'(x)\Delta x = 0$$

$$\Delta x = -\frac{f(x)}{f'(x)}$$

**Update rule:**

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Geometric Intuition:**
Draw a tangent line at current point $x_n$. Where this tangent crosses the x-axis is your next guess $x_{n+1}$.

**Example - Finding Square Roots:**

To find $\sqrt{a}$, solve: $f(x) = x^2 - a = 0$

Newton's method gives:

$$x_{n+1} = x_n - \frac{x_n^2 - a}{2x_n}$$

$$x_{n+1} = \frac{x_n + \frac{a}{x_n}}{2}$$

This is the ancient Babylonian method for square roots!

**Adaptation for Optimization:**

To find minimum of $g(x)$, find where derivative is zero: $g'(x) = 0$

Apply Newton's root finding to $f(x) = g'(x)$:

$$x_{n+1} = x_n - \frac{g'(x_n)}{g''(x_n)}$$

This is Newton's method for optimization!

**Multivariate Version:**

For function $L(\theta)$ with gradient $\nabla L$ and Hessian $H$:

$$\theta_{n+1} = \theta_n - H^{-1} \nabla L(\theta_n)$$

**Taylor Expansion View:**

Newton's method uses second-order Taylor approximation:

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H \Delta\theta$$

Minimize this quadratic approximation by setting derivative to zero:

$$\nabla L(\theta) + H \Delta\theta = 0$$

$$\Delta\theta = -H^{-1} \nabla L(\theta)$$

**Why It Works:**

Near the minimum, most functions look like parabolas (quadratics). Newton's method solves the quadratic exactly, so it converges very quickly near the optimum.

**Convergence Properties:**

- **Far from optimum**: May not converge, can oscillate
- **Near optimum**: Quadratic convergence (error squared each iteration!)
- **Exactly quadratic functions**: Converges in one step

**Practical Modifications:**

**1. Damped Newton's Method**
Add learning rate $\eta$:

$$\theta_{n+1} = \theta_n - \eta H^{-1} \nabla L(\theta_n)$$

Start with small $\eta$, increase as you get closer to minimum.

**2. Trust Region Methods**
Only trust Newton's quadratic approximation in a small region:
- Solve: $\min \|\Delta\theta\|$ s.t. $\Delta\theta = -H^{-1}\nabla L$ and $\|\Delta\theta\| \leq r$
- Adjust trust region radius $r$ based on how good the approximation is

**3. Gauss-Newton (for least squares)**
For $f(\theta) = \frac{1}{2}\|r(\theta)\|^2$, where $r$ is residual:
- Hessian $\approx J^\top J$ ($J$ is Jacobian of residuals)
- Avoids computing second derivatives
- Update: $\theta_{n+1} = \theta_n - (J^\top J)^{-1} J^\top r$

---

## Part 2: Linear Algebra

### What is a Jacobian matrix?

**Definition:**

The Jacobian is a matrix of all first-order partial derivatives of a vector-valued function.

For function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ that maps $n$ inputs to $m$ outputs:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Dimensions:** $m \times n$ matrix ($m$ outputs $\times$ $n$ inputs)

**Example - Simple Case:**

Function: $f(x, y) = [x^2 + y,\ xy]$

Jacobian:
```
J = [∂f₁/∂x  ∂f₁/∂y]   [2x   1]
    [∂f₂/∂x  ∂f₂/∂y] = [y    x]
```

**Intuition:**

The Jacobian tells you how small changes in inputs affect outputs:

$$\Delta f \approx J \Delta x$$

It's the multi-dimensional generalization of the derivative.

**Special Cases:**

**1. Gradient ($m=1$):**
When $f: \mathbb{R}^n \rightarrow \mathbb{R}$ (scalar output), Jacobian is just the gradient:

$$J = \nabla f = \left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \ldots,\ \frac{\partial f}{\partial x_n}\right]$$

**2. Divergence:**
When $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$ (same input/output dimensions), trace of Jacobian gives divergence:

$$\text{div}(f) = \text{tr}(J) = \sum_i \frac{\partial f_i}{\partial x_i}$$

**3. Determinant:**
For $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$, $|\det(J)|$ measures how much $f$ scales volumes:
- $|\det(J)| > 1$: $f$ expands volumes
- $|\det(J)| < 1$: $f$ contracts volumes
- $\det(J) = 0$: $f$ collapses dimensions

**Applications in ML:**

**1. Backpropagation:**

Neural network as composition: $y = f_3(f_2(f_1(x)))$

Chain rule using Jacobians:

$$\frac{dy}{dx} = \frac{dy}{df_3} \cdot \frac{df_3}{df_2} \cdot \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

Each term is a Jacobian matrix.

**2. Normalizing Flows:**

Transformation $x \rightarrow z = f(x)$ changes probability density:

$$p_z(z) = p_x(x)\,|\det(J_{f^{-1}})|$$

The Jacobian determinant accounts for volume changes.

**3. Change of Variables:**

When transforming random variables:

$$\int p_x(x)\,dx = \int p_y(y)\,|\det(J)|\,dy$$

**4. Sensitivity Analysis:**

Jacobian shows how sensitive outputs are to input changes:
- Large entry $J_{ij}$: output $i$ is sensitive to input $j$
- Small entry: output $i$ doesn't depend much on input $j$

**Computing Jacobians Efficiently:**

For neural networks with $n$ inputs and $m$ outputs:
- **Forward mode**: Efficient when $n \ll m$
- **Reverse mode** (backprop): Efficient when $m \ll n$
- For $m=1$ (typical loss function), reverse mode is $O(n)$ vs $O(n^2)$ for forward

**Example in Deep Learning:**

Layer: $z = Wx + b$

Jacobian: $\frac{\partial z}{\partial x} = W$

This is why weight matrices ARE the Jacobians in linear layers!

Activation: $a = \sigma(z)$

Jacobian: $\frac{\partial a}{\partial z} = \text{diag}(\sigma'(z))$

A diagonal matrix with derivatives of activation function.

**Jacobian vs Hessian:**

| Matrix | Definition | Dimensions | Order |
|--------|------------|------------|-------|
| Jacobian | First derivatives of vector function | $m \times n$ | 1st |
| Hessian | Second derivatives of scalar function | $n \times n$ | 2nd |
| Gradient | First derivatives of scalar function | $n \times 1$ | 1st |

**Practical Tips:**

1. **Check dimensions**: Jacobian should be (outputs × inputs)
2. **Use automatic differentiation**: Don't compute by hand for complex functions
3. **Watch for degeneracies**: If $\det(J) = 0$, function has singularities
4. **Numerical stability**: Small changes in input shouldn't cause huge changes in output (check $\|J\|$)

---

## Part 3: Tensors and Linear Algebra

### Explain tensors, eigenvalues, and matrix decomposition

**Tensors - Generalized Arrays:**

**Hierarchy:**
- **Scalar** (0D tensor): Just a number
- **Vector** (1D tensor): [1, 2, 3]
- **Matrix** (2D tensor): [[1,2], [3,4]]
- **3D Tensor**: Video data (time × height × width)
- **4D Tensor**: Batch of images (batch × channels × height × width)
- **nD Tensor**: General n-dimensional array

**In Deep Learning:**

Common tensor shapes:
- **Input image**: (batch, channels, height, width) = (32, 3, 224, 224)
- **Text sequence**: (batch, sequence_length, embedding_dim) = (64, 128, 512)
- **Hidden states**: (batch, sequence, hidden_size)

**Tensor Operations:**

**Element-wise operations:**
- Addition: $C = A + B$
- Multiplication: $C = A \odot B$ (Hadamard product)
- Activation: $\sigma(A)$

**Reduction operations:**
- Sum: $\sum_{ij} A_{ij}$
- Mean: $\frac{1}{n} \sum_{ij} A_{ij}$
- Max: $\max_{ij} A_{ij}$

**Tensor contraction (generalized matrix multiplication):**
- Matrix-vector: $y = Wx$
- Matrix-matrix: $C = AB$
- Einstein summation: einsum('ij,jk->ik', A, B)

---

### Eigenvalues and Eigenvectors

**Definition:**

For matrix $A$, if $Av = \lambda v$, then:
- $v$ is an eigenvector
- $\lambda$ is the corresponding eigenvalue

**Meaning:**

Eigenvectors are special directions where $A$ only stretches/shrinks (doesn't rotate).

**Example:**

Matrix $A = [[2, 0], [0, 3]]$

Eigenvectors: $[1,0]$ and $[0,1]$
Eigenvalues: $\lambda_1 = 2$, $\lambda_2 = 3$

Meaning: $A$ stretches x-direction by $2\times$, y-direction by $3\times$.

**Properties:**

1. **Trace = sum of eigenvalues**: $\text{tr}(A) = \sum_i \lambda_i$
2. **Determinant = product of eigenvalues**: $\det(A) = \prod_i \lambda_i$
3. **Symmetric matrices** have real eigenvalues and orthogonal eigenvectors

**Applications in ML:**

**1. PCA (Principal Component Analysis):**

Find directions of maximum variance in data:
- Compute covariance matrix $\Sigma$
- Find eigenvectors of $\Sigma$
- These are principal components (directions of variance)
- Eigenvalues tell you how much variance each component captures

**2. Spectral Clustering:**

Use eigenvectors of graph Laplacian to cluster data.

**3. Neural Network Analysis:**

Eigenvalues of weight matrices indicate:
- Stability of training (eigenvalues near 1 = stable)
- Vanishing/exploding gradients (eigenvalues $\ll 1$ or $\gg 1$)

**4. Power Iteration:**

Find dominant eigenvector (largest eigenvalue):
- Start with random vector $v$
- Repeat: $v \leftarrow \frac{Av}{\|Av\|}$
- Converges to eigenvector with largest $|\lambda|$

---

### Matrix Decompositions

**Why Decompose Matrices?**

Break complex matrix into simpler pieces:
- Easier to analyze
- Faster computation
- Reveal structure
- Enable compression

**1. Eigendecomposition (Spectral Decomposition):**

For symmetric matrix $A$:

$$A = Q \Lambda Q^\top$$

where:
- $Q$: matrix of eigenvectors (orthonormal columns)
- $\Lambda$: diagonal matrix of eigenvalues

**Use cases:**
- PCA
- Understanding linear transformations
- Computing matrix powers: $A^n = Q \Lambda^n Q^\top$

**2. Singular Value Decomposition (SVD):**

For any matrix $A$ ($m \times n$):

$$A = U \Sigma V^\top$$

where:
- $U$: $m \times m$ orthogonal (left singular vectors)
- $\Sigma$: $m \times n$ diagonal (singular values)
- $V$: $n \times n$ orthogonal (right singular vectors)

**Interpretation:**
Any matrix represents rotation → scaling → rotation

**Applications:**
- **Dimensionality reduction**: Keep top $k$ singular values
- **Matrix completion**: Recommender systems
- **Low-rank approximation**: Compress images, data
- **Pseudoinverse**: $A^+ = V \Sigma^+ U^\top$ for solving least squares

**Example - Image Compression:**

Original image: $1000 \times 1000 = 1\text{M}$ pixels
SVD: Keep top 50 singular values
Storage: $50 \times (1000 + 1000 + 1) \approx 100\text{K}$ values
Compression ratio: $10\times$

**3. QR Decomposition:**

$$A = QR$$

where:
- $Q$: orthogonal matrix
- $R$: upper triangular matrix

**Use case:** Numerically stable way to solve linear systems

**4. Cholesky Decomposition:**

For positive definite matrix $A$:

$$A = LL^\top$$

where $L$ is lower triangular.

**Use cases:**
- Sampling from multivariate Gaussian
- Solving systems $Ax = b$ efficiently
- Checking positive definiteness

**5. LU Decomposition:**

$$A = LU$$

where:
- $L$: lower triangular
- $U$: upper triangular

**Use case:** Efficient solution of multiple systems with same $A$

**Comparison Table:**

| Decomposition | Works For | Main Use |
|---------------|-----------|----------|
| Eigendecomposition | Square, symmetric | PCA, understanding transformations |
| SVD | Any matrix | Dimensionality reduction, compression |
| QR | Any matrix | Solving least squares |
| Cholesky | Positive definite | Sampling, fast linear solves |
| LU | Square | Solving multiple systems |

---


- Expert 3: Handles conversational text
- Gating network learns: "This input looks technical → use Expert 1"

**Benefits:**

1. **Specialization**: Each expert learns different patterns
2. **Scalability**: Add more experts without exponential parameter growth
3. **Efficiency**: Sparse MoE activates only relevant experts
4. **Conditional computation**: Adapt computation to input

**Challenges:**

1. **Load balancing**: Some experts may be underutilized
   - Solution: Add auxiliary loss to encourage balanced usage

2. **Training difficulty**: Gating network hard to train
   - Solution: Pre-train experts, gradually introduce gating

3. **Expert collapse**: All inputs routed to few experts
   - Solution: Diversity-encouraging losses

**Modern Applications:**

**Switch Transformer (Google):**
- Replace feedforward layer in Transformer with MoE
- Each token routed to one expert
- Scales to trillions of parameters efficiently

**GPT-4 (rumored):**
- Uses MoE architecture
- Different experts for different types of knowledge
- Allows massive model size with manageable inference cost

**Load Balancing in MoE:**

Auxiliary loss:

$$L_\text{aux} = \alpha \cdot \sum_i f_i \cdot P_i$$

where:
- $f_i$: fraction of inputs routed to expert $i$
- $P_i$: average gating value for expert $i$
- Encourages: uniform $f_i$ (balanced usage)

**Comparison with Ensembles:**

| Feature | MoE | Ensemble |
|---------|-----|----------|
| Training | Joint (end-to-end) | Independent |
| Gating | Learned | Fixed (e.g., average) |
| Specialization | Automatic | Manual/random |
| Efficiency | Can be sparse | All models run |

**When to Use MoE:**

- Very large datasets with diverse patterns
- Need to scale model size efficiently
- Heterogeneous data (different domains/tasks)
- Want conditional computation (save compute on easy inputs)

---

## Part 4: Backpropagation & Derivatives

### What is backpropagation?

**Backpropagation** (backprop): An algorithm to compute the gradient of a loss function with respect to every parameter in a neural network, by applying the chain rule backwards through the computational graph.

**Why it matters**: Training a neural network = minimizing a loss L(θ). To do gradient descent we need ∂L/∂θ for every weight θ. Backprop gives us all of them in one backward pass.

**Two-phase process:**

**1. Forward pass** – compute predictions and loss
```
Input x → [Layer 1] → h₁ → [Layer 2] → h₂ → ... → ŷ → Loss L
```

**2. Backward pass** – propagate error signal back through the network
```
∂L/∂hₙ → ∂L/∂hₙ₋₁ → ... → ∂L/∂h₁ → ∂L/∂W₁, ∂L/∂b₁
```

**Key insight**: We cache the intermediate values from the forward pass (activations, pre-activations) because they are needed to compute the backward pass gradients.

**Complexity**:
- Forward pass: O(P) where P = number of parameters
- Backward pass: O(P) — same order as forward
- Naïve finite differences: O(P²) — impractical for millions of params

---

### Explain the chain rule. How does it apply to neural networks?

**Chain rule (single variable):**

If y = f(u) and u = g(x), then:
```
dy/dx = dy/du · du/dx
```

**Example**:
```
y = sin(x²)    let u = x²
dy/du = cos(u) = cos(x²)
du/dx = 2x
dy/dx = cos(x²) · 2x
```

**Chain rule (multivariable):**

If L is a function of z, and z is a function of x₁, x₂, ..., xₙ:
```
∂L/∂xᵢ = ∂L/∂z · ∂z/∂xᵢ
```

If L depends on z via multiple paths (e.g., z = f(x) and g(x) both depend on x):
```
∂L/∂x = Σₖ ∂L/∂zₖ · ∂zₖ/∂x    (sum over all paths)
```

**How it applies to neural networks:**

A neural network is a composition of functions. For a 3-layer net:
```
L = loss(ŷ)
ŷ = σ(h₂)
h₂ = W₂ · a₁ + b₂
a₁ = σ(h₁)
h₁ = W₁ · x + b₁
```

To find ∂L/∂W₁, chain rule says:
```
∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂h₂ · ∂h₂/∂a₁ · ∂a₁/∂h₁ · ∂h₁/∂W₁
```

Each factor is a local gradient computed at that node. Backprop multiplies these factors in sequence, going right to left.

**Vector form (Jacobians):**

When x, z are vectors, the scalar chain rule generalises to:
```
∂L/∂x = (∂z/∂x)ᵀ · ∂L/∂z
```
where ∂z/∂x is the Jacobian matrix (entries ∂zᵢ/∂xⱼ).

In practice most layers have diagonal Jacobians (e.g. element-wise activations), which reduces the matrix-vector product to an element-wise multiplication:
```
∂L/∂h = σ'(h) ⊙ ∂L/∂a    (⊙ = element-wise)
```

---

### What are common derivatives every deep learning practitioner must know?

**1. Activation functions**

| Activation | Formula | Derivative | Notes |
|---|---|---|---|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | σ(x)(1−σ(x)) | Saturates; max grad = 0.25 |
| Tanh | tanh(x) | 1 − tanh²(x) | Saturates; max grad = 1 |
| ReLU | max(0, x) | 1 if x>0, else 0 | Not differentiable at 0 |
| Leaky ReLU | max(αx, x) | 1 if x>0, else α | Fixes dead ReLU |
| GELU | x·Φ(x) | Φ(x) + x·φ(x) | Used in BERT/GPT |
| Softmax | eˣⁱ/Σeˣʲ | Sᵢ(δᵢⱼ − Sⱼ) | Jacobian is not diagonal |

**Sigmoid derivation** (important to know how to derive this):
```
σ(x) = 1/(1 + e⁻ˣ)

dσ/dx = e⁻ˣ / (1 + e⁻ˣ)²
      = [1/(1 + e⁻ˣ)] · [e⁻ˣ/(1 + e⁻ˣ)]
      = σ(x) · (1 − σ(x))
```

**Softmax + cross-entropy combined** (used in practice):
```
L = -log(Sᵧ)    (cross-entropy for true class y)

∂L/∂zᵢ = Sᵢ − 1[i=y]    (Sᵢ − target)
```
The combined gradient is simply the softmax output minus the one-hot label — a beautiful result.

**2. Loss functions**

| Loss | Formula | Gradient ∂L/∂ŷ |
|---|---|---|
| MSE | (1/N)Σ(ŷ−y)² | (2/N)(ŷ−y) |
| MAE | (1/N)Σ|ŷ−y| | (1/N)·sign(ŷ−y) |
| Cross-Entropy | −Σy·log(ŷ) | −y/ŷ |
| BCE | −[y·log(ŷ)+(1−y)·log(1−ŷ)] | (ŷ−y)/(ŷ(1−ŷ)) |
| Hinge | max(0, 1−y·ŷ) | −y if y·ŷ<1, else 0 |

**3. Linear layer**

For z = Wx + b with L scalar:
```
∂L/∂W = ∂L/∂z · xᵀ        (outer product; shape = shape of W)
∂L/∂b = ∂L/∂z             (same shape as b)
∂L/∂x = Wᵀ · ∂L/∂z        (for passing gradient to previous layer)
```

**Intuition for ∂L/∂W = δ·xᵀ:**
Weight Wᵢⱼ connects input xⱼ to output zᵢ. If output zᵢ has error δᵢ, the weight's contribution is proportional to xⱼ and δᵢ → δᵢxⱼ. Written for all i,j: δ·xᵀ.

**4. Batch normalisation**

```
x̂ = (x − μ) / σ     (normalise)
y = γx̂ + β          (scale and shift)

∂L/∂γ = Σ ∂L/∂y ⊙ x̂
∂L/∂β = Σ ∂L/∂y
∂L/∂x = (1/σ)[∂L/∂y·γ − mean(∂L/∂y·γ) − x̂·mean(∂L/∂y·γ·x̂)]
```

**5. Attention (scaled dot-product)**

```
A = softmax(QKᵀ/√d)    scores
out = AV

∂L/∂V = Aᵀ · ∂L/∂out
∂L/∂A = ∂L/∂out · Vᵀ
∂L/∂(QKᵀ/√d) = dsoftmax · ∂L/∂A    (apply softmax Jacobian)
∂L/∂Q = (1/√d) ∂L/∂(QKᵀ/√d) · K
∂L/∂K = (1/√d) ∂L/∂(QKᵀ/√d)ᵀ · Q
```

---

### How do we backpropagate through a single linear layer?

Consider a linear layer followed by a nonlinearity:
```
h = Wx + b      (pre-activation, shape [m])
a = σ(h)        (activation, shape [m])
```

We receive from the next layer: `δₐ = ∂L/∂a` (upstream gradient, shape [m])

**Step 1: Backprop through activation σ**
```
δₕ = ∂L/∂h = ∂L/∂a ⊙ σ'(h)    element-wise multiply
```
σ'(h) is the derivative of σ evaluated at the pre-activation h — cached during forward pass.

**Step 2: Backprop through weights W**
```
∂L/∂W = δₕ · xᵀ    outer product, shape [m × n]
∂L/∂b = δₕ          shape [m]
```

**Step 3: Backprop through input x (to pass to previous layer)**
```
∂L/∂x = Wᵀ · δₕ    shape [n]
```

This `∂L/∂x` becomes the upstream gradient `δₐ` for the previous layer.

**Batched version** (x is [n × B], B = batch size):
```
∂L/∂W = δₕ · xᵀ / B        average over batch
∂L/∂b = mean(δₕ, axis=batch)
∂L/∂x = Wᵀ · δₕ
```

**Code (numpy)**:
```python
# Forward
h = W @ x + b[:, None]   # [m, B]
a = sigmoid(h)            # [m, B]

# Backward (given da = dL/da from next layer)
dh = da * sigmoid_deriv(h)      # [m, B]  element-wise
dW = dh @ x.T / B               # [m, n]
db = dh.mean(axis=1)             # [m]
dx = W.T @ dh                   # [n, B]
```

---

### What is a computational graph and how does backprop traverse it?

A **computational graph** is a directed acyclic graph (DAG) where:
- **Nodes** = operations or variables
- **Edges** = data flow (tensors)

**Example** (z = (x + y) × w):
```
     x ──┐
         ├──[+]── s ──[×]── z
     y ──┘            │
                      w
```

**Backprop on a graph:**

1. Start at the output node with gradient = 1 (∂L/∂L = 1)
2. For each node in **reverse topological order**:
   - Multiply the incoming upstream gradient by the local Jacobian
   - Pass the result to each input of that node
   - If a node has multiple downstream consumers, **sum the incoming gradients**

**The sum rule** (branch): if variable x feeds into two nodes f and g:
```
∂L/∂x = ∂L/∂f · ∂f/∂x  +  ∂L/∂g · ∂g/∂x
```

**Concrete graph example** (L = (a·b + c)²):
```
     a ──[×]── ab ──[+]── s ──[²]── L
     b ──┘         │
     c ────────────┘

Forward:
  ab = a·b = 2·3 = 6
  s  = ab + c = 6 + 1 = 7
  L  = s² = 49

Backward (upstream grad = 1):
  ∂L/∂s  = 2s = 14
  ∂L/∂ab = ∂L/∂s · 1 = 14
  ∂L/∂c  = ∂L/∂s · 1 = 14
  ∂L/∂a  = ∂L/∂ab · b = 14·3 = 42
  ∂L/∂b  = ∂L/∂ab · a = 14·2 = 28
```

**Modes of automatic differentiation:**

| Mode | Traversal | Efficient when |
|---|---|---|
| **Forward mode** | Left-to-right (with input) | Few inputs, many outputs |
| **Reverse mode** (backprop) | Right-to-left (with output) | Many inputs, scalar output |

For neural nets: scalar loss L, millions of parameters → **reverse mode always wins**.

---

### Walk through a full backprop example on a 2-layer network

**Setup:**
```
x   input,  shape [2]        (example: [0.5, 0.8])
W₁  weights layer 1, shape [3, 2]
b₁  biases  layer 1, shape [3]
W₂  weights layer 2, shape [1, 3]
b₂  bias    layer 2, shape [1]
y   target  (scalar, e.g. 1)

Forward:
  h₁ = W₁x + b₁                  pre-activation layer 1, [3]
  a₁ = sigmoid(h₁)               activation layer 1, [3]
  h₂ = W₂a₁ + b₂                 pre-activation layer 2, [1]
  ŷ  = sigmoid(h₂)               output, [1]
  L  = −[y·log(ŷ) + (1−y)·log(1−ŷ)]   binary cross-entropy
```

**Forward pass (concrete numbers):**
```
x = [0.5, 0.8]

(assume W₁, b₁, W₂, b₂ are initialised; track shapes)

h₁ = W₁ @ x + b₁    → [3]
a₁ = σ(h₁)          → [3]
h₂ = W₂ @ a₁ + b₂   → [1]
ŷ  = σ(h₂)          → scalar
L  = BCE(ŷ, y)       → scalar
```

**Backward pass (step by step):**

**Step 0 – seed**
```
∂L/∂L = 1
```

**Step 1 – BCE + sigmoid output combined**
```
∂L/∂h₂ = ŷ − y                     (beautiful combined gradient)
```
Shape: [1] — this is δ₂.

**Step 2 – Layer 2 weights**
```
∂L/∂W₂ = δ₂ · a₁ᵀ                  shape [1, 3]
∂L/∂b₂ = δ₂                         shape [1]
```

**Step 3 – Pass gradient to a₁**
```
∂L/∂a₁ = W₂ᵀ · δ₂                  shape [3]
```

**Step 4 – Through activation in layer 1**
```
∂L/∂h₁ = ∂L/∂a₁ ⊙ σ'(h₁)
        = ∂L/∂a₁ ⊙ a₁ ⊙ (1 − a₁)  shape [3]
```
This is δ₁.

**Step 5 – Layer 1 weights**
```
∂L/∂W₁ = δ₁ · xᵀ                   shape [3, 2]
∂L/∂b₁ = δ₁                         shape [3]
```

**Step 6 – (Optional) gradient w.r.t. input x**
```
∂L/∂x = W₁ᵀ · δ₁                   shape [2]
```
(not needed for weight update but needed if x itself is a trainable parameter, e.g. embeddings)

**Summary of gradients computed:**
```
δ₂ = ŷ − y                              ← output error
∂L/∂W₂ = δ₂ · a₁ᵀ
∂L/∂b₂ = δ₂

δ₁ = (W₂ᵀδ₂) ⊙ a₁ ⊙ (1−a₁)            ← backprop through W₂ then activation
∂L/∂W₁ = δ₁ · xᵀ
∂L/∂b₁ = δ₁
```

**Weight update (gradient descent):**
```
W₂ ← W₂ − η · ∂L/∂W₂
b₂ ← b₂ − η · ∂L/∂b₂
W₁ ← W₁ − η · ∂L/∂W₁
b₁ ← b₁ − η · ∂L/∂b₁
```

**Code (full numpy implementation):**
```python
import numpy as np

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def sig_d(a):     return a * (1 - a)    # derivative when a = sigmoid(x)
def bce(y_hat, y): return -(y*np.log(y_hat+1e-9) + (1-y)*np.log(1-y_hat+1e-9))

# Hyperparameters
lr = 0.1

# Parameters (random init)
W1 = np.random.randn(3, 2) * 0.01
b1 = np.zeros(3)
W2 = np.random.randn(1, 3) * 0.01
b2 = np.zeros(1)

x = np.array([0.5, 0.8])
y = 1.0

# ---- Forward pass ----
h1 = W1 @ x + b1           # [3]
a1 = sigmoid(h1)            # [3]
h2 = W2 @ a1 + b2           # [1]
y_hat = sigmoid(h2)         # [1]
loss = bce(y_hat, y)

# ---- Backward pass ----
# Step 1: BCE + sigmoid combined
delta2 = y_hat - y                       # [1]

# Step 2: Layer 2 grads
dW2 = np.outer(delta2, a1)              # [1, 3]
db2 = delta2                             # [1]

# Step 3: Pass through W2 to a1
da1 = W2.T @ delta2                      # [3]

# Step 4: Through sigmoid of layer 1
delta1 = da1 * sig_d(a1)               # [3]

# Step 5: Layer 1 grads
dW1 = np.outer(delta1, x)              # [3, 2]
db1 = delta1                             # [3]

# ---- Update ----
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
```

---

### What is the vanishing/exploding gradient problem and how does it relate to backprop?

**Vanishing gradients**: Gradients shrink exponentially as they propagate back through many layers, making early layers learn very slowly (or not at all).

**Exploding gradients**: Gradients grow exponentially, causing numerical instability (NaN/Inf).

**Why it happens — the chain rule view:**

For an L-layer network, the gradient w.r.t. layer 1 involves a product of L Jacobians:
```
∂L/∂W₁ = ∂L/∂aL · J_L · J_{L-1} · ... · J₂ · J₁
```

where Jₖ = ∂aₖ/∂aₖ₋₁ is the Jacobian of layer k.

If the largest singular value of each Jₖ is:
- `< 1` → product → 0 exponentially (**vanishing**)
- `> 1` → product → ∞ exponentially (**exploding**)

**Sigmoid makes this worse:** σ'(x) ≤ 0.25 for all x. With 10 layers:
```
gradient magnitude ≤ 0.25¹⁰ ≈ 0.000001    (essentially zero)
```

**Solutions and their mechanisms:**

| Problem | Solution | Why it helps |
|---|---|---|
| Vanishing | ReLU activation | σ'(x) = 1 for x>0, no saturation |
| Vanishing | Residual connections | Gradient can flow directly via skip: ∂L/∂xₗ = ∂L/∂xₗ₊₁ + ... |
| Vanishing | BatchNorm | Normalises pre-activations, keeps them in active regime |
| Vanishing | LSTM / GRU gates | Gates selectively pass gradients over time |
| Exploding | Gradient clipping | Cap gradient norm: g ← g · min(1, c/‖g‖) |
| Exploding | Weight init (Xavier/He) | Initialises weights so variance is preserved layer-to-layer |
| Both | Layer normalisation | Normalises within each sample, stable at all layers |

**Residual connections (skip connections) explained:**
```
Output: y = F(x) + x        (ResNet block)

Gradient: ∂L/∂x = ∂L/∂y · (∂F/∂x + I)

The '+I' term means gradient always has a direct path back,
regardless of what ∂F/∂x is → vanishing is prevented.
```

**Xavier initialisation** (for sigmoid/tanh):
```
W ~ Uniform(-√(6/(nᵢₙ+nₒᵤₜ)),  +√(6/(nᵢₙ+nₒᵤₜ)))
```
Keeps variance of activations ≈ 1 through the forward pass,
and variance of gradients ≈ 1 through the backward pass.

**He initialisation** (for ReLU):
```
W ~ Normal(0, √(2/nᵢₙ))
```
Factor of 2 compensates for ReLU zeroing half its inputs.

---

## Summary: Key Takeaways

**For Interviews:**

**Optimization:**
- Understand gradient descent and its variants (batch, stochastic, mini-batch)
- Know when to use first-order vs second-order methods
- Explain Newton's method and why it converges faster
- Understand the computational tradeoffs

**Linear Algebra:**
- Know what Jacobian, Hessian, and gradient are
- Understand eigenvalues/eigenvectors and their applications
- Be familiar with key matrix decompositions (SVD, eigendecomposition)
- Explain tensors and their operations in deep learning

**Generative Models:**
- VAE: Probabilistic encoder-decoder with KL regularization
- GAN: Adversarial game between generator and discriminator
- Know tradeoffs and when to use each

**Advanced Concepts:**
- Mixture of Experts: Multiple specialized models with learned routing
- Understand how these scale to large models

**Mathematical Fluency:**
- Be comfortable deriving gradients
- Understand chain rule for backpropagation
- Know basic matrix calculus

Remember: Interviewers want to see you **understand** the math, not just memorize formulas. Focus on intuition and be ready to derive key results.
