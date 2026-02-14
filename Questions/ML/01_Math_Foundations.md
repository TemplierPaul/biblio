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
- [[#Part 4: Generative Models]]
  - [[#Explain VAE (Variational Autoencoder)]]
  - [[#Explain GAN (Generative Adversarial Network)]]
- [[#Part 5: Advanced Topics]]
  - [[#What is Mixture of Experts (MoE)?]]
- [[#Summary: Key Takeaways]]

---

## Part 1: Optimization Theory

### Explain gradient descent. What are second-order optimization algorithms?

**Gradient Descent - The Foundation:**

**Core Idea:**
Move in the direction of steepest descent (negative gradient) to find a local minimum of a function.

**Update Rule:**
θ_{t+1} = θ_t - η∇L(θ_t)

where:
- θ: parameters to optimize
- η: learning rate (step size)
- ∇L(θ): gradient of loss function

**Intuition:**
Imagine you're on a mountain in fog and want to reach the valley. You can't see far, but you can feel the slope under your feet. Always step downhill (in the direction of steepest descent).

**Types of Gradient Descent:**

**1. Batch Gradient Descent**
- Compute gradient using ALL training data
- Update formula: θ = θ - η × (1/n) × Σᵢ ∇L(xᵢ, yᵢ; θ)
- **Pros**: Stable updates, converges to true gradient
- **Cons**: Slow for large datasets, memory intensive

**2. Stochastic Gradient Descent (SGD)**
- Compute gradient using ONE random sample at a time
- Update formula: θ = θ - η × ∇L(xᵢ, yᵢ; θ)
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
- Computational cost: O(n) per update

**Second-Order Methods (use gradient + curvature):**
- Newton's method, L-BFGS, Natural gradient
- Information: Direction AND curvature (how fast function curves)
- Computational cost: O(n²) or O(n³) per update

**Why Use Second Derivatives?**

The gradient tells you which direction to go, but the **Hessian** (matrix of second derivatives) tells you:
1. How much to step (curvature information)
2. Whether you're in a valley or on a ridge
3. Which directions are "steep" vs "flat"

**Newton's Method:**

**Update Rule:**
θ_{t+1} = θ_t - H⁻¹∇L(θ_t)

where H is the Hessian matrix:
H_ij = ∂²L/∂θᵢ∂θⱼ

**Intuition:**
Newton's method approximates the loss function as a quadratic (parabola) near the current point, then jumps directly to the minimum of that parabola.

**Advantages:**
- Converges much faster than gradient descent (quadratic convergence)
- Automatically adapts step size based on curvature
- Near the optimum, takes larger steps in flat directions, smaller in steep directions

**Disadvantages:**
- Computing and inverting Hessian is expensive: O(n³) for n parameters
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
| Gradient Descent | O(n) | Linear | Simple problems |
| SGD | O(1) | Sublinear | Large-scale ML |
| Adam | O(n) | Linear | Deep learning (default) |
| Newton | O(n³) | Quadratic | Small-scale, smooth problems |
| L-BFGS | O(n × m) | Superlinear | Medium-scale ML |

**When to Use Each:**

- **Deep Learning**: Adam or SGD (first-order only, due to parameter count)
- **Convex Optimization**: Newton or L-BFGS (if you can afford it)
- **Large-Scale**: Stochastic methods only (L-BFGS doesn't scale)
- **Online Learning**: SGD or adaptive methods

---

### Describe Newton's algorithm. Where does it come from? How can it be adapted to find a minimum?

**Newton's Method - Origins:**

Originally designed for **finding roots** of equations (where f(x) = 0), later adapted for optimization.

**Root Finding Version:**

Problem: Find x such that f(x) = 0

**Taylor approximation:**
f(x + Δx) ≈ f(x) + f'(x)Δx

Set this to zero and solve for Δx:
f(x) + f'(x)Δx = 0
Δx = -f(x)/f'(x)

**Update rule:**
x_{n+1} = x_n - f(x_n)/f'(x_n)

**Geometric Intuition:**
Draw a tangent line at current point x_n. Where this tangent crosses the x-axis is your next guess x_{n+1}.

**Example - Finding Square Roots:**

To find √a, solve: f(x) = x² - a = 0

Newton's method gives:
x_{n+1} = x_n - (x_n² - a)/(2x_n)
x_{n+1} = (x_n + a/x_n)/2

This is the ancient Babylonian method for square roots!

**Adaptation for Optimization:**

To find minimum of g(x), find where derivative is zero: g'(x) = 0

Apply Newton's root finding to f(x) = g'(x):

x_{n+1} = x_n - g'(x_n)/g''(x_n)

This is Newton's method for optimization!

**Multivariate Version:**

For function L(θ) with gradient ∇L and Hessian H:

θ_{n+1} = θ_n - H⁻¹∇L(θ_n)

**Taylor Expansion View:**

Newton's method uses second-order Taylor approximation:

L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀΔθ + ½Δθᵀ H Δθ

Minimize this quadratic approximation by setting derivative to zero:
∇L(θ) + H Δθ = 0
Δθ = -H⁻¹∇L(θ)

**Why It Works:**

Near the minimum, most functions look like parabolas (quadratics). Newton's method solves the quadratic exactly, so it converges very quickly near the optimum.

**Convergence Properties:**

- **Far from optimum**: May not converge, can oscillate
- **Near optimum**: Quadratic convergence (error squared each iteration!)
- **Exactly quadratic functions**: Converges in one step

**Practical Modifications:**

**1. Damped Newton's Method**
Add learning rate η:
θ_{n+1} = θ_n - η H⁻¹∇L(θ_n)

Start with small η, increase as you get closer to minimum.

**2. Trust Region Methods**
Only trust Newton's quadratic approximation in a small region:
- Solve: min ||Δθ|| s.t. Δθ = -H⁻¹∇L and ||Δθ|| ≤ r
- Adjust trust region radius r based on how good the approximation is

**3. Gauss-Newton (for least squares)**
For f(θ) = ½||r(θ)||², where r is residual:
- Hessian ≈ JᵀJ (J is Jacobian of residuals)
- Avoids computing second derivatives
- Update: θ_{n+1} = θ_n - (JᵀJ)⁻¹Jᵀr

---

## Part 2: Linear Algebra

### What is a Jacobian matrix?

**Definition:**

The Jacobian is a matrix of all first-order partial derivatives of a vector-valued function.

For function f: ℝⁿ → ℝᵐ that maps n inputs to m outputs:

J_ij = ∂f_i/∂x_j

**Dimensions:** m × n matrix (m outputs × n inputs)

**Example - Simple Case:**

Function: f(x, y) = [x² + y, xy]

Jacobian:
```
J = [∂f₁/∂x  ∂f₁/∂y]   [2x   1]
    [∂f₂/∂x  ∂f₂/∂y] = [y    x]
```

**Intuition:**

The Jacobian tells you how small changes in inputs affect outputs:

Δf ≈ J Δx

It's the multi-dimensional generalization of the derivative.

**Special Cases:**

**1. Gradient (m=1):**
When f: ℝⁿ → ℝ (scalar output), Jacobian is just the gradient:
J = ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

**2. Divergence:**
When f: ℝⁿ → ℝⁿ (same input/output dimensions), trace of Jacobian gives divergence:
div(f) = tr(J) = Σᵢ ∂fᵢ/∂xᵢ

**3. Determinant:**
For f: ℝⁿ → ℝⁿ, |det(J)| measures how much f scales volumes:
- |det(J)| > 1: f expands volumes
- |det(J)| < 1: f contracts volumes
- det(J) = 0: f collapses dimensions

**Applications in ML:**

**1. Backpropagation:**

Neural network as composition: y = f₃(f₂(f₁(x)))

Chain rule using Jacobians:
dy/dx = (dy/df₃)(df₃/df₂)(df₂/df₁)(df₁/dx)

Each term is a Jacobian matrix.

**2. Normalizing Flows:**

Transformation x → z = f(x) changes probability density:

p_z(z) = p_x(x) |det(J_f⁻¹)|

The Jacobian determinant accounts for volume changes.

**3. Change of Variables:**

When transforming random variables:
∫ p_x(x) dx = ∫ p_y(y) |det(J)| dy

**4. Sensitivity Analysis:**

Jacobian shows how sensitive outputs are to input changes:
- Large entry J_ij: output i is sensitive to input j
- Small entry: output i doesn't depend much on input j

**Computing Jacobians Efficiently:**

For neural networks with n inputs and m outputs:
- **Forward mode**: Efficient when n << m
- **Reverse mode** (backprop): Efficient when m << n
- For m=1 (typical loss function), reverse mode is O(n) vs O(n²) for forward

**Example in Deep Learning:**

Layer: z = Wx + b

Jacobian: ∂z/∂x = W

This is why weight matrices ARE the Jacobians in linear layers!

Activation: a = σ(z)

Jacobian: ∂a/∂z = diag(σ'(z))

A diagonal matrix with derivatives of activation function.

**Jacobian vs Hessian:**

| Matrix | Definition | Dimensions | Order |
|--------|------------|------------|-------|
| Jacobian | First derivatives of vector function | m × n | 1st |
| Hessian | Second derivatives of scalar function | n × n | 2nd |
| Gradient | First derivatives of scalar function | n × 1 | 1st |

**Practical Tips:**

1. **Check dimensions**: Jacobian should be (outputs × inputs)
2. **Use automatic differentiation**: Don't compute by hand for complex functions
3. **Watch for degeneracies**: If det(J) = 0, function has singularities
4. **Numerical stability**: Small changes in input shouldn't cause huge changes in output (check ||J||)

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
- Addition: C = A + B
- Multiplication: C = A ⊙ B (Hadamard product)
- Activation: σ(A)

**Reduction operations:**
- Sum: Σᵢⱼ A_ij
- Mean: (1/n) Σᵢⱼ A_ij
- Max: maxᵢⱼ A_ij

**Tensor contraction (generalized matrix multiplication):**
- Matrix-vector: y = Wx
- Matrix-matrix: C = AB
- Einstein summation: einsum('ij,jk->ik', A, B)

---

### Eigenvalues and Eigenvectors

**Definition:**

For matrix A, if Av = λv, then:
- v is an eigenvector
- λ is the corresponding eigenvalue

**Meaning:**

Eigenvectors are special directions where A only stretches/shrinks (doesn't rotate).

**Example:**

Matrix A = [[2, 0], [0, 3]]

Eigenvectors: [1,0] and [0,1]
Eigenvalues: λ₁=2, λ₂=3

Meaning: A stretches x-direction by 2×, y-direction by 3×.

**Properties:**

1. **Trace = sum of eigenvalues**: tr(A) = Σᵢ λᵢ
2. **Determinant = product of eigenvalues**: det(A) = Πᵢ λᵢ
3. **Symmetric matrices** have real eigenvalues and orthogonal eigenvectors

**Applications in ML:**

**1. PCA (Principal Component Analysis):**

Find directions of maximum variance in data:
- Compute covariance matrix Σ
- Find eigenvectors of Σ
- These are principal components (directions of variance)
- Eigenvalues tell you how much variance each component captures

**2. Spectral Clustering:**

Use eigenvectors of graph Laplacian to cluster data.

**3. Neural Network Analysis:**

Eigenvalues of weight matrices indicate:
- Stability of training (eigenvalues near 1 = stable)
- Vanishing/exploding gradients (eigenvalues << 1 or >> 1)

**4. Power Iteration:**

Find dominant eigenvector (largest eigenvalue):
- Start with random vector v
- Repeat: v ← Av / ||Av||
- Converges to eigenvector with largest |λ|

---

### Matrix Decompositions

**Why Decompose Matrices?**

Break complex matrix into simpler pieces:
- Easier to analyze
- Faster computation
- Reveal structure
- Enable compression

**1. Eigendecomposition (Spectral Decomposition):**

For symmetric matrix A:
A = QΛQᵀ

where:
- Q: matrix of eigenvectors (orthonormal columns)
- Λ: diagonal matrix of eigenvalues

**Use cases:**
- PCA
- Understanding linear transformations
- Computing matrix powers: A^n = QΛⁿQᵀ

**2. Singular Value Decomposition (SVD):**

For any matrix A (m × n):
A = UΣVᵀ

where:
- U: m × m orthogonal (left singular vectors)
- Σ: m × n diagonal (singular values)
- V: n × n orthogonal (right singular vectors)

**Interpretation:**
Any matrix represents rotation → scaling → rotation

**Applications:**
- **Dimensionality reduction**: Keep top k singular values
- **Matrix completion**: Recommender systems
- **Low-rank approximation**: Compress images, data
- **Pseudoinverse**: A⁺ = VΣ⁺Uᵀ for solving least squares

**Example - Image Compression:**

Original image: 1000 × 1000 = 1M pixels
SVD: Keep top 50 singular values
Storage: 50 × (1000 + 1000 + 1) ≈ 100K values
Compression ratio: 10×

**3. QR Decomposition:**

A = QR

where:
- Q: orthogonal matrix
- R: upper triangular matrix

**Use case:** Numerically stable way to solve linear systems

**4. Cholesky Decomposition:**

For positive definite matrix A:
A = LLᵀ

where L is lower triangular.

**Use cases:**
- Sampling from multivariate Gaussian
- Solving systems Ax = b efficiently
- Checking positive definiteness

**5. LU Decomposition:**

A = LU

where:
- L: lower triangular
- U: upper triangular

**Use case:** Efficient solution of multiple systems with same A

**Comparison Table:**

| Decomposition | Works For | Main Use |
|---------------|-----------|----------|
| Eigendecomposition | Square, symmetric | PCA, understanding transformations |
| SVD | Any matrix | Dimensionality reduction, compression |
| QR | Any matrix | Solving least squares |
| Cholesky | Positive definite | Sampling, fast linear solves |
| LU | Square | Solving multiple systems |

---

## Part 4: Generative Models

### Explain VAE (Variational Autoencoder)

**The Goal:**

Learn a generative model of data p(x) that can:
1. Generate new samples similar to training data
2. Learn meaningful latent representations
3. Handle uncertainty

**Architecture:**

**Encoder:** q(z|x) - Maps data x to latent code z
**Decoder:** p(x|z) - Reconstructs x from latent code z

**Key Idea - Variational Inference:**

We want to learn p(z|x), but it's intractable. Instead, learn an approximate distribution q(z|x) that's close to the true posterior.

**The VAE Loss (ELBO):**

Maximize Evidence Lower Bound:

ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

**Two terms:**

1. **Reconstruction loss**: E[log p(x|z)]
   - How well can we reconstruct x from z?
   - Typically: MSE for continuous data, BCE for binary

2. **KL divergence**: KL(q(z|x) || p(z))
   - How different is q(z|x) from prior p(z)?
   - Regularizes latent space
   - Typically: KL between Gaussian q(z|x) and standard normal p(z)

**Reparameterization Trick:**

Problem: Can't backpropagate through sampling z ~ q(z|x)

Solution: Reparameterize as:
- Sample ε ~ N(0, I)
- Compute z = μ(x) + σ(x) ⊙ ε

Now gradients flow through μ and σ!

**Intuition:**

- **Encoder** compresses x into mean μ and variance σ²
- **Sample** z from N(μ, σ²) using reparameterization
- **Decoder** reconstructs x from z
- **KL term** keeps latent space organized (prevents overfitting)

**Why the KL term matters:**

Without it, encoder could map each x to a unique z with zero variance → no generalization.

With it, similar inputs get mapped to overlapping regions in latent space → smooth interpolation.

**Training Process:**

1. Input image x
2. Encode to μ(x), σ(x)
3. Sample z = μ + σ ⊙ ε
4. Decode to x̂
5. Compute loss: ||x - x̂||² + KL(N(μ,σ²) || N(0,I))
6. Backpropagate, update weights

**Generating New Samples:**

1. Sample z ~ N(0, I) from prior
2. Decode z to get x = decoder(z)
3. Since latent space is continuous, can interpolate between points

**Applications:**

- Image generation
- Data compression
- Anomaly detection (high reconstruction error)
- Semi-supervised learning
- Disentangled representations

**VAE vs Autoencoder:**

| Feature | Autoencoder | VAE |
|---------|-------------|-----|
| Latent space | Deterministic | Probabilistic |
| Loss | Reconstruction only | Reconstruction + KL |
| Generation | No (irregular latent space) | Yes (smooth prior) |
| Interpolation | Poor | Smooth |

---

### Explain GAN (Generative Adversarial Network)

**The Core Idea:**

Two neural networks compete in a game:
- **Generator (G)**: Creates fake data
- **Discriminator (D)**: Distinguishes real from fake

**The Adversarial Game:**

- **D's goal**: Maximize ability to classify real vs fake
  - D(x_real) → 1
  - D(G(z)) → 0

- **G's goal**: Fool D by generating realistic data
  - D(G(z)) → 1

**Objective Function:**

min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

**Training Alternates:**

1. **Update D** (discriminator):
   - Sample real data x
   - Sample noise z, generate fake G(z)
   - Train D to maximize V (classify correctly)

2. **Update G** (generator):
   - Sample noise z
   - Train G to minimize V (fool D)

**Intuition - Counterfeiter vs Police:**

- Generator = counterfeiter making fake money
- Discriminator = police detecting fakes
- As police get better at detection, counterfeiter improves quality
- Eventually, counterfeits become indistinguishable from real money

**Training Dynamics:**

**Nash Equilibrium:**
When G generates perfect samples, D outputs 0.5 (can't tell real from fake).

At this point:
- D can't improve (samples are perfect)
- G can't improve (already generating real distribution)

**Practical Challenges:**

**1. Mode Collapse:**
- Generator produces limited variety (only a few types of outputs)
- **Solution**: Unrolled GAN, minibatch discrimination

**2. Training Instability:**
- D becomes too strong → G gets no gradient signal
- G becomes too strong → D gives up
- **Solution**: Careful tuning, Wasserstein GAN

**3. Vanishing Gradients:**
- When D is perfect, log(1-D(G(z))) saturates
- **Solution**: Use -log D(G(z)) instead (non-saturating loss)

**GAN Variants:**

**DCGAN (Deep Convolutional GAN):**
- Use conv layers instead of fully connected
- BatchNorm in G and D
- LeakyReLU in D, ReLU in G
- More stable training

**WGAN (Wasserstein GAN):**
- Replace JS divergence with Wasserstein distance
- More meaningful loss metric
- More stable training
- Clip weights or use gradient penalty

**Conditional GAN:**
- Condition on class label y
- G(z, y) and D(x, y)
- Can control what type of image to generate

**StyleGAN:**
- State-of-the-art image generation
- Progressive growing
- Style transfer at different resolutions
- Generates incredibly realistic faces

**Applications:**

- **Image generation**: High-quality synthetic images
- **Image-to-image translation**: Pix2Pix, CycleGAN (day→night, sketch→photo)
- **Super-resolution**: Upscale low-res images
- **Data augmentation**: Generate training data
- **Art and creativity**: Generate art, music, text

**VAE vs GAN:**

| Feature | VAE | GAN |
|---------|-----|-----|
| Training | Stable | Can be unstable |
| Sample quality | Blurry | Sharp, realistic |
| Latent space | Interpretable | Less interpretable |
| Diversity | Good | Risk of mode collapse |
| Likelihood | Tractable lower bound | Intractable |

**When to Use Each:**

- **VAE**: Need stable training, interpretable latent space, anomaly detection
- **GAN**: Need highest quality samples, image synthesis, don't care about exact likelihood

---

## Part 5: Advanced Topics

### What is Mixture of Experts (MoE)?

**Core Concept:**

Instead of one large model, use multiple specialized "expert" networks, and learn when to use each expert.

**Architecture:**

1. **Multiple expert networks**: E₁, E₂, ..., Eₙ
2. **Gating network**: Decides which experts to use
3. **Combiner**: Aggregates expert outputs

**How It Works:**

For input x:
1. Gating network outputs weights: g(x) = [g₁, g₂, ..., gₙ]
2. Each expert produces output: yᵢ = Eᵢ(x)
3. Final output: y = Σᵢ gᵢ(x) · yᵢ

**Gating Function:**

Typically softmax over expert scores:
g(x) = softmax(Wx)

This ensures weights sum to 1 and are non-negative.

**Types of MoE:**

**Soft MoE:**
- All experts contribute (weighted)
- Smooth, differentiable
- More computation (run all experts)

**Hard MoE (Sparse MoE):**
- Only activate top-k experts
- Efficient (fewer computations)
- Requires tricks for gradients (Gumbel-softmax, straight-through estimators)

**Example - Language Model:**

- Expert 1: Handles technical text
- Expert 2: Handles creative writing
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
L_aux = α · Σᵢ fᵢ · Pᵢ

where:
- fᵢ: fraction of inputs routed to expert i
- Pᵢ: average gating value for expert i
- Encourages: uniform fᵢ (balanced usage)

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
