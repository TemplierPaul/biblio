# Mathematical Glossary for Machine Learning

A quick reference for classic mathematical terms and definitions used in ML.

## Calculus & Optimization

*   **Gradient ($\nabla f$)**: A vector of partial derivatives pointing in the direction of steepest ascent. If $f(x_1, \dots, x_n)$ is a scalar field, $\nabla f = [\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}]^T$.
*   **Jacobian Matrix ($J$)**: A matrix of all first-order partial derivatives of a vector-valued function. If $f: \mathbb{R}^n \to \mathbb{R}^m$, then $J$ is an $m \times n$ matrix where $J_{ij} = \frac{\partial f_i}{\partial x_j}$. It represents the linear approximation of the function at a point.
*   **Hessian Matrix ($H$)**: A square matrix of second-order partial derivatives of a scalar function. $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$. It describes the local curvature of the function.
    *   **Positive Definite**: All eigenvalues $> 0$ (Local Minimum).
    *   **Negative Definite**: All eigenvalues $< 0$ (Local Maximum).
    *   **Indefinite**: Mixed eigenvalues (Saddle Point).
*   **Laplacian ($\Delta f$)**: The divergence of the gradient, $\nabla \cdot \nabla f$. It is the sum of unmixed second partial derivatives: $\sum \frac{\partial^2 f}{\partial x_i^2}$. Useful in edge detection and physics.
*   **Lagrange Multipliers**: A strategy for finding the local maxima/minima of a function subject to equality constraints using a Lagrangian function $\mathcal{L} = f(x) - \lambda g(x)$.

## Linear Algebra

*   **Eigenvector**: A non-zero vector $v$ that changes only by a scalar factor when a linear transformation $A$ is applied to it ($Av = \lambda v$).
*   **Eigenvalue ($\lambda$)**: The scalar factor by which the eigenvector is scaled.
*   **Determinant ($\det(A)$)**: A scalar value that describes the scaling factor of the linear transformation. If $\det(A) = 0$, the matrix is singular (non-invertible) and collapses space into lower dimensions.
*   **Trace ($\text{tr}(A)$)**: The sum of the diagonal elements of a matrix. It is also the sum of the eigenvalues.
*   **Rank**: The maximum number of linearly independent column vectors (or row vectors) in a matrix. It represents the dimension of the image of the linear transformation.
*   **Singular Value Decomposition (SVD)**: A factorization of any matrix $A = U\Sigma V^T$, where $U$ and $V$ are orthogonal matrices and $\Sigma$ is a diagonal matrix of singular values. Fundamental for PCA and dimensionality reduction.
*   **Positive Definite Matrix**: A symmetric matrix $M$ where $x^T M x > 0$ for all non-zero vectors $x$. Crucial for optimization (convexity) and covariance matrices.
*   **Orthogonal Matrix**: A square matrix $Q$ whose columns and rows are orthonormal vectors ($Q^T Q = Q Q^T = I$). Rotations and reflections are orthogonal transformations.

## Probability & Information Theory

*   **Prior $P(\theta)$**: The probability distribution representing knowledge about a quantity before observing data.
*   **Posterior $P(\theta|X)$**: The conditional probability distribution of the quantity after observing data $X$.
*   **Likelihood $P(X|\theta)$**: The probability of the data $X$ given parameters $\theta$.
*   **Marginal Probability $P(X)$**: The probability of an event irrespective of the outcome of another variable (summing/integrating out the other variable).
*   **Entropy ($H$)**: A measure of the average uncertainty or "information" content in a random variable. High entropy = high unpredictability.
*   **Cross-Entropy**: A measure of the difference between two probability distributions. Commonly used as a loss function in classification. $H(p, q) = -\sum p(x) \log q(x)$.
*   **KL Divergence ($D_{KL}$)**: A non-symmetric measure of how one probability distribution $P$ diverges from a second expected probability distribution $Q$. "The expected amount of extra information required to encode samples from P using a code optimized for Q."
*   **Evidence Lower Bound (ELBO)**: A lower bound on the log-likelihood of observed data, maximized in Variational Inference (e.g., VAEs) to approximate the posterior.

## Machine Learning Concepts

*   **Bias**: The error introduced by approximating a real-world problem with a simplified model. High bias $\to$ Underfitting.
*   **Variance**: The error introduced by sensitivity to small fluctuations in the training set. High variance $\to$ Overfitting.
*   **Manifold Hypothesis**: The assumption that real-world high-dimensional data (like images) actually lies on a lower-dimensional manifold embedded within that space.
*   **Convex Function**: A function where a line segment connecting any two points on the graph lies above or on the graph. Ensures a single global minimum.
*   **I.I.D. (Independent and Identically Distributed)**: A common assumption that training examples are independent of each other and drawn from the same probability distribution.
