# Optimizers & Regularization - Interview Q&A

Comprehensive coverage of optimization algorithms and regularization techniques.

---


## Table of Contents

- [[#Part 4: Optimizers]]
  - [[#Name different types of optimizers and their intuitions (e.g., Adam, RMSProp)]]
- [[#Part 5: Regularization & Overfitting]]
  - [[#What is the difference between L1 and L2 regularization? When would you use each?]]
  - [[#Explain overfitting and strategies to mitigate it]]

---

## Part 4: Optimizers

### Name different types of optimizers and their intuitions (e.g., Adam, RMSProp)

**1. Stochastic Gradient Descent (SGD)**

```python
import torch.optim as optim

# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov momentum
)
```

**Update Rule**:
```
θ = θ - η × ∇L(θ)
```

**With Momentum**:
```
v_t = β × v_{t-1} + ∇L(θ)
θ = θ - η × v_t
```

**Intuition**:
- Basic: Follow negative gradient direction
- Momentum: Accumulate velocity, smooth out oscillations
- Nesterov: "Look ahead" before computing gradient

**Pros**:
- Simple, well-understood
- Works well with momentum
- Good for convex problems

**Cons**:
- Same learning rate for all parameters
- Struggles with saddle points
- Sensitive to learning rate

**2. AdaGrad (Adaptive Gradient)**

```python
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    eps=1e-10
)
```

**Update Rule**:
```
G_t = G_{t-1} + (∇L(θ))²
θ = θ - (η / sqrt(G_t + ε)) × ∇L(θ)
```

**Intuition**:
- Adapt learning rate for each parameter
- Larger updates for infrequent parameters
- Smaller updates for frequent parameters

**Pros**:
- Good for sparse data
- No manual learning rate tuning

**Cons**:
- Learning rate decays too aggressively
- Can stop learning too early

**3. RMSProp (Root Mean Square Propagation)**

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,  # Decay rate
    eps=1e-8
)
```

**Update Rule**:
```
E[g²]_t = α × E[g²]_{t-1} + (1-α) × (∇L(θ))²
θ = θ - (η / sqrt(E[g²]_t + ε)) × ∇L(θ)
```

**Intuition**:
- Fixes AdaGrad's aggressive decay
- Uses exponential moving average of squared gradients
- Adapts learning rate per parameter

**Pros**:
- Fixes AdaGrad's decay problem
- Works well with RNNs
- Good for non-stationary objectives

**Cons**:
- Still requires manual learning rate
- Can oscillate

**4. Adam (Adaptive Moment Estimation)**

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # β₁, β₂
    eps=1e-8,
    weight_decay=0.0
)
```

**Update Rule**:
```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L(θ)        # 1st moment (mean)
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L(θ))²     # 2nd moment (variance)

m̂_t = m_t / (1 - β₁^t)  # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ = θ - η × m̂_t / (sqrt(v̂_t) + ε)
```

**Intuition**:
- Combines momentum (1st moment) and RMSProp (2nd moment)
- Bias correction for initial estimates
- Adaptive learning rates

**Pros**:
- Works well in practice
- Little hyperparameter tuning needed
- Handles sparse gradients well
- Fast convergence

**Cons**:
- Can have poor generalization vs SGD
- May not converge to optimal solution
- Memory intensive (stores two moments)

**5. AdamW (Adam with Weight Decay)**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Decoupled weight decay
)
```

**Key Difference from Adam**:
- Decouples weight decay from gradient update
- More effective regularization
- Better generalization

**Update Rule**:
```
θ = θ - η × (m̂_t / (sqrt(v̂_t) + ε) + λ × θ)
```

**When to use**: Transformer models, modern deep learning (now preferred over Adam)

**6. RAdam (Rectified Adam)**

```python
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)
```

**Intuition**:
- Fixes Adam's warmup problem
- Automatic warmup based on variance
- More robust to learning rate

**7. Lookahead**

```python
from torch_optimizer import Lookahead

# Wrap another optimizer
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

**Intuition**:
- Maintains two sets of weights: fast and slow
- Fast weights explore, slow weights maintain stability
- Improves convergence and generalization

**8. LAMB (Layer-wise Adaptive Moments for Batch training)**

```python
from torch_optimizer import Lamb

optimizer = Lamb(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

**Intuition**:
- Designed for large batch training
- Layer-wise adaptation
- Used in BERT training

**Optimizer Comparison:**

| Optimizer | Learning Rate Needed | Memory | Speed | Generalization | Best For |
|-----------|---------------------|---------|-------|----------------|----------|
| SGD | Yes | Low | Fast | Excellent | Simple problems, CV |
| SGD + Momentum | Yes | Low | Fast | Excellent | CV, proven architectures |
| AdaGrad | Less | Medium | Medium | Good | Sparse data, NLP |
| RMSProp | Yes | Medium | Fast | Good | RNNs, non-stationary |
| Adam | Less | High | Fast | Good | General purpose, RL |
| AdamW | Less | High | Fast | Better | Transformers, modern DL |
| RAdam | Less | High | Fast | Better | Unstable training |
| LAMB | Less | High | Fast | Good | Large batch training |

**Practical Guidelines:**

```python
# Computer Vision
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# NLP / Transformers
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Reinforcement Learning
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,
    eps=1e-5
)

# Large Batch Training
optimizer = Lamb(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

**Learning Rate Scheduling:**

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts
)

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# One-cycle policy (recommended for SGD)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=epochs * len(train_loader)
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
    scheduler.step()
```


---


## Part 5: Regularization & Overfitting

### What is the difference between L1 and L2 regularization? When would you use each?

**L2 Regularization (Ridge)**

**Formula**: Loss_total = Loss_original + λ × ∑ᵢ wᵢ²

```python
import torch.nn as nn

# In PyTorch, use weight_decay parameter
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)

# Or add manually to loss
l2_lambda = 0.01
l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
loss = loss_original + l2_lambda * l2_reg
```

**Effects**:
- Penalizes large weights quadratically
- Weights shrink but rarely become exactly zero
- Spreads weight across all features
- Differentiable everywhere

**When to use**:
- Have many relevant features
- Want to reduce multicollinearity
- Prefer smooth, distributed weights
- Default choice for deep learning

**L1 Regularization (Lasso)**

**Formula**: Loss_total = Loss_original + λ × ∑ᵢ |wᵢ|

```python
# L1 not directly supported in PyTorch optimizers
# Add manually to loss
l1_lambda = 0.01
l1_reg = sum(param.abs().sum() for param in model.parameters())
loss = loss_original + l1_lambda * l1_reg
```

**Effects**:
- Penalizes weights linearly
- Drives weights to exactly zero (sparsity)
- Performs feature selection
- Not differentiable at zero

**When to use**:
- Have many irrelevant features
- Want automatic feature selection
- Prefer sparse, interpretable models
- Feature engineering/selection

**Elastic Net (Combine L1 and L2)**

```python
# Combine L1 and L2
alpha = 0.01  # Overall regularization strength
l1_ratio = 0.5  # Balance between L1 and L2

l1_reg = sum(param.abs().sum() for param in model.parameters())
l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())

reg_loss = alpha * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
loss = loss_original + reg_loss
```

**Comparison**:

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|------------|------------|-------------|
| Penalty | ∑\|w\| | ∑w² | α(ρ∑\|w\| + (1-ρ)∑w²) |
| Sparsity | Yes | No | Moderate |
| Feature Selection | Yes | No | Yes |
| Differentiable | No (at 0) | Yes | Mixed |
| Computational | Slower | Faster | Medium |
| Multi-collinearity | Picks one | Handles well | Handles well |

**Sklearn Example**:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# L2 (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# L1 (Lasso)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# Compare coefficients
print(f"Ridge non-zero: {np.sum(ridge.coef_ != 0)}")
print(f"Lasso non-zero: {np.sum(lasso.coef_ != 0)}")
```

---

### Explain overfitting and strategies to mitigate it

**What is Overfitting?**

Model learns training data too well, including noise and outliers, leading to poor generalization on unseen data.

**Symptoms**:
- High training accuracy, low validation accuracy
- Large gap between train and validation loss
- Model memorizes rather than learns patterns

**Detection**:

```python
import matplotlib.pyplot as plt

# Plot training and validation curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.show()

# If val_loss increases while train_loss decreases → overfitting
```

**Mitigation Strategies:**

**1. More Training Data**

```python
# Data augmentation (for images)
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Synthetic data generation
# Use GANs, SMOTE, or other techniques
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**2. Regularization (L1, L2)**

```python
# L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# L1 regularization
l1_lambda = 0.01
l1_reg = sum(param.abs().sum() for param in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_reg
```

**3. Dropout**

```python
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.3)  # Drop 30% of neurons
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Only active during training
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Model automatically handles train/eval mode
model.train()  # Dropout active
model.eval()   # Dropout inactive
```

**4. Early Stopping**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

**5. Simpler Model Architecture**

```python
# Complex model (prone to overfitting)
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 10)

# Simpler model (better generalization)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 10)
```

**6. Batch Normalization**

```python
class NetworkWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

**7. Cross-Validation**

```python
from sklearn.model_selection import KFold, cross_val_score

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    # Train model on this fold
    model = create_model()
    train(model, X_train_fold, y_train_fold)
    score = evaluate(model, X_val_fold, y_val_fold)
    print(f"Fold {fold+1} score: {score:.4f}")
```

**8. Ensemble Methods**

```python
# Bagging (Bootstrap Aggregating)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

# Random Forest (built-in bagging + feature randomness)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Boosting
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
ada = AdaBoostClassifier(n_estimators=50)
```

**Complete Overfitting Prevention Pipeline:**

```python
class AntiOverfitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        # Simpler architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Training with regularization and early stopping
model = AntiOverfitModel(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 reg
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
early_stopping = EarlyStopping(patience=10)

for epoch in range(100):
    # Training with data augmentation
    train_loss = train_with_augmentation(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        break
```

**Summary of Strategies:**

| Strategy | Effectiveness | Computational Cost | When to Use |
|----------|---------------|-------------------|-------------|
| More Data | High | High | Always preferred |
| Data Augmentation | High | Low-Medium | Images, text |
| L2 Regularization | Medium | Low | Default choice |
| L1 Regularization | Medium | Low | Feature selection |
| Dropout | High | Low | Deep networks |
| Early Stopping | High | None | Always use |
| Simpler Model | High | Negative | Small datasets |
| Batch Normalization | Medium | Low | Deep networks |
| Cross-Validation | High | High | Model selection |
| Ensembles | Very High | High | Production models |

