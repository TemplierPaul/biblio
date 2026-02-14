# ML Systems & Scalability - Interview Q&A

Comprehensive coverage of ML system design, scalability, deployment, and production best practices.

---


## Table of Contents

- [[#Part 1: Scalable ML Systems]]
  - [[#How would you scale an ML model to handle billions of queries per second?]]
- [[#Part 2: Recommendation Systems]]
  - [[#Design a scalable recommendation system for YouTube videos]]

---

## Part 1: Scalable ML Systems

### How would you scale an ML model to handle billions of queries per second?

**Architecture Components:**

**1. Model Optimization**

```python
# Quantization: Reduce model size and latency
import torch
import torch.quantization as quantization

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Post-training dynamic quantization
quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# Compare sizes
import os
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1e6  # MB
    os.remove("temp.pth")
    return size

print(f"Original size: {get_model_size(model):.2f} MB")
print(f"Quantized size: {get_model_size(quantized_model):.2f} MB")

# Measure inference time
import time

x = torch.randn(1, 100)

start = time.time()
for _ in range(1000):
    _ = model(x)
original_time = time.time() - start

start = time.time()
for _ in range(1000):
    _ = quantized_model(x)
quantized_time = time.time() - start

print(f"Original inference: {original_time:.4f}s")
print(f"Quantized inference: {quantized_time:.4f}s")
print(f"Speedup: {original_time/quantized_time:.2f}x")
```

**Knowledge Distillation:**

```python
import torch.nn as nn
import torch.nn.functional as F

class TeacherModel(nn.Module):
    """Large, accurate model"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StudentModel(nn.Module):
    """Small, fast model"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Combine soft targets from teacher with hard labels
    """
    # Soft targets (knowledge from teacher)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
    distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)

    # Hard targets (ground truth)
    student_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * distillation_loss + (1 - alpha) * student_loss

# Training loop
teacher = TeacherModel()  # Pre-trained
student = StudentModel()
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

teacher.eval()  # Teacher in eval mode
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(batch_x)

        student_logits = student(batch_x)

        loss = distillation_loss(student_logits, teacher_logits, batch_y)
        loss.backward()
        optimizer.step()
```

**2. Distributed Serving Architecture**

```python
# FastAPI with multiple workers
from fastapi import FastAPI
import uvicorn
import torch
import numpy as np

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load("model_traced.pt")
    model.eval()

@app.post("/predict")
async def predict(data: dict):
    """
    Single prediction endpoint
    """
    features = np.array(data['features'])
    input_tensor = torch.FloatTensor(features)

    with torch.no_grad():
        prediction = model(input_tensor)

    return {"prediction": prediction.tolist()}

@app.post("/predict_batch")
async def predict_batch(data: dict):
    """
    Batch prediction for efficiency
    """
    features = np.array(data['features'])
    input_tensor = torch.FloatTensor(features)

    with torch.no_grad():
        predictions = model(input_tensor)

    return {"predictions": predictions.tolist()}

# Run with multiple workers
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Deployment configuration
"""
# docker-compose.yml
version: '3.8'
services:
  ml-api:
    image: ml-model:latest
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
      - MODEL_PATH=/models/model.pt
    deploy:
      replicas: 10  # 10 instances
      resources:
        limits:
          cpus: '2'
          memory: 4G
    volumes:
      - ./models:/models

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    depends_on:
      - ml-api
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
"""
```

**Nginx Load Balancer Configuration:**

```nginx
# nginx.conf
upstream ml_backend {
    least_conn;  # Load balance based on connections
    server ml-api-1:8000;
    server ml-api-2:8000;
    server ml-api-3:8000;
    # ... more servers
}

server {
    listen 80;

    location /predict {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

**3. Caching Strategy**

```python
from functools import lru_cache
import redis
import hashlib
import pickle

class ModelCache:
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.ttl = ttl  # Cache TTL in seconds

    def _hash_input(self, features):
        """Create hash of input features"""
        return hashlib.md5(str(features).encode()).hexdigest()

    def get_prediction(self, features, model_fn):
        """Get prediction with caching"""
        cache_key = self._hash_input(features)

        # Try to get from cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)

        # Compute prediction
        prediction = model_fn(features)

        # Store in cache
        self.redis_client.setex(
            cache_key,
            self.ttl,
            pickle.dumps(prediction)
        )

        return prediction

# Usage
cache = ModelCache()

def predict_with_model(features):
    input_tensor = torch.FloatTensor(features)
    with torch.no_grad():
        return model(input_tensor).numpy()

# Get prediction (cached if available)
features = [0.1, 0.2, 0.3, ...]
prediction = cache.get_prediction(features, predict_with_model)

# In-memory cache for hot data
@lru_cache(maxsize=10000)
def predict_cached(features_tuple):
    """LRU cache for most frequent predictions"""
    features = list(features_tuple)
    return predict_with_model(features)
```

**4. Model Serving with TensorFlow Serving**

```python
# Export model for TF Serving
import tensorflow as tf

# Save model in SavedModel format
tf.saved_model.save(model, "models/my_model/1")

# Docker deployment
"""
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models \
  tensorflow/serving \
  --model_config_file=/models/models.config
"""

# Client code
import requests
import json

def predict_tf_serving(features):
    url = "http://localhost:8501/v1/models/my_model:predict"
    data = json.dumps({"instances": features.tolist()})
    headers = {"content-type": "application/json"}

    response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(response.text)['predictions']
    return predictions
```

**5. Horizontal Scaling with Kubernetes**

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 50  # Scale to 50 pods
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-api
        image: ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**6. Request Batching for Throughput**

```python
import asyncio
from collections import deque
import time

class BatchPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.futures = []

    async def predict(self, features):
        """Add request to batch and wait for result"""
        future = asyncio.Future()
        self.queue.append((features, future))

        # Start batch processing if not already running
        if len(self.queue) == 1:
            asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        await asyncio.sleep(self.max_wait_time)

        if not self.queue:
            return

        # Collect batch
        batch = []
        futures = []

        while self.queue and len(batch) < self.max_batch_size:
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)

        # Run batch inference
        batch_tensor = torch.FloatTensor(batch)
        with torch.no_grad():
            predictions = self.model(batch_tensor)

        # Return results to individual requests
        for i, future in enumerate(futures):
            future.set_result(predictions[i].numpy())

# Usage
batch_predictor = BatchPredictor(model)

@app.post("/predict_async")
async def predict_async(data: dict):
    features = data['features']
    prediction = await batch_predictor.predict(features)
    return {"prediction": prediction.tolist()}
```

**Complete Scalable Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                     CDN / Edge Caching                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│   API Gateway    │                  │   API Gateway    │
│   (Rate Limit)   │                  │   (Rate Limit)   │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│  Redis Cache     │←─────────────────→│  Redis Cache     │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│  Model Service 1 │                  │  Model Service 2 │
│  (Quantized)     │                  │  (Quantized)     │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────────────────────────────────────────┐
│              GPU Cluster (TensorRT)                   │
└──────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────┐
│          Monitoring (Prometheus/Grafana)              │
└──────────────────────────────────────────────────────┘
```

**Scaling Checklist:**
- ✅ Model optimization (quantization, distillation, pruning)
- ✅ Horizontal scaling (multiple instances, load balancing)
- ✅ Caching (Redis, CDN, in-memory)
- ✅ Request batching (increase throughput)
- ✅ Auto-scaling (based on CPU/memory/QPS)
- ✅ Model versioning and A/B testing
- ✅ Monitoring and alerting
- ✅ Database optimization (if feature lookup needed)

---

## Part 2: Recommendation Systems

### Design a scalable recommendation system for YouTube videos

**System Requirements:**
- Billions of users, millions of videos
- Real-time recommendations
- Personalized for each user
- Must handle cold-start problem
- Balance exploration vs exploitation

**Architecture:**

**1. Candidate Generation**

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class MatrixFactorization:
    """Collaborative filtering for candidate generation"""

    def __init__(self, n_factors=100, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization

    def fit(self, user_item_matrix, n_epochs=20):
        """
        Factor user-item matrix into user and item embeddings
        """
        n_users, n_items = user_item_matrix.shape

        # Initialize embeddings
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # SGD training
        for epoch in range(n_epochs):
            for u in range(n_users):
                for i in range(n_items):
                    if user_item_matrix[u, i] > 0:
                        # Compute error
                        pred = self.predict_single(u, i)
                        error = user_item_matrix[u, i] - pred

                        # Update factors
                        self.user_factors[u] += self.lr * (
                            error * self.item_factors[i] - self.reg * self.user_factors[u]
                        )
                        self.item_factors[i] += self.lr * (
                            error * self.user_factors[u] - self.reg * self.item_factors[i]
                        )

    def predict_single(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])

    def recommend(self, user_id, n=10, exclude_seen=None):
        """Get top-N recommendations for user"""
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        if exclude_seen is not None:
            scores[exclude_seen] = -np.inf

        top_items = np.argsort(scores)[::-1][:n]
        return top_items, scores[top_items]

# Two-tower neural network for learning embeddings
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128):
        super().__init__()

        # User tower
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Item tower
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        user_vec = self.user_fc(user_emb)

        item_emb = self.item_embedding(item_ids)
        item_vec = self.item_fc(item_emb)

        # Dot product for similarity
        return (user_vec * item_vec).sum(dim=1)

    def get_user_embedding(self, user_id):
        user_emb = self.user_embedding(user_id)
        return self.user_fc(user_emb)

    def get_item_embedding(self, item_id):
        item_emb = self.item_embedding(item_id)
        return self.item_fc(item_emb)

# Training
model = TwoTowerModel(n_users=1000000, n_items=1000000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    for user_ids, item_ids, labels in train_loader:
        optimizer.zero_grad()
        scores = model(user_ids, item_ids)
        loss = criterion(scores, labels.float())
        loss.backward()
        optimizer.step()
```

**2. Fast Candidate Retrieval with ANN**

```python
# Use FAISS for fast approximate nearest neighbor search
import faiss

class FAISSIndex:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # Use IVF (Inverted File) with PQ (Product Quantization) for billions of vectors
        quantizer = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            embedding_dim,
            nlist=1000,  # Number of clusters
            M=8,  # Number of subquantizers
            nbits=8  # Bits per subquantizer
        )

    def train_and_add(self, item_embeddings):
        """Train index on item embeddings"""
        # Train on subset
        self.index.train(item_embeddings[:100000])

        # Add all embeddings
        self.index.add(item_embeddings)

    def search(self, query_embedding, k=100):
        """Find k nearest neighbors"""
        distances, indices = self.index.search(query_embedding, k)
        return indices[0], distances[0]

# Build index
item_embeddings = model.item_fc(model.item_embedding.weight).detach().numpy()
faiss_index = FAISSIndex(embedding_dim=64)
faiss_index.train_and_add(item_embeddings)

# Fast retrieval for user
user_id = 12345
user_emb = model.get_user_embedding(torch.tensor([user_id])).detach().numpy()
candidate_items, scores = faiss_index.search(user_emb, k=500)
```

**3. Ranking Model**

```python
class RankingModel(nn.Module):
    """Deep ranking model with rich features"""

    def __init__(self, n_users, n_items, n_categories):
        super().__init__()

        # Embeddings
        self.user_emb = nn.Embedding(n_users, 64)
        self.item_emb = nn.Embedding(n_items, 64)
        self.category_emb = nn.Embedding(n_categories, 32)

        # Feature dimension calculation
        # User emb (64) + Item emb (64) + Category emb (32)
        # + User features (10) + Item features (20) + Context features (15)
        feature_dim = 64 + 64 + 32 + 10 + 20 + 15

        # Deep network
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_ids, item_ids, categories, user_features, item_features, context_features):
        # Get embeddings
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)
        cat_emb = self.category_emb(categories)

        # Concatenate all features
        combined = torch.cat([
            user_emb,
            item_emb,
            cat_emb,
            user_features,      # e.g., age, country, viewing history stats
            item_features,      # e.g., video length, upload date, popularity
            context_features    # e.g., time of day, device, location
        ], dim=1)

        # Predict engagement probability
        return self.fc(combined).squeeze()

# Features to include:
"""
User features:
- Demographics (age, gender, location)
- Viewing history (categories watched, watch time)
- Engagement patterns (like rate, share rate)
- Device type

Item features:
- Video metadata (title, description embeddings)
- Category, tags
- Upload date, video length
- Popularity metrics (views, likes, comments)
- Creator info

Context features:
- Time of day, day of week
- User's current session behavior
- Trending topics
"""
```

**4. Complete Recommendation Pipeline**

```python
class YouTubeRecommender:
    def __init__(self):
        self.candidate_model = TwoTowerModel(...)
        self.ranking_model = RankingModel(...)
        self.faiss_index = FAISSIndex(...)

    def recommend(self, user_id, n_recommendations=10, context=None):
        """
        Two-stage recommendation: candidate generation + ranking
        """

        # Stage 1: Candidate Generation (100-1000 candidates)
        user_emb = self.candidate_model.get_user_embedding(
            torch.tensor([user_id])
        ).detach().numpy()

        candidate_items, _ = self.faiss_index.search(user_emb, k=500)

        # Stage 2: Ranking
        # Prepare features for all candidates
        user_features = self.get_user_features(user_id)
        item_features = self.get_item_features(candidate_items)
        context_features = self.get_context_features(user_id, context)

        # Score all candidates
        with torch.no_grad():
            scores = self.ranking_model(
                torch.tensor([user_id] * len(candidate_items)),
                torch.tensor(candidate_items),
                item_features['categories'],
                user_features,
                item_features['features'],
                context_features
            )

        # Get top-N
        top_indices = torch.argsort(scores, descending=True)[:n_recommendations]
        recommended_items = candidate_items[top_indices]

        return recommended_items, scores[top_indices]

    def get_user_features(self, user_id):
        # Fetch from user database/cache
        pass

    def get_item_features(self, item_ids):
        # Fetch from item database/cache
        pass

    def get_context_features(self, user_id, context):
        # Current time, device, etc.
        pass
```

**5. Handling Cold Start**

```python
class ColdStartHandler:
    """Handle new users and new items"""

    def recommend_for_new_user(self, user_demographics=None):
        """Use content-based and popularity for new users"""
        recommendations = []

        # 1. Popular items (trending)
        popular = self.get_trending_videos(n=5)
        recommendations.extend(popular)

        # 2. Demographic-based recommendations
        if user_demographics:
            demo_recs = self.get_demographic_recommendations(
                user_demographics, n=5
            )
            recommendations.extend(demo_recs)

        # 3. Diverse categories (exploration)
        diverse = self.get_diverse_categories(n=5, exclude=recommendations)
        recommendations.extend(diverse)

        return recommendations

    def recommend_new_item(self, item_id, user_id):
        """Score new item using content features"""
        # Use item metadata (title, description, category)
        item_features = self.get_item_content_features(item_id)

        # Find similar items
        similar_items = self.find_similar_by_content(item_features, n=100)

        # Use collaborative filtering on similar items
        # Assume user's preferences transfer to similar content
        return similar_items

    def get_trending_videos(self, n=10, time_window='24h'):
        """Get currently popular videos"""
        # Query from real-time analytics
        # Consider: views, likes, shares in last 24h
        pass
```

**6. Online Learning & A/B Testing**

```python
class OnlineLearner:
    """Continuously update model based on user feedback"""

    def __init__(self, model, buffer_size=10000):
        self.model = model
        self.experience_buffer = []
        self.buffer_size = buffer_size

    def log_interaction(self, user_id, item_id, engagement):
        """Log user interaction"""
        self.experience_buffer.append({
            'user_id': user_id,
            'item_id': item_id,
            'engagement': engagement,  # click, watch_time, like, etc.
            'timestamp': time.time()
        })

        # Update model when buffer is full
        if len(self.experience_buffer) >= self.buffer_size:
            self.update_model()

    def update_model(self):
        """Incremental model update"""
        # Create mini-batch from buffer
        batch = self.experience_buffer[-self.buffer_size:]

        # Train for one epoch
        # ... training code ...

        # Clear old experiences
        self.experience_buffer = self.experience_buffer[-self.buffer_size//2:]

# Multi-armed bandit for exploration
class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select(self, recommendations, exploration_pool):
        """Mix recommended items with exploration"""
        n_recommendations = len(recommendations)
        n_explore = int(n_recommendations * self.epsilon)
        n_exploit = n_recommendations - n_explore

        # Exploit: use top recommendations
        final_recs = recommendations[:n_exploit]

        # Explore: random items from exploration pool
        explore_items = np.random.choice(
            exploration_pool,
            size=n_explore,
            replace=False
        )
        final_recs.extend(explore_items)

        # Shuffle to mix exploration and exploitation
        np.random.shuffle(final_recs)
        return final_recs
```

**System Design Summary:**

```
User Request
     ↓
┌─────────────────────┐
│  Request Handler     │
│  (User ID, Context)  │
└─────────────────────┘
     ↓
┌─────────────────────┐
│  Feature Store       │ ← Real-time features
│  (User/Item/Context) │
└─────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Candidate Generation                │
│  - Two-tower model                   │
│  - FAISS for ANN search              │
│  - Generates 500 candidates          │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Ranking Model                       │
│  - Deep NN with rich features        │
│  - Scores all candidates             │
│  - Outputs top-10                    │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Re-ranking & Business Logic         │
│  - Diversity                         │
│  - Freshness                         │
│  - Exploration (epsilon-greedy)      │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Return Recommendations              │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Log Interactions for Online Learning│
└─────────────────────────────────────┘
```

**Key Metrics:**
- **Precision@K**: Fraction of recommended items that are engaged with
- **Recall@K**: Fraction of engaging items that are recommended
- **NDCG (Normalized Discounted Cumulative Gain)**: Quality of ranking
- **Diversity**: Avoid filter bubbles
- **Serendipity**: Recommend unexpected but relevant items
- **Business metrics**: Watch time, CTR, user retention

---

(Continued in final part on deployment best practices and monitoring...)
