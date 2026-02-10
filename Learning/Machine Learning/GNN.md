# Graph Neural Networks (GNN)

## Definition
Graph Neural Networks are neural architectures designed to operate on graph-structured data, learning representations by aggregating information from a node's neighbors through message passing.

## Graph Structure
**Graph**: $G = (V, E)$
- $V$: Nodes/vertices (entities)
- $E$: Edges (relationships)
- Node features: $x_v \in \mathbb{R}^d$
- Edge features: $e_{uv} \in \mathbb{R}^{d_e}$ (optional)

**Examples**:
- Social networks (users = nodes, friendships = edges)
- Molecules (atoms = nodes, bonds = edges)
- Citation networks (papers = nodes, citations = edges)
- Road networks (intersections = nodes, roads = edges)

## Core Idea: Message Passing

### Basic Framework
Each layer updates node representations by:
1. **Aggregate** messages from neighbors
2. **Combine** aggregated message with own features
3. **Update** node representation

**General form**:
$$h_v^{(k+1)} = \text{UPDATE}^{(k)}\left(h_v^{(k)}, \text{AGGREGATE}^{(k)}\left(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)\right)$$

Where:
- $h_v^{(k)}$: Node $v$'s representation at layer $k$
- $\mathcal{N}(v)$: Neighbors of node $v$
- Initial: $h_v^{(0)} = x_v$ (input features)

## Major GNN Architectures

### 1. GCN (Graph Convolutional Network)
**Update rule**:
$$h_v^{(k+1)} = \sigma\left(W^{(k)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{h_u^{(k)}}{\sqrt{|\mathcal{N}(u)| \cdot |\mathcal{N}(v)|}}\right)$$

**Characteristics**:
- Symmetric normalization (by degree)
- Includes self-loops ($v$ in its own neighborhood)
- Simple, effective baseline
- **Limitation**: All neighbors weighted equally (no attention)

### 2. GraphSAGE (Sample and Aggregate)
**Update rule**:
$$h_v^{(k+1)} = \sigma\left(W^{(k)} \cdot \text{CONCAT}\left(h_v^{(k)}, \text{AGG}\left(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)\right)\right)$$

**Aggregators**:
- **Mean**: $\text{AGG} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k)}$
- **Pool**: $\text{AGG} = \max(\{\sigma(W_{pool} h_u^{(k)}) : u \in \mathcal{N}(v)\})$
- **LSTM**: Apply LSTM to random permutation of neighbors

**Key Innovation**:
- **Sampling**: Sample fixed-size neighborhood (e.g., 10-25 neighbors)
- Enables mini-batch training on large graphs
- Inductive: can generalize to unseen nodes

### 3. GAT (Graph Attention Network)
**Update rule**:
$$h_v^{(k+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W^{(k)} h_u^{(k)}\right)$$

**Attention weights**:
$$\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{u' \in \mathcal{N}(v)} \exp(e_{vu'})}$$
$$e_{vu} = \text{LeakyReLU}(a^T [W h_v \| W h_u])$$

**Key Innovation**:
- **Learned attention**: Different neighbors have different importance
- **Multi-head attention**: Like Transformers (8-16 heads)
- No need for degree normalization

### 4. GIN (Graph Isomorphism Network)
**Update rule**:
$$h_v^{(k+1)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot h_v^{(k)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k)}\right)$$

**Key Property**:
- **Maximally expressive**: As powerful as Weisfeiler-Lehman graph isomorphism test
- Provably more powerful than GCN/GraphSAGE
- $\epsilon$: Learnable or fixed parameter

## Tasks

### 1. Node-Level Tasks
**Goal**: Predict property of each node
- **Examples**: Node classification (user interests), node regression
- **Output**: Use final node embeddings $h_v^{(K)}$
- **Loss**: Cross-entropy for classification, MSE for regression

### 2. Edge-Level Tasks
**Goal**: Predict property of edge or existence of edge
- **Examples**: Link prediction, relation classification
- **Output**: Combine node embeddings: $e_{uv} = f(h_u^{(K)}, h_v^{(K)})$
- **Methods**: Dot product, concatenation + MLP, cosine similarity

### 3. Graph-Level Tasks
**Goal**: Predict property of entire graph
- **Examples**: Molecule property prediction, graph classification
- **Output**: Aggregate all node embeddings
- **Readout functions**:
  - **Sum**: $h_G = \sum_{v \in V} h_v^{(K)}$
  - **Mean**: $h_G = \frac{1}{|V|} \sum_{v \in V} h_v^{(K)}$
  - **Max**: $h_G = \max_{v \in V} h_v^{(K)}$
  - **Attention-based**: Weighted sum with learned attention

## Training Considerations

### Over-smoothing
**Problem**: After many layers, all node representations converge to same value
- Nodes become indistinguishable
- Information from distant nodes dilutes local structure

**Solutions**:
- **Fewer layers**: Typically 2-4 layers (vs 50+ for CNNs)
- **Residual connections**: $h_v^{(k+1)} = h_v^{(k+1)} + h_v^{(k)}$
- **Jumping knowledge**: Combine representations from all layers

### Scalability
**Problem**: Message passing requires access to entire graph
- Full-batch training doesn't scale to millions of nodes

**Solutions**:
- **Sampling** (GraphSAGE): Sample fixed-size neighborhoods
- **Cluster-GCN**: Partition graph into clusters, train on subgraphs
- **Mini-batching**: Sample subgraphs for each batch

### Graph Pooling
**For graph-level tasks**: Need to downsample graph
- **Hierarchical pooling**: DiffPool, TopK pooling
- Analogous to pooling in CNNs
- Creates coarser graph representations

## Applications

### Drug Discovery
- **Nodes**: Atoms
- **Edges**: Chemical bonds
- **Task**: Predict molecular properties (toxicity, binding affinity)
- **Models**: GIN, GCN, GAT

### Social Networks
- **Nodes**: Users
- **Edges**: Friendships/interactions
- **Task**: Recommend friends, detect communities, predict user behavior
- **Models**: GraphSAGE (inductive), GCN

### Knowledge Graphs
- **Nodes**: Entities (people, places, concepts)
- **Edges**: Relations (typed edges)
- **Task**: Link prediction, entity classification, reasoning
- **Models**: R-GCN (relation-aware), CompGCN

### Recommendation Systems
- **Nodes**: Users + Items
- **Edges**: User-item interactions
- **Task**: Predict user-item affinity
- **Models**: PinSage (Pinterest), LightGCN

### Traffic/Route Prediction
- **Nodes**: Road segments, intersections
- **Edges**: Connectivity
- **Task**: Predict traffic flow, travel time
- **Models**: Spatiotemporal GNNs

## GNN vs Other Architectures

| Aspect | CNN | RNN | GNN |
|--------|-----|-----|-----|
| Input structure | Grid | Sequence | Graph |
| Locality | Fixed kernel | Temporal | Neighborhood |
| Permutation | Spatial order | Temporal order | Permutation-invariant |
| Use case | Images | Text, time series | Social, molecular |

**Key property**: GNNs are **permutation-equivariant** - node ordering doesn't matter

## Interview Relevance

**Common Questions**:
1. **What is message passing?** Aggregate neighbor information, update node representation
2. **GCN vs GAT?** GCN: degree normalization, equal weights; GAT: learned attention weights
3. **Over-smoothing problem?** Too many layers → all nodes converge to same representation
4. **Why few layers?** 2-4 layers typical (vs 50+ in CNNs) due to over-smoothing
5. **Graph-level prediction?** Aggregate node embeddings (sum/mean/max pooling)
6. **Inductive vs transductive?** Inductive: generalize to unseen nodes (GraphSAGE); Transductive: fixed graph (GCN)
7. **Applications?** Molecules, social networks, recommendations, knowledge graphs
8. **Scalability?** Sampling (GraphSAGE), clustering (Cluster-GCN), mini-batching

**Key Formulas**:
- Message passing: $h_v^{(k+1)} = \text{UPDATE}(h_v^{(k)}, \text{AGG}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$
- GCN: Degree normalization
- GAT: Attention weights $\alpha_{vu}$
- Graph readout: $h_G = \text{AGG}(\{h_v^{(K)} : v \in V\})$

**Key Insight**: GNNs extend deep learning to non-Euclidean domains (graphs) through iterative neighborhood aggregation, enabling learning on structured relational data.
