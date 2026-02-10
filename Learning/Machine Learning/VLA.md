# Vision-Language-Action Models (VLA)

## Definition
Vision-Language-Action (VLA) models are end-to-end transformer architectures that map visual observations and natural language instructions directly to robot actions, enabling generalist robotic manipulation policies.

## Core Idea

### Traditional Robot Learning
```
Perception → Planning → Control → Action
(separate modules, hand-engineered)
```

### VLA Approach
```
(Vision + Language) → Transformer → Action
(end-to-end, single model)
```

**Key innovation**: Treat robot control as sequence modeling problem, leverage pre-trained vision-language models

## RT-1 (Robotics Transformer 1)

### Architecture
**Input**:
- **Image**: Robot camera view (300×300 RGB)
- **Instruction**: Natural language ("pick up the apple")
- **History**: Previous 6 image-action pairs (optional)

**Encoder**:
- **Vision**: EfficientNet-B3 (pre-trained on ImageNet)
- **Language**: Universal Sentence Encoder
- **FiLM**: Feature-wise Linear Modulation to condition vision on language

**Decoder**:
- Token Learner: Compress spatial features
- Transformer: 8 layers, causal attention
- Action tokenization: Discretize continuous actions

**Output**:
- 7-DoF action (3D position, 3D rotation, gripper)
- Discretized into 256 bins per dimension

### Training
- **Dataset**: 130K episodes from 13 robots over 17 months
- **Tasks**: 700+ tasks (pick, place, open, close, etc.)
- **Imitation learning**: Behavioral cloning (predict expert actions)
- **Data augmentation**: Image augmentation, language paraphrasing

### Key Innovations
1. **Action tokenization**: Discretize continuous actions (better for transformers)
2. **FiLM conditioning**: Efficient vision-language fusion
3. **Token Learner**: Reduce sequence length (efficiency)

### Results
- **Generalization**: 97% success on training tasks, 76% on novel tasks
- **Language**: Follows complex instructions
- **Objects**: Generalizes to unseen objects

## RT-2 (Robotics Transformer 2)

### Core Innovation
**Co-fine-tune** vision-language model (VLM) on robot data:
1. Start from pre-trained VLM (PaLI-X, PaLM-E)
2. Add action tokens to vocabulary
3. Fine-tune on robot trajectories

**Key insight**: Leverage internet-scale vision-language knowledge for robotics

### Architecture
Based on VLMs:
- **Vision encoder**: ViT (Vision Transformer)
- **Language decoder**: PaLM (LLM)
- **Training**: Co-trained on image captioning + robot actions

**Input format**:
```
Image: [robot camera view]
Text: "pick up the blue cup"
Output: action tokens [0.1, 0.2, -0.3, ...]
```

### Training Strategy
**Two-stage**:
1. **Pre-training**: Vision-language tasks (captioning, VQA) on web data
2. **Fine-tuning**: Robot trajectories (predict actions as text tokens)

**Data mixture**:
- Web data: Billions of image-text pairs
- Robot data: 100K+ episodes

### Key Capabilities

#### 1. Emergent Skills from Internet Knowledge
- **Reasoning**: "move banana to sum of 2+2" → places at 4th position
- **Symbol grounding**: Understands "extinct animal" → moves toy dinosaur
- **Visual reasoning**: "move to number matching Taylor Swift albums" → 10

#### 2. Chain-of-Thought for Robotics
- Generate reasoning before action
- Example: "I should pick X because..." → action

#### 3. Multimodal Understanding
- Combines visual and semantic understanding
- "Pick up the fruit" → recognizes apple is fruit

### Results
- **Performance**: 62% success (vs 32% for RT-1) on novel tasks
- **Emergent capabilities**: Reasoning, symbol grounding
- **Generalization**: Better on long-tail/novel scenarios

## Action Representation

### Continuous Actions (Traditional RL)
- 7-DoF: $(x, y, z, roll, pitch, yaw, gripper)$
- Continuous values $\in \mathbb{R}^7$

**Problem for Transformers**:
- Designed for discrete tokens
- Regression head works but suboptimal

### Discretization (RT-1)
- Bin each dimension into 256 values
- Action becomes sequence of discrete tokens
- Predict via classification (cross-entropy)

**Advantages**:
- Leverage transformer strength (classification)
- Multi-modal action distributions
- Better for noisy data

### Hybrid Approaches
- Predict discretized + offset
- Mixture of Gaussians
- Diffusion for actions (recent)

## Training Data Requirements

### Scale
- **RT-1**: 130K episodes
- **RT-2**: 100K episodes + web VLM data
- Smaller than typical RL (millions of steps)

### Data Collection
- **Teleoperation**: Humans control robot
- **Autonomous**: Previous policies
- **Simulation**: Sim-to-real transfer

### Data Composition
- Diverse tasks (pick, place, open, close, push, pull)
- Diverse objects (kitchen, office, toys)
- Diverse scenes (backgrounds, lighting)

**Key**: Diversity more important than quantity

## Limitations

### 1. Sample Efficiency
- Still requires substantial robot data (100K+ episodes)
- Expensive to collect real-world data

### 2. Action Space
- Discretization limits precision
- Fixed action dimensions (no dynamic morphology)

### 3. Long-Horizon Tasks
- Struggles with multi-step planning
- Typically 1-3 steps (pick and place)

### 4. Safety
- No safety guarantees
- Can produce unexpected behaviors

### 5. Sim-to-Real Gap
- Pre-training on images, but robot data distribution different
- Domain adaptation needed

## Related Approaches

### GATO (DeepMind)
- Generalist agent: vision, language, control
- Single transformer for 600+ tasks
- Includes Atari, robot control, image captioning

### PaLM-E (Google)
- Embodied multimodal LLM
- 562B parameters
- Vision-language-action

### OpenVLA (Open-source)
- Open-source VLA based on LLaMA and CLIP
- Community-driven datasets

### Octo (Open X-Embodiment)
- Generalist policy trained on diverse robot data
- 800K+ trajectories from 60+ robots

## Future Directions

### Foundation Models for Robotics
- Scale up: billions of robot interactions (data bottleneck)
- Sim-to-real: leverage simulation
- Embodied internet data: egocentric videos (Ego4D)

### Multimodal Integration
- Audio, tactile, proprioception
- Temporal understanding (video)

### Planning and Reasoning
- Integrate with classical planning
- Hierarchical policies (high-level + low-level)

## Interview Relevance

**Common Questions**:
1. **What is VLA?** End-to-end vision-language-action model for robotics
2. **RT-1 vs RT-2?** RT-1: trained from scratch; RT-2: fine-tuned from VLM
3. **Why co-fine-tune VLM?** Leverage internet-scale knowledge (reasoning, semantics)
4. **Action representation?** Discretize continuous actions into tokens
5. **Key innovation?** Treat robot control as sequence modeling (language model approach)
6. **Emergent skills?** Reasoning, symbol grounding from pre-training
7. **Data requirements?** 100K+ robot episodes, but leverages web data
8. **Limitations?** Sample efficiency, long-horizon tasks, safety
9. **Why transformers?** Unify vision-language-action, leverage pre-training, scalable

**Key Concepts**:
- **FiLM**: Condition vision on language
- **Action tokenization**: Discretize actions for transformers
- **Co-fine-tuning**: VLM → robot policy
- **Emergent capabilities**: Internet knowledge enables novel skills

**Key Insight**: VLAs represent a paradigm shift in robotics - from modular pipelines to end-to-end learning, and from task-specific to generalist policies by leveraging foundation models.
