# Voyager: Technical Implementation Details

**Complete technical guide for implementing Voyager from scratch.**

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Voyager Agent Loop                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  Automatic   │─────>│    Skill     │─────>│Iterative  │ │
│  │  Curriculum  │      │   Library    │      │Prompting  │ │
│  │  (GPT-4)     │      │(Vector DB)   │      │(GPT-4)    │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         v                      v                     v       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Minecraft Environment (MineDojo)           │   │
│  │              Mineflayer JavaScript APIs              │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                              │     │
│         └──────────────> Feedback <───────────────────┘     │
│              (Environment, Errors, Verification)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Automatic Curriculum

### Objective

Generate suitable next task based on:
- Agent's current exploration progress
- Agent's current capabilities (skill level)
- Agent's current world state (biome, inventory, nearby resources)

### GPT-4 Prompt Structure

**System Prompt Components**:

1. **Directives** (constraints + goals):
```
You are a helpful assistant that tells me the next immediate task to do in Minecraft.
My ultimate goal is to discover as many diverse things as possible, accomplish as many
diverse tasks as possible and become the best Minecraft player in the world.

I will give you the following information:
Question 1: ...
Answer: ...
Question 2: ...
Answer: ...
...
Inventory (xx/36): ...
Chests: ...
Completed tasks so far: ...
Failed tasks that are too hard: ...

You must follow the following criteria:
1) You should act as a mentor and guide me to the next task based on my current learning
   progress.
2) The next task should follow a concise format, such as "Mine 3 cobblestone", "Craft
   a furnace", "Collect 10 wheat seeds".
3) The next task should not be too hard since I may not have the necessary resources or
   have learned enough skills to complete it yet.
4) The next task should be novel and diverse. I should look for new items to collect,
   or new mobs to fight, or new tools to craft, or new structures to build.
5) I may sometimes need to repeat some tasks if I need to collect more resources to
   complete more difficult tasks. Only repeat tasks if necessary.
```

2. **Agent's Current State**:
```
Inventory (15/36): {
    'cobblestone': 4, 'furnace': 1, 'stone_pickaxe': 1, 'oak_planks': 7,
    'dirt': 6, 'wooden_pickaxe': 1, 'crafting_table': 1, 'raw_iron': 4, 'coal': 1
}

Equipment: None

Nearby blocks: dirt, water, spruce_planks, grass_block, dirt_path, sugar_cane, fern

Nearby entities: pig, cat, villager, zombie

Biome: plains

Time: sunset

Health: 20.0/20

Hunger: 20.0/20

Position: x=123.5, y=64.0, z=-45.2

Chests: None
```

3. **Completed/Failed Tasks**:
```
Completed tasks so far: Mine 3 wood log, Craft 1 crafting table, Craft 1 wooden pickaxe,
Mine 11 cobblestone, Craft 1 stone pickaxe, Craft 1 furnace, Mine 3 iron ore

Failed tasks that are too hard: Craft 1 diamond pickaxe (no diamonds yet)
```

4. **Additional Context** (self-asked Q&A via GPT-3.5):
```
Question 1: What is the best way to find iron ore?
Answer: Iron ore generates below y=72. Mine underground with a stone pickaxe or better.

Question 2: How do I smelt iron ore?
Answer: Place iron ore and fuel (coal/wood) in a furnace to get iron ingots.
```

**Response Format** (chain-of-thought):
```
Reasoning: I have 4 raw iron and 1 coal. I should smelt the iron to make iron ingots,
which will allow me to craft iron tools. This is the natural next step in the tech tree.

Task: Smelt 3 raw iron
```

### Warm-Up Schedule

**Purpose**: Gradually increase prompt complexity as agent gains experience

| Information | After N tasks completed |
|-------------|-------------------------|
| Core inventory (log, planks, stick, crafting table, furnace, dirt, coal, pickaxe, sword, axe) | 0 |
| Equipment | 0 |
| Nearby blocks | 0 |
| Position | 0 |
| Nearby entities | 5 |
| Full inventory | 7 |
| Other blocks recently seen | 10 |
| Biome | 10 |
| Health bar | 15 |
| Hunger bar | 15 |
| Time | 15 |
| Additional context (Q&A) | 15 |

### Implementation Details

**GPT-3.5 for Context Q&A**:
```python
# Self-ask questions
questions = gpt35_complete(
    prompt=f"You are a helpful assistant that asks questions about Minecraft. "
           f"Based on the agent's state: {agent_state}, "
           f"ask 5 relevant questions about what to do next."
)

# Self-answer questions (optionally using wiki)
answers = []
for q, concept in questions:
    wiki_doc = retrieve_wiki(concept)  # Optional
    answer = gpt35_complete(
        prompt=f"Question: {q}\nContext: {wiki_doc}\nAnswer:"
    )
    answers.append((q, answer))
```

**Temperature**: 0.1 (slight randomness for task diversity)

---

## Component 2: Skill Library

### Data Structure

**Vector Database** (Qdrant, Pinecone, or similar):
```python
class SkillLibrary:
    def __init__(self):
        self.skills = {}  # skill_name -> code
        self.descriptions = {}  # skill_name -> description
        self.embeddings = {}  # skill_name -> embedding vector

    def add_skill(self, code: str):
        # Generate description
        description = gpt35_complete(
            prompt=f"Describe this JavaScript function in one sentence:\n{code}"
        )

        # Generate function name from code
        name = extract_function_name(code)

        # Compute embedding
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=description
        )

        # Store
        self.skills[name] = code
        self.descriptions[name] = description
        self.embeddings[name] = embedding

    def retrieve(self, query: str, top_k: int = 5):
        # Embed query
        query_embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )

        # Compute cosine similarity
        scores = {
            name: cosine_similarity(query_embedding, emb)
            for name, emb in self.embeddings.items()
        }

        # Return top-k
        top_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.skills[name] for name, _ in top_skills]
```

### Skill Retrieval Strategy

**Query construction**:
```python
# Combine task description + environment feedback
task = "Craft an iron pickaxe"
env_feedback = "I have 3 iron ingots and 2 sticks in my inventory"

# Generate general suggestions
suggestions = gpt35_complete(
    prompt=f"How to {task}? Give step-by-step suggestions."
)

# Combine for retrieval
query = f"{task}. {suggestions}. {env_feedback}"

# Retrieve top-5 relevant skills
relevant_skills = skill_library.retrieve(query, top_k=5)
```

**Retrieval accuracy** (evaluated on 309 samples):
- Top-1: 80.2%
- Top-3: 93.2%
- Top-5: 96.5% ✓ (used in practice)

### Example Skills

**Simple skill** (craftWoodenPlanks):
```javascript
async function craftWoodenPlanks(bot) {
  const logTypes = ['oak', 'birch', 'spruce', 'jungle', 'acacia', 'dark_oak'];

  for (const logType of logTypes) {
    const log = bot.inventory.findInventoryItem(mcdata.itemsByName[`${logType}_log`].id);
    if (log) {
      await bot.craftRecipe(mcdata.recipes.find(
        recipe => recipe.result.name === `${logType}_planks`
      ));
      bot.chat(`Crafted ${logType} planks`);
      return;
    }
  }

  bot.chat("I don't have any logs to craft planks.");
}
```

**Complex skill** (smeltFiveRawIronV2):
```javascript
async function smeltFiveRawIronV2(bot) {
  const rawIronCount = bot.inventory.count(mcdata.itemsByName.raw_iron.id);
  const coalCount = bot.inventory.count(mcdata.itemsByName.coal.id);

  if (rawIronCount < 5) {
    bot.chat(`Not enough raw iron. Have ${rawIronCount}, need 5.`);
    return;
  }

  if (coalCount < 5) {
    bot.chat(`Not enough coal. Have ${coalCount}, need 5.`);
    return;
  }

  // Find furnace
  const furnaceBlock = bot.findBlock({
    matching: mcdata.blocksByName.furnace.id,
    maxDistance: 32
  });

  if (!furnaceBlock) {
    bot.chat("No furnace nearby.");
    return;
  }

  // Smelt iron
  await smeltItem(bot, "raw_iron", "coal", 5);
  bot.chat("Smelted 5 raw iron into iron ingots.");
}
```

---

## Component 3: Iterative Prompting Mechanism

### Control Primitive APIs

**Custom wrappers** (implemented by authors):

```javascript
// Explore in a direction until callback returns true
async function exploreUntil(bot, direction, maxTime = 60, callback) {
  const startTime = Date.now();
  while (Date.now() - startTime < maxTime * 1000) {
    await bot.pathfinder.goto(new GoalXZ(
      bot.entity.position.x + direction.x * 10,
      bot.entity.position.z + direction.z * 10
    ));
    if (await callback(bot)) break;
  }
}

// Mine specific block type
async function mineBlock(bot, blockName, count = 1) {
  let mined = 0;
  while (mined < count) {
    const block = bot.findBlock({
      matching: mcdata.blocksByName[blockName].id,
      maxDistance: 32
    });
    if (!block) {
      bot.chat(`No ${blockName} nearby.`);
      return;
    }
    await bot.dig(block);
    mined++;
    bot.chat(`Mined ${blockName}. Progress: ${mined}/${count}`);
  }
}

// Craft item (requires crafting table nearby)
async function craftItem(bot, itemName, count = 1) {
  const recipe = mcdata.recipes.find(r => r.result.name === itemName);
  if (!recipe) {
    bot.chat(`No recipe for ${itemName}.`);
    return;
  }

  const craftingTable = bot.findBlock({
    matching: mcdata.blocksByName.crafting_table.id,
    maxDistance: 32
  });

  if (!craftingTable) {
    bot.chat("No crafting table nearby.");
    return;
  }

  await bot.pathfinder.goto(new GoalNear(
    craftingTable.position.x,
    craftingTable.position.y,
    craftingTable.position.z,
    2
  ));

  await bot.craft(recipe, count, craftingTable);
  bot.chat(`Crafted ${count} ${itemName}.`);
}
```

**Mineflayer APIs** (provided to GPT-4):
- `bot.pathfinder.goto(goal)`: Navigate to position
- `bot.equip(item, destination)`: Equip item
- `bot.consume()`: Eat/drink item in hand
- `bot.fish()`: Fish with equipped rod
- `bot.sleep(bedBlock)`: Sleep in bed
- `bot.activateBlock(block)`: Right-click block
- `bot.activateItem()`: Right-click item in hand
- `bot.useOn(entity)`: Right-click entity

### Code Generation Prompt Structure

**System Prompt**:
```
You are a helpful assistant that writes Mineflayer JavaScript code to complete
any Minecraft task specified by me.

Here are some useful programs written with Mineflayer APIs:
[Retrieved skills from skill library]

At each round of conversation, I will give you:
Code from the last round: ...
Execution error: ...
Chat log: ...
Biome: ...
Time: ...
Nearby blocks: ...
Nearby entities: ...
Health: ...
Hunger: ...
Position: ...
Equipment: ...
Inventory (xx/36): ...
Chests: ...
Task: ...
Context: ...
Critique: ...

You should then respond to me with:
Explain (if applicable): ...
Plan:
1) ...
2) ...
3) ...
Code:
```async function yourFunctionName(bot) {
  // ...
}```

Guidelines:
1) Your function will be reused for building more complex functions. Make it
   generic and reusable.
2) Anything defined outside the function will be ignored. Do not write code
   outside the function.
3) Use `bot.chat()` to show intermediate progress.
4) Use try-catch to handle errors gracefully.
5) Name the function clearly based on what it does.
```

**Iterative Refinement Loop**:

```python
def iterative_prompting(task, max_rounds=4):
    code = None
    for round in range(max_rounds):
        # Generate/refine code
        prompt = build_prompt(
            task=task,
            code_from_last_round=code,
            execution_error=execution_error,
            environment_feedback=env_feedback,
            critique=critique,
            agent_state=agent_state,
            relevant_skills=skill_library.retrieve(task)
        )

        code = gpt4_complete(prompt, temperature=0)

        # Execute code
        env_feedback, execution_error = execute_code(code)

        # Self-verification
        success, critique = self_verify(task, agent_state)

        if success:
            # Add to skill library
            skill_library.add_skill(code)
            return code

    # Failed after max rounds
    return None
```

### Three Feedback Types

#### 1. Environment Feedback

**Generated via `bot.chat()`** inside control primitives:

```javascript
async function craftItem(bot, itemName, count) {
  // Check materials
  const recipe = getRecipe(itemName);
  for (const [mat, needed] of recipe.materials) {
    const have = bot.inventory.count(mat);
    if (have < needed) {
      bot.chat(`I cannot make ${itemName} because I need: ${needed - have} more ${mat}`);
      return;
    }
  }

  // Craft
  await bot.craft(recipe, count);
  bot.chat(`Crafted ${count} ${itemName}.`);
}
```

**Example feedback shown to GPT-4**:
```
Chat log:
[09:23:45] <bot> I cannot make an iron chestplate because I need: 7 more iron ingots
```

#### 2. Execution Errors

**From JavaScript interpreter**:

```
Execution error:
ReferenceError: craftAcaciaAxe is not defined
    at async craftAcaciaAxe (eval:3:5)
```

**GPT-4 response**:
```
Explain: There is no "acacia_axe" in Minecraft. I should craft a "wooden_axe" instead.

Code:
```async function craftWoodenAxe(bot) {
  await craftItem(bot, "wooden_axe", 1);
}```
```

#### 3. Self-Verification

**Prompt to GPT-4**:
```
You are an assistant that assesses my progress of playing Minecraft and provides
guidance on the next tasks.

Inventory: {iron_ingot: 3, stick: 2, crafting_table: 1}
Equipment: None
Nearby blocks: ...
Task: Craft an iron pickaxe
Context: An iron pickaxe requires 3 iron ingots and 2 sticks.

Reasoning: ...
Success: true/false
Critique (if failed): ...
```

**Response format** (chain-of-thought):
```
Reasoning: I have 3 iron ingots and 2 sticks in my inventory, which are the exact
materials needed for an iron pickaxe. Since the task is to craft an iron pickaxe
and I have the materials, the task is successful.

Success: true
```

**If failed**:
```
Reasoning: I only have 2 iron ingots but need 3. The task is not complete.

Success: false

Critique: Mine 1 more iron ore and smelt it to get the third iron ingot.
```

### Few-Shot Examples for Self-Verification

```
Example 1:
Inventory: {diamond: 2}
Task: Mine 3 diamonds
Success: false
Critique: Mine 1 more diamond to reach the goal of 3.

Example 2:
Inventory: {crafting_table: 1}
Task: Place a crafting table
Success: false
Critique: You only crafted the table but didn't place it. Use placeItem() to place it.

Example 3:
Inventory: {spider_string: 5}
Task: Kill a spider
Success: true (spider string is proof of killing a spider)
```

---

## Complete Algorithm

```python
def voyager_main_loop():
    skill_library = SkillLibrary()
    completed_tasks = []
    failed_tasks = []

    while True:
        # 1. Get agent state
        agent_state = get_agent_state()  # inventory, position, biome, etc.

        # 2. Automatic curriculum proposes next task
        task = gpt4_complete(
            system_prompt=curriculum_system_prompt,
            user_prompt=format_curriculum_prompt(
                agent_state=agent_state,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks
            ),
            temperature=0.1
        )

        print(f"Next task: {task}")

        # 3. Iterative prompting to complete task
        code = None
        execution_error = None
        env_feedback = None
        critique = None

        for round in range(4):  # Max 4 rounds
            # 3a. Retrieve relevant skills
            relevant_skills = skill_library.retrieve(
                query=f"{task} {env_feedback or ''}",
                top_k=5
            )

            # 3b. Generate/refine code
            code = gpt4_complete(
                system_prompt=code_generation_system_prompt,
                user_prompt=format_code_prompt(
                    task=task,
                    agent_state=agent_state,
                    relevant_skills=relevant_skills,
                    code_from_last_round=code,
                    execution_error=execution_error,
                    env_feedback=env_feedback,
                    critique=critique
                ),
                temperature=0
            )

            print(f"Generated code (round {round+1}):\n{code}")

            # 3c. Execute code
            try:
                execute_javascript(code)
                execution_error = None
            except Exception as e:
                execution_error = str(e)
                print(f"Execution error: {execution_error}")

            # 3d. Get environment feedback
            env_feedback = get_chat_log()

            # 3e. Self-verification
            agent_state = get_agent_state()
            success, critique = gpt4_self_verify(
                task=task,
                agent_state=agent_state
            )

            print(f"Success: {success}")
            if success:
                # 3f. Add skill to library
                skill_library.add_skill(code)
                completed_tasks.append(task)
                print(f"✓ Task completed: {task}")
                break

        else:
            # Failed after 4 rounds
            failed_tasks.append(task)
            print(f"✗ Task failed: {task}")
```

---

## Implementation Details

### Environment Setup

**MineDojo** (Minecraft simulation):
```python
import minedojo

env = minedojo.make(
    task_id="open-ended",
    image_size=(640, 360),
    world_seed=None,  # Random world
    fast_reset=False  # Persistent world
)
```

**Mineflayer** (JavaScript bot control):
```javascript
const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder');

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'voyager'
});

bot.loadPlugin(pathfinder.pathfinder);
```

### Chat Log Extraction

**Filter bot messages from environment**:
```python
def get_environment_feedback():
    chat_log = []
    for msg in bot.chat_history:
        if msg.sender == "bot" and "bot.chat()" in msg.source:
            chat_log.append(msg.content)
    return "\n".join(chat_log)
```

### Token Usage Optimization

**Concatenate system + user prompts** (avoid multi-turn):
```python
# Instead of:
# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": user_prompt}
# ]

# Use single call:
response = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0
)
```

### Chest Management

**External storage for inventory overflow**:
```javascript
// Check chests
const chests = bot.findBlocks({
  matching: mcdata.blocksByName.chest.id,
  maxDistance: 32,
  count: 10
});

// Store chest positions and contents
const chest_info = chests.map(pos => ({
  position: pos,
  contents: bot.openChest(bot.blockAt(pos)).items()
}));
```

---

## Evaluation Metrics

### 1. Discovered Items

**Count unique items** acquired during exploration:
```python
discovered_items = set()
for timestep in exploration_history:
    for item in timestep.inventory:
        discovered_items.add(item)

score = len(discovered_items)
```

### 2. Tech Tree Unlocking

**Hierarchical milestones**:
- Wooden tier: wooden_pickaxe, wooden_axe, wooden_sword
- Stone tier: stone_pickaxe, stone_axe, stone_sword
- Iron tier: iron_pickaxe, iron_axe, iron_sword
- Diamond tier: diamond_pickaxe, diamond_axe, diamond_sword

**Metric**: Prompting iterations to unlock each tier

### 3. Map Coverage

**Distance traveled**:
```python
def compute_distance_traveled(trajectory):
    total_distance = 0
    for i in range(len(trajectory) - 1):
        pos1 = trajectory[i]
        pos2 = trajectory[i+1]
        total_distance += euclidean_distance(pos1, pos2)
    return total_distance
```

**Biome diversity**: Count unique biomes visited

### 4. Zero-Shot Generalization

**Setup**:
1. Clear inventory
2. Reset to new world
3. Provide unseen tasks

**Tasks**:
- Craft diamond pickaxe
- Craft golden sword
- Collect lava bucket
- Craft compass

**Metric**: Success rate (3 trials each) + iterations to complete

---

## Model Specifications

### GPT-4 Usage

**Code generation**: `gpt-4-0314`
- Temperature: 0 (deterministic)
- Max tokens: 1500
- Stop sequences: "```" (end of code block)

**Automatic curriculum**: `gpt-4-0314`
- Temperature: 0.1 (slight diversity)
- Max tokens: 200

**Self-verification**: `gpt-4-0314`
- Temperature: 0 (deterministic)
- Max tokens: 300

### GPT-3.5 Usage

**Question-answering**: `gpt-3.5-turbo-0301`
- Temperature: 0
- Max tokens: 200
- Purpose: Save cost for simple NLP tasks

**Embeddings**: `text-embedding-ada-002`
- Dimension: 1536
- Purpose: Skill library retrieval

### Cost Considerations

**GPT-4 is 15× more expensive than GPT-3.5**, but:
- Necessary for code generation quality (ablation: 5.7× performance drop with GPT-3.5)
- Used selectively (not for all tasks)
- GPT-3.5 handles: embeddings, simple Q&A, descriptions

---

## Robustness

### Model Version Consistency

**Tested on**:
- `gpt-4-0314`: Baseline
- `gpt-4-0613`: Similar performance (~same item count)

**Takeaway**: Voyager robust to GPT-4 version updates

### Error Handling

**Bot death**: Respawn near closest ground, preserve inventory

**Stuck states**: Automatic curriculum retries task later

**Code execution timeout**: 300 seconds per task (configurable)

---

## Extensions

### Multimodal Perception

**Current**: Text-only (GPT-4 API limitation at time of paper)

**Future**: Vision-language models for:
- 3D structure building (with human feedback)
- Visual critique of spatial details
- Perceiving complex structures

**Demonstrated with human feedback**:
- Built Nether Portal
- Built house
- Two feedback modes:
  1. Human as critic (visual feedback)
  2. Human as curriculum (break down complex tasks)

### Transfer Learning

**Skill library enables transfer**:
- New world → reuse all skills
- Novel tasks → compose from existing skills
- AutoGPT + Voyager's skill library → improved performance

---

## Key Implementation Insights

1. **Warm-up schedule crucial**: Prevents overwhelming GPT-4 with too much info early
2. **Top-5 skill retrieval sufficient**: 96.5% accuracy
3. **Self-verification > self-reflection**: Check success + provide critique
4. **Chain-of-thought prompting**: Essential for GPT-4 to reason before acting
5. **Environment feedback most informative**: Shows "why" task failed
6. **4 rounds sufficient**: Most tasks succeed within 4 iterations
7. **Temperature 0.1 for curriculum**: Balances diversity and consistency

**This implementation guide provides everything needed to replicate Voyager from scratch.**
