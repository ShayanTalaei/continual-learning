## Multi-Step Environments Guide

This guide shows you how to use multi-step environments where agents take multiple actions to complete a task.

### Overview

**Single-turn environments** (QA, Math, MCQ):
- Agent sees question
- Agent gives one answer
- Episode ends

**Multi-step environments** (ALFWorld, interactive tasks):
- Agent sees initial state
- Agent takes action
- Environment updates
- Agent sees new state
- Repeat until task complete

---

### ALFWorld: Embodied Household Tasks

ALFWorld is an interactive text-based environment for household tasks like "put apple in fridge" or "heat potato".

#### Quick Start

**Installation:**
```bash
# Clone ALFWorld
git clone https://github.com/alfworld/alfworld.git third_party/alfworld
cd third_party/alfworld
pip install -e .

# Download data
alfworld-download
```

**Basic configuration:**
```yaml
runtime:
  max_envs_to_visit: 10
  max_steps_per_episode: 50  # Important: cap steps

train_dataset:
  type: alfworld
  base_config_path: "third_party/alfworld/configs/base_config.yaml"
  split: "eval_out_of_distribution"
  num_episodes: 10
  prompts_path: "src/data/prompts/alfworld/alfworld_3prompts.json"

agent:
  type: reflexion_agent  # Recommended: learns from mistakes
  # ... agent config
```

#### Task Types

ALFWorld supports 6 task types:
1. **pick_and_place**: "put X in/on Y"
2. **pick_clean_then_place**: "clean X then put in/on Y"
3. **pick_heat_then_place**: "heat X then put in/on Y"
4. **pick_cool_then_place**: "cool X then put in/on Y"
5. **look_at_obj**: "examine X"
6. **pick_two_obj**: "put X and Y in/on Z"

#### Action Format

**Movement actions:**
```
go to <object> <number>
```
Examples: `go to cabinet 1`, `go to fridge 1`

**Interaction actions:**
```
take <object> <number> from <container> <number>
open <object> <number>
close <object> <number>
use <object> <number>
put <object> <number> in/on <container> <number>
```

**Thinking (no environment change):**
```
think: <reasoning>
```
Example: `think: I should check the fridge first`

#### Example Episode

**Initial observation:**
```
You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 2, 
a fridge 1, a countertop 1, a microwave 1.

Your task is to: put a clean apple in the fridge.
```

**Agent actions:**
```
Step 1: go to countertop 1
  → You arrive at loc 1. On the countertop 1, you see an apple 1.

Step 2: take apple 1 from countertop 1
  → You pick up the apple 1 from countertop 1.

Step 3: go to sink 1
  → You arrive at loc 2. On the sink 1, you see nothing.

Step 4: clean apple 1 with sink 1
  → You clean the apple 1 using the sink 1.

Step 5: go to fridge 1
  → You arrive at loc 3. The fridge 1 is closed.

Step 6: open fridge 1
  → You open the fridge 1. The fridge 1 is open. In it, you see nothing.

Step 7: put apple 1 in fridge 1
  → You put the apple 1 in fridge 1. Task complete!
```

#### Few-Shot Prompting

ALFWorld environments can inject task-specific examples:

**Prompts JSON** (`alfworld_3prompts.json`):
```json
{
  "react_clean_1": "Example 1: clean task...",
  "react_clean_0": "Example 2: clean task...",
  "react_heat_1": "Example 1: heat task...",
  ...
}
```

**Enable in config:**
```yaml
train_dataset:
  prompts_path: "src/data/prompts/alfworld/alfworld_3prompts.json"
```

The environment automatically selects 2 relevant examples based on task type.

#### Agent Configuration

**ReflexionAgent (recommended):**
```yaml
agent:
  type: reflexion_agent
  lm_config:
    model: "gpt-4o-mini"
    temperature: 0.2
    max_output_tokens: 512
  memory_config:
    _type: history_list
    max_length: 200
  history_k: 50
  enable_reflection: true
  agent_system_prompt: |
    You are an embodied household assistant in ALFWorld.
    - Issue one action per turn (e.g., "go to fridge 1", "take apple 1 from table 1")
    - You may think using "think: ..." which does not change the environment
    - Stop repeating failed actions
    - Complete the task efficiently
  reflection_few_shot_examples: "${file:prompts/alfworld/reflection_examples.txt}"
```

**HistoryAgent:**
```yaml
agent:
  type: history_agent
  # ... similar config
```

#### Best Practices

1. **Set max_steps_per_episode:** Prevent infinite loops
   ```yaml
   runtime:
     max_steps_per_episode: 50
   ```

2. **Use ReflexionAgent:** Learns to avoid repeating mistakes

3. **Enable thinking:** Let agent plan without acting
   ```
   think: I should check if the apple is already clean
   ```

4. **Monitor repeated actions:** If agent loops, add to system prompt:
   ```
   - If an action fails twice, try a different approach
   - Don't repeat the same failed action more than 2 times
   ```

5. **Few-shot examples:** Dramatically improve success rate

---

### Custom Multi-Step Environments

You can create custom multi-step environments by extending the base `Environment` class.

#### Basic Structure

```python
from src.data.env import Environment
from typing import Dict, Any, Optional, Tuple

class MyInteractiveEnv(Environment):
    def __init__(self, env_id: str, task_description: str):
        super().__init__(env_id=env_id, env_type="interactive")
        self.task = task_description
        self.state = self._init_state()
        self.step_count = 0
        
    def _init_state(self) -> Dict[str, Any]:
        """Initialize environment state."""
        return {"status": "running", ...}
    
    def reset(self) -> str:
        """Reset environment and return initial observation."""
        self.state = self._init_state()
        self.step_count = 0
        return self._build_observation()
    
    def step(self, action: str) -> Tuple[Optional[str], Dict[str, Any], bool, Dict[str, Any]]:
        """
        Execute action and return (observation, feedback, done, info).
        
        Returns:
            observation: Next state description
            feedback: {score: float, message: str, ...}
            done: True if episode should end
            info: Additional metadata
        """
        self.step_count += 1
        
        # Parse action
        parsed = self._parse_action(action)
        
        # Update state
        self.state = self._apply_action(parsed)
        
        # Build response
        obs = self._build_observation()
        done = self._is_terminal()
        feedback = self._evaluate_state()
        info = {"step": self.step_count}
        
        return obs, feedback, done, info
    
    def _build_observation(self) -> str:
        """Construct observation string from current state."""
        return f"Current status: {self.state['status']}\nTask: {self.task}"
    
    def _parse_action(self, action: str) -> Dict[str, Any]:
        """Parse action string into structured command."""
        # Custom parsing logic
        return {"type": "...", "args": [...]}
    
    def _apply_action(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parsed action to state."""
        new_state = dict(self.state)
        # Update state based on action
        return new_state
    
    def _is_terminal(self) -> bool:
        """Check if episode should end."""
        return self.state["status"] == "complete" or self.step_count >= 50
    
    def _evaluate_state(self) -> Dict[str, Any]:
        """Evaluate current state and return feedback."""
        if self.state["status"] == "complete":
            return {"score": 1.0, "message": "Task complete!"}
        return {"score": 0.0, "message": "In progress..."}
```

#### Dataset Loader

```python
from src.data.env import EnvDataset, EnvDatasetConfig
from typing import List
from pydantic import BaseModel

class MyEnvDatasetConfig(EnvDatasetConfig):
    tasks_file: str
    max_tasks: Optional[int] = None

class MyEnvDataset(EnvDataset):
    def __init__(self, config: MyEnvDatasetConfig, logger=None):
        self.config = config
        self.logger = logger
        self.dataset = self.load_dataset()
    
    def load_dataset(self) -> List[Environment]:
        envs = []
        with open(self.config.tasks_file) as f:
            for i, line in enumerate(f):
                if self.config.max_tasks and i >= self.config.max_tasks:
                    break
                task = line.strip()
                envs.append(MyInteractiveEnv(
                    env_id=f"task_{i}",
                    task_description=task
                ))
        return envs
```

#### Registration

Register your environment for use in configs:

```python
# In src/data/registry.py
from src.data.my_env import MyEnvDataset

DATASET_REGISTRY["my_env"] = MyEnvDataset
```

#### Configuration

```yaml
train_dataset:
  type: my_env
  tasks_file: "data/my_tasks.txt"
  max_tasks: 100
```

---

### Tips for Multi-Step Tasks

1. **Cap episode length:** Always set `max_steps_per_episode`

2. **Monitor step counts:** Track average steps per episode
   ```python
   avg_steps = total_steps / num_episodes
   ```

3. **Intermediate rewards:** Provide partial credit for progress
   ```python
   feedback = {
       "score": 0.5,  # 50% complete
       "message": "Found the apple, now clean it"
   }
   ```

4. **State description:** Make observations informative
   ```
   Good: "You see: cabinet 1 (closed), table 1 (apple on it), fridge 1 (open)"
   Bad: "You are in the kitchen"
   ```

5. **Action validation:** Return helpful feedback for invalid actions
   ```python
   if action not in valid_actions:
       return obs, {"score": 0, "message": "Invalid action. Try: go, take, use"}, False, {}
   ```

6. **Think actions:** Support deliberation without state changes

7. **Memory management:** Multi-step episodes generate lots of history
   ```yaml
   memory_config:
     max_length: 500  # Increase for multi-step
   ```

8. **Termination conditions:** Clear success and failure states
   ```python
   done = (
       self.state["success"] or 
       self.step_count >= self.max_steps or
       self.state["failed"]
   )
   ```

---

### Debugging Multi-Step Environments

**Enable verbose logging:**
```yaml
agent:
  verbose: true

train_dataset:
  verbose: true

runtime:
  verbose: true
  scores_path: scores.jsonl
  verbose_score_logging: true  # See all observations/actions
```

**Check score logs:**
```bash
tail -50 scores/train/scores.jsonl
```

**Inspect LLM calls:**
```yaml
agent:
  lm_config:
    log_calls: true
```

Then check:
```bash
ls llm_calls/train/actions/
cat llm_calls/train/actions/action_episode_1_step_5_*.json
```

**Watch episode unfold:**
```python
# In custom logging code
def log_step(step_num, obs, action, feedback):
    print(f"\n=== Step {step_num} ===")
    print(f"Observation: {obs}")
    print(f"Action: {action}")
    print(f"Feedback: {feedback}")
```

---

### See Also

- [Environments Concepts](../concepts/environments.md) - Environment architecture
- [Runtime Concepts](../concepts/runtime.md) - Episode execution loop
- [Configuration Reference](../reference/config.md) - All environment options

