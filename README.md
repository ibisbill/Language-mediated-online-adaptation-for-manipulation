# Prompt2MetaWorld

Zero-shot robot control in [MetaWorld](https://meta-world.github.io/) using pure prompt engineering — no training, no fine-tuning. A GPT-4 agent reads the raw observation vector, reasons about the robot's current state and goal, and outputs the next 4-DoF action.

---

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│                     Control Loop                         │
│                                                          │
│  step 0:  get_first_input(obs)                           │
│           → few-shot demo + task description + obs       │
│           → GPT-4 predicts action + next obs             │
│                                                          │
│  step t (even):  get_input(obs, action, history, ...)    │
│           → interact_prompt: reason about movement       │
│           → GPT-4 predicts action + next obs             │
│                                                          │
│  step t (odd):   get_input(obs, ...)                     │
│           → cot_prompt: self-reflect on prediction error │
│           → GPT-4 explains discrepancy, adjusts plan     │
│                                                          │
│  repeat until done or 100 steps                          │
└──────────────────────────────────────────────────────────┘
```

**Two-prompt alternating strategy:**
- **Even steps** — action prediction: given the current observation and previous action, the model reasons about the robot's position relative to the goal and outputs the next action and predicted next observation.
- **Odd steps** — chain-of-thought self-reflection: the model compares its predicted observation to the actual one, explains why they differ, and adjusts its understanding before the next action.

---

## Repository Structure

```
.
├── llm_clean.py      # Main control loop: API calls, response parsing, env stepping
├── mw.py             # MetaWorld environment wrapper (gym + dm_env)
├── prompt_cot.py     # CoT prompts: system, demo, interact, cot, and new-task prompts
├── prompt_meta.py    # Meta-learning style prompts with cross-task trajectory demonstrations
├── requirements.txt
└── LICENSE
```

---

## Observation & Action Space

| Space | Dim | Description |
|---|---|---|
| Observation | 39 | Gripper pos (3) + gripper state (1) + object 1 pos/quat (7) + object 2 pos/quat (7) × 2 timesteps + goal pos (3) |
| Action | 4 | Δ gripper position (3) + gripper force (1), all in [−1, 1] |

---

## Installation

```bash
git clone https://github.com/ibisbill/prompt2metaworld.git
cd prompt2metaworld
pip install -r requirements.txt
```

Install MetaWorld following the [official instructions](https://github.com/Farama-Foundation/Metaworld).

Set your API key:

```bash
export OPENAI_API_KEY=<your_key>
```

---

## Usage

```bash
python llm_clean.py
```

This runs the `door-open` task by default. To change the task, edit the `make(name=...)` call in the `__main__` block of `llm_clean.py`. Output is saved step-by-step to `output/run.json`.

To use a different task or prompt style, swap `prompt_cot` for `prompt_meta` in the import and adjust the prompt variables accordingly.

---

## Prompting Strategy

Two prompt files are provided, each encoding a different prior:

| File | Strategy |
|---|---|
| `prompt_cot.py` | Chain-of-thought with alternating action/reflection turns |
| `prompt_meta.py` | Meta-learning style: demonstrates full success trajectories from related tasks before asking the model to solve a new one |

---

## Requirements

- Python ≥ 3.8
- OpenAI API key with GPT-4 access
- MetaWorld + MuJoCo installed

---

## License

[MIT License](LICENSE)
