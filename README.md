# GUI-RL: A Generalist Reinforcement Learning Framework for GUI Agents

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/franskey-0112/GUI-RL)](https://github.com/franskey-0112/GUI-RL/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## Overview

**GUI-RL** is a reinforcement learning training framework specifically designed for GUI (Graphical User Interface) agents. Built on top of [verl](https://github.com/verl-project/verl) (Volcano Engine Reinforcement Learning for LLMs), GUI-RL enables training vision-language models to interact with graphical user interfaces through reinforcement learning with rule-based rewards.

The framework supports training GUI agents that can understand screenshots, reason about the required actions, and output structured action predictions (click, type, scroll, etc.) with precise coordinates.

## Key Features

- **Multi-Model Support**: Supports multiple model architectures including:
  - **Qwen2.5-VL** (Vision-Language models, e.g., Qwen2.5-VL-3B-Instruct)
  - **Qwen3.5** (Latest generation models, e.g., Qwen3.5-9B)
  - Any HuggingFace-compatible vision-language or language model

- **GRPO-based RL Training**: Uses Group Relative Policy Optimization (GRPO) for efficient on-policy reinforcement learning without a separate critic model.

- **Rule-based Reward System**: Comprehensive reward function (`r1gui.py`) that evaluates:
  - **Format Reward (20%)**: Validates the model's output follows the `<think>...</think><answer>...</answer>` structure with correct action schema
  - **Accuracy Reward (80%)**: Measures action correctness including:
    - Point accuracy (within 70px Euclidean distance or inside bounding box)
    - Text matching (F1 score >= 0.5)
    - Scroll direction / press key matching

- **12 GUI Action Types**: `click`, `type`, `scroll`, `complete`, `open_app`, `wait`, `long_press`, `moveto`, `doubleclick`, `impossible`, `rightclick`, `press`

- **Flexible Dataset Pipeline**: Custom `GUIRLDataset` class that handles:
  - Image loading from multiple formats (base64, file path, bytes, PIL Image)
  - Automatic image resizing with configurable max/min pixels
  - Action history context integration
  - Prompt length filtering

- **Multi-GPU & Multi-Node Training**: Pre-configured scripts for:
  - Single-node 8x GPU (H800)
  - Multi-node 16x GPU (2x8 H200)
  - Configurable tensor parallelism, data parallelism, and FSDP

- **Production-Ready Infrastructure**:
  - vLLM async rollout for fast generation
  - FSDP with parameter/optimizer offloading for memory efficiency
  - Shared job queue for distributed cluster management
  - Checkpoint saving and experiment tracking (wandb, tensorboard)

## Architecture

```
GUI-RL
├── verl/utils/dataset/gui_rl_dataset.py   # Custom dataset for GUI RL training
├── verl/utils/reward_score/r1gui.py       # Rule-based reward function
├── examples/gui_rl/                        # Training scripts
│   ├── run_qwen2_5_vl_3b_gui_rl_grpo.sh  # Qwen2.5-VL-3B training
│   ├── run_qwen3_5_gui_rl_grpo.sh        # Qwen3.5-9B base training
│   ├── run_qwen3_5_gui_rl_grpo_h800_8gpu.sh  # H800 8-GPU optimized
│   └── run_qwen3_5_gui_rl_grpo_h200_16gpu.sh # H200 16-GPU optimized
└── scripts/shared_queue/                   # Distributed job submission
```

## Quick Start

### Installation

```bash
pip install -e .
```

### Data Format

Training data should be in Parquet or JSON/JSONL format with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `image` | bytes/str/dict | Screenshot image (base64, file path, or PIL-compatible) |
| `instruction` | str | The task instruction for the agent |
| `history` | str | Previous action history |
| `gt_action` | str | Ground truth action type |
| `gt_bbox` | list | Ground truth bounding box [x1, y1, x2, y2] or point [x, y] (normalized 0-1) |
| `gt_input_text` | str | Ground truth input text (for type/scroll/press actions) |

### Training

**Qwen2.5-VL-3B (Vision-Language Model):**
```bash
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
bash examples/gui_rl/run_qwen2_5_vl_3b_gui_rl_grpo.sh
```

**Qwen3.5-9B:**
```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
bash examples/gui_rl/run_qwen3_5_gui_rl_grpo.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 128 | Training batch size |
| `ROLLOUT_N` | 4 | Number of rollout samples per prompt |
| `MAX_PROMPT_LENGTH` | 16384 | Maximum prompt token length |
| `MAX_RESPONSE_LENGTH` | 1024 | Maximum response token length |
| `MAX_PIXELS` | 3000000 | Maximum image pixels |
| `MIN_PIXELS` | 262144 | Minimum image pixels |

## Model Output Format

The trained model produces structured reasoning and action output:

```
<think> reasoning about what action to take... </think>
<answer>[{'action': 'click', 'point': [123, 456], 'input_text': 'no input text'}]</answer>
```

## Reward Function

The reward function (`r1gui_compute_score`) combines:
- **Format reward** (weight 0.2): Checks structural validity of the output
- **Accuracy reward** (weight 0.8): Checks correctness of the predicted action

Overall score = 0.8 * accuracy + 0.2 * format

## Acknowledgements

This project is improved from [GUI-R1](https://github.com/ritzz-ai/GUI-R1). The original GUI-R1 only supports Qwen2.5-VL models, while GUI-RL extends the framework to support multiple model architectures (including Qwen3.5 and other HuggingFace-compatible models), provides optimized multi-node training configurations, production-ready distributed job queue infrastructure, and enhanced reward computation.

---

This project is part of the paper **WebFactory**: [https://arxiv.org/pdf/2603.05044](https://arxiv.org/pdf/2603.05044)
