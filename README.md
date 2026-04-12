# CodeReviewEnv: Security-Aware Code Review 🛡️🔍

CodeReviewEnv is a Reinforcement Learning (RL) environment built on the OpenEnv framework, specifically designed to train and evaluate AI agents on structured, taxonomy-driven code review tasks.

Unlike simple linting or unstructured LLM prompting, CodeReviewEnv challenges agents to understand Python code vulnerabilities in context. It requires agents to identify issues using a strict taxonomy, assess severity, and provide articulate review comments, enforcing precision and accuracy through a multi-dimensional reward system.

## �� Key Features

- **Taxonomy-Driven Constraints**: Tasks require agents to map vulnerabilities to a strict 12-tag taxonomy (e.g., `sql_injection`, `race_condition`, `null_pointer`).
- **Tiered Difficulty Levels**:
    - 🟢 **Easy**: Basic logic errors like missing return statements and null pointer dereferences.
    - 🟡 **Medium**: Standard application security flaws such as SQL injections and hardcoded secrets.
    - 🔴 **Hard**: Complex concurrency and side-channel flaws like race conditions and timing attacks.
- **Multi-Dimensional Reward Function**: Designed for real-world engineering review standards. It rewards recall (finding the planted bugs), applies a precision penalty for False Positives (over-flagging or hallucinating bugs), and gives an articulation bonus for high-quality review comments.
- **GRPO-Ready**: Built for Group Relative Policy Optimization (GRPO), ensuring the environment easily integrates with generation-based LLM RL tuning via HuggingFace's TRL.
- **OpenEnv Architecture**: Fully conforms to the Meta OpenEnv standard with asynchronous WebSocket/Docker client-server execution.

## 📁 Project Structure

```text
code_review_env/
├── baselines/
│   └── openai_baseline.py       # Zero-shot evaluation script using GPT-4o
├── client.py                    # OpenEnv client implementation
├── models.py                    # Pydantic schemas for Actions and Observations
├── server/                      # OpenEnv server and environment logic
│   ├── app.py                   # FastAPI server and Gradio UI
│   ├── code_review_env_environment.py # Core environment implementation
│   ├── graders.py               # Deterministic grading and reward logic
│   ├── reward.py                # Wrapper for GRPO reward calculation
│   └── tasks.py                 # Structured tasks and planted vulnerabilities
├── training/
│   └── grpo_train.py            # Scaffold for GRPO training loop
├── inference.py                 # Standardized inference and evaluation runner
└── .env                         # Configuration variables for models and environments
```

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.10+ installed, then install the necessary dependencies:

```bash
# Create and activate your virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install openai python-dotenv httpx pydantic fastapi uvicorn
```

### 2. Configuration

Create or modify your `.env` file in the root directory to configure the baseline scripts and inference runners:

```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
HF_TOKEN=your_api_key_here
IMAGE_NAME=code_review_env:latest
CODE_REVIEW_TASK=code_review
CODE_REVIEW_BENCHMARK=code_review_env
```

### 3. Usage

**Running the OpenAI Baseline**

To test the environment purely using a standard LLM (GPT-4o) with zero-shot prompting across all difficulty tiers:

```bash
python baselines/openai_baseline.py
```

**Running Standard Inference**

OpenEnv benchmarks utilize a standardized inference file that connects to the environment (locally or via Docker) and emits `[START]`, `[STEP]`, and `[END]` evaluation blocks. 

Make sure your environment server is running first:
```bash
python -m server.app
```
Then, execute the inference script:
```bash
python inference.py
```

## 📈 Reward Structure

The heart of CodeReviewEnv is its multi-dimensional reward calculation (`server/graders.py` and `server/reward.py`):

- **Recall (Base Score)**: Finding the actual planted issues in the code. `|Correctly Found ∩ Planted| / |Planted|`
- **False Positive Penalty**: Over-aggressive linting or hallucinated bugs. (Penalty: -0.1 per false positive)
- **Articulation Bonus**: Mentioning root causes and specific keywords in the free-text `review_comment`. (Bonus: +0.05 per keyword match)
- **Final Score**: The combined total is strictly clamped between `[0.0, 1.0]`.

This enforces the agent to prioritize high-confidence issues and communicate them clearly, rather than blindly selecting every taxonomy tag from the list.

## 🛠️ Development & Training

To train your own small language models (SLMs) on this environment:

- Wrap the `CodeReviewEnv` inside a standard Gym/Gymnasium wrapper if using traditional RL frameworks.
- For LLMs, integrate the step interactions directly inside a TRL-compatible GRPO wrapper. The scaffold is located in `training/grpo_train.py` and automatically computes zero-mean, unit-variance local advantage rewards.
