import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI
from client import CodeReviewEnv
from models import ReviewAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
IMAGE_NAME = os.getenv("IMAGE_NAME") or "code_review_env:latest"

TASK_NAME = os.getenv("CODE_REVIEW_TASK", "code_review")
BENCHMARK = os.getenv("CODE_REVIEW_BENCHMARK", "code_review_env")
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.95  # normalized score in [0, 1]

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a senior Python code reviewer.
    Return ONLY a valid JSON object with keys:
    - issues_found: array of issue tags from the allowed taxonomy only
    - review_comment: concise explanation of the identified issues
    - severity: one of low|medium|high|critical
    Do not include markdown, code fences, or extra prose.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    tags = ", ".join(getattr(obs, "available_issue_tags", []))
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        New Observation:
        TASK ID: {getattr(obs, 'task_id', 'unknown')}
        FILE: {getattr(obs, 'file_name', 'unknown')}
        INSTRUCTION: {getattr(obs, 'task_description', 'N/A')}
        LAST FEEDBACK: {getattr(obs, 'feedback', 'N/A')}

        ALLOWED ISSUE TAGS:
        {tags}

        CODE UNDER REVIEW:
        {getattr(obs, 'code_snippet', '')}

        Return strictly JSON with keys: issues_found, review_comment, severity.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        elif "```" in text:
            text = text.replace("```", "").strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


def extract_json_object(text: str) -> dict:
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_action(payload: dict) -> dict:
    issues_found_raw = payload.get("issues_found", [])
    if not isinstance(issues_found_raw, list):
        issues_found_raw = []

    issues_found = [str(issue) for issue in issues_found_raw]
    review_comment = str(payload.get("review_comment", "")).strip()
    severity = str(payload.get("severity", "medium")).lower()
    if severity not in {"low", "medium", "high", "critical"}:
        severity = "medium"

    return {
        "issues_found": issues_found,
        "review_comment": review_comment,
        "severity": severity,
    }


def run_task(client: OpenAI, env: CodeReviewEnv, task_id: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        url = env.base_url + "/reset"
        import httpx
        try:
            resp = httpx.post(url, json={"task_id": task_id}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            result = env._parse_result(payload)
        except Exception as e:
            print(f"Error resetting env via HTTP: {e}")
            return
            
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            message = get_model_message(client, step, obs, last_reward, history)

            try:
                action_payload = normalize_action(extract_json_object(message))
            except BaseException:
                action_payload = {"issues_found": [], "review_comment": "Failed to parse model output.", "severity": "medium"}
                
            action_str = json.dumps(action_payload, separators=(",", ":"))
            try:
                action = ReviewAction(**action_payload)
                result = env.step(action)
                obs = result.observation
                error = None
            except Exception as e:
                error = str(e)
                obs = getattr(result, "observation", None)

            reward = result.reward or 0.0
            done = result.done or False

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: Action: {action_str!r} -> Reward: {reward:+.2f}")

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as e:
        print(f"Task runtime error: {e}")

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    try:
        env = CodeReviewEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        env = CodeReviewEnv(base_url="http://localhost:8000")

    try:
        for task_id in TASKS:
            run_task(client, env, task_id)
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main()
