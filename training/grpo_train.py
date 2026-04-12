import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from client import CodeReviewEnv
import json

def get_reward_from_env(prompts, completions, **kwargs):
    rewards = []
    # Initialize the OpenEnv client connecting to our FastAPI server
    client = CodeReviewEnv(base_url="http://127.0.0.1:8000")
    
    for prompt, completion in zip(prompts, completions):
        try:
            parsed = json.loads(completion.strip())
            issues_found = parsed.get("issues_found", [])
            review_comment = parsed.get("review_comment", "")
            severity = parsed.get("severity", "medium")
        except:
            issues_found = []
            review_comment = ""
            severity = "medium"

        # TRL uses this mostly statically or dynamically, we assume 'task_easy' for demonstration
        import httpx
        try:
            resp = httpx.post(client.base_url + "/reset", json={"task_id": "task_easy"}, timeout=10)
            payload = resp.json()
            result = client._parse_result(payload)
        except Exception:
            pass

        action = {"issues_found": issues_found, "review_comment": review_comment, "severity": severity}
        try:
            step_result = client.step(action)
            rewards.append(step_result.reward)
        except:
            rewards.append(0.0)
            
    return rewards

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Lightweight model for testing
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    train_dataset = [
        {"prompt": "Review this code for issues: def get_user_age(user): ... "},
        {"prompt": "Check auth.py for SQL injection and hardcoded secrets..."}
    ]
    
    training_args = GRPOConfig(
        output_dir="./grpo_code_review_model",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_prompt_length=128,
        max_completion_length=64,
        num_generations=4, # Group size
        max_steps=100,
        logging_steps=10,
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=get_reward_from_env,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("Starting GRPO Training...")
    trainer.train()
    print("Training Complete!")

if __name__ == "__main__":
    main()
