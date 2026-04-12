import os
import json
from openai import OpenAI
from server.tasks import TASKS
from server.reward import calculate_review_reward
from models import ReviewAction

def run_baseline():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))
    
    for task_id, task in TASKS.items():
        print(f"--- Running task {task_id} ({task.difficulty}) ---")
        print(f"[START] {json.dumps({'difficulty': task.difficulty, 'task_id': task_id})}")
        
        prompt = f"""
TASK ID: {task.task_id}
FILE: {task.file_name}
INSTRUCTION: {task.description}

CODE UNDER REVIEW:
{task.code}

Return ONLY a valid JSON object with keys:
- issues_found: array of issue tags from the allowed taxonomy only
- review_comment: concise explanation of the identified issues
- severity: one of low|medium|high|critical
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior Python code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif "```" in response_text:
                response_text = response_text.replace("```", "").strip()
                
            parsed = json.loads(response_text)
            action = ReviewAction(**parsed)
            
        except Exception as e:
            print(f"Error calling OpenAI API or parsing result: {e}")
            action = ReviewAction(issues_found=[], review_comment="Error", severity="medium")
            
        reward = calculate_review_reward(action.issues_found, action.review_comment, task)
        print(f"[END] {json.dumps({'issues_found': action.issues_found, 'reward': reward})}")

if __name__ == "__main__":
    run_baseline()
