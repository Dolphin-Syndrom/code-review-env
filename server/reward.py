import numpy as np
from typing import List

from .graders import grade_review
from .tasks import Task

def calculate_review_reward(
    predicted_issues: List[str],
    predicted_comment: str,
    task: Task,
) -> float:
    """
    Calculates the reward strictly in the [0.0, 1.0] range based on accuracy.
    """
    return grade_review(predicted_issues, predicted_comment, task)

def calculate_grpo_rewards(
    group_predicted_issues: List[List[str]],
    group_predicted_comments: List[str],
    task: Task,
) -> List[float]:
    """
    Calculates Group Relative Policy Optimization (GRPO) rewards for a group of predictions.
    This function computes the raw review rewards for every rollout in the group,
    and applies GRPO scoring (zero-mean, unit-variance normalization) so that the
    agent learns relative to the group's average performance.
    """
    raw_rewards = [
        calculate_review_reward(issues, comment, task)
        for issues, comment in zip(group_predicted_issues, group_predicted_comments)
    ]
    
    if len(raw_rewards) <= 1:
        # Cannot normalize a group of 1 or fewer
        return raw_rewards
        
    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards)
    
    if std_reward < 1e-8:
        # All actions resulted in the same reward
        return [0.0 for _ in raw_rewards]
        
    grpo_rewards = [(float(r) - mean_reward) / std_reward for r in raw_rewards]
    return grpo_rewards
