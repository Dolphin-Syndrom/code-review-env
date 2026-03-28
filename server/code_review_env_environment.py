# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code Review environment implementation for OpenEnv."""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ReviewAction, ReviewObservation, ReviewState
    from .graders import grade_review_with_breakdown
    from .tasks import get_task
except ImportError:
    from models import ReviewAction, ReviewObservation, ReviewState
    from server.graders import grade_review_with_breakdown
    from server.tasks import get_task


MAX_STEPS = 3


class CodeReviewEnvironment(Environment):
    """Environment where an agent reviews code and tags planted issues."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        default_task = get_task("task_easy")
        self._state = ReviewState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_id=default_task.task_id,
            max_steps=MAX_STEPS,
        )
        self._current_task = default_task

    def reset(self, task_id: str = "task_easy", **kwargs) -> ReviewObservation:
        """Reset episode and load selected task (fallback to task_easy)."""
        _ = kwargs
        task = get_task(task_id)
        self._current_task = task
        self._state = ReviewState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_id=task.task_id,
            max_steps=MAX_STEPS,
        )

        return ReviewObservation(
            task_id=task.task_id,
            file_name=task.file_name,
            task_description=task.description,
            code_snippet=task.code,
            feedback="Environment reset. Submit issues_found and review_comment.",
            step_number=0,
            reward=0.0,
            done=False,
            metadata={
                "difficulty": task.difficulty,
                "planted_issue_count": len(task.planted_issues),
            },
        )

    def step(self, action: ReviewAction) -> ReviewObservation:  # type: ignore[override]
        """Grade one review action and return updated observation."""
        self._state.step_count += 1

        breakdown = grade_review_with_breakdown(
            action_issues=action.issues_found,
            action_comment=action.review_comment,
            task=self._current_task,
        )

        score = breakdown.score
        done = (score >= 0.95) or (self._state.step_count >= MAX_STEPS)

        correctly_found = sorted(breakdown.correctly_found)
        missed_count = len(breakdown.missed)
        false_positive_count = len(breakdown.false_positives)

        feedback = (
            f"Score: {score:.3f} | Found: {correctly_found} | "
            f"Missed: {missed_count} remaining | False positives: {false_positive_count}"
        )

        return ReviewObservation(
            task_id=self._current_task.task_id,
            file_name=self._current_task.file_name,
            task_description=self._current_task.description,
            code_snippet=self._current_task.code,
            feedback=feedback,
            step_number=self._state.step_count,
            reward=score,
            done=done,
            metadata={
                "correctly_found": correctly_found,
                "missed": sorted(breakdown.missed),
                "false_positives": sorted(breakdown.false_positives),
                "submitted_severity": action.severity,
            },
        )

    @property
    def state(self) -> ReviewState:
        """Return current episode state."""
        return self._state
