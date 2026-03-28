from dataclasses import dataclass

from .tasks import Task


ISSUE_KEYWORDS: dict[str, list[str]] = {
    "null_pointer": ["null", "none", "not check", "missing check"],
    "missing_return": ["return", "missing", "no return", "never returns", "none returned"],
    "sql_injection": ["sql", "injection", "f-string", "sanitize", "parameterize"],
    "hardcoded_secret": ["hardcoded", "secret", "credential", "env var", "plaintext"],
    "race_condition": ["race", "atomic", "concurrent", "lock", "thread"],
    "timing_attack": ["timing", "constant time", "hmac", "compare_digest"],
    "improper_error_handling": ["except", "swallow", "silent", "bare except"],
    "type_error": ["type", "string", "int", "cast"],
    "index_out_of_bounds": ["index", "bounds", "length", "len("],
}


@dataclass(frozen=True)
class GradeBreakdown:
    score: float
    correctly_found: set[str]
    missed: set[str]
    false_positives: set[str]


def _comment_has_quality_signal(issue_tag: str, comment: str) -> bool:
    keywords = ISSUE_KEYWORDS.get(issue_tag, [])
    lowered_comment = comment.lower()
    return any(keyword in lowered_comment for keyword in keywords)


def grade_review(
    action_issues: list[str],
    action_comment: str,
    task: Task,
) -> float:
    """
    Deterministic grader for code review actions.

    Formula:
        base_score = |correct| / |planted|
        quality_bonus = +0.05 for each correct issue with matching keywords in comment
        precision_penalty = -0.1 for each false-positive issue
        final = clamp(base + bonus - penalty, 0.0, 1.0)
    """
    try:
        submitted = set(action_issues or [])
        planted = set(task.planted_issues or [])

        if not submitted or not planted:
            return 0.0

        correctly_found = submitted & planted
        false_positives = submitted - planted

        base_score = len(correctly_found) / len(planted)

        quality_bonus = 0.0
        safe_comment = action_comment or ""
        for issue_tag in correctly_found:
            if _comment_has_quality_signal(issue_tag, safe_comment):
                quality_bonus += 0.05

        precision_penalty = 0.1 * len(false_positives)

        raw_score = base_score + quality_bonus - precision_penalty
        return float(max(0.0, min(1.0, raw_score)))
    except Exception:
        return 0.0


def grade_review_with_breakdown(
    action_issues: list[str],
    action_comment: str,
    task: Task,
) -> GradeBreakdown:
    """Utility helper for environment feedback text and endpoint diagnostics."""
    try:
        submitted = set(action_issues or [])
        planted = set(task.planted_issues or [])

        correctly_found = submitted & planted
        false_positives = submitted - planted
        missed = planted - submitted

        score = grade_review(action_issues, action_comment, task)
        return GradeBreakdown(
            score=score,
            correctly_found=correctly_found,
            missed=missed,
            false_positives=false_positives,
        )
    except Exception:
        return GradeBreakdown(score=0.0, correctly_found=set(), missed=set(), false_positives=set())
