"""
Canonical home for the conciseness LLM judge.

A single source of truth for the conciseness judge's instructions and its
``make_judge`` construction, shared by ``setup.py`` (which registers it as an
experiment scorer in loop 2) and any per-phase script that needs the judge.

Imports only from mlflow — NOT from ``setup`` — so that ``setup.py`` can import
from here without a circular import.
"""

from mlflow.genai.judges import make_judge

# Instructions match the inline judge used during alignment (02_star_judge_align.py)
# verbatim so the registered scorer behaves identically.
CONCISENESS_INSTRUCTIONS = (
    "Evaluate if {{ outputs }} provides a concise, direct answer to "
    "{{ inputs }}. Respond true if the answer is concise, false if it is verbose."
)


def build_conciseness_judge(model):
    """Build the conciseness judge for *model* (e.g. ``"openai:/gpt-4.1-mini"``).

    Returns an unregistered judge; callers that want a first-class experiment
    scorer (e.g. ``setup.get_conciseness_scorer``) register the result.
    """
    return make_judge(
        name="conciseness",
        instructions=CONCISENESS_INSTRUCTIONS,
        model=model,
        feedback_value_type=bool,
    )
