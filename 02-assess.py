"""
02 — ASSESS phase (coSTAR loop 2).

Scores the existing traces with the registered conciseness judge and then adds
a small set of (simulated) human assessments that encode a *learnable*
distinction: explanatory "how/why/differences" questions deserve a thorough,
multi-paragraph answer and should NOT be penalized for length (value=True),
whereas simple factual questions are expected to be short, so a long answer is
unnecessarily verbose (value=False). This is the A in STAR for loop 2:
attach both the judge's verdicts and human labels to the loop-1 traces so the
next phase can align the generic judge to that principle.

The generic conciseness judge (v1) is a one-sentence "is it concise? true/false"
judge that tends to penalize ANY long answer. The human labels here teach the
exception via their *rationales* — the natural-language signal MemAlign learns
from — so the aligned judge stops penalizing length on explanatory questions.

There is no TRACE phase in loop 2 — the traces already exist from loop 1
(the '01-refine' run). We read them back and assess them in place.

After running, open the traces in the MLflow UI to see the conciseness judge
verdicts alongside the human feedback.

Run loop 1 (01-trace / 01-assess / 01-refine) first, then 02-add-judge.py —
this phase reads the '01-refine' traces and the registered conciseness judge.
"""

import json
import sys

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import get_scorer
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from setup import experiment, traces_for_run

# ── Category-based human-label scheme (exactly 5 labels) ──────────────────
#
# The baseline prompt is deliberately verbose, so the agent produces long
# answers and the generic conciseness judge tends to say "not concise" for
# every question. The labels encode a LEARNABLE rule that the judge is missing:
#   * EXPLANATORY questions — a thorough, multi-paragraph answer is appropriate,
#     so a long answer should NOT be penalized: human value=True. Where the
#     generic judge said False, this label DISAGREES — the signal MemAlign learns.
#   * FACTUAL questions — a long, elaborate answer IS unnecessarily verbose:
#     human value=False. This matches what the judge already says for verbose
#     answers, but the rationale teaches *why* (a short answer was expected),
#     anchoring the contrast against the explanatory group.
EXPLANATORY_RATIONALE = (
    "This is an explanatory 'how/why' question — a thorough, multi-paragraph "
    "answer is appropriate here and should NOT be penalized for length."
)
FACTUAL_RATIONALE = (
    "This is a simple factual question — a long, elaborate answer is "
    "unnecessarily verbose; a short, direct answer is expected."
)

# Each entry: (exact question text, category, human value, rationale).
LABEL_SCHEME = [
    ("How does CRISPR gene editing work?", "explanatory", True, EXPLANATORY_RATIONALE),
    ("How do mRNA vaccines work?", "explanatory", True, EXPLANATORY_RATIONALE),
    (
        "How does quantum computing differ from classical computing?",
        "explanatory",
        True,
        EXPLANATORY_RATIONALE,
    ),
    ("What is the current population of Tokyo?", "factual", False, FACTUAL_RATIONALE),
    (
        "Who won the most recent FIFA World Cup and where was it held?",
        "factual",
        False,
        FACTUAL_RATIONALE,
    ),
]


def trace_question(trace):
    """Return the user question text from a trace's request inputs.

    The agent is invoked with ``{"messages": [{"role": "user", "content": q}]}``
    (see ``setup.run_scenarios``), serialized as the trace request JSON. We pull
    the first user message's content so traces can be matched to target
    questions by exact text rather than fragile positional index alignment.
    """
    request = json.loads(trace.data.request)
    return request["messages"][0]["content"]


# ── Source traces from the most-recent loop-1 agent run ───────────────────
traces = traces_for_run("01-refine")
if not traces:
    print("No traces found for run '01-refine'. Run loop 1 first (01-refine.py).")
    sys.exit(1)

# ── Load the registered conciseness judge (v1 from 02-add-judge) ──────────
try:
    conciseness = get_scorer(name="conciseness", experiment_id=experiment.experiment_id)
except MlflowException as e:
    if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
        raise
    print("No 'conciseness' scorer registered. Run `python 02-add-judge.py` first.")
    sys.exit(1)

print(f"Loaded registered judge '{conciseness.name}'; assessing {len(traces)} traces.")

# Score and add human feedback under a dedicated "02-assess" run so the next
# phase (and the UI) can find this assessment work by name.
with mlflow.start_run(run_name="02-assess"):
    # ── A (part 1): run the judge on all traces ──────────────────────────
    print("\n" + "=" * 70)
    print("Running registered conciseness judge on all traces …")
    print("=" * 70)
    mlflow.genai.evaluate(data=traces, scorers=[conciseness])

    # Re-fetch traces to pick up the judge's assessments, then extract verdicts.
    traces = [mlflow.get_trace(t.info.trace_id) for t in traces]

    judge_verdicts = {}
    for trace in traces:
        assessment = next((a for a in trace.info.assessments if a.name == "conciseness"), None)
        judge_verdicts[trace.info.trace_id] = assessment.value if assessment else False

    for tid, verdict in judge_verdicts.items():
        print(f"  trace {tid}: judge says '{verdict}'")

    # ── A (part 2): log category-based human feedback on 5 traces ────────
    #
    # The story for loop 2 is "align the judge from a *small* amount of human
    # feedback": a domain expert labels only 5 of the 10 questions. Rather than
    # arbitrary flips, the labels encode a learnable distinction (see
    # LABEL_SCHEME above) — explanatory questions may run long (value=True),
    # simple factual ones should stay short (value=False). Both the values and
    # the contrasting RATIONALES carry the principle MemAlign aligns to.
    print("\n" + "=" * 70)
    print("Logging human feedback (category-based) on a small labeled subset …")
    print("=" * 70)

    human_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="domain_expert",
    )

    # Match traces to target questions by exact question text (robust to the
    # non-deterministic order of search_traces).
    traces_by_question = {trace_question(t): t for t in traces}

    n_explanatory = 0
    n_factual = 0
    n_disagree = 0
    for question, category, human_val, rationale in LABEL_SCHEME:
        trace = traces_by_question.get(question)
        if trace is None:
            print(f"  SKIP (no trace matched): {question!r}")
            continue

        judge_val = judge_verdicts[trace.info.trace_id]
        disagrees = bool(human_val) != bool(judge_val)
        if disagrees:
            n_disagree += 1
        if category == "explanatory":
            n_explanatory += 1
        else:
            n_factual += 1

        mlflow.log_feedback(
            trace_id=trace.info.trace_id,
            name="conciseness",  # must match judge name
            value=human_val,
            source=human_source,
            rationale=rationale,
        )
        marker = "DISAGREE" if disagrees else "agree"
        print(
            f"  [{category:<11}] judge={str(judge_val):<5}  human={str(human_val):<5}  "
            f"[{marker}]  {question}"
        )

    n_labeled = n_explanatory + n_factual
    print(
        f"\n  Labeled {n_labeled}/{len(traces)} "
        f"({n_explanatory} explanatory, {n_factual} factual); "
        f"{n_disagree} disagree with the generic judge."
    )

print("\nOpen the traces in the MLflow UI to see judge verdicts + human feedback.")
