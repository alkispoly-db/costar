"""
02 — ASSESS phase (coSTAR loop 2).

Logs a small set of (simulated) human assessments on the loop-1 traces. The
labels encode a *learnable* distinction: explanatory "how/why/differences"
questions deserve a thorough, multi-paragraph answer and should NOT be penalized
for length (value=True), whereas simple factual questions are expected to be
short, so a long answer is unnecessarily verbose (value=False). This is the A in
STAR for loop 2: attach human labels to the loop-1 traces so the next phase can
align the generic judge to that principle.

The conciseness LLM judge is NOT applied here. It is only *applied* later, after
alignment (in 03-loop, with the aligned v2 judge). 02-refine builds the generic
judge itself to compute its before-alignment verdicts — this phase logs human
labels only, so the demo narrative stays clean.

There is no TRACE phase in loop 2 — the traces already exist from the baseline
agent run (the '01-trace' run). We read them back and label them in place.
Loop 2 only needs baseline agent traces to assess conciseness on, so it sources
from '01-trace' rather than the (slower, optimizer-driven) '01-refine' step.

After running, open the traces in the MLflow UI to see the human feedback.

Run 01-trace.py first, then 02-add-judge.py — this phase reads the
'01-trace' traces.
"""

import json
import sys

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from setup import traces_for_run

# ── Category-based human-label scheme (exactly 5 labels) ──────────────────
#
# The baseline prompt is deliberately verbose, so the agent produces long
# answers and the generic conciseness judge tends to say "not concise" for
# every question. The labels encode a LEARNABLE rule that the judge is missing:
#   * EXPLANATORY questions — a thorough, multi-paragraph answer is appropriate,
#     so a long answer should NOT be penalized: human value=True.
#   * FACTUAL questions — a long, elaborate answer IS unnecessarily verbose:
#     human value=False.
# Both the values and the contrasting RATIONALES carry the principle MemAlign
# aligns to (the natural-language signal it learns from).
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


# ── Source traces from the baseline agent run ─────────────────────────────
traces = traces_for_run("01-trace")
if not traces:
    print("No traces found for run '01-trace'. Run 01-trace.py first.")
    sys.exit(1)

print(f"Logging human labels on {len(traces)} baseline traces.")

# Log human feedback under a dedicated "02-assess" run so the next phase (and
# the UI) can find this assessment work by name.
with mlflow.start_run(run_name="02-assess"):
    # ── A: log category-based human feedback on 5 traces ─────────────────
    #
    # The story for loop 2 is "align the judge from a *small* amount of human
    # feedback": a domain expert labels only 5 of the questions. The labels
    # encode a learnable distinction (see LABEL_SCHEME above) — explanatory
    # questions may run long (value=True), simple factual ones should stay short
    # (value=False). Both the values and the contrasting RATIONALES carry the
    # principle MemAlign aligns to. The judge itself is applied later, after
    # alignment — not here.
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
    for question, category, human_val, rationale in LABEL_SCHEME:
        trace = traces_by_question.get(question)
        if trace is None:
            print(f"  SKIP (no trace matched): {question!r}")
            continue

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
        print(f"  [{category:<11}] human={str(human_val):<5}  {question}")

    n_labeled = n_explanatory + n_factual
    print(
        f"\n  Logged {n_labeled} human labels: "
        f"{n_explanatory} explanatory (concise=True), "
        f"{n_factual} factual (concise=False). "
        f"The judge is applied later, after alignment."
    )

print("\nOpen the traces in the MLflow UI to see the human feedback.")
