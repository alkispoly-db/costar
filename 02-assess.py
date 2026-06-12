"""
02 — ASSESS phase (coSTAR loop 2).

Scores the existing traces with the registered conciseness judge and then adds
(simulated) human assessments that disagree with the judge on about half of
them. This is the A in STAR for loop 2: attach both the judge's verdicts and
human labels to the loop-1 traces so the next phase can align the judge to the
human preferences.

There is no TRACE phase in loop 2 — the traces already exist from loop 1
(the '01-refine' run). We read them back and assess them in place.

After running, open the traces in the MLflow UI to see the conciseness judge
verdicts alongside the human feedback.

Run loop 1 (01-trace / 01-assess / 01-refine) first, then 02-add-judge.py —
this phase reads the '01-refine' traces and the registered conciseness judge.
"""

import sys

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import get_scorer
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from setup import experiment, load_scenarios, traces_for_run

# Source scenarios from the eval dataset; reused below so the per-trace human
# feedback rationales stay index-aligned with the traces.
scenarios = load_scenarios()

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

    # ── A (part 2): simulate human feedback ──────────────────────────────
    #
    # Humans have domain-specific opinions about conciseness that the generic
    # judge doesn't capture. We simulate that by logging human assessments that
    # *disagree* with the judge on a subset of traces.
    #
    # To ensure ~30-40% disagreement regardless of how the judge scores, we
    # split traces where the judge said "yes" and "no", then flip about half
    # of each group:
    #   - Judge said "no" → human says "yes" for half (nuanced questions deserve
    #     a few sentences; the judge was too strict)
    #   - Judge said "yes" → human says "no" for half (answers ramble despite
    #     the judge calling them concise)
    print("\n" + "=" * 70)
    print("Logging human feedback (simulated) …")
    print("=" * 70)

    human_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="domain_expert",
    )

    # Split traces by judge verdict so we can flip a balanced subset
    judge_yes_idxs = [i for i, t in enumerate(traces) if judge_verdicts[t.info.trace_id] is True]
    judge_no_idxs = [i for i, t in enumerate(traces) if judge_verdicts[t.info.trace_id] is False]

    # Flip roughly half of each group (at least 2 from each side)
    flip_yes_to_no = set(judge_yes_idxs[: max(2, len(judge_yes_idxs) // 2)])
    flip_no_to_yes = set(judge_no_idxs[: max(2, len(judge_no_idxs) // 2)])

    for i, trace in enumerate(traces):
        judge_val = judge_verdicts[trace.info.trace_id]
        if i in flip_yes_to_no:
            human_val = False  # human thinks the answer rambles
        elif i in flip_no_to_yes:
            human_val = True  # human thinks a longer answer is fine here
        else:
            human_val = judge_val  # agree with the judge

        agrees = "agree" if human_val == judge_val else "DISAGREE"

        mlflow.log_feedback(
            trace_id=trace.info.trace_id,
            name="conciseness",  # must match judge name
            value=human_val,
            source=human_source,
            rationale=(
                f"Human {'agrees' if human_val == judge_val else 'disagrees'} "
                f"with judge. Question: {scenarios[i]['question'][:50]}…"
            ),
        )
        print(f"  trace {i:>2}: judge={str(judge_val):<5}  human={str(human_val):<5}  [{agrees}]")

    n_disagree = len(flip_yes_to_no) + len(flip_no_to_yes)
    print(f"\n  Disagreements: {n_disagree}/{len(traces)} ({n_disagree/len(traces):.0%})")

print("\nOpen the traces in the MLflow UI to see judge verdicts + human feedback.")
