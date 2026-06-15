"""
02 — REFINE phase (coSTAR loop 2).

Aligns the conciseness judge to the (simulated) human labels with MemAlign and
registers the result as the next version of the experiment's 'conciseness'
scorer. This is the R in STAR for loop 2: turn the human assessments into a
judge we trust, then hand it off through the registry.

After running, browse the Judges panel: 'conciseness' now has two versions —
v1 (the generic judge from 02-add-judge) and v2 (the MemAligned judge).

This phase is self-contained: it builds the generic conciseness judge itself,
reads the human labels logged by 02-assess, computes the generic judge's
*before* verdicts by calling the judge directly (02-assess no longer stores any
judge verdict), aligns against the human labels, then computes the *after*
verdicts with the aligned judge for the "Agreement with humans: X/5 → Y/5"
demo print. MemAlign learns purely from the human feedback + each trace's
input/output, so the judge does not need to have scored the traces beforehand.

Run 02-assess.py first — this phase re-reads the '01-refine' traces and aligns
against their human feedback. The registry is the handoff to loop 3 (no JSON
file).
"""

import sys

import mlflow
from mlflow.genai.judges.optimizers import MemAlignOptimizer

from conciseness_judge import build_conciseness_judge
from setup import JUDGE_MODEL, experiment, traces_for_run

# ── Build the generic conciseness judge to align ──────────────────────────
# Same instructions as the registered v1 (single source of truth in
# conciseness_judge.py). align() operates on a make_judge object, and we also
# call this judge directly to compute the before-alignment verdicts — so we
# build one here rather than loading the registered scorer.
generic_judge = build_conciseness_judge(JUDGE_MODEL)

# ── Re-fetch the loop-1 traces WITH their human assessments ───────────────
base_traces = traces_for_run("01-refine")
if not base_traces:
    print("No traces found for run '01-refine'. Run loop 1 first (01-refine.py).")
    sys.exit(1)

traces = [mlflow.get_trace(t.info.trace_id) for t in base_traces]

# Find the human conciseness label on each trace (02-assess only labels a small
# subset). Both the alignment and the before/after verification operate over
# just those labeled traces — the others carry no human signal to align against.
human_verdicts = {}
for trace in traces:
    human_a = next(
        (a for a in trace.info.assessments if a.name == "conciseness" and a.source.source_type == "HUMAN"),
        None,
    )
    human_verdicts[trace.info.trace_id] = human_a.value if human_a else None

labeled_traces = [t for t in traces if human_verdicts[t.info.trace_id] is not None]
if not labeled_traces:
    print("Traces carry no human 'conciseness' feedback. Run `python 02-assess.py` first.")
    sys.exit(1)
print(f"Found human labels on {len(labeled_traces)}/{len(traces)} traces.")

# ── Compute the generic judge's BEFORE verdicts ourselves ─────────────────
# 02-assess logs only human labels — it no longer scores traces with the judge.
# So we compute the generic (pre-alignment) verdict on each labeled trace here
# by calling the judge directly. MemAlign does not require these to be stored on
# the trace; this is purely for the before/after demo comparison.
print("\n" + "=" * 70)
print("Scoring labeled traces with the generic judge (before alignment) …")
print("=" * 70)
generic_verdicts = {t.info.trace_id: generic_judge(trace=t).value for t in labeled_traces}

# ── R: align the judge with human feedback via MemAlign ───────────────────
print("\n" + "=" * 70)
print("Aligning judge with human feedback via MemAlign …")
print("=" * 70)

# Pass only the labeled traces: MemAlign learns from the human feedback, and
# unlabeled traces would contribute no signal (and would in fact raise, since
# they carry no human assessment for the judge).
optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL)
aligned_judge = generic_judge.align(traces=labeled_traces, optimizer=optimizer)

print("\n  Original instructions:")
print(f"    {generic_judge.instructions[:200]}…")
print("\n  Aligned instructions:")
print(f"    {aligned_judge.instructions[:400]}…")

# ── Verify: aligned judge matches human preferences ──────────────────────
print("\n" + "=" * 70)
print("Verifying aligned judge against human labels …")
print("=" * 70)

matches_before = 0
matches_after = 0
total = len(labeled_traces)

for i, trace in enumerate(labeled_traces):
    human_val = human_verdicts[trace.info.trace_id]
    judge_val = generic_verdicts[trace.info.trace_id]
    aligned_val = aligned_judge(trace=trace).value

    before_ok = judge_val == human_val
    after_ok = aligned_val == human_val
    matches_before += before_ok
    matches_after += after_ok

    marker = ""
    if not before_ok and after_ok:
        marker = " << FIXED"
    elif before_ok and not after_ok:
        marker = " << REGRESSED"
    print(
        f"  trace {i:>2}: human={str(human_val):<5}  "
        f"original={str(judge_val):<5}  aligned={str(aligned_val):<5}{marker}"
    )

print(f"\n  Agreement with humans: {matches_before}/{total} → {matches_after}/{total}")

# ── Register aligned judge as a new 'conciseness' scorer version ──────────
#
# 02-add-judge already registered the generic conciseness judge as v1.
# Registering the aligned judge here creates v2 (register() versions on each
# call), so the experiment scorer's latest version is now the MemAligned one.
# Loop 3 loads the latest version — the registry is the handoff, no JSON file.
aligned_judge.register(name="conciseness", experiment_id=experiment.experiment_id)

print("\nRegistered MemAligned judge as scorer 'conciseness' (new version).")
print("  'conciseness' now has v1 (generic) and v2 (aligned).")
print("Open the Judges panel in the MLflow UI to see both versions.")
