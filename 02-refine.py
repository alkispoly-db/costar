"""
02 — REFINE phase (coSTAR loop 2).

Aligns the conciseness judge to the (simulated) human labels with MemAlign and
registers the result as the next version of the experiment's 'conciseness'
scorer. This is the R in STAR for loop 2: turn the human assessments into a
judge we trust, then hand it off through the registry.

After running, browse the Judges panel: 'conciseness' now has two versions —
v1 (the generic judge from 02-add-judge) and v2 (the MemAligned judge).

Run 02-assess.py first — this phase re-reads the '01-refine' traces *with* their
judge + human assessments and aligns against them. The registry is the handoff
to loop 3 (no JSON file).
"""

import sys

import mlflow
from mlflow.genai.judges.optimizers import MemAlignOptimizer

from conciseness_judge import build_conciseness_judge
from setup import JUDGE_MODEL, experiment, load_scenarios, traces_for_run

# Reused for the verification print below; index-aligned with the traces.
scenarios = load_scenarios()

# ── Build the conciseness judge to align ──────────────────────────────────
# Same instructions as the registered v1 (single source of truth in
# conciseness_judge.py). align() operates on a make_judge object, so we build
# one here rather than loading the registered scorer.
conciseness_judge = build_conciseness_judge(JUDGE_MODEL)

# ── Re-fetch the assessed traces WITH assessments ─────────────────────────
base_traces = traces_for_run("01-refine")
if not base_traces:
    print("No traces found for run '01-refine'. Run loop 1 first (01-refine.py).")
    sys.exit(1)

traces = [mlflow.get_trace(t.info.trace_id) for t in base_traces]

# Capture judge + human verdicts so we can verify the alignment afterwards, and
# confirm the human feedback from 02-assess is present.
judge_verdicts = {}
human_verdicts = {}
for trace in traces:
    assessments = trace.info.assessments
    judge_a = next(
        (a for a in assessments if a.name == "conciseness" and a.source.source_type != "HUMAN"),
        None,
    )
    human_a = next(
        (a for a in assessments if a.name == "conciseness" and a.source.source_type == "HUMAN"),
        None,
    )
    judge_verdicts[trace.info.trace_id] = judge_a.value if judge_a else False
    human_verdicts[trace.info.trace_id] = human_a.value if human_a else None

if all(v is None for v in human_verdicts.values()):
    print("Traces carry no human 'conciseness' feedback. Run `python 02-assess.py` first.")
    sys.exit(1)

# ── R: align the judge with human feedback via MemAlign ───────────────────
print("=" * 70)
print("Aligning judge with human feedback via MemAlign …")
print("=" * 70)

optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL)
aligned_judge = conciseness_judge.align(traces=traces, optimizer=optimizer)

print("\n  Original instructions:")
print(f"    {conciseness_judge.instructions[:200]}…")
print("\n  Aligned instructions:")
print(f"    {aligned_judge.instructions[:400]}…")

# ── Verify: aligned judge matches human preferences ──────────────────────
print("\n" + "=" * 70)
print("Verifying aligned judge against human labels …")
print("=" * 70)

matches_before = 0
matches_after = 0
total = len(traces)

for i, trace in enumerate(traces):
    human_val = human_verdicts[trace.info.trace_id]
    judge_val = judge_verdicts[trace.info.trace_id]
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
