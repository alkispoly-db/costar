"""
STAR Loop 2 — Judge Alignment for Conciseness

Before we can refine the agent for conciseness, we need a judge we trust.
This script:

  1. Loads the optimized prompt v2 from the prompt registry (created by Loop 1).
  2. Creates a generic conciseness LLM judge via make_judge().
  3. Runs agent v2 on all scenarios.
  4. Runs the judge on every trace — the judge logs its assessments automatically.
  5. Simulates human feedback that disagrees with the judge on specific cases.
  6. Aligns the judge with human preferences using MemAlignOptimizer.
  7. Verifies the aligned judge now matches human opinions.
  8. Saves the aligned judge to _aligned_judge.json for Loop 3.

Key insight: you can't trust the agent refinement loop until you trust the judge.
"""

import json
from pathlib import Path

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer

from setup import JUDGE_MODEL, PROMPT_NAME, SCENARIOS, create_agent, find_prompt_by_tag, run_scenarios

# ── Load the v2 prompt from the registry (created by 01_star_objective.py) ──

prompt_v2 = find_prompt_by_tag(PROMPT_NAME, "costar_step", "optimized")
print(f"Loaded prompt '{PROMPT_NAME}' v{prompt_v2.version} from registry.")
print(f"  Template: {prompt_v2.template[:120]}…\n")

# ── Generic conciseness judge ─────────────────────────────────────────────

conciseness_judge = make_judge(
    name="conciseness",
    instructions=(
        "Evaluate if {{ outputs }} provides a concise, direct answer to "
        "{{ inputs }}. A concise answer gets to the point quickly without "
        "unnecessary elaboration, filler phrases, or repeated information.\n\n"
        "Respond true if the answer is concise, false if it is verbose."
    ),
    model=JUDGE_MODEL,
    feedback_value_type=bool,
)

print(f"Judge: '{conciseness_judge.name}' (model={JUDGE_MODEL})")
print(f"  Instructions: {conciseness_judge.instructions[:120]}…\n")

# ── S & T: Run agent v2 and collect traces ────────────────────────────────

print("=" * 70)
print("STAR Loop 2 — Running agent v2 to generate traces for judge alignment")
print("=" * 70)

agent_v2 = create_agent(prompt_v2.template)
traces = run_scenarios(agent_v2, SCENARIOS, run_name="agent-v2-for-alignment")

# ── A (part 1): Run conciseness judge on all traces ──────────────────────

print("\n" + "=" * 70)
print("Running generic judge on all traces …")
print("=" * 70)
mlflow.genai.evaluate(data=traces, scorers=[conciseness_judge])

# Re-fetch traces to pick up the judge's assessments, then extract verdicts.
traces = [mlflow.get_trace(t.info.trace_id) for t in traces]

judge_verdicts = {}
for trace in traces:
    assessment = next((a for a in trace.info.assessments if a.name == "conciseness"), None)
    judge_verdicts[trace.info.trace_id] = assessment.value if assessment else False

for tid, verdict in judge_verdicts.items():
    print(f"  trace {tid}: judge says '{verdict}'")

# ── A (part 2): Simulate human feedback ──────────────────────────────────
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

human_verdicts = {}
for i, trace in enumerate(traces):
    judge_val = judge_verdicts[trace.info.trace_id]
    if i in flip_yes_to_no:
        human_val = False  # human thinks the answer rambles
    elif i in flip_no_to_yes:
        human_val = True  # human thinks a longer answer is fine here
    else:
        human_val = judge_val  # agree with the judge

    human_verdicts[trace.info.trace_id] = human_val
    agrees = "agree" if human_val == judge_val else "DISAGREE"

    mlflow.log_feedback(
        trace_id=trace.info.trace_id,
        name="conciseness",  # must match judge name
        value=human_val,
        source=human_source,
        rationale=(
            f"Human {'agrees' if human_val == judge_val else 'disagrees'} "
            f"with judge. Question: {SCENARIOS[i]['question'][:50]}…"
        ),
    )
    print(f"  trace {i:>2}: judge={str(judge_val):<5}  human={str(human_val):<5}  [{agrees}]")

n_disagree = len(flip_yes_to_no) + len(flip_no_to_yes)
print(f"\n  Disagreements: {n_disagree}/{len(traces)} ({n_disagree/len(traces):.0%})")

# ── R: Align the judge ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Aligning judge with human feedback via MemAlign …")
print("=" * 70)

optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL)

# Re-fetch traces so they include all assessments (judge + human)
traces = [mlflow.get_trace(t.info.trace_id) for t in traces]
aligned_judge = conciseness_judge.align(traces=traces, optimizer=optimizer)

print("\n  Original instructions:")
print(f"    {conciseness_judge.instructions[:200]}…")
print("\n  Aligned instructions:")
print(f"    {aligned_judge.instructions[:400]}…")

# ── Verify: aligned judge matches human preferences ──────────────────────

print("\n" + "=" * 70)
print("Verifying aligned judge on traces where human disagreed …")
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

# ── Save aligned judge for Loop 3 ─────────────────────────────────────────

Path("_aligned_judge.json").write_text(json.dumps(aligned_judge.model_dump()))
print("\nSaved aligned judge to _aligned_judge.json")
print("Done. The aligned conciseness judge is ready for Loop 3.")
