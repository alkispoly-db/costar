"""
STAR Loop 3 — Agent Refinement with the Aligned Conciseness Judge

Now that we have a conciseness judge aligned to human preferences (Loop 2),
we can safely use it to refine the agent — the "coupled" in coSTAR.

This script:

  1. Loads the aligned conciseness judge saved by Loop 2.
  2. Loads prompt v2 from the prompt registry.
  3. Runs agent v2 on all scenarios and evaluates with BOTH the citation
     scorer and the aligned conciseness judge via mlflow.genai.evaluate().
  4. Uses optimize_prompts() with MetaPromptOptimizer to automatically
     generate an improved prompt (v3) that adds conciseness.
  5. Re-evaluates v3 with both scorers to confirm conciseness improves
     without regressing on citations.
  6. Prints a final 3-version comparison table (v1 → v2 → v3).

Key insight: evaluate with ALL criteria at every step to catch regressions.

Usage:
  python 03_star_subjective.py                     # default: metaprompt
  python 03_star_subjective.py --refine=claude-code # use Claude Code headless
"""

import argparse
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Scorer

from setup import (
    JUDGE_MODEL,
    PROMPT_NAME,
    SCENARIOS,
    TRAIN_DATA,
    create_agent,
    find_prompt_by_tag,
    has_sources,
    predict_fn,
    run_scenarios,
)

parser = argparse.ArgumentParser(description="STAR Loop 3 — Subjective Judge")
parser.add_argument(
    "--refine",
    choices=["metaprompt", "claude-code"],
    default="metaprompt",
    help="Which Refine engine to use (default: metaprompt)",
)
args = parser.parse_args()

# ── Load the aligned conciseness judge ────────────────────────────────────

print("=" * 70)
print("Loading aligned conciseness judge from Loop 2 …")
print("=" * 70)

judge_file = Path("_aligned_judge.json")
if not judge_file.exists():
    raise RuntimeError("No aligned judge found. Run 02_star_judge_align.py first.")

aligned_judge = Scorer.model_validate_json(judge_file.read_text())
print(f"  Loaded aligned judge '{aligned_judge.name}' from {judge_file}\n")


# ── Helpers ───────────────────────────────────────────────────────────────


def eval_both(traces, label: str) -> tuple[float, float]:
    """Evaluate traces with both has_sources and aligned conciseness judge."""
    result = mlflow.genai.evaluate(data=traces, scorers=[has_sources, aligned_judge])
    cite = result.metrics.get("has_sources/mean", 0.0)
    concise = result.metrics.get("conciseness/mean", 0.0)
    print(f"  {label}  has_sources  = {cite:.0%}")
    print(f"  {label}  conciseness  = {concise:.0%}")
    return cite, concise


# ── Load prompt v2 from the registry ─────────────────────────────────────

prompt_v2 = find_prompt_by_tag(PROMPT_NAME, "costar_step", "optimized")
print(f"Loaded prompt '{PROMPT_NAME}' v{prompt_v2.version} from registry.\n")

# ── S & T: Run agent v2 ──────────────────────────────────────────────────

print("=" * 70)
print("STAR Loop 3 — Agent v2 (citations, no conciseness instructions)")
print("=" * 70)

agent_v2 = create_agent(prompt_v2.template)
traces_v2 = run_scenarios(agent_v2, SCENARIOS, run_name="agent-v2-loop3")

# ── A: Evaluate v2 with BOTH scorers ─────────────────────────────────────

print("\nEvaluating agent v2 with citation scorer + aligned conciseness judge …")
v2_cite, v2_concise = eval_both(traces_v2, "v2")

# ── R: Automated prompt optimization → v3 ────────────────────────────────
# We pass only has_sources to optimize_prompts scorers (the aligned judge has
# issues inside the optimizer's eval loop). The conciseness objective is
# conveyed via guidelines= so the optimizer knows to improve it. We verify
# conciseness with the aligned judge after optimization.

if args.refine == "metaprompt":
    from mlflow.genai.optimize import optimize_prompts
    from mlflow.genai.optimize.optimizers import MetaPromptOptimizer

    print("\n" + "=" * 70)
    print("Optimizing prompt for conciseness with MetaPromptOptimizer …")
    print("=" * 70)

    opt_result = optimize_prompts(
        predict_fn=predict_fn,
        train_data=TRAIN_DATA,
        prompt_uris=[prompt_v2.uri],
        optimizer=MetaPromptOptimizer(
            reflection_model=JUDGE_MODEL,
            guidelines=(
                "Responses must cite sources with URLs AND be concise "
                "(one short paragraph, no filler)."
            ),
        ),
        scorers=[has_sources],
    )

    prompt_v3 = opt_result.optimized_prompts[0]
    print(f"\nOptimized prompt registered as v{prompt_v3.version}:")
    print(f"  {prompt_v3.template[:200]}…")
    print(f"\n  Baseline score: {opt_result.initial_eval_score}")
    print(f"  Optimized score: {opt_result.final_eval_score}")

elif args.refine == "claude-code":
    from refine_claude_code import HAS_SOURCES_SCORER, refine_with_claude_code

    print("\n" + "=" * 70)
    print("Optimizing prompt for conciseness with Claude Code …")
    print("=" * 70)

    prompt_v3 = refine_with_claude_code(
        prompt_name=PROMPT_NAME,
        prompt_version=prompt_v2.version,
        scores={"has_sources": v2_cite, "conciseness": v2_concise},
        goal=(
            "Responses must cite sources with URLs AND be concise "
            "(one short paragraph, no filler)."
        ),
        project_dir=str(Path(__file__).parent),
        scorer_source=HAS_SOURCES_SCORER,
    )

    print(f"\nOptimized prompt registered as v{prompt_v3.version}:")
    print(f"  {prompt_v3.template[:200]}…")

# ── S & T: Run agent v3 ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STAR Loop 3 — Agent v3 (citations + conciseness)")
print("=" * 70)

agent_v3 = create_agent(prompt_v3.template)
traces_v3 = run_scenarios(agent_v3, SCENARIOS, run_name="agent-v3")

# ── A: Evaluate v3 with BOTH scorers ─────────────────────────────────────

print("\nEvaluating agent v3 …")
v3_cite, v3_concise = eval_both(traces_v3, "v3")

# ── Side-by-side comparison: v2 vs v3 ────────────────────────────────────

print("\n" + "=" * 70)
print("Loop 3 Results: v2 → v3")
print("=" * 70)
print(f"  {'Metric':<20} {'v2':>8} {'v3':>8} {'Delta':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
print(
    f"  {'has_sources':<20} {v2_cite:>7.0%} {v3_cite:>7.0%}"
    f" {v3_cite - v2_cite:>+7.0%}"
)
print(
    f"  {'conciseness':<20} {v2_concise:>7.0%} {v3_concise:>7.0%}"
    f" {v3_concise - v2_concise:>+7.0%}"
)

# ── Final 3-version summary ──────────────────────────────────────────────
#
# v1 scores come from Loop 1. We hard-code placeholders here; in practice
# you'd load them from the MLflow experiment. For the blog narrative we
# print the structure — the actual numbers depend on the LLM's behavior.

print("\n" + "=" * 70)
print("coSTAR Journey: v1 → v2 → v3")
print("=" * 70)
print(
    f"  {'Version':<12} {'Loop':<12} {'has_sources':>12} {'conciseness':>12}  Notes"
)
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}  {'-'*30}")
print(f"  {'v1':<12} {'Loop 1':<12} {'(low)':>12} {'—':>12}  Baseline agent")
print(
    f"  {'v2':<12} {'Loop 1→3':<12} {v2_cite:>11.0%} {v2_concise:>11.0%}"
    f"  + citations, still verbose"
)
print(
    f"  {'v3':<12} {'Loop 3':<12} {v3_cite:>11.0%} {v3_concise:>11.0%}"
    f"  + citations + conciseness"
)
print()
print("Key takeaway:")
print("  - Objective criteria (citations) → 1 STAR loop")
print("  - Subjective criteria (conciseness) → 2 STAR loops (align judge, then refine agent)")
print("  - This is the 'coupled' in coSTAR: trust the judge before trusting its scores.")
print("\nDone. Browse the MLflow UI to see all runs, traces, evaluations, and prompt versions.")
print("Check the 'Prompts' tab for 'research-agent' with 3 versions and diffs between them.")
