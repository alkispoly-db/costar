"""
03 — LOOP phase (coSTAR loop 3).

The 3rd STAR loop: REFINE the agent prompt for conciseness using the ALIGNED
conciseness judge (Loop 2's output, now the registry's latest 'conciseness'
version) while guarding citations from regressing.

This is optimize_prompts() — NOT Claude Code. It optimizes against BOTH judges:
has_sources guards citations while the aligned conciseness judge actively drives
conciseness up, with guidelines= reinforcing the same smart-conciseness rule. It
registers research-agent v3 and logs the final evaluation (both metrics) under
one run named '03-loop'.

Run loops 1 and 2 first so the registry latest is the right version: loop 1
gives the citations-optimized prompt, and loop 2 aligns the conciseness judge so
the latest 'conciseness' version is the aligned (v2) one — we load the latest,
never a hardcoded version.
"""

import argparse
from pathlib import Path

import mlflow
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.scorers import get_scorer

from common import (
    create_agent,
    experiment,
    get_has_sources_scorer,
    get_train_data,
    latest_prompt,
    predict_fn,
    reflection_optimizer,
    run_scenarios,
)

parser = argparse.ArgumentParser(
    description="Loop 3 Refine: improve the agent prompt for conciseness."
)
parser.add_argument(
    "--refine",
    choices=["metaprompt", "claude-code"],
    default="metaprompt",
    help="Which Refine engine to use (default: metaprompt)",
)
args = parser.parse_args()

# ── Load both scorers ──────────────────────────────────────────────────────
# Run loops 1 and 2 first so the latest 'conciseness' version is the aligned
# (v2) judge from loop 2; we load the latest, never a hardcoded version.
has_sources = get_has_sources_scorer()
conciseness = get_scorer(name="conciseness", experiment_id=experiment.experiment_id)

if args.refine == "metaprompt":
    # ── Load the citations-optimized prompt from loop 1 ────────────────────────
    prompt = latest_prompt()

    # ── R: optimize the prompt for conciseness → new version ──────────────────
    # Optimize against BOTH judges: has_sources guards citations and the aligned
    # conciseness judge actively drives conciseness up (not just via a guideline).
    # Guidelines reinforce the smart-conciseness rule the aligned judge encodes.
    print("=" * 70)
    print("Optimizing prompt for conciseness with MetaPromptOptimizer …")
    print("=" * 70)

    opt = optimize_prompts(
        predict_fn=predict_fn,
        train_data=get_train_data(),
        prompt_uris=[prompt.uri],
        optimizer=reflection_optimizer(
            guidelines=(
                "Cite sources with URLs. Be concise and direct for simple factual "
                "questions; a thorough, multi-paragraph answer is appropriate for "
                "explanatory 'how/why' questions."
            ),
        ),
        scorers=[has_sources, conciseness],
    )

    new_prompt = opt.optimized_prompts[0]

    print(f"\nOptimized prompt registered as v{new_prompt.version}.")
    # With two optimizer scorers, initial/final_eval_score are the AGGREGATE.
    print(f"  baseline  aggregate = {opt.initial_eval_score}")
    print(f"  optimized aggregate = {opt.final_eval_score}")

    # If the optimizer exposes per-scorer breakdowns, surface them too. These
    # attributes may be absent (e.g. scalar scores), so guard and never break.
    for label, score in (("baseline", opt.initial_eval_score), ("optimized", opt.final_eval_score)):
        for scorer_name in ("has_sources", "conciseness"):
            per_scorer = getattr(score, scorer_name, None)
            if per_scorer is not None:
                print(f"  {label} {scorer_name} = {per_scorer}")
else:
    # ── R: refine the prompt with Claude Code → new version ──────────────────
    from refine_claude_code import refine_with_claude_code

    print("=" * 70)
    print("Refining prompt for conciseness with Claude Code …")
    print("=" * 70)

    # Establish baseline scores (both criteria) for the current prompt.
    baseline_prompt = latest_prompt()
    baseline_traces = run_scenarios(
        create_agent(baseline_prompt.template),
        run_name="03-loop-baseline",
        scorers=[has_sources, conciseness],
    )
    baseline_result = mlflow.genai.evaluate(
        data=baseline_traces, scorers=[has_sources, conciseness]
    )
    scores = {
        s.name: baseline_result.metrics[f"{s.name}/mean"]
        for s in (has_sources, conciseness)
    }

    new_prompt = refine_with_claude_code(
        prompt_name=baseline_prompt.name,
        prompt_version=baseline_prompt.version,
        scores=scores,
        goal=(
            "Achieve at least an 80% pass rate on the conciseness scorer "
            "(conciseness/mean >= 0.8) while keeping has_sources at or near 100%. "
            "Stop as soon as conciseness reaches 0.8."
        ),
        project_dir=str(Path(__file__).parent),
    )

# ── Final eval of v3 under one '03-loop' run, scoring BOTH criteria ───────
# run_scenarios opens the '03-loop' run, associates all agent traces with it,
# and (via scorers=) runs evaluate inside that same run so has_sources/mean and
# conciseness/mean both land on it. It prints each "{name}/mean".
agent = create_agent(new_prompt.template)
traces = run_scenarios(agent, run_name="03-loop", scorers=[has_sources, conciseness])

# ── Report ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Key takeaway:")
print("=" * 70)
print("  - Objective criteria (citations) → 1 STAR loop")
print("  - Subjective criteria (conciseness) → 2 STAR loops (align judge, then refine agent)")
print("  - This is the 'coupled' in coSTAR: trust the judge before trusting its scores.")
print(f"\nOptimized prompt registered as v{new_prompt.version}.")
print("\nBrowse the Prompts registry for 'research-agent' v3,")
print("and the Evaluation runs view for the '03-loop' run (both metrics).")
