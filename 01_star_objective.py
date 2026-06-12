"""
STAR Loop 1 — Agent Refinement with an Objective Judge (Citation Quality)

This script demonstrates the simplest coSTAR loop: refining an agent using an
objective "has_sources" LLM judge that checks for cited URLs.

Flow:
  S — Load 15 Q&A scenarios
  T — Run agent v1 on all scenarios, auto-capture traces
  A — Score each trace with the registered "has_sources" judge that checks
      whether the answer cites at least one http/https URL.
  R — Use optimize_prompts() with MetaPromptOptimizer to automatically generate
      an improved prompt (v2). Re-run and compare scores side by side.

Usage:
  python 01_star_objective.py                     # default: metaprompt
  python 01_star_objective.py --refine=claude-code # use Claude Code headless
"""

import argparse
from pathlib import Path

import mlflow

from setup import (
    JUDGE_MODEL,
    PROMPT_NAME,
    create_agent,
    get_conciseness_scorer,
    get_has_sources_scorer,
    get_train_data,
    predict_fn,
    prompt_v1,
    run_scenarios,
)

# Registered experiment scorers; reused for both v1 and v2 evaluation below.
# Loop 1 uses the registered has_sources judge (not the deterministic regex
# has_sources @scorer) because custom @scorer functions can't be registered on
# OSS MLflow.
has_sources = get_has_sources_scorer()
conciseness = get_conciseness_scorer()

parser = argparse.ArgumentParser(description="STAR Loop 1 — Objective Judge")
parser.add_argument(
    "--refine",
    choices=["metaprompt", "claude-code"],
    default="metaprompt",
    help="Which Refine engine to use (default: metaprompt)",
)
args = parser.parse_args()

# ── S & T: Run agent v1 ──────────────────────────────────────────────────

print("=" * 70)
print("STAR Loop 1 — Agent v1 (baseline)")
print("=" * 70)

agent_v1 = create_agent(prompt_v1.template)
traces_v1 = run_scenarios(agent_v1, run_name="agent-v1")

# ── A: Evaluate v1 ───────────────────────────────────────────────────────

print("\nEvaluating agent v1 …")
eval_v1 = mlflow.genai.evaluate(data=traces_v1, scorers=[has_sources, conciseness])

v1_cite = eval_v1.metrics["has_sources/mean"]
v1_concise = eval_v1.metrics["conciseness/mean"]
print(f"  v1  has_sources = {v1_cite:.0%}  conciseness = {v1_concise:.0%}")

# ── R: Automated prompt optimization → v2 ────────────────────────────────

if args.refine == "metaprompt":
    from mlflow.genai.optimize import optimize_prompts
    from mlflow.genai.optimize.optimizers import MetaPromptOptimizer

    print("\n" + "=" * 70)
    print("Optimizing prompt with MetaPromptOptimizer …")
    print("=" * 70)

    opt_result = optimize_prompts(
        predict_fn=predict_fn,
        train_data=get_train_data(),
        prompt_uris=[prompt_v1.uri],
        optimizer=MetaPromptOptimizer(
            reflection_model=JUDGE_MODEL,
            guidelines="Responses MUST cite sources with URLs.",
        ),
        # Loop 1 optimizes against the registered has_sources judge only;
        # conciseness is reported above/below as the pre-alignment read, not an
        # optimization target.
        scorers=[has_sources],
    )

    prompt_v2 = opt_result.optimized_prompts[0]
    print(f"\nOptimized prompt registered as v{prompt_v2.version}:")
    print(f"  {prompt_v2.template[:200]}…")
    print(f"\n  Baseline score: {opt_result.initial_eval_score}")
    print(f"  Optimized score: {opt_result.final_eval_score}")

elif args.refine == "claude-code":
    from refine_claude_code import refine_with_claude_code

    print("\n" + "=" * 70)
    print("Optimizing prompt with Claude Code …")
    print("=" * 70)

    prompt_v2 = refine_with_claude_code(
        prompt_name=PROMPT_NAME,
        prompt_version=prompt_v1.version,
        scores={"has_sources": v1_cite},
        goal="Responses MUST cite sources with Wikipedia URLs.",
        project_dir=str(Path(__file__).parent),
    )

    print(f"\nOptimized prompt registered as v{prompt_v2.version}:")
    print(f"  {prompt_v2.template[:200]}…")

# Tag the optimized prompt so Loops 2 and 3 can find it
mlflow.genai.set_prompt_version_tag(PROMPT_NAME, prompt_v2.version, "costar_step", "optimized")

# ── S & T: Run agent v2 ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STAR Loop 1 — Agent v2 (optimized prompt)")
print("=" * 70)

agent_v2 = create_agent(prompt_v2.template)
traces_v2 = run_scenarios(agent_v2, run_name="agent-v2")

# ── A: Evaluate v2 ───────────────────────────────────────────────────────

print("\nEvaluating agent v2 …")
eval_v2 = mlflow.genai.evaluate(data=traces_v2, scorers=[has_sources, conciseness])

v2_cite = eval_v2.metrics["has_sources/mean"]
v2_concise = eval_v2.metrics["conciseness/mean"]
print(f"  v2  has_sources = {v2_cite:.0%}  conciseness = {v2_concise:.0%}")

# ── Side-by-side comparison ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("Loop 1 Results: v1 → v2")
print("=" * 70)
print(f"  {'Metric':<20} {'v1':>8} {'v2':>8} {'Delta':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
print(
    f"  {'has_sources':<20} {v1_cite:>7.0%} {v2_cite:>7.0%}"
    f" {v2_cite - v1_cite:>+7.0%}"
)
print(
    f"  {'conciseness':<20} {v1_concise:>7.0%} {v2_concise:>7.0%}"
    f" {v2_concise - v1_concise:>+7.0%}"
)
print("\nDone. Inspect runs in the MLflow UI under the 'costar-research-agent' experiment.")
print("Browse the 'Prompts' tab to see the 'research-agent' prompt with v1 and v2 diffs.")
