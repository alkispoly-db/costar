"""
STAR Loop 1 — Agent Refinement with an Objective Judge (Citation Quality)

This script demonstrates the simplest coSTAR loop: refining an agent using a
deterministic scorer that requires no LLM judge.

Flow:
  S — Load 15 Q&A scenarios
  T — Run agent v1 on all scenarios, auto-capture traces
  A — Score each trace with a deterministic "has_sources" scorer that checks
      whether the answer contains URLs.
  R — Use optimize_prompts() with MetaPromptOptimizer to automatically generate
      an improved prompt (v2). Re-run and compare scores side by side.

Usage:
  python 01_star_objective.py                     # default: metaprompt
  python 01_star_objective.py --refine=claude-code # use Claude Code headless
"""

import argparse
import re
from pathlib import Path

import mlflow
from mlflow.genai import evaluate
from mlflow.genai.scorers import scorer

from setup import (
    MODEL,
    PROMPT_NAME,
    SCENARIOS,
    TRAIN_DATA,
    create_agent,
    predict_fn,
    prompt_v1,
    run_scenarios,
)

parser = argparse.ArgumentParser(description="STAR Loop 1 — Objective Judge")
parser.add_argument(
    "--refine",
    choices=["metaprompt", "claude-code"],
    default="metaprompt",
    help="Which Refine engine to use (default: metaprompt)",
)
args = parser.parse_args()

# ── Scorer: deterministic, no LLM needed ─────────────────────────────────

URL_PATTERN = re.compile(r"https?://\S+")


@scorer
def has_sources(outputs) -> bool:
    """Check whether the agent's answer contains at least one URL."""
    return bool(URL_PATTERN.search(str(outputs)))


# ── S & T: Run agent v1 ──────────────────────────────────────────────────

print("=" * 70)
print("STAR Loop 1 — Agent v1 (baseline)")
print("=" * 70)

agent_v1 = create_agent(prompt_v1.template)
traces_v1 = run_scenarios(agent_v1, SCENARIOS, run_name="agent-v1")

# ── A: Evaluate v1 ───────────────────────────────────────────────────────

print("\nEvaluating agent v1 …")
eval_v1 = evaluate(data=traces_v1, scorers=[has_sources])

v1_citation = eval_v1.metrics["has_sources/mean"]
print(f"  v1  has_sources = {v1_citation:.0%}")

# ── R: Automated prompt optimization → v2 ────────────────────────────────

if args.refine == "metaprompt":
    from mlflow.genai.optimize import optimize_prompts
    from mlflow.genai.optimize.optimizers import MetaPromptOptimizer

    print("\n" + "=" * 70)
    print("Optimizing prompt with MetaPromptOptimizer …")
    print("=" * 70)

    result = optimize_prompts(
        predict_fn=predict_fn,
        train_data=TRAIN_DATA,
        prompt_uris=[prompt_v1.uri],
        optimizer=MetaPromptOptimizer(
            reflection_model=MODEL,
            guidelines="Responses MUST cite sources with Wikipedia URLs.",
        ),
        scorers=[has_sources],
    )

    prompt_v2 = result.optimized_prompts[0]
    print(f"\nOptimized prompt registered as v{prompt_v2.version}:")
    print(f"  {prompt_v2.template[:200]}…")
    print(f"\n  Baseline score: {result.initial_eval_score}")
    print(f"  Optimized score: {result.final_eval_score}")

elif args.refine == "claude-code":
    from refine_claude_code import HAS_SOURCES_SCORER, refine_with_claude_code

    print("\n" + "=" * 70)
    print("Optimizing prompt with Claude Code …")
    print("=" * 70)

    prompt_v2 = refine_with_claude_code(
        prompt_name=PROMPT_NAME,
        prompt_version=prompt_v1.version,
        scores={"has_sources": v1_citation},
        goal="Responses MUST cite sources with Wikipedia URLs.",
        project_dir=str(Path(__file__).parent),
        scorer_source=HAS_SOURCES_SCORER,
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
traces_v2 = run_scenarios(agent_v2, SCENARIOS, run_name="agent-v2")

# ── A: Evaluate v2 ───────────────────────────────────────────────────────

print("\nEvaluating agent v2 …")
eval_v2 = evaluate(data=traces_v2, scorers=[has_sources])

v2_citation = eval_v2.metrics["has_sources/mean"]
print(f"  v2  has_sources = {v2_citation:.0%}")

# ── Side-by-side comparison ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("Loop 1 Results: v1 → v2")
print("=" * 70)
print(f"  {'Metric':<20} {'v1':>8} {'v2':>8} {'Delta':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
print(
    f"  {'has_sources':<20} {v1_citation:>7.0%} {v2_citation:>7.0%}"
    f" {v2_citation - v1_citation:>+7.0%}"
)
print("\nDone. Inspect runs in the MLflow UI under the 'costar-research-agent' experiment.")
print("Browse the 'Prompts' tab to see the 'research-agent' prompt with v1 and v2 diffs.")
