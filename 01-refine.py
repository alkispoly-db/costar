"""
01 — REFINE phase (coSTAR loop 1).

Improves the agent prompt automatically with optimize_prompts() +
MetaPromptOptimizer, optimizing against the registered "has_sources" judge.
This is the R in STAR: turn the assessment into a better prompt (a new
research-agent version), then run a final evaluation of that prompt under a run
named "01-refine".

After running, browse the Prompts registry to see the new prompt version, and
the Evaluation runs view for the "01-refine" eval run.
"""

from mlflow.genai.optimize import optimize_prompts

from setup import (
    create_agent,
    get_has_sources_scorer,
    get_train_data,
    latest_prompt,
    predict_fn,
    reflection_optimizer,
    run_scenarios,
)

has_sources = get_has_sources_scorer()

# ── R: optimize the prompt → new version ─────────────────────────────────
prompt_v1 = latest_prompt()

print("=" * 70)
print("Optimizing prompt with MetaPromptOptimizer …")
print("=" * 70)

opt = optimize_prompts(
    predict_fn=predict_fn,
    train_data=get_train_data(),
    prompt_uris=[prompt_v1.uri],
    optimizer=reflection_optimizer(
        guidelines="Responses MUST cite sources with URLs.",
    ),
    scorers=[has_sources],
)

new_prompt = opt.optimized_prompts[0]

# ── Final eval of the optimized prompt under the "01-refine" run ──────────
# Generate fresh traces with the new prompt and score them in one shot:
# run_scenarios opens the "01-refine" run, associates all 10 agent traces with
# it, and (via scorers=) runs evaluate inside that same run so the
# has_sources/mean metric lands on it too. It also prints the final mean.
agent = create_agent(new_prompt.template)
traces = run_scenarios(agent, run_name="01-refine", scorers=[has_sources])

# ── Report ────────────────────────────────────────────────────────────────
print(f"\nOptimized prompt registered as v{new_prompt.version}.")
print(f"  baseline  has_sources = {opt.initial_eval_score}")
print(f"  optimized has_sources = {opt.final_eval_score}")
print("\nBrowse the Prompts registry for the new prompt version,")
print("and the Evaluation runs view for the '01-refine' eval run.")
