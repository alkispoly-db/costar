"""Evaluate a prompt version against test scenarios.

Usage: uv run --no-project --python .venv -- python .claude/skills/costar-refine/eval.py <prompt_name> <version>

Prints a JSON line with scores, e.g.:
  EVAL_RESULT: {"has_sources": 0.87, "conciseness": 0.6}
"""

import json
import sys

import mlflow
from mlflow.genai.scorers import get_scorer

from setup import create_agent, experiment, has_sources, run_scenarios

# Load the aligned conciseness judge from the registry (loop 2 registers it as
# the latest 'conciseness' version). If no such scorer is registered yet —
# e.g. only loop 1 has run — fall back to scoring with just has_sources.
scorers = [has_sources]
try:
    aligned = get_scorer(name="conciseness", experiment_id=experiment.experiment_id)
    scorers.append(aligned)
except Exception:
    pass

prompt_name, version = sys.argv[1], int(sys.argv[2])
prompt = mlflow.genai.load_prompt(prompt_name, version=version)
traces = run_scenarios(
    create_agent(prompt.template), run_name=f"refine-eval-v{version}"
)
result = mlflow.genai.evaluate(data=traces, scorers=scorers)
scores = {
    k.removesuffix("/mean"): v for k, v in result.metrics.items() if k.endswith("/mean")
}
print(f"EVAL_RESULT: {json.dumps(scores)}")
