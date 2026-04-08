"""Evaluate a prompt version against test scenarios.

Usage: uv run --no-project --python .venv -- python .claude/skills/costar-refine/eval.py <prompt_name> <version>

Prints a JSON line with scores, e.g.:
  EVAL_RESULT: {"has_sources": 0.87, "conciseness": 0.6}
"""

import json
import sys
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Scorer

from setup import SCENARIOS, create_agent, has_sources, run_scenarios

# Load the aligned conciseness judge if available
judge_file = Path("_aligned_judge.json")
scorers = [has_sources]
if judge_file.exists():
    aligned_judge = Scorer.model_validate_json(judge_file.read_text())
    scorers.append(aligned_judge)

prompt_name, version = sys.argv[1], int(sys.argv[2])
prompt = mlflow.genai.load_prompt(prompt_name, version=version)
traces = run_scenarios(
    create_agent(prompt.template), SCENARIOS, run_name=f"refine-eval-v{version}"
)
result = mlflow.genai.evaluate(data=traces, scorers=scorers)
scores = {
    k.removesuffix("/mean"): v for k, v in result.metrics.items() if k.endswith("/mean")
}
print(f"EVAL_RESULT: {json.dumps(scores)}")
