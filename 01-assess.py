"""
01 — ASSESS phase (coSTAR loop 1).

Scores the traces produced by 01-trace.py with the registered "has_sources"
judge (does the answer cite at least one http/https URL?). This is the A in
STAR: attach assessments to the existing traces so we can see how the baseline
agent does before refining it.

After running, switch to the Evaluation runs view in the MLflow UI; the
assessments show up on the traces under the "01-assess" run.

Run 01-trace.py first — this phase reads its traces back by run name.
"""

import sys

import mlflow

from common import get_has_sources_scorer, traces_for_run

has_sources = get_has_sources_scorer()

traces = traces_for_run("01-trace")
if not traces:
    print("No traces found for run '01-trace'. Run `python 01-trace.py` first.")
    sys.exit(1)

# Score the existing traces under a dedicated "01-assess" run so the next phase
# (and the UI) can find the assessments by name.
with mlflow.start_run(run_name="01-assess"):
    result = mlflow.genai.evaluate(data=traces, scorers=[has_sources])

print(f"\nhas_sources/mean = {result.metrics['has_sources/mean']:.0%}")
print("Open the Evaluation runs view in the MLflow UI to see the assessments.")
