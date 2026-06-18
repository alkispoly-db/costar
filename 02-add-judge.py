"""
02 — SCENARIOS phase (coSTAR loop 2).

Introduces the conciseness LLM judge by registering it as a first-class scorer
on the experiment. This is the S in STAR for loop 2: before we can refine the
agent for conciseness, we need a judge to assess it — so we add the judge here.

After a clean 00-setup the conciseness scorer is absent, so this registers it as
v1 (the generic judge). It is idempotent: re-running just looks the scorer back
up rather than creating another version.

After running, open the Judges panel in the MLflow UI and show the newly
registered conciseness judge.

No traces and no scoring happen here — see 02-assess.py for the A phase.
"""

from mlflow.genai.scorers import list_scorers

from common import experiment, get_conciseness_scorer

# Get-or-register the conciseness judge as an experiment scorer. On the clean
# state this registers v1 (the generic judge); on a re-run it just fetches it.
conciseness = get_conciseness_scorer()

print(f"Registered conciseness judge as scorer '{conciseness.name}'.")

# Show the experiment's registered scorers — expect has_sources + conciseness.
scorer_names = [x.name for x in list_scorers(experiment_id=experiment.experiment_id)]
print(f"  Registered scorers: {scorer_names}")

print("\nOpen the Judges panel in the MLflow UI to see the conciseness judge.")
