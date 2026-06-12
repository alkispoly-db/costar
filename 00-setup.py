"""
00 — Reset the costar-research-agent experiment to a clean "Scenarios" state.

This is the precondition for the per-phase demo. After running, the experiment
contains exactly:

  * the ``research-scenarios`` eval dataset (15 records),
  * the ``has_sources`` judge registered as an experiment scorer,
  * the ``research-agent`` prompt at v1,

and nothing else: no ``conciseness`` scorer (that belongs to loop 2), no
traces, no eval/agent runs.

The reset is programmatic — it talks to the live MLflow server on :5000 via the
SDK; it does NOT touch the sqlite file or restart the server. It is idempotent:
running it twice lands on the same clean state with no errors and no growth.

No OpenAI key is required — registering a judge and seeding a dataset do not
call the model.
"""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.genai import datasets
from mlflow.genai.scorers import delete_scorer, list_scorers

# Importing setup configures the tracking URI + experiment and (as a side
# effect) registers the research-agent prompt v1 if it is missing.
import setup
from setup import (
    PROMPT_NAME,
    SCENARIO_DATASET_NAME,
    experiment,
    get_has_sources_scorer,
    get_scenario_dataset,
)

EID = experiment.experiment_id
_client = MlflowClient()


def _reset():
    """Tear the experiment down to an empty shell (no dataset/scorers/traces/runs)."""
    # --- dataset ---------------------------------------------------------
    # On OSS MLflow delete_dataset only accepts dataset_id (the name kwarg is
    # Databricks-only), so look the id up first; an absent dataset is the
    # idempotent case.
    try:
        ds = datasets.get_dataset(name=SCENARIO_DATASET_NAME)
    except MlflowException as e:
        ds = None
        print(f"  dataset '{SCENARIO_DATASET_NAME}' not present ({type(e).__name__})")
    if ds is not None:
        datasets.delete_dataset(dataset_id=ds.dataset_id)
        print(f"  deleted dataset '{SCENARIO_DATASET_NAME}'")

    # --- scorers ---------------------------------------------------------
    # delete_scorer REQUIRES the version kwarg; version="all" removes every
    # version. We target both judges that the loops may have registered.
    for name in ("conciseness", "has_sources"):
        try:
            delete_scorer(name=name, experiment_id=EID, version="all")
            print(f"  deleted scorer '{name}' (all versions)")
        except MlflowException as e:
            print(f"  scorer '{name}' not present ({type(e).__name__})")
    # setup's per-process "already registered" guards would otherwise stop the
    # later seed from re-registering has_sources after we just deleted it.
    setup._has_sources_registered = False
    setup._conciseness_registered = False

    # --- traces ----------------------------------------------------------
    # delete_traces caps each call at max_traces; loop until the experiment is
    # empty so we drain however many accumulated across prior demo runs.
    total_traces = 0
    while True:
        remaining = mlflow.search_traces(locations=[EID], return_type="list", max_results=1)
        if not remaining:
            break
        deleted = _client.delete_traces(experiment_id=EID, max_timestamp_millis=_now_ms())
        if deleted == 0:
            break
        total_traces += deleted
    print(f"  deleted {total_traces} traces")

    # --- runs ------------------------------------------------------------
    runs = mlflow.search_runs(experiment_ids=[EID], output_format="list")
    for run in runs:
        _client.delete_run(run.info.run_id)
    print(f"  deleted {len(runs)} runs")

    # --- prompt versions -------------------------------------------------
    # Reduce the registry to a single v1. delete_prompt_version exists, so we
    # drop every version > 1 (and any stray non-1 ids); v1 itself is left in
    # place and re-created by the seed step if it was somehow removed.
    _prune_prompt_to_v1()


def _now_ms():
    import time

    return int(time.time() * 1000)


def _prune_prompt_to_v1():
    """Delete all research-agent prompt versions except v1, if the API allows."""
    try:
        versions = list(_client.search_prompt_versions(PROMPT_NAME))
    except MlflowException as e:
        print(f"  prompt '{PROMPT_NAME}' not present ({type(e).__name__})")
        return
    pruned = 0
    leftover = []
    for pv in versions:
        if str(pv.version) == "1":
            continue
        try:
            _client.delete_prompt_version(PROMPT_NAME, str(pv.version))
            pruned += 1
        except MlflowException as e:
            leftover.append((pv.version, type(e).__name__))
    print(f"  pruned {pruned} extra prompt versions (kept v1)")
    if leftover:
        print(f"  NOTE: could not delete prompt versions {leftover}; left in place")


def _seed():
    """Recreate the clean-state artifacts: dataset + has_sources judge + prompt v1."""
    ds = get_scenario_dataset()
    print(f"  seeded dataset '{SCENARIO_DATASET_NAME}' ({len(ds.to_df())} records)")

    get_has_sources_scorer()
    print("  registered 'has_sources' judge")

    # Importing setup already registered prompt v1 if it was missing. If the
    # reset removed it, re-register here so v1 is guaranteed present.
    try:
        mlflow.genai.load_prompt(PROMPT_NAME, version=1)
    except MlflowException:
        mlflow.genai.register_prompt(
            name=PROMPT_NAME,
            template=setup.SYSTEM_PROMPT_V1,
            commit_message="v1: basic research assistant",
        )
        print("  re-registered prompt v1")


def _summary():
    print("\n" + "=" * 60)
    print("CLEAN STATE SUMMARY")
    print("=" * 60)

    ds = datasets.get_dataset(name=SCENARIO_DATASET_NAME)
    print(f"  dataset '{SCENARIO_DATASET_NAME}': {len(ds.to_df())} records")

    scorer_names = sorted(s.name for s in list_scorers(experiment_id=EID))
    print(f"  registered scorers: {scorer_names}")

    pvs = list(_client.search_prompt_versions(PROMPT_NAME))
    print(f"  prompt '{PROMPT_NAME}' versions: {sorted(str(pv.version) for pv in pvs)}")

    traces = mlflow.search_traces(locations=[EID], return_type="list")
    print(f"  traces: {len(traces)}")

    runs = mlflow.search_runs(experiment_ids=[EID], output_format="list")
    print(f"  runs: {len(runs)}")
    print("=" * 60)


def main():
    print("=" * 60)
    print(f"RESET experiment '{setup.EXPERIMENT_NAME}' (id={EID})")
    print("=" * 60)
    _reset()
    print("\nSEED clean state")
    _seed()
    _summary()


if __name__ == "__main__":
    main()
