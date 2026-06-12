"""
Shared configuration for the coSTAR blog post examples.

Sets up the MLflow experiment, Deep Agent factory, tools, prompt registry,
and evaluation scenarios. Run this module before the numbered scripts.
"""

import functools
import os
import re

import mlflow
from mlflow.genai.scorers import scorer

from conciseness_judge import CONCISENESS_INSTRUCTIONS, build_conciseness_judge

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
# autolog() pulls in langchain, and the agent factory pulls in deepagents.
# Both are deferred to create_agent() so that the dataset-management helpers
# below can be imported and run (e.g. to seed the eval dataset) without the
# agent stack or an OpenAI key installed.
# Honor MLFLOW_TRACKING_URI so tests can point at a temp sqlite backend without
# a running server; default to the local server used by the blog walkthrough.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

EXPERIMENT_NAME = "costar-research-agent"
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

PROMPT_NAME = "research-agent"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
AGENT_MODEL = "openai:gpt-4.1-mini"  # Deep Agent uses "provider:model" format
JUDGE_MODEL = "openai:/gpt-4.1-mini"  # MLflow judges/optimizers use "openai:/model" format

# ---------------------------------------------------------------------------
# Wikipedia search tool (no API key needed)
# ---------------------------------------------------------------------------


def search_wikipedia(query: str, max_results: int = 3) -> str:
    """Search Wikipedia and return article summaries.

    Returns titles and text content. Does NOT return URLs — if you need
    to cite a source, construct the Wikipedia URL from the page title
    (e.g. https://en.wikipedia.org/wiki/Page_Title).
    """
    import time

    import requests
    import wikipedia

    # The `wikipedia` library defaults to http://, which Wikipedia 301-redirects
    # to an empty body (-> JSONDecodeError). Pin to https and set a real UA.
    wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
    wikipedia.wikipedia.USER_AGENT = "costar-demo/1.0 (https://github.com/alkispoly-db/costar)"

    # A transient/empty Wikipedia API response makes search()/page() raise a
    # JSONDecodeError ("Expecting value: line 1 column 1"), a RequestException /
    # ValueError subclass. Left to propagate it crashes the whole agent run
    # mid-demo, so every API call here degrades gracefully instead of raising.
    titles = None
    for delay in (0.5, 1.0):
        try:
            titles = wikipedia.search(query, results=max_results)
            break
        except (requests.exceptions.RequestException, ValueError):
            time.sleep(delay)
    if titles is None:
        # Still failing after retries — degrade instead of crashing the agent.
        try:
            titles = wikipedia.search(query, results=max_results)
        except (requests.exceptions.RequestException, ValueError):
            return "No results found."

    results = []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            results.append(f"Title: {page.title}\nSummary: {page.summary[:500]}\n")
        except (
            wikipedia.exceptions.DisambiguationError,
            wikipedia.exceptions.PageError,
        ):
            continue
        except (requests.exceptions.RequestException, ValueError):
            # Transient API junk for this title (JSONDecodeError etc.): one quick
            # retry, then skip the title rather than crash the agent.
            try:
                time.sleep(0.5)
                page = wikipedia.page(title, auto_suggest=False)
                results.append(f"Title: {page.title}\nSummary: {page.summary[:500]}\n")
            except (
                wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
                requests.exceptions.RequestException,
                ValueError,
            ):
                continue
    return "\n---\n".join(results) if results else "No results found."


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V1 = """\
You are a research assistant. Answer the user's question using the
search tool. Provide a thorough, accurate answer based on search results."""


_autolog_enabled = False


def create_agent(system_prompt: str):
    """Create a Deep Agent with the given system prompt and search tool."""
    # Imported and enabled here (not at module load) so the dataset helpers
    # stay usable without the langchain/deepagents stack.
    from deepagents import create_deep_agent

    # create_agent runs per agent and per predict_fn call; enable autolog only
    # once to avoid re-patching langchain on every invocation.
    global _autolog_enabled
    if not _autolog_enabled:
        mlflow.langchain.autolog()
        _autolog_enabled = True
    return create_deep_agent(
        model=AGENT_MODEL, tools=[search_wikipedia], system_prompt=system_prompt
    )


# ---------------------------------------------------------------------------
# Prompt registry — register v1 only if it doesn't exist yet
# ---------------------------------------------------------------------------
try:
    prompt_v1 = mlflow.genai.load_prompt(PROMPT_NAME, version=1)
except mlflow.MlflowException:
    prompt_v1 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=SYSTEM_PROMPT_V1,
        commit_message="v1: basic research assistant",
    )


# ---------------------------------------------------------------------------
# predict_fn for optimize_prompts — loads prompt from registry, runs agent
# ---------------------------------------------------------------------------
def predict_fn(question: str) -> str:
    """Load the current prompt from the registry, create an agent, and run it.

    optimize_prompts() monkey-patches PromptVersion.template internally, so
    loading the prompt each time picks up the candidate template being tested.
    """
    prompt = mlflow.genai.load_prompt(PROMPT_NAME)
    agent = create_agent(prompt.template)
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content


# ---------------------------------------------------------------------------
# Evaluation scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "question": "What is the current population of Tokyo?",
        "expected_facts": ["population", "Tokyo", "million"],
    },
    {
        "question": "Who won the most recent FIFA World Cup and where was it held?",
        "expected_facts": ["Argentina", "Qatar", "2022"],
    },
    {
        "question": "What programming language is used most for machine learning?",
        "expected_facts": ["Python"],
    },
    {
        "question": "What are the health benefits of intermittent fasting?",
        "expected_facts": ["weight", "insulin", "metabolism"],
    },
    {
        "question": "How does CRISPR gene editing work?",
        "expected_facts": ["DNA", "Cas9", "guide RNA"],
    },
    {
        "question": "What caused the 2008 financial crisis?",
        "expected_facts": ["subprime", "mortgage", "Lehman"],
    },
    {
        "question": "What is the James Webb Space Telescope's primary mission?",
        "expected_facts": ["infrared", "galaxies", "universe"],
    },
    {
        "question": "How do mRNA vaccines work?",
        "expected_facts": ["mRNA", "spike protein", "immune"],
    },
    {
        "question": "What are the main differences between TCP and UDP?",
        "expected_facts": ["connection", "reliable", "speed"],
    },
    {
        "question": "What is the significance of the Rosetta Stone?",
        "expected_facts": ["Egyptian", "hieroglyphs", "translation"],
    },
    {
        "question": "How does quantum computing differ from classical computing?",
        "expected_facts": ["qubit", "superposition", "entanglement"],
    },
    {
        "question": "What are the environmental impacts of fast fashion?",
        "expected_facts": ["waste", "water", "pollution"],
    },
    {
        "question": "What is the current state of fusion energy research?",
        "expected_facts": ["plasma", "tokamak", "energy"],
    },
    {
        "question": "How did the internet originate?",
        "expected_facts": ["ARPANET", "TCP/IP", "1960s"],
    },
    {
        "question": "What are the main causes of coral reef decline?",
        "expected_facts": ["bleaching", "temperature", "ocean acidification"],
    },
]

# ---------------------------------------------------------------------------
# Evaluation dataset — the runtime source of truth for scenarios
# ---------------------------------------------------------------------------
# The in-code SCENARIOS list above is only the *seed*. At runtime everything
# (run_scenarios, train data, evaluate) is fed from an MLflow GenAI evaluation
# dataset so the scenarios live in the experiment and can be versioned/shared.
SCENARIO_DATASET_NAME = "research-scenarios"

_dataset_seeded = False


def get_scenario_dataset():
    """Return the ``research-scenarios`` eval dataset, seeding it if needed.

    Get-or-create: create_dataset() does not check for an existing name (it
    would silently make a duplicate, after which get_dataset(name=...) becomes
    ambiguous), so we look the dataset up first and only create on a genuine
    miss. merge_records() de-duplicates on ``inputs``, so re-seeding on every
    call is idempotent.

    Requires a SQL-backed tracking server (GenAI datasets are unavailable on
    the file store). Works without the agent stack or an OpenAI key.
    """
    from mlflow.genai import datasets
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

    try:
        dataset = datasets.get_dataset(name=SCENARIO_DATASET_NAME)
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        dataset = datasets.create_dataset(
            name=SCENARIO_DATASET_NAME, experiment_id=experiment.experiment_id
        )

    # merge_records is idempotent, but seeding once per process avoids a
    # redundant upsert on every read path (load_scenarios, repeated run_scenarios).
    global _dataset_seeded
    if not _dataset_seeded:
        dataset.merge_records(
            [
                {
                    "inputs": {"question": s["question"]},
                    "expectations": {"expected_facts": s["expected_facts"]},
                }
                for s in SCENARIOS
            ]
        )
        _dataset_seeded = True
    return dataset


def load_scenarios():
    """Read the eval dataset back into the SCENARIOS dict shape.

    Returns ``[{"question": ..., "expected_facts": [...]}, ...]`` so callers
    that index scenarios (e.g. for human-feedback rationales) keep working.
    """
    # to_df() does not preserve insertion order (records are keyed by uuid), so
    # we re-impose the SCENARIOS question order to keep scenarios[i]<->trace
    # alignment in the numbered scripts deterministic.
    df = get_scenario_dataset().to_df()
    by_question = {
        row["inputs"]["question"]: row["expectations"]["expected_facts"]
        for _, row in df.iterrows()
    }
    return [
        {"question": s["question"], "expected_facts": by_question[s["question"]]}
        for s in SCENARIOS
    ]


# ---------------------------------------------------------------------------
# Training data for optimize_prompts (sourced from the eval dataset)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_train_data():
    """Build optimize_prompts train_data from the eval dataset.

    Lazy (not a module-level constant) so ``import setup`` performs no dataset
    or tracking-server I/O; the result is cached for repeated callers.
    """
    return [
        {"inputs": {"question": s["question"]}, "outputs": ", ".join(s["expected_facts"])}
        for s in load_scenarios()
    ]


# ---------------------------------------------------------------------------
# Scorer: deterministic has_sources, no LLM needed
# ---------------------------------------------------------------------------
# This is the deterministic regex has_sources scorer, used by loop 3
# (03_star_subjective.py) and .claude/skills/costar-refine/eval.py. It is
# distinct from the registered "has_sources" judge below (get_has_sources_scorer),
# which loop 1 uses. Custom @scorer functions can't be registered on OSS MLflow,
# so loop 1 needs the judge form; the regex form is kept for the loops/eval that
# call it directly.
URL_PATTERN = re.compile(r"https?://\S+")


@scorer
def has_sources(outputs) -> bool:
    """Check whether the agent's answer contains at least one URL."""
    return bool(URL_PATTERN.search(str(outputs)))


# ---------------------------------------------------------------------------
# Scorer: conciseness LLM judge, registered as a first-class experiment scorer
# ---------------------------------------------------------------------------
# CONCISENESS_INSTRUCTIONS and the judge construction now live in
# conciseness_judge.py (single source of truth, imported above); the
# constant is re-exported here so existing `from setup import
# CONCISENESS_INSTRUCTIONS` callers keep working.
CONCISENESS_SCORER_NAME = "conciseness"

_conciseness_registered = False


def get_conciseness_scorer():
    """Return the ``conciseness`` judge as a registered experiment scorer.

    Get-or-register: we register the judge so the scorer lives in the
    experiment itself — reusable across loops and available for production
    monitoring — rather than being re-instantiated ad hoc in each script.
    get_scorer() looks it up first; only a genuine miss (RESOURCE_DOES_NOT_EXIST,
    same pattern as get_dataset) triggers registration.

    Requires a SQL-backed tracking server. Registration does NOT call the model,
    so this works without an OpenAI key.
    """
    # Imported locally (like get_scenario_dataset) so `import setup` stays light
    # and doesn't pull in the judge stack just to use the dataset helpers.
    from mlflow.genai.scorers import get_scorer
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

    try:
        return get_scorer(
            name=CONCISENESS_SCORER_NAME, experiment_id=experiment.experiment_id
        )
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        judge = build_conciseness_judge(JUDGE_MODEL)
        # Guard against registering a new version if a concurrent miss + register
        # races within this process (register() versions on each call).
        global _conciseness_registered
        if _conciseness_registered:
            return get_scorer(
                name=CONCISENESS_SCORER_NAME, experiment_id=experiment.experiment_id
            )
        _conciseness_registered = True
        return judge.register(
            name=CONCISENESS_SCORER_NAME, experiment_id=experiment.experiment_id
        )


# ---------------------------------------------------------------------------
# Scorer: has_sources LLM judge, registered as a first-class experiment scorer
# ---------------------------------------------------------------------------
# The registered "has_sources" judge (loop 1) is distinct from the deterministic
# regex has_sources @scorer above (loops 3/eval): custom @scorer functions can't
# be registered on OSS MLflow, so loop 1 uses this judge form. The judge's
# registered NAME is "has_sources", so its eval metric key is "has_sources/mean"
# — the same key loop 1 already prints, so reporting code stays unchanged.
HAS_SOURCES_SCORER_NAME = "has_sources"
HAS_SOURCES_INSTRUCTIONS = (
    "Evaluate whether {{ outputs }} cites at least one source as an http or "
    "https URL (a clickable link). Respond true if the answer contains at "
    "least one such URL, false otherwise."
)

_has_sources_registered = False


def get_has_sources_scorer():
    """Return the ``has_sources`` judge as a registered experiment scorer.

    Get-or-register, mirroring get_conciseness_scorer(): we register the judge
    so the scorer lives in the experiment itself — reusable across loops and
    available for production monitoring — rather than being re-instantiated ad
    hoc in each script. get_scorer() looks it up first; only a genuine miss
    (RESOURCE_DOES_NOT_EXIST, same pattern as get_dataset) triggers registration.

    Requires a SQL-backed tracking server. Registration does NOT call the model,
    so this works without an OpenAI key.
    """
    # Imported locally (like get_conciseness_scorer) so `import setup` stays
    # light and doesn't pull in the judge stack just to use the dataset helpers.
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import get_scorer
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

    try:
        return get_scorer(
            name=HAS_SOURCES_SCORER_NAME, experiment_id=experiment.experiment_id
        )
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        judge = make_judge(
            name=HAS_SOURCES_SCORER_NAME,
            instructions=HAS_SOURCES_INSTRUCTIONS,
            model=JUDGE_MODEL,
            feedback_value_type=bool,
        )
        # Guard against registering a new version if a concurrent miss + register
        # races within this process (register() versions on each call).
        global _has_sources_registered
        if _has_sources_registered:
            return get_scorer(
                name=HAS_SOURCES_SCORER_NAME, experiment_id=experiment.experiment_id
            )
        _has_sources_registered = True
        return judge.register(
            name=HAS_SOURCES_SCORER_NAME, experiment_id=experiment.experiment_id
        )


# ---------------------------------------------------------------------------
# Helper: run agent on every scenario and collect traces
# ---------------------------------------------------------------------------
def run_scenarios(agent, scenarios=None, *, run_name, scorers=None):
    """Invoke *agent* on each scenario under its own run and return the traces.

    When *scenarios* is omitted, questions are sourced from the eval dataset
    (the runtime source of truth); existing callers may still pass an explicit
    scenario list.

    A new run named *run_name* is always opened; the agent traces associate with
    it. Logging into a caller's active run instead silently fails to associate
    the agent traces (only the caller's own validation traces land on it), so
    *run_name* is required and run_scenarios owns the run.

    When *scorers* is provided, ``mlflow.genai.evaluate`` runs on the collected
    traces inside the same run, so the score metric lands on this run; each
    scorer's ``"{name}/mean"`` value is printed. The return contract is
    unchanged: the list of traces.
    """
    if scenarios is None:
        scenarios = load_scenarios()

    with mlflow.start_run(run_name=run_name):
        trace_ids = []
        for scenario in scenarios:
            agent.invoke(
                {"messages": [{"role": "user", "content": scenario["question"]}]}
            )
            trace_id = mlflow.get_last_active_trace_id()
            trace_ids.append(trace_id)
            print(f"  [{run_name}] {scenario['question'][:60]}…  trace={trace_id}")

        mlflow.flush_trace_async_logging()
        traces = [mlflow.get_trace(tid) for tid in trace_ids]

        if scorers is not None:
            result = mlflow.genai.evaluate(data=traces, scorers=scorers)
            for scorer_obj in scorers:
                key = f"{scorer_obj.name}/mean"
                print(f"  [{run_name}] {key} = {result.metrics[key]:.0%}")

    return traces


# ---------------------------------------------------------------------------
# Helper: find a prompt version by tag
# ---------------------------------------------------------------------------
def find_prompt_by_tag(prompt_name, tag_key, tag_value):
    client = mlflow.MlflowClient()
    for pv in client.search_prompt_versions(prompt_name):
        if pv.tags.get(tag_key) == tag_value:
            return pv
    raise RuntimeError(f"No '{prompt_name}' version with tag {tag_key}={tag_value}")


# ---------------------------------------------------------------------------
# Convention-based lookup helpers
# ---------------------------------------------------------------------------
# The per-phase demo scripts hand work off to each other "by convention": a
# phase writes a run under a well-known run_name, and the next phase looks that
# run (and its traces) back up by name rather than threading a run id through.
# These helpers are import-light and side-effect-free so any phase script can
# call them without seeding or registering anything.
def latest_run_id(run_name):
    """Return the most-recent run id named *run_name* in the experiment.

    Returns None if no such run exists.
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def traces_for_run(run_name):
    """Return the Trace objects logged under the latest run named *run_name*.

    The list is suitable to pass directly to
    ``mlflow.genai.evaluate(data=...)`` (which accepts a list of traces, as the
    numbered scripts do). Returns an empty list if the run is absent.
    """
    run_id = latest_run_id(run_name)
    if run_id is None:
        return []
    # return_type="list" yields Trace objects (vs the default pandas DataFrame);
    # run_id alone scopes the search to that run's experiment, no location needed.
    return mlflow.search_traces(run_id=run_id, return_type="list")


def latest_prompt():
    """Return the latest registered version of the research-agent prompt."""
    return mlflow.genai.load_prompt(PROMPT_NAME)
