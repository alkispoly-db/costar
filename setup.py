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
    import wikipedia

    # The `wikipedia` library defaults to http://, which Wikipedia 301-redirects
    # to an empty body (-> JSONDecodeError). Pin to https and set a real UA.
    wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
    wikipedia.wikipedia.USER_AGENT = "costar-demo/1.0 (https://github.com/alkispoly-db/costar)"

    titles = wikipedia.search(query, results=max_results)
    results = []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            results.append(f"Title: {page.title}\nSummary: {page.summary[:500]}\n")
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
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
# Scorer: deterministic, no LLM needed
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+")


@scorer
def has_sources(outputs) -> bool:
    """Check whether the agent's answer contains at least one URL."""
    return bool(URL_PATTERN.search(str(outputs)))


# ---------------------------------------------------------------------------
# Helper: run agent on every scenario and collect traces
# ---------------------------------------------------------------------------
def run_scenarios(agent, scenarios=None, *, run_name: str):
    """Invoke *agent* on each scenario and return the resulting traces.

    When *scenarios* is omitted, questions are sourced from the eval dataset
    (the runtime source of truth); existing callers may still pass an explicit
    scenario list.
    """
    if scenarios is None:
        scenarios = load_scenarios()
    trace_ids = []
    with mlflow.start_run(run_name=run_name):
        for scenario in scenarios:
            agent.invoke(
                {"messages": [{"role": "user", "content": scenario["question"]}]}
            )
            trace_id = mlflow.get_last_active_trace_id()
            trace_ids.append(trace_id)
            print(f"  [{run_name}] {scenario['question'][:60]}…  trace={trace_id}")

    mlflow.flush_trace_async_logging()
    return [mlflow.get_trace(tid) for tid in trace_ids]


# ---------------------------------------------------------------------------
# Helper: find a prompt version by tag
# ---------------------------------------------------------------------------
def find_prompt_by_tag(prompt_name, tag_key, tag_value):
    client = mlflow.MlflowClient()
    for pv in client.search_prompt_versions(prompt_name):
        if pv.tags.get(tag_key) == tag_value:
            return pv
    raise RuntimeError(f"No '{prompt_name}' version with tag {tag_key}={tag_value}")
