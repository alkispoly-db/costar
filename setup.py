"""
Shared configuration for the coSTAR blog post examples.

Sets up the MLflow experiment, Deep Agent factory, tools, prompt registry,
and evaluation scenarios. Run this module before the numbered scripts.
"""

import wikipedia

import mlflow
from deepagents import create_deep_agent

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
mlflow.langchain.autolog()
mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "costar-research-agent"
mlflow.set_experiment(EXPERIMENT_NAME)

PROMPT_NAME = "research-agent"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
AGENT_MODEL = "openai:gpt-4.1-mini"  # Deep Agent uses "provider:model" format
MODEL = "openai:/gpt-4.1-mini"  # MLflow judges use "openai:/model" format

# ---------------------------------------------------------------------------
# Wikipedia search tool (no API key needed)
# ---------------------------------------------------------------------------


def search_wikipedia(query: str, max_results: int = 3) -> str:
    """Search Wikipedia and return article summaries.

    Returns titles and text content. Does NOT return URLs — if you need
    to cite a source, construct the Wikipedia URL from the page title
    (e.g. https://en.wikipedia.org/wiki/Page_Title).
    """
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


def create_agent(system_prompt: str):
    """Create a Deep Agent with the given system prompt and search tool."""
    return create_deep_agent(
        model=AGENT_MODEL, tools=[search_wikipedia], system_prompt=system_prompt
    )


# ---------------------------------------------------------------------------
# Prompt registry — register v1 only if it doesn't exist yet
# ---------------------------------------------------------------------------
try:
    prompt_v1 = mlflow.genai.load_prompt(PROMPT_NAME, version=1)
except Exception:
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
    agent = create_deep_agent(
        model=AGENT_MODEL, tools=[search_wikipedia], system_prompt=prompt.template
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


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
# Training data for optimize_prompts (reformatted from SCENARIOS)
# ---------------------------------------------------------------------------
TRAIN_DATA = [
    {"inputs": {"question": s["question"]}, "outputs": ", ".join(s["expected_facts"])}
    for s in SCENARIOS
]


# ---------------------------------------------------------------------------
# Helper: run agent on every scenario and collect traces
# ---------------------------------------------------------------------------
def run_scenarios(agent, scenarios, *, run_name: str):
    """Invoke *agent* on each scenario and return the resulting traces."""
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
