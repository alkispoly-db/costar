# coSTAR: Ship AI Agents Fast Without Breaking Things

Code examples for the [coSTAR blog post](https://www.databricks.com/blog/costar-how-we-ship-ai-agents-databricks-fast-without-breaking-things), demonstrating how to use MLflow to iteratively refine both an AI agent and the LLM judges that evaluate it.

coSTAR runs **STAR loops** — Scenario → Trace → Assess → Refine — to improve an agent systematically. The demo walks through three loops over a Wikipedia research agent:

- **Loop 1 — objective (citations).** Refine the agent against a citation judge (`has_sources`).
- **Loop 2 — align the judge.** Conciseness is subjective, so before refining for it we first align a generic LLM judge to human preferences.
- **Loop 3 — subjective (conciseness).** Refine the agent for conciseness using the *aligned* judge, while guarding citations from regressing.

The "coupled" in coSTAR is loop 2: trust the judge before you trust its scores.

## Setup

This is a [uv](https://docs.astral.sh/uv/) project. Dependencies are pinned in `pyproject.toml` / `uv.lock` (the agent stack: `mlflow`, `deepagents`, `langchain-openai`, `dspy`, `wikipedia`, `openai`, `litellm`, `requests-cache`, `python-dotenv`). There is no manual venv to create or activate — `uv run` resolves the environment on first use.

Run every script with `uv run`:

```bash
uv run 00-setup.py
```

`common.py` loads your `OPENAI_API_KEY` automatically from `~/.env` (an already-exported key still wins), so no manual `export` is needed. The key is required for the LLM judge and the agent; seeding and the dataset test do not need it.

There is **no manual MLflow server to start**. `uv run 00-setup.py` resets the experiment to a clean state *and* auto-starts a local sqlite-backed MLflow server on `:5000` (detached, so the later scripts reuse it). If a server is already up, it is left alone.

## Demo flow

Run the eight per-phase scripts in order. Each maps to one STAR phase, and each hands off to the next through MLflow (by run name or the registry) — no files threaded between them.

```bash
# Loop 1 — objective: citations
uv run 00-setup.py        # reset experiment, seed dataset, start server
uv run 01-trace.py        # Trace:  run baseline agent (prompt v1) over the scenarios
uv run 01-assess.py       # Assess: score traces with the has_sources judge
uv run 01-refine.py       # Refine: optimize_prompts() → research-agent v2

# Loop 2 — align the conciseness judge
uv run 02-add-judge.py    # register the generic conciseness judge (v1)
uv run 02-assess.py       # log simulated human conciseness labels on 5 of the traces
uv run 02-refine.py       # MemAlign the judge to the labels → aligned conciseness v2

# Loop 3 — subjective: conciseness
uv run 03-loop.py         # Refine the prompt for conciseness with the aligned judge → v3
```

### Alternative Refine engine: Claude Code

By default the Refine phases use MLflow's `optimize_prompts()`, which rewrites the prompt text while treating tools and agent logic as fixed. Claude Code can be used as a more general optimization engine instead — it can read traces, inspect failure patterns, and go beyond prompt rewrites (rewrite tools, add tools, rewire agent logic). It is driven by the `costar-refine` skill under `.claude/skills/`, which evaluates each candidate prompt via `eval.py`.

Pass `--refine claude-code` to either Refine script to use it:

```bash
uv run 01-refine.py --refine claude-code
uv run 03-loop.py --refine claude-code
```

This requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and available as `claude` on your PATH. It runs Claude Code headless and is noticeably slower than the default `metaprompt` engine.

## What you'll see in the MLflow UI

Open <http://localhost:5000> and browse the `costar-research-agent` experiment:

- **Datasets** — `research-scenarios`, the eval dataset of **10** research scenarios.
- **Prompts** — `research-agent` with three versions: v1 (verbose baseline), v2 (optimized for citations), v3 (optimized for citations + conciseness). Click a version to diff iterations.
- **Judges** — two registered scorers: `has_sources` (an LLM judge: does the answer cite a source URL?) and `conciseness` with two versions, v1 (generic) → v2 (aligned to human labels).
- **Traces** — full span trees per scenario: planning, Wikipedia tool calls, LLM reasoning.
- **Evaluation runs** — one run per phase (`01-trace`, `01-assess`, `01-refine`, `03-loop`, …) with the scorer means, plus the `optimize_prompts()` optimization runs and the judge-alignment results.

## File Structure

```
├── README.md              # This file
├── pyproject.toml         # uv project + pinned dependencies (uv.lock alongside)
├── common.py              # Shared: agent factory, tools, MLflow experiment, scenarios, scorers
├── conciseness_judge.py   # The generic conciseness judge (instructions + builder)
├── 00-setup.py            # Reset + seed the experiment; auto-start the MLflow server
├── 01-trace.py            # Loop 1 · Trace
├── 01-assess.py           # Loop 1 · Assess (has_sources judge)
├── 01-refine.py           # Loop 1 · Refine (optimize_prompts → prompt v2)
├── 02-add-judge.py        # Loop 2 · register the generic conciseness judge
├── 02-assess.py           # Loop 2 · Assess (log human conciseness labels)
├── 02-refine.py           # Loop 2 · Refine (MemAlign → aligned conciseness judge v2)
├── 03-loop.py             # Loop 3 · Refine for conciseness with the aligned judge → prompt v3
├── tests/                 # Standalone smoke test for the eval dataset (no OpenAI key needed)
└── .claude/skills/costar-refine/
    ├── SKILL.md           # Skill teaching Claude Code the coSTAR refine workflow
    └── eval.py            # Scores a prompt version against the scenarios
```
