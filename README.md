# coSTAR: Ship AI Agents Fast Without Breaking Things

Code examples for the [coSTAR blog post](https://www.databricks.com/blog/costar-how-we-ship-ai-agents-databricks-fast-without-breaking-things), demonstrating how to use MLflow to iteratively refine both AI agents and the judges that evaluate them.

## The Three-Loop Narrative

coSTAR uses **STAR loops** (Scenario → Trace → Assess → Refine) to improve agents systematically:

| Loop | Script | What it does |
|------|--------|-------------|
| **Loop 1** | `01_star_objective.py` | Refine the agent with an objective citation scorer |
| **Loop 2** | `02_star_judge_align.py` | Align a generic conciseness LLM judge to match human preferences (subjective criterion — must align the judge first) |
| **Loop 3** | `03_star_subjective.py` | Refine the agent for conciseness with the aligned judge, while ensuring citations don't regress |


## Prerequisites

```bash
pip install mlflow>=3.10 deepagents wikipedia openai litellm
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."      # Required for the LLM judge and Deep Agent
```

## Start MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000 to see traces, evaluations, and feedback.

## Running the Examples

Run the scripts in order — each builds on the previous:

```bash
# Loop 1: Agent refinement with citation scorer
python 01_star_objective.py

# Loop 2: Judge alignment for conciseness
python 02_star_judge_align.py

# Loop 3: Agent refinement with aligned conciseness judge
python 03_star_subjective.py
```

### Alternative Refine engine: Claude Code

By default, Loops 1 and 3 use the `optimize_prompts()` SDK in MLflow for the Refinement step. An alternative is to use Claude Code as the "engine" for refinement. In this setup, Claude Code is equipped with a skill that teaches the basic steps of the coSTAR framework:

```bash
python 01_star_objective.py --refine=claude-code
python 03_star_subjective.py --refine=claude-code
```

This requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and available as `claude` on your PATH.

Each script prints a comparison table showing improvement across agent versions.

## What You'll See in the MLflow UI

- **Prompts** tab: "research-agent" with 3 versions (v1: baseline, v2: optimized for citations, v3: optimized for citations + conciseness) — click any version to see diffs between prompt iterations
- **Traces** with full span trees: planning, tool calls (Wikipedia search), LLM reasoning
- **Assessments** from both automated scorers and simulated human feedback
- **Evaluation results** comparing agent versions side by side
- **Optimization runs** logged by `optimize_prompts()` with baseline → optimized scores
- **Judge alignment** showing how human feedback refines the judge's instructions

## File Structure

```
├── README.md                # This file
├── setup.py                 # Shared: agent factory, tools, MLflow experiment, scenarios
├── 01_star_objective.py     # Loop 1: agent refinement with citation scorer
├── 02_star_judge_align.py   # Loop 2: judge alignment for conciseness
├── 03_star_subjective.py    # Loop 3: agent refinement with aligned judge
├── refine_claude_code.py    # Claude Code headless Refine engine
└── .claude/skills/costar-refine/
    └── SKILL.md             # Skill for Claude Code prompt refinement
```
