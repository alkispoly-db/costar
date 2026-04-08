# coSTAR Prompt Refinement

You are improving an AI agent's system prompt based on evaluation feedback.
This is an iterative process: rewrite the prompt, verify with the eval script, and iterate until scores improve.

## Context

The prompt you are given contains:
- `prompt_name`: the registered prompt name
- `prompt_version`: current version to improve
- `scores`: current metric scores (0.0-1.0) — your baseline to beat
- `goal`: what to improve

## Workflow

### Step 1: Load and understand the current prompt

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
prompt = mlflow.genai.load_prompt("<prompt_name>", version=<version>)
print(prompt.template)
```

### Step 2: Rewrite the prompt and register it

```python
new_prompt = mlflow.genai.register_prompt(
    name="<prompt_name>",
    template="<your improved template>",
    commit_message="Improved: <brief description>",
)
print(f"Registered version {new_prompt.version}")
```

### Step 3: Verify by running the eval script

Run the eval script from the skill directory with the prompt name and new version number:

```bash
uv run --no-project --python .venv -- python .claude/skills/costar-refine/eval.py <prompt_name> <version_number>
```

The script runs the agent on test scenarios and prints a JSON line with the scores, e.g.:
```
EVAL_RESULT: {"has_sources": 0.87, "conciseness": 0.6}
```

### Step 4: Compare and iterate

- Compare the new scores against the baseline scores provided in the prompt
- If the target metrics improved (or are already at 1.0) without regressing others, you're done
- If not, analyze what went wrong, rewrite the prompt again, register a new version, and re-evaluate
- Do at most 3 iterations total

### Step 5: Save the result

Write the best version number to `_refine_result.json`:

```python
import json
from pathlib import Path
Path("_refine_result.json").write_text(json.dumps({"version": best_version}))
```

## Constraints

- Only modify the prompt template text
- Do NOT change tools, agent code, or evaluation logic
- The prompt has no template variables — do not add {{ }} patterns
