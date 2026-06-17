# coSTAR Prompt Refinement

You are improving an AI agent's system prompt based on evaluation feedback.
This is a target-driven process: rewrite the prompt, verify with the eval script,
and iterate ONLY until the goal's target pass rate is met — then stop immediately.

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

Run this command in the FOREGROUND and WAIT for it to print its `EVAL_RESULT:` line
before doing anything else. NEVER run the eval in the background. Do not end your turn
while an eval is still running.

The script runs the agent on test scenarios and prints a JSON line with the scores, e.g.:
```
EVAL_RESULT: {"has_sources": 0.87, "conciseness": 0.6}
```

### Step 4: Check the target and stop as soon as it is met

The `goal` you are given states a **TARGET pass rate for a named scorer** (e.g.
"conciseness/mean >= 0.8"). This target is your stop condition:

- After each eval, read the named scorer's mean from the `EVAL_RESULT:` line.
- **As soon as that scorer's mean is at or above the target, STOP immediately and
  finalize the current version — do NOT keep iterating to find a "better" prompt.**
  Meeting the target is the whole job; optimizing past it wastes time.
- If the target is not yet met, analyze what went wrong, rewrite the prompt,
  register a new version, and re-evaluate.
- Respect any guard metrics in the goal (e.g. keep `has_sources` at or near 1.0):
  do not regress them while pushing the target metric up.
- Safety cap: do at most 3 iterations total. If the target is still not met after
  3 iterations, finalize the best version you have. But the target — not the cap —
  is the primary stop condition: stop the instant it is met, even on iteration 1.

### Step 5: Save the result

Write the best version number to `_refine_result.json`. This MUST be your LAST action,
performed only after the eval has returned its `EVAL_RESULT:` line. Do not end your turn
while an eval is still running — the file MUST exist before you finish.

```python
import json
from pathlib import Path
Path("_refine_result.json").write_text(json.dumps({"version": best_version}))
```

## Constraints

- Only modify the prompt template text
- Do NOT change tools, agent code, or evaluation logic
- The prompt has no template variables — do not add {{ }} patterns
- Do not run the eval asynchronously or end your turn waiting for a background task;
  `_refine_result.json` must exist before you finish.
