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

**You are running in a HEADLESS, ONE-SHOT session. There is NO next turn: you will
NOT be re-invoked, notified, woken up, or resumed.** Everything — including waiting for
this eval — must happen in THIS single response.

- Run the eval as ONE blocking FOREGROUND Bash command and WAIT for it to print its
  `EVAL_RESULT:` line before doing anything else.
- The eval takes a few minutes — set a long Bash timeout (e.g. `timeout: 600000` ms /
  10 minutes) so it completes in the foreground. Do NOT background it to avoid a timeout.
- NEVER use `run_in_background`, the Monitor tool, background tasks, or scheduled
  wakeups, and NEVER say "I'll continue when it finishes". Anything you defer or
  background WILL silently die and you WILL FAIL the task — there is no mechanism that
  will wake you back up.
- Do NOT end your turn while an eval is still running. Block on the single command until
  the `EVAL_RESULT:` line appears, then read the real scores from that line.

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
performed only after the eval has returned its `EVAL_RESULT:` line. Because this is a
HEADLESS, ONE-SHOT session, you will NOT get another turn — so do not end your response
until `_refine_result.json` exists on disk. The file MUST exist before you finish; if it
does not, the task has failed.

```python
import json
from pathlib import Path
Path("_refine_result.json").write_text(json.dumps({"version": best_version}))
```

## Constraints

- Only modify the prompt template text
- Do NOT change tools, agent code, or evaluation logic
- The prompt has no template variables — do not add {{ }} patterns
- You are in a HEADLESS, ONE-SHOT session: there is no next turn, no wakeup, no resume.
  NEVER use `run_in_background`, the Monitor tool, background tasks, scheduled wakeups, or
  any "I'll continue when it finishes" deferral — anything deferred or backgrounded WILL
  silently die and you WILL FAIL.
- Run the eval as a single blocking FOREGROUND Bash command with a long (~10 minute /
  600000 ms) timeout; do not end your turn while it runs.
- Stop the instant the goal's target pass rate is met (see Step 4), then write
  `_refine_result.json` in-turn as your final action. The file MUST exist before you finish.
