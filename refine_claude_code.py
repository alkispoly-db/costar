"""Claude Code headless mode as an alternative Refine engine for coSTAR."""

import json
import os
import re
import subprocess
from pathlib import Path

import mlflow
from mlflow.entities.model_registry.prompt_version import PromptVersion

# Scorer source code that can be injected into the eval script.
# Kept here so both 01 and 03 scripts can import it.
HAS_SOURCES_SCORER = '''\
import re

URL_PATTERN = re.compile(r"https?://\\S+")

@scorer
def has_sources(outputs) -> bool:
    return bool(URL_PATTERN.search(str(outputs)))
'''


def refine_with_claude_code(
    prompt_name: str,
    prompt_version: int,
    scores: dict[str, float],
    goal: str,
    project_dir: str,
    scorer_source: str = "",
) -> PromptVersion:
    """Refine a prompt using Claude Code in headless mode.

    Writes evaluation context and an eval script, invokes Claude Code
    with the costar-refine skill, and returns the newly registered prompt.

    Args:
        scorer_source: Python source code that defines the scorer functions.
            Each scorer must be decorated with @scorer and importable after
            exec'ing this code. The eval script will use these scorers.
    """
    project = Path(project_dir)

    # Write the eval script that Claude Code will use to verify improvements
    _write_eval_script(project, prompt_name, scorer_source)

    # Write context for Claude Code
    context = {
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "scores": scores,
        "goal": goal,
        "eval_script": "_refine_eval.py",
    }
    context_file = project / "_refine_context.json"
    context_file.write_text(json.dumps(context, indent=2))

    # Clean up any previous result
    result_file = project / "_refine_result.json"
    if result_file.exists():
        result_file.unlink()

    # Build the headless prompt
    prompt = (
        "Read the costar-refine skill and _refine_context.json. "
        f"Improve the '{prompt_name}' prompt to achieve the goal described in the context file. "
        "Use the eval script to verify your changes improve the scores. "
        "Register the best prompt and write the version number to _refine_result.json."
    )

    print(f"\nInvoking Claude Code headlessly …")
    print(f"  Context: {context_file}")
    print(f"  Goal: {goal}")

    # Strip CLAUDECODE env var so claude doesn't refuse to run inside
    # an existing Claude Code session (the parent that launched this script).
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--dangerously-skip-permissions",
            "--allowedTools", "Bash,Read,Write,Edit",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"  Claude Code stderr:\n{result.stderr}")
        raise RuntimeError(f"Claude Code exited with code {result.returncode}")

    print(f"  Claude Code finished successfully.")
    if result.stdout:
        # Print last few lines of output as summary
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            print(f"    {line}")

    if not result_file.exists():
        raise RuntimeError(
            "Claude Code did not write _refine_result.json. "
            "Check the output above for errors."
        )

    new_version = json.loads(result_file.read_text())["version"]
    print(f"  New prompt version: v{new_version}")

    return mlflow.genai.load_prompt(prompt_name, version=new_version)


def _write_eval_script(project: Path, prompt_name: str, scorer_source: str):
    """Write _refine_eval.py that evaluates a prompt version against scenarios."""
    scorer_names = _scorer_names(scorer_source)
    script = f'''\
"""Evaluate a prompt version against test scenarios. Used by Claude Code during refinement."""

import json
import sys

import mlflow
from mlflow.genai.scorers import scorer

from setup import SCENARIOS, create_agent

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("costar-research-agent")

# ── Scorers (injected from the calling script) ──
{scorer_source}

# ── Main ──
version = int(sys.argv[1])
prompt = mlflow.genai.load_prompt("{prompt_name}", version=version)
agent = create_agent(prompt.template)

trace_ids = []
with mlflow.start_run(run_name=f"refine-eval-v{{version}}"):
    for s in SCENARIOS:
        agent.invoke({{"messages": [{{"role": "user", "content": s["question"]}}]}})
        trace_ids.append(mlflow.get_last_active_trace_id())

mlflow.flush_trace_async_logging()
traces = [mlflow.get_trace(tid) for tid in trace_ids]

result = mlflow.genai.evaluate(data=traces, scorers=[{scorer_names}])

scores = {{}}
for key, val in result.metrics.items():
    if key.endswith("/mean"):
        scores[key.removesuffix("/mean")] = val

print(f"EVAL_RESULT: {{json.dumps(scores)}}")
'''
    (project / "_refine_eval.py").write_text(script)


def _scorer_names(scorer_source: str) -> str:
    """Extract scorer function names from source code."""
    names = re.findall(r"@scorer\s*\ndef\s+(\w+)", scorer_source)
    return ", ".join(names)
