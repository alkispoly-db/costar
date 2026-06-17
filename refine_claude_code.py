"""Claude Code headless mode as an alternative Refine engine for coSTAR."""

import json
import os
import subprocess
from pathlib import Path

import mlflow
from mlflow.entities.model_registry.prompt_version import PromptVersion


def refine_with_claude_code(
    prompt_name: str,
    prompt_version: int,
    scores: dict[str, float],
    goal: str,
    project_dir: str,
) -> PromptVersion:
    """Refine a prompt using Claude Code in headless mode.

    Passes context directly in the prompt, invokes Claude Code with the
    costar-refine skill, and returns the newly registered prompt version.
    """
    project = Path(project_dir)

    # Clean up any previous result
    result_file = project / "_refine_result.json"
    if result_file.exists():
        result_file.unlink()

    scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
    prompt = (
        f"Read the costar-refine skill. "
        f"Improve the '{prompt_name}' prompt (currently version {prompt_version}, "
        f"scores: {scores_str}). "
        f"Goal: {goal} "
        f"Use the eval script in the skill directory to verify your changes improve the scores. "
        f"CRITICAL: Run the eval script in the FOREGROUND and WAIT for it to print its "
        f"'EVAL_RESULT:' line before doing anything else. Do NOT run the eval in the "
        f"background, and do NOT end your turn while an eval is still running. "
        f"When you have found the best version, register it, then write its version number "
        f"to _refine_result.json. "
        f"Writing _refine_result.json MUST be your final action, and the file MUST exist "
        f"before you finish. Do not stop or hand off while waiting on a background task."
    )

    print(f"\nInvoking Claude Code headlessly …")
    print(f"  Prompt: {prompt_name} v{prompt_version}")
    print(f"  Scores: {scores_str}")
    print(f"  Goal: {goal}")

    # Strip CLAUDECODE env var so claude doesn't refuse to run inside
    # an existing Claude Code session (the parent that launched this script).
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    claude_cmd = [
        "claude",
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--allowedTools", "Bash,Read,Write,Edit",
    ]

    def run_claude() -> subprocess.CompletedProcess:
        return subprocess.run(
            claude_cmd,
            cwd=project,
            capture_output=True,
            text=True,
            env=env,
        )

    result = run_claude()

    if result.returncode != 0:
        print(f"  Claude Code stderr:\n{result.stderr}")
        raise RuntimeError(f"Claude Code exited with code {result.returncode}")

    print(f"  Claude Code finished successfully.")
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            print(f"    {line}")

    # A single retry catches the rare case where the nested claude ended
    # without writing the result file (e.g. it backgrounded the eval).
    if not result_file.exists():
        print("  _refine_result.json missing — retrying Claude Code once …")
        result = run_claude()
        if result.returncode != 0:
            print(f"  Claude Code stderr:\n{result.stderr}")
            raise RuntimeError(f"Claude Code exited with code {result.returncode}")
        print(f"  Claude Code retry finished successfully.")
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-5:]:
                print(f"    {line}")

    if not result_file.exists():
        raise RuntimeError(
            "Claude Code did not write _refine_result.json. "
            "Check the output above for errors."
        )

    new_version = json.loads(result_file.read_text())["version"]
    print(f"  New prompt version: v{new_version}")

    return mlflow.genai.load_prompt(prompt_name, version=new_version)
