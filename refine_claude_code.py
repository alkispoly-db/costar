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
        f"When the target pass rate in the goal is met, register the best version, then write "
        f"its version number to _refine_result.json. "
        f"\n\n"
        f"=== HEADLESS ONE-SHOT EXECUTION — READ CAREFULLY ===\n"
        f"You are running in a HEADLESS, ONE-SHOT session. There is NO next turn: you will "
        f"NOT be re-invoked, notified, woken up, or resumed. Everything must happen in THIS "
        f"single response.\n"
        f"NEVER use run_in_background, the Monitor tool, background tasks, scheduled wakeups, "
        f"or 'I'll continue when it finishes' deferral. Anything you defer or background WILL "
        f"silently die and you WILL FAIL the task. There is no mechanism that will wake you "
        f"back up — a backgrounded eval and an armed Monitor are guaranteed failures.\n"
        f"Run the eval as ONE blocking FOREGROUND Bash command and WAIT for it to print its "
        f"'EVAL_RESULT:' line. The eval takes a few minutes — set a long Bash timeout (e.g. "
        f"timeout: 600000 ms / 10 minutes) so it completes in the foreground; do NOT "
        f"background it to avoid a timeout, and do NOT end your turn while an eval is still "
        f"running. Block on the single command until the 'EVAL_RESULT:' line appears in its "
        f"output, then read the real scores from that line.\n"
        f"Do not end your response until _refine_result.json exists on disk. Writing that "
        f"file (after reading the real 'EVAL_RESULT:' scores) is your FINAL action. The file "
        f"MUST exist before you finish — if it does not, the task has failed."
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
