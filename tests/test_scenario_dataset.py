"""Standalone smoke test for the research-scenarios eval dataset.

Runs WITHOUT an OpenAI key and WITHOUT the agent stack, against a temporary
sqlite tracking backend. Verifies seeding, idempotency, and deterministic
scenario ordering.

Usage: python tests/test_scenario_dataset.py
Exits 0 on success, non-zero on failure.
"""

import os
import sys
import tempfile

# Point setup at a throwaway sqlite backend BEFORE importing it, so the test
# needs no running MLflow server.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_tmp_db.name}"

# Import from the repo root regardless of the cwd the test is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import setup  # noqa: E402


def main():
    expected_questions = [s["question"] for s in setup.SCENARIOS]
    expected_facts = {s["question"]: s["expected_facts"] for s in setup.SCENARIOS}

    # Call twice to exercise idempotency of the seed/get-or-create path.
    for call in (1, 2):
        setup.get_scenario_dataset()
        scenarios = setup.load_scenarios()

        assert len(scenarios) == len(setup.SCENARIOS), (
            f"call {call}: expected {len(setup.SCENARIOS)} records, "
            f"got {len(scenarios)}"
        )

        questions = [s["question"] for s in scenarios]
        assert set(questions) == set(expected_questions), (
            f"call {call}: question set mismatch"
        )

        # Deterministic order must match the in-code SCENARIOS order (fix #1).
        assert questions == expected_questions, (
            f"call {call}: scenario order does not match SCENARIOS"
        )

        for s in scenarios:
            facts = s["expected_facts"]
            assert isinstance(facts, list), (
                f"call {call}: expected_facts is not a list for {s['question']!r}"
            )
            assert facts == expected_facts[s["question"]], (
                f"call {call}: expected_facts mismatch for {s['question']!r}"
            )

    print(
        f"OK: research-scenarios seeds {len(setup.SCENARIOS)} records, "
        "idempotent, ordered."
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    finally:
        os.unlink(_tmp_db.name)
