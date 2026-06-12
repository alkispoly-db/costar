"""
01 — TRACE phase (coSTAR loop 1).

Runs the baseline agent (prompt v1) over the research-scenarios dataset and
captures one trace per scenario under a run named "01-trace". This is the T in
STAR: produce the traces we will assess and refine in the next two phases. No
scoring happens here.

After running, switch to the experiment's Traces tab in the MLflow UI to
inspect the captured traces.

No standalone scoring — see 01-assess.py for the A phase.
"""

from setup import create_agent, latest_prompt, run_scenarios

# v1 in the clean state; create_agent loads the langchain/deepagents stack.
prompt = latest_prompt()
agent = create_agent(prompt.template)

traces = run_scenarios(agent, run_name="01-trace")

print(f"\nGenerated {len(traces)} traces under run '01-trace'.")
print("Open the Traces tab in the MLflow UI to inspect them.")
