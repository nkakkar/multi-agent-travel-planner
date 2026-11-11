"""Run the multi-agent travel planner from the command line."""

from __future__ import annotations

# argparse reads options such as --query from the command line. os reads API
# keys from environment variables, and sys lets us send errors to stderr and
# return a proper exit status to the shell.
import argparse
import os
import sys

# load_dotenv reads key/value settings from a local .env file into the process
# environment. The workflow builder contains the LangGraph implementation.
from dotenv import load_dotenv

from travel_planner import build_travel_planner


# This makes the program useful immediately after setup: users can run
# "python demo.py" without first composing a query on the command line.
DEFAULT_QUERY = (
    "Plan a three-day visit to Kyoto for someone interested in history and food."
)


def parse_args() -> argparse.Namespace:
    """Read command-line options and return the travel request to run."""
    parser = argparse.ArgumentParser(
        description="Plan a trip with a LangGraph multi-agent workflow."
    )
    # The supplied query replaces DEFAULT_QUERY; argparse also creates the
    # help text shown by "python demo.py --help".
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Travel request for the planner (wrap it in quotes).",
    )
    return parser.parse_args()


def main() -> int:
    # Load configuration before creating agents: their model and search tools
    # read the API keys from the environment.
    load_dotenv()
    args = parse_args()

    # Fail early with actionable setup instructions rather than waiting for an
    # API call to fail with a more confusing provider error.
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and add your "
            "OpenAI API key.",
            file=sys.stderr,
        )
        return 1

    # Web research is enabled in the example workflow, so Tavily credentials
    # are required even if the router ultimately chooses a single agent.
    if not os.environ.get("TAVILY_API_KEY"):
        print(
            "Missing TAVILY_API_KEY. Copy .env.example to .env and add your "
            "Tavily API key.",
            file=sys.stderr,
        )
        return 1

    # Build the graph once and invoke it with the user's request as its input
    # state. Provider or graph errors are displayed on stderr and produce a
    # non-zero command-line exit code.
    try:
        planner = build_travel_planner()
        result = planner.invoke({"user_query": args.query})
    except Exception as exc:
        print(f"Travel planner failed: {exc}", file=sys.stderr)
        return 1

    # Print the routing decision so a beginner can see which specialists the
    # router selected before reading each agent's result.
    print("=" * 72)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("=" * 72)
    print(f"\nYour request:\n{result['user_query']}")

    print("\nAgents assigned:")
    for task in result["tasks"]:
        print(f"- {task.source}: {task.focus or task.user_query}")

    for item in result.get("agent_results", []):
        print(f"\n--- {item['agent']} ---")
        print(item["result"])

    # This is the consolidated response returned by the graph's final node.
    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(result["final_answer"])
    return 0


# This guard allows the file to be imported by another Python module without
# starting the command-line program. When run directly, it passes main()'s
# success/failure code back to the operating system.
if __name__ == "__main__":
    raise SystemExit(main())
