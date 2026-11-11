# Multi-Agent Travel Planner

This project is a small command-line travel assistant. Give it a travel
request and it chooses the specialist work needed, runs those specialists,
and prints the results. For example, it can research a destination and build
an itinerary, search flights or hotels, or do both for one request.

It is an educational example of **multi-agent systems**, **LangChain tools**,
and **LangGraph workflows**. The agents suggest and research options; they do
not book tickets or hotel rooms.

## How the project works

The program follows this path:

1. **The router reads the request.** A language model returns a structured
   list of tasks: destination research, flight/hotel search, or both.
2. **LangGraph dispatches the tasks.** If both kinds of work are needed,
   LangGraph sends them to separate nodes at the same time.
3. **Specialist agents do their work.**
   - The research agent uses Tavily to search the web for current travel
     information and itinerary ideas.
   - The booking agent uses the flight and hotel tools, which query SerpApi.
4. **The graph gathers the results.** Its shared state collects the output
   from each agent.
5. **The synthesizer prepares the final answer.** One-agent answers are
   returned directly; answers from multiple agents are combined by the model.
6. **`demo.py` prints the task assignments and results.**

### Beginner's guide to the concepts

- **LLM (large language model):** The language model interprets the request,
  decides which specialist should handle it, and writes a clear response.
  This project uses ChatOpenAI. The model name can be changed with
  `OPENAI_MODEL`.
- **Tool:** A regular function or service the model can call for information
  it cannot reliably know by itself. Here, Tavily searches the web, while the
  flight and hotel tools query travel search results.
- **Agent:** A model equipped with instructions and tools. An agent can decide
  when a tool is useful, call it, read the result, and then respond. It does
  not have independent goals or run on its own.
- **Multi-agent system:** Multiple specialist agents share a larger job. This
  project separates broad travel research from flight/hotel searching, so
  each can concentrate on the part it handles best.
- **LangGraph:** A workflow framework that represents the program as nodes
  and connections. Nodes perform steps (routing, research, booking, and
  synthesis); edges determine what runs next. In this project, `Send` fans
  independent tasks out to the specialists. The `agent_results` state field
  uses a reducer so parallel results are collected rather than overwriting
  one another.
- **LangChain:** The library used for the model, messages, tools, and agent
  tool-calling behavior. LangGraph coordinates those pieces into the
  multi-step workflow.

## Project files

| File | Purpose |
| --- | --- |
| `demo.py` | Command-line entry point; loads settings, runs the graph, and prints output. |
| `travel_planner.py` | Router, specialist agent setup, graph nodes, parallel dispatch, and answer synthesis. |
| `flightSearchTool.py` | Structured flight search tool backed by SerpApi. |
| `hotelSearchTool.py` | Structured hotel search tool backed by SerpApi. |
| `requirements.txt` | Python package dependencies. |
| `.env.example` | Template for API keys and optional model setting. |

## Setup

Use Python 3.10 or newer. From this project folder, create and activate a
virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Copy `.env.example` to `.env` and fill in the keys:

```bash
cp .env.example .env
```

The planner needs:

- `OPENAI_API_KEY` — required for routing and agent responses.
- `TAVILY_API_KEY` — required for destination research and itinerary requests.
- `SERPAPI_API_KEY` — required only when the request asks for flight or hotel
  search results.

The example query is an itinerary request, so the first two keys are enough
to run it. A request that searches flights or hotels also needs the SerpApi
key. Keep `.env` private; it is excluded from Git.

## Run it

Run the built-in example:

```bash
python demo.py
```

Provide a different request:

```bash
python demo.py --query "Plan four days in Rome focused on ancient history and food"
```

Ask for flight or hotel search (provide the SerpApi key in `.env`):

```bash
python demo.py --query "Find a flight from JFK to LHR on 2026-12-10 for one adult"
```

Ask for both research and booking information:

```bash
python demo.py --query "Plan three days in Lisbon and find a hotel for two adults from 2026-12-10 to 2026-12-13"
```

Search prices and availability can change. Always check the airline, hotel,
and official destination websites before making travel decisions.

## Troubleshooting

- **A key is missing:** Add the required value to `.env`; the program reports
  which key is needed before or during the relevant search.
- **`ModuleNotFoundError`:** Activate `.venv` and run
  `python -m pip install -r requirements.txt` again.
- **API errors or rate limits:** Check that the relevant key is valid and
  that the service account allows the requested API.
- **No matching flight/hotel results:** Try an airport code, clearer dates,
  or broader search details. No-results responses are not bookings.
