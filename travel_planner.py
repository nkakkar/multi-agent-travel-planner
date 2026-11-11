"""LangGraph workflow that routes travel requests to specialist agents."""

from __future__ import annotations

# operator.add is used below as a LangGraph state reducer: results from
# multiple parallel agent nodes are appended together instead of replacing
# each other. The typing helpers describe the graph's shared state.
import operator
import os
from typing import Annotated, Literal, TypedDict

# LangChain provides the tool-using agent and chat messages/model; LangGraph
# provides workflow nodes, routing, and parallel task dispatch.
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# These project-local functions are exposed to the booking agent as tools.
from flightSearchTool import search_flights
from hotelSearchTool import search_hotels


class AgentTask(BaseModel):
    """One focused piece of work assigned to a specialist.

    The source field limits routing to nodes that actually exist in the graph.
    Each task includes a tailored query and an optional short description of
    what the agent should focus on.
    """

    source: Literal["research_agent", "booking_agent"]
    user_query: str
    focus: str = ""


class RoutePlan(BaseModel):
    """Structured decision returned by the request router.

    Structured output makes the model return validated task data instead of
    an unstructured paragraph that the program would have to parse itself.
    """

    tasks: list[AgentTask] = Field(
        description="One or more focused tasks for the travel specialist agents."
    )


class TravelState(TypedDict, total=False):
    """Shared data passed between the graph's nodes.

    total=False means each node can return only the fields it updates.
    Parallel specialists each return one agent_results item; Annotated with
    operator.add tells LangGraph to combine those lists in the shared state.
    """

    user_query: str
    tasks: list[AgentTask]
    task: AgentTask
    agent_results: Annotated[list[dict[str, str]], operator.add]
    final_answer: str


# These prompts define each model's responsibilities. Keeping the router,
# specialists, and synthesizer instructions separate makes the workflow
# easier to understand and tune without changing its graph structure.
ROUTER_PROMPT = """You are the request router for a travel-planning assistant.
Choose one or both specialists based on what the traveler asks:

- research_agent: destination guidance, activities, attractions, travel tips,
  or day-by-day itineraries. It can search the web for current information.
- booking_agent: flight and/or hotel search requests. It can query live
  Google Flights and Google Hotels results through SerpApi.

Use only research_agent for travel advice or itinerary requests.
Use only booking_agent for flight/hotel requests.
Use both when the traveler asks for research/planning and flight/hotel results.
Create a precise subtask for each selected specialist, preserving the user's
dates, locations, party size, and preferences. Do not invent missing details.
Return at least one task. The specialist will ask follow-up questions if
required booking information is missing.
"""

RESEARCH_PROMPT = """You are a practical travel research and itinerary agent.
Answer travel-related questions using Tavily web search for current facts.
For itineraries, make a realistic day-by-day plan with locations, approximate
times, transportation guidance, and useful practical tips. Distinguish
estimates from verified facts, avoid making up prices or opening hours, and
include source links when search results provide them. If the request lacks
important trip details, state your assumptions or ask a concise question.
"""

BOOKING_PROMPT = """You are a flight and hotel search agent.
Use the flight and hotel tools for requests that include enough information
to search. Never claim that a booking has been made; these tools only search.
For flights, collect origin, destination, departure date, passenger counts,
and a return date if the traveler wants a round trip. For hotels, collect
location, check-in and check-out dates, and guest/room counts. If a required
detail is missing or ambiguous, ask for it instead of guessing or calling
the tool. Summarize the returned options and clearly label prices and
availability as search results that can change.
"""

SYNTHESIS_PROMPT = """Combine the specialist results into one helpful answer
to the traveler's original request.

Keep useful sections distinct (for example: itinerary, flights, hotels, and
practical notes). Do not invent facts or claim that a booking was made. Call
out estimates, missing information, and important trade-offs. Avoid repeating
the same details.

Original request:
{query}

Specialist results:
{results}
"""


def _make_agents():
    """Create the shared chat model and the two specialist agents.

    The research agent receives a Tavily web-search tool. The booking agent
    receives the flight and hotel search tools. The same model also handles
    routing and, when necessary, combining the agents' answers.
    """
    # OPENAI_MODEL is optional; this default is inexpensive and widely
    # available. ChatOpenAI reads OPENAI_API_KEY from the environment.
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(model=model_name, temperature=0.2)

    # TavilySearch is a LangChain tool: the research agent can call it when it
    # needs current web information rather than relying only on model memory.
    web_search = TavilySearch(max_results=5, search_depth="advanced")

    # create_agent manages the model/tool loop: it can call a tool, receive
    # its observation, and continue until it has a user-facing response.
    research_agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt=RESEARCH_PROMPT,
    )
    booking_agent = create_agent(
        model=model,
        tools=[search_flights, search_hotels],
        system_prompt=BOOKING_PROMPT,
    )
    return model, research_agent, booking_agent


def _route_request(state: TravelState, model: ChatOpenAI) -> dict:
    """Ask the model to convert the request into validated specialist tasks."""
    # Bind the Pydantic schema to the model so its answer is a RoutePlan with
    # AgentTask objects, not free-form prose.
    router = model.with_structured_output(RoutePlan)
    plan = router.invoke(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    if not plan.tasks:
        raise ValueError("The router did not assign any travel-planning tasks.")
    # Nodes return dictionaries containing only the state fields they update.
    return {"tasks": plan.tasks}


def _dispatch_tasks(state: TravelState) -> list[Send]:
    """Create one LangGraph Send instruction for each assigned specialist.

    If the router selected both specialists, LangGraph can execute these
    sends in parallel. Each worker receives its own task as local input.
    """
    sends: list[Send] = []
    for task in state.get("tasks", []):
        sends.append(Send(task.source, {"task": task}))
    if not sends:
        raise ValueError("Cannot dispatch the request because no tasks were assigned.")
    return sends


def _run_research_agent(state: TravelState, agent) -> dict:
    """Run the destination-research specialist for its assigned task."""
    task = state["task"]
    # Agent state uses a message list. A HumanMessage carries the routed
    # sub-query into the agent's tool-calling loop.
    result = agent.invoke({"messages": [HumanMessage(content=task.user_query)]})
    answer = str(result["messages"][-1].content)
    # Return a one-element list; the state reducer appends it to results from
    # any other specialist dispatched for this same request.
    return {
        "agent_results": [
            {
                "agent": "Research agent",
                "focus": task.focus,
                "result": answer,
            }
        ]
    }


def _run_booking_agent(state: TravelState, agent) -> dict:
    """Run the flight/hotel specialist for its assigned task."""
    task = state["task"]
    result = agent.invoke({"messages": [HumanMessage(content=task.user_query)]})
    answer = str(result["messages"][-1].content)
    return {
        "agent_results": [
            {
                "agent": "Booking agent",
                "focus": task.focus,
                "result": answer,
            }
        ]
    }


def _synthesize(state: TravelState, model: ChatOpenAI) -> dict:
    """Return a single agent result directly or combine parallel results."""
    results = state.get("agent_results", [])
    if not results:
        raise ValueError("The specialist agents returned no results.")

    # A single-agent request needs no extra model call; retaining its answer
    # directly avoids adding latency and avoids unnecessary rewriting.
    if len(results) == 1:
        answer = results[0]["result"]
    else:
        # Label each specialist result so the model can keep research and
        # booking information organized while producing one response.
        formatted_results = "\n\n".join(
            f"{item['agent']} — {item['focus']}\n{item['result']}"
            for item in results
        )
        response = model.invoke(
            SYNTHESIS_PROMPT.format(
                query=state["user_query"],
                results=formatted_results,
            )
        )
        answer = str(response.content)

    # This is the final state field read by demo.py for display.
    return {"final_answer": answer}


def build_travel_planner():
    """Create and compile the router → specialists → synthesis workflow.

    A LangGraph StateGraph is similar to a flowchart: nodes perform work,
    edges describe what runs next, and the shared TravelState carries data
    between steps.
    """
    model, research_agent, booking_agent = _make_agents()
    graph = StateGraph(TravelState)

    # Each node is one function in the flowchart. Lambdas supply the model or
    # agent dependencies that LangGraph does not keep inside TravelState.
    graph.add_node("router", lambda state: _route_request(state, model))
    graph.add_node(
        "research_agent",
        lambda state: _run_research_agent(state, research_agent),
    )
    graph.add_node(
        "booking_agent",
        lambda state: _run_booking_agent(state, booking_agent),
    )
    graph.add_node("synthesizer", lambda state: _synthesize(state, model))

    # Every run starts at the router. Its conditional edge chooses specialist
    # nodes dynamically; the named destinations are the allowed graph nodes.
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _dispatch_tasks,
        ["research_agent", "booking_agent"],
    )

    # Both specialists lead to synthesis. If only one was selected, it still
    # passes through synthesis, which returns its single result unchanged.
    graph.add_edge("research_agent", "synthesizer")
    graph.add_edge("booking_agent", "synthesizer")
    graph.add_edge("synthesizer", END)
    # Compilation validates and turns the configured graph into an invokable
    # runnable object used by demo.py.
    return graph.compile()
