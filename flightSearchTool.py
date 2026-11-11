"""SerpApi-backed flight search tool used by the travel planner."""

import json
import os
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch


# Pydantic describes and validates the arguments that the language model is
# allowed to pass to the flight search tool. Field descriptions help the
# model understand what each value means.
class FlightSearchInput(BaseModel):
    """Input schema for flight search."""

    departure_airport: str = Field(..., description="Departure airport code (e.g., 'JFK')")
    arrival_airport: str = Field(..., description="Arrival airport code (e.g., 'LAX')")
    outbound_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    stops: Optional[int] = Field(None, description="0=Any, 1=Nonstop, 2=1 stop or fewer")


# The @tool decorator exposes this Python function to LangChain agents. The
# argument names here must match the fields in FlightSearchInput so model
# tool calls can be validated before the function runs.
@tool(args_schema=FlightSearchInput)
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    stops: Optional[int] = None
) -> str:
    """Search SerpApi for flights and return a compact JSON summary.

    This searches options only; it does not book a flight.
    """

    # Never send a request without credentials. Raising an error gives the
    # agent and user a clear explanation instead of returning fake empty data.
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Flight search requires SERPAPI_API_KEY. Add it to your .env file."
        )

    # SerpApi's Google Flights engine uses these parameter names. The
    # departure/arrival airport IDs are normalized to uppercase, and "type"
    # distinguishes a one-way search from a round trip.
    params = {
        'api_key': api_key,
        'engine': 'google_flights',
        'departure_id': departure_airport.upper(),
        'arrival_id': arrival_airport.upper(),
        'outbound_date': outbound_date,
        'adults': adults,
        'children': children,
        'currency': 'USD',
        'type': 1 if return_date else 2
    }

    # Include optional filters only when the user supplied them, so defaults
    # do not unintentionally restrict the search.
    if return_date:
        params['return_date'] = return_date
    if stops is not None:
        params['stops'] = stops

    # get_dict() makes the API request and returns a Python dictionary.
    # SerpApi may report provider errors inside that dictionary, so surface
    # those explicitly instead of treating them as no matching flights.
    results = GoogleSearch(params).get_dict()
    if results.get("error"):
        raise RuntimeError(f"SerpApi flight search failed: {results['error']}")
    flights = results.get('best_flights', []) + results.get('other_flights', [])

    # An empty result is a valid search outcome (unlike an API error above).
    if not flights:
        return json.dumps({"message": "No flights found for those search details."})

    # Keep the first ten options and extract fields useful for comparing
    # flights. Each flight can have multiple legs, for example when it has a
    # connection, so summarize every leg's carrier and airport/time.
    output = []
    for flight in flights[:10]:
        output.append({
            "price": flight.get("price"),
            "duration_mins": flight.get("total_duration"),
            "airline_logo": flight.get("airline_logo"),
            "legs": [{
                "airline": leg.get("airline"),
                "flight_number": leg.get("flight_number"),
                "departure": (
                    f"{leg.get('departure_airport', {}).get('id')} "
                    f"{leg.get('departure_airport', {}).get('time')}"
                ),
                "arrival": (
                    f"{leg.get('arrival_airport', {}).get('id')} "
                    f"{leg.get('arrival_airport', {}).get('time')}"
                ),
                "airline_logo": leg.get("airline_logo"),
            } for leg in flight.get("flights", [])],
        })

    # Returning JSON keeps the tool result compact and easy for the agent to
    # read and summarize.
    return json.dumps(output, indent=2)