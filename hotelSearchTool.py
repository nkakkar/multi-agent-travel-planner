"""SerpApi-backed hotel search tool used by the travel planner."""

import json
import os
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch


# These fields form the structured input contract for hotel searches. The
# model uses their names and descriptions to build a valid tool call.
class HotelSearchInput(BaseModel):
    """Input schema for hotel search."""

    location: str = Field(..., description="Location (e.g., 'New York', 'Paris')")
    check_in_date: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(..., description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    rooms: int = Field(1, description="Number of rooms")
    hotel_class: Optional[str] = Field(None, description="Star rating (e.g., '3,4,5')")
    sort_by: int = Field(8, description="3=Price, 8=Rating, 13=Reviews")


# @tool lets the booking agent invoke this function with validated arguments.
@tool(args_schema=HotelSearchInput)
def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    hotel_class: Optional[str] = None,
    sort_by: int = 8
) -> str:
    """Search SerpApi for hotels and return a compact JSON summary.

    This searches available options; it does not make a reservation.
    """

    # Check credentials before building a request so a missing key is reported
    # as a setup problem, not misleadingly shown as zero hotels found.
    api_key = os.environ.get('SERPAPI_API_KEY')
    if not api_key:
        raise RuntimeError(
            "Hotel search requires SERPAPI_API_KEY. Add it to your .env file."
        )

    # These are the Google Hotels engine parameters accepted by SerpApi.
    # Dates and party/room counts describe the stay; currency and language
    # settings make the returned prices and text consistent.
    params = {
        'api_key': api_key,
        'engine': 'google_hotels',
        'q': location,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'adults': adults,
        'children': children,
        'rooms': rooms,
        'sort_by': sort_by,
        'currency': 'USD',
        'hl': 'en',
        'gl': 'us'
    }

    # Hotel class is a filter, so omit it unless the user actually requested a
    # star class. sort_by controls the order returned by the provider.
    if hotel_class:
        params['hotel_class'] = hotel_class

    # Make the search and surface errors from the provider separately from a
    # valid search that simply returned no properties.
    results = GoogleSearch(params).get_dict()
    if results.get("error"):
        raise RuntimeError(f"SerpApi hotel search failed: {results['error']}")
    properties = results.get('properties', [])

    if not properties:
        return json.dumps({"message": "No hotels found for those search details."})

    # Limit the response to five properties and retain the comparison fields
    # most useful to a traveler. SerpApi can omit fields, so .get() avoids
    # failing when optional details (such as photos) are unavailable.
    output = []
    for hotel in properties[:5]:
        output.append({
            "name": hotel.get("name"),
            "price": hotel.get("rate_per_night", {}).get("lowest"),
            "rating": hotel.get("overall_rating"),
            "reviews": hotel.get("reviews"),
            "stars": hotel.get("hotel_class"),
            "thumbnail": (
                hotel.get("images", [{}])[0].get("thumbnail")
                if hotel.get("images")
                else None
            ),
        })

    # A JSON string gives the language model a predictable, readable tool
    # result that it can summarize for the traveler.
    return json.dumps(output, indent=2)