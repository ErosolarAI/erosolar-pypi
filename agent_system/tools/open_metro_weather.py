"""Open-Meteo powered weather lookup tool."""
from __future__ import annotations

from typing import Any, Dict, List

import requests

from ..tool_registry import MCPTool

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _normalize_hours(value: Any) -> int:
    if value is None:
        return 24
    try:
        hours = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'hours' must be an integer") from exc
    return max(1, min(72, hours))


def _geocode_location(location: str) -> Dict[str, Any]:
    """
    Geocode a location using Open-Meteo's geocoding API with intelligent retry strategies.

    Tries multiple search strategies to maximize success rate:
    1. Original location string (e.g., "Burlington, MA")
    2. City name only (e.g., "Burlington")
    3. City + State/Province full name (e.g., "Burlington, Massachusetts")
    4. City + Country (e.g., "Burlington, USA")
    """
    # US State abbreviation to full name mapping
    state_expansions = {
        "MA": "Massachusetts", "CA": "California", "NY": "New York",
        "TX": "Texas", "FL": "Florida", "IL": "Illinois",
        "PA": "Pennsylvania", "OH": "Ohio", "GA": "Georgia",
        "NC": "North Carolina", "MI": "Michigan", "NJ": "New Jersey",
        "VA": "Virginia", "WA": "Washington", "AZ": "Arizona",
        "IN": "Indiana", "TN": "Tennessee", "MO": "Missouri",
        "MD": "Maryland", "WI": "Wisconsin", "CO": "Colorado",
        "MN": "Minnesota", "SC": "South Carolina", "AL": "Alabama",
        "LA": "Louisiana", "KY": "Kentucky", "OR": "Oregon",
        "OK": "Oklahoma", "CT": "Connecticut", "UT": "Utah",
        "IA": "Iowa", "NV": "Nevada", "AR": "Arkansas",
        "MS": "Mississippi", "KS": "Kansas", "NM": "New Mexico",
        "NE": "Nebraska", "WV": "West Virginia", "ID": "Idaho",
        "HI": "Hawaii", "NH": "New Hampshire", "ME": "Maine",
        "MT": "Montana", "RI": "Rhode Island", "DE": "Delaware",
        "SD": "South Dakota", "ND": "North Dakota", "AK": "Alaska",
        "VT": "Vermont", "WY": "Wyoming",
    }

    # Parse the location to extract components
    location_parts = [part.strip() for part in location.split(',')]

    # Build search strategies in order of preference
    search_strategies = [location]  # Always try original first

    if len(location_parts) == 2:
        city, region = location_parts
        # Try different variations
        search_strategies.extend([
            city,  # City only
            f"{city}, USA",  # Assume USA if not specified
            f"{city}, United States",  # Full country name
        ])

        # If region looks like a US state abbreviation (2 letters), expand it
        if len(region) == 2 and region.isalpha():
            full_state = state_expansions.get(region.upper())
            if full_state:
                search_strategies.extend([
                    f"{city}, {full_state}",
                    f"{city}, {full_state}, USA",
                ])

    elif len(location_parts) == 1:
        # Single part - try with common country additions
        search_strategies.extend([
            f"{location}, USA",
            f"{location}, United States",
        ])

    # Try each strategy until one succeeds
    # Request multiple results to allow for better filtering
    # Use count=15 to ensure we get enough results to find smaller cities
    last_error = None
    for strategy in search_strategies:
        try:
            params = {"name": strategy, "count": 15, "language": "en", "format": "json"}
            response = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []

            if results:
                # Filter results based on the original location string
                # If a US state was specified, try to match it exactly
                if len(location_parts) == 2:
                    region = location_parts[1].strip()

                    # Check if it's a state abbreviation
                    if len(region) == 2 and region.isalpha():
                        full_state = state_expansions.get(region.upper())
                        if full_state:
                            # Try to find a match for the specific state
                            state_matches = [
                                r for r in results
                                if r.get("country_code") == "US" and
                                r.get("admin1") == full_state
                            ]
                            if state_matches:
                                return state_matches[0]

                    # If region is a full state/province name, try to match it
                    else:
                        region_matches = [
                            r for r in results
                            if r.get("admin1", "").lower() == region.lower()
                        ]
                        if region_matches:
                            return region_matches[0]

                    # Fallback: prioritize US results if state abbreviation was used
                    if len(region) == 2 and region.isalpha():
                        us_results = [r for r in results if r.get("country_code") == "US"]
                        if us_results:
                            return us_results[0]

                # Return the first result
                return results[0]
        except Exception as e:
            last_error = e
            continue

    # All strategies failed
    raise ValueError(
        f"No matching location found for '{location}'. "
        f"Tried variations: {', '.join(repr(s) for s in search_strategies[:3])}... "
        f"Please try a different location format like 'City, Country' or 'City, State, Country'."
    )


def _fetch_forecast(latitude: float, longitude: float, timezone: str | None, hours: int) -> Dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weathercode,wind_speed_10m,wind_direction_10m",
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,relative_humidity_2m,wind_speed_10m",
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": timezone or "auto",
        "forecast_days": max(1, min(7, (hours + 23) // 24)),
    }
    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _summarize_hourly(hourly: Dict[str, List[Any]], hours: int) -> List[Dict[str, Any]]:
    if not hourly:
        return []
    times = hourly.get("time") or []
    fields = (
        "temperature_2m",
        "apparent_temperature",
        "precipitation_probability",
        "precipitation",
        "relative_humidity_2m",
        "wind_speed_10m",
    )
    window = min(len(times), hours)
    summaries: List[Dict[str, Any]] = []
    for idx in range(window):
        entry = {"time": times[idx]}
        for field in fields:
            values = hourly.get(field)
            if values and idx < len(values):
                entry[field] = values[idx]
        summaries.append(entry)
    return summaries


def open_metro_weather_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
    location = (inputs.get("location") or inputs.get("query") or "").strip()
    if not location:
        raise ValueError("'location' is required")
    hours = _normalize_hours(inputs.get("hours"))
    place = _geocode_location(location)
    forecast = _fetch_forecast(place["latitude"], place["longitude"], place.get("timezone"), hours)
    hourly_summary = _summarize_hourly(forecast.get("hourly") or {}, hours)
    return {
        "location": {
            "name": place.get("name"),
            "country": place.get("country"),
            "admin1": place.get("admin1"),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": place.get("timezone"),
        },
        "requested_hours": hours,
        "current": forecast.get("current"),
        "current_units": forecast.get("current_units"),
        "hourly": hourly_summary,
        "hourly_units": forecast.get("hourly_units"),
        "daily": forecast.get("daily"),
        "daily_units": forecast.get("daily_units"),
        "source": {
            "provider": "open-meteo.com",
            "license": "CC-BY 4.0",
        },
    }


def build_open_metro_weather_tool() -> List[MCPTool]:
    return [
        MCPTool(
            name="open_metro_weather_lookup",
            description=(
                "Look up current weather conditions and short-term forecasts using the Open-Meteo API. "
                "Supports intelligent location matching with automatic retry strategies. "
                "Recommended location formats: 'City, State' (e.g., 'Boston, MA'), "
                "'City, Country' (e.g., 'Paris, France'), or 'City' (e.g., 'London'). "
                "For US locations, both state abbreviations and full names work (e.g., 'Burlington, MA' or 'Burlington, Massachusetts')."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Location to look up. Accepts multiple formats:\n"
                            "- City name: 'Tokyo'\n"
                            "- City, State: 'Austin, TX' or 'Austin, Texas'\n"
                            "- City, Country: 'Sydney, Australia'\n"
                            "- City, State, Country: 'Portland, OR, USA'\n"
                            "The tool will automatically try variations if the first format doesn't work."
                        ),
                    },
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 72,
                        "description": "How many hours of the hourly forecast to return (default 24).",
                    },
                },
                "required": ["location"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "object"},
                    "requested_hours": {"type": "integer"},
                    "current": {"type": "object"},
                    "hourly": {"type": "array", "items": {"type": "object"}},
                    "daily": {"type": "object"},
                    "source": {"type": "object"},
                },
            },
            handler=open_metro_weather_handler,
            metadata={
                "categories": ["weather", "utilities"],
                "latency_budget_ms": 800,
            },
        )
    ]
