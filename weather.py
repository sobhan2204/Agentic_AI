import os
import asyncio
import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any

os.environ["MCP_SERVER_PORT"] = "8000"

mcp = FastMCP("Weather")

OPEN_METEO_forcast_API = "https://api.open-meteo.com/v1/forecast"

@mcp.tool()
async def get_current_weather(lat: float, lon: float) -> dict:
    """
    Get current weather using latitude and longitude.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            OPEN_METEO_forcast_API,
            params=params,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

    current = data.get("current_weather")

    if not current:
        return {"error": "Weather data unavailable"}

    return {
        "temperature_c": current["temperature"],
        "windspeed_kmh": current["windspeed"],
        "wind_direction": current["winddirection"],
        "weather_code": current["weathercode"],
        "time": current["time"]
    }

OPEN_METEO_AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"

@mcp.tool()
async def get_air_quality(lat: float, lon: float) -> dict:
    """
    Get latest pollutant levels (hourly) and return the most recent sample.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "auto",
        "models": "cams_global",
        "past_days": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                OPEN_METEO_AQI_API,
                params=params,
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError:
        return {"error": "AQI data unavailable"}

    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        return {"error": "AQI data unavailable"}

    times = hourly["time"]

    def value_at(key: str, i: int):
        series = hourly.get(key)
        return series[i] if series and len(series) > i else None

    # pick the latest index with at least one non-null pollutant
    chosen_idx = None
    for i in range(len(times) - 1, -1, -1):
        vals = [
            value_at("pm2_5", i),
            value_at("pm10", i),
            value_at("carbon_monoxide", i),
            value_at("nitrogen_dioxide", i),
            value_at("sulphur_dioxide", i),
            value_at("ozone", i),
        ]
        if any(v is not None for v in vals):
            chosen_idx = i
            break

    if chosen_idx is None:
        return {"error": "AQI data unavailable"}

    return {
        "pm2_5": value_at("pm2_5", chosen_idx),
        "pm10": value_at("pm10", chosen_idx),
        "no2": value_at("nitrogen_dioxide", chosen_idx),
        "o3": value_at("ozone", chosen_idx),
        "co": value_at("carbon_monoxide", chosen_idx),
        "so2": value_at("sulphur_dioxide", chosen_idx),
        "timestamp": times[chosen_idx],
    }

NOMINATIM_API = "https://nominatim.openstreetmap.org/search"

@mcp.tool()
async def get_geo_details(place: str) -> dict:
    """
    Get latitude, longitude, country, and region for a place name.
    """
    params = {
        "q": place,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "geo-weather-agent/1.0"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            NOMINATIM_API,
            params=params,
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

    if not data:
        return {"error": "Location not found"}

    loc = data[0]

    return {
        "place": place,
        "latitude": float(loc["lat"]),
        "longitude": float(loc["lon"]),
        "display_name": loc["display_name"],
        "country": loc.get("display_name", "").split(",")[-1].strip()
    }
@mcp.tool()
async def get_environment_report(place: str) -> dict:
    """
    Get location, weather, and air quality for a place.
    """

    geo = await get_geo_details(place)
    if "error" in geo:
        return geo

    weather = await get_current_weather(
        geo["latitude"], geo["longitude"]
    )

    aqi = await get_air_quality(
        geo["latitude"], geo["longitude"]
    )

    return {
        "location": geo,
        "weather": weather,
        "air_quality": aqi
    }


# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")


if __name__ == "__main__":
    result = asyncio.run(get_environment_report("Delhi, India"))
    print(result)