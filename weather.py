import asyncio
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
NOMINATIM_API = "https://nominatim.openstreetmap.org/search"


async def _geocode(place: str) -> dict:
    """Internal: convert place name to lat/lon."""
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "geo-weather-mcp/1.0"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(NOMINATIM_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    if not data:
        return {"error": f"Location '{place}' not found"}
    loc = data[0]
    return {
        "lat": float(loc["lat"]),
        "lon": float(loc["lon"]),
        "display_name": loc.get("display_name", place),
    }


async def _fetch_weather(lat: float, lon: float) -> dict:
    """Internal: fetch weather from Open-Meteo."""
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_FORECAST_API, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def _fetch_aqi(lat: float, lon: float) -> dict:
    """Internal: fetch air quality from Open-Meteo."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "auto", "models": "cams_global", "past_days": 1,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_AQI_API, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Get current weather for a city or place by name.

    Args:
        city: Name of the city or place (e.g., "London", "New York", "Tokyo")

    Returns:
        str: A human-readable weather report for that location.
    """
    try:
        geo = await _geocode(city)
        if "error" in geo:
            return geo["error"]

        data = await _fetch_weather(geo["lat"], geo["lon"])
        current = data.get("current_weather")
        if not current:
            return f"Weather data unavailable for {city}"

        condition = WEATHER_CODES.get(current.get("weathercode", -1), "Unknown")
        temp = current.get("temperature", "N/A")
        wind = current.get("windspeed", "N/A")
        wind_dir = current.get("winddirection", "N/A")

        return (
            f"Weather in {geo['display_name']}:\n"
            f"  Condition: {condition}\n"
            f"  Temperature: {temp}°C\n"
            f"  Wind: {wind} km/h (direction: {wind_dir}°)"
        )
    except Exception as e:
        return f"Failed to get weather for '{city}': {str(e)}"


@mcp.tool()
async def get_air_quality(city: str) -> str:
    """
    Get current air quality for a city or place by name.

    Args:
        city: Name of the city or place (e.g., "London", "Delhi", "Beijing")

    Returns:
        str: A human-readable air quality report.
    """
    try:
        geo = await _geocode(city)
        if "error" in geo:
            return geo["error"]

        data = await _fetch_aqi(geo["lat"], geo["lon"])
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            return f"Air quality data unavailable for {city}"

        times = hourly["time"]

        def val(key, i):
            series = hourly.get(key)
            return series[i] if series and len(series) > i else None

        # Find latest data point with values
        latest_idx = None
        for i in range(len(times) - 1, -1, -1):
            if any(val(k, i) is not None for k in ["pm2_5", "pm10", "ozone"]):
                latest_idx = i
                break

        if latest_idx is None:
            return f"Air quality data unavailable for {city}"

        pm25 = val("pm2_5", latest_idx)
        pm10 = val("pm10", latest_idx)
        o3 = val("ozone", latest_idx)
        no2 = val("nitrogen_dioxide", latest_idx)

        lines = [f"Air Quality in {geo['display_name']} (as of {times[latest_idx]}):"]
        if pm25 is not None: lines.append(f"  PM2.5: {pm25} µg/m³")
        if pm10 is not None: lines.append(f"  PM10: {pm10} µg/m³")
        if o3 is not None:   lines.append(f"  Ozone: {o3} µg/m³")
        if no2 is not None:  lines.append(f"  NO2: {no2} µg/m³")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to get air quality for '{city}': {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
