import asyncio
import httpx
import traceback
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQI_API      = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_GEOCODE_API  = "https://geocoding-api.open-meteo.com/v1/search"


async def _geocode(place: str) -> dict:
    """Internal: convert place name to lat/lon."""
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    headers = {"User-Agent": "geo-weather-mcp/1.0"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_GEOCODE_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") or []
    if not results:
        return {"error": f"Location '{place}' not found. Try a different spelling or a nearby major city."}
    loc = results[0]
    city = loc.get("name", place)
    country = loc.get("country")
    admin1 = loc.get("admin1")
    display_name = ", ".join([x for x in [city, admin1, country] if x])
    return {
        "lat":          float(loc["latitude"]),
        "lon":          float(loc["longitude"]),
        "display_name": display_name or place,
    }


async def _fetch_weather(lat: float, lon: float) -> dict:
    """Internal: fetch weather from Open-Meteo."""
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "current_weather": True,
        # Also fetch feels-like and humidity for richer output
        "hourly":          "relativehumidity_2m",
        "timezone":        "auto",
    }
    headers = {"User-Agent": "geo-weather-mcp/1.0"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_FORECAST_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()


async def _fetch_aqi(lat: float, lon: float) -> dict:
    """Internal: fetch air quality from Open-Meteo."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone":  "auto",
        "models":    "cams_global",
        "past_days": 1,
    }
    headers = {"User-Agent": "geo-weather-mcp/1.0"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_AQI_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()


WEATHER_CODES = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Get the current weather conditions for any city or place by name.

    Use this tool whenever the user asks about weather, temperature, wind,
    or current conditions in a location.

    Args:
        city: The name of the city or place to get weather for.
              Examples: "London", "New York", "Tokyo", "Delhi", "Paris"

    Returns:
        A human-readable weather report including temperature, conditions,
        and wind speed for that location.
    """
    if not city or not city.strip():
        return "Please provide a city name to get the weather for."

    city = city.strip()

    try:
        geo = await _geocode(city)
        if "error" in geo:
            return geo["error"]

        data    = await _fetch_weather(geo["lat"], geo["lon"])
        current = data.get("current_weather")

        if not current:
            return f"Weather data is currently unavailable for '{city}'. Please try again later."

        weather_code = current.get("weathercode", -1)
        condition    = WEATHER_CODES.get(weather_code, f"Unknown (code {weather_code})")
        temp         = current.get("temperature", "N/A")
        wind         = current.get("windspeed",   "N/A")
        wind_dir     = current.get("winddirection", "N/A")
        is_day       = current.get("is_day", 1)
        time_of_day  = "daytime" if is_day else "nighttime"

        return (
            f"Weather in {geo['display_name']}:\n"
            f"  Condition    : {condition} ({time_of_day})\n"
            f"  Temperature  : {temp}°C\n"
            f"  Wind Speed   : {wind} km/h\n"
            f"  Wind Direction: {wind_dir}°"
        )

    except httpx.TimeoutException:
        return f"Request timed out while fetching weather for '{city}'. Please try again."
    except httpx.HTTPStatusError as e:
        return f"Weather API error for '{city}': HTTP {e.response.status_code}"
    except Exception as e:
        return (
            f"Failed to get weather for '{city}': {type(e).__name__}: {repr(e)}\n"
            f"Trace: {traceback.format_exc(limit=1).strip()}"
        )


@mcp.tool()
async def get_air_quality(city: str) -> str:
    """
    Get the current air quality index and pollutant levels for any city or place.

    Use this tool when the user asks about air quality, pollution, AQI,
    PM2.5, smog, or whether the air is safe to breathe in a location.

    Args:
        city: The name of the city or place to check air quality for.
              Examples: "Delhi", "Beijing", "London", "Los Angeles"

    Returns:
        A human-readable air quality report with PM2.5, PM10, ozone,
        and nitrogen dioxide levels.
    """
    if not city or not city.strip():
        return "Please provide a city name to get air quality for."

    city = city.strip()

    try:
        geo = await _geocode(city)
        if "error" in geo:
            return geo["error"]

        data   = await _fetch_aqi(geo["lat"], geo["lon"])
        hourly = data.get("hourly")

        if not hourly or "time" not in hourly:
            return f"Air quality data is currently unavailable for '{city}'."

        times = hourly["time"]

        def val(key, i):
            series = hourly.get(key)
            return series[i] if series and len(series) > i else None

        # Find the most recent index that actually has data
        latest_idx = None
        for i in range(len(times) - 1, -1, -1):
            if any(val(k, i) is not None for k in ["pm2_5", "pm10", "ozone"]):
                latest_idx = i
                break

        if latest_idx is None:
            return f"No air quality readings available for '{city}' right now."

        pm25 = val("pm2_5",            latest_idx)
        pm10 = val("pm10",             latest_idx)
        o3   = val("ozone",            latest_idx)
        no2  = val("nitrogen_dioxide", latest_idx)
        co   = val("carbon_monoxide",  latest_idx)

        lines = [f"Air Quality in {geo['display_name']} (as of {times[latest_idx]}):"]
        if pm25 is not None: lines.append(f"  PM2.5           : {pm25:.1f} µg/m³")
        if pm10 is not None: lines.append(f"  PM10            : {pm10:.1f} µg/m³")
        if o3   is not None: lines.append(f"  Ozone (O₃)      : {o3:.1f} µg/m³")
        if no2  is not None: lines.append(f"  Nitrogen Dioxide: {no2:.1f} µg/m³")
        if co   is not None: lines.append(f"  Carbon Monoxide : {co:.1f} µg/m³")

        # Simple AQI hint based on PM2.5
        if pm25 is not None:
            if pm25 <= 12:
                hint = "Good 🟢"
            elif pm25 <= 35.4:
                hint = "Moderate 🟡"
            elif pm25 <= 55.4:
                hint = "Unhealthy for sensitive groups 🟠"
            elif pm25 <= 150.4:
                hint = "Unhealthy 🔴"
            else:
                hint = "Very Unhealthy / Hazardous 🟣"
            lines.append(f"  Overall AQI     : {hint} (based on PM2.5)")

        return "\n".join(lines)

    except httpx.TimeoutException:
        return f"Request timed out while fetching air quality for '{city}'. Please try again."
    except httpx.HTTPStatusError as e:
        return f"Air quality API error for '{city}': HTTP {e.response.status_code}"
    except Exception as e:
        return (
            f"Failed to get air quality for '{city}': {type(e).__name__}: {repr(e)}\n"
            f"Trace: {traceback.format_exc(limit=1).strip()}"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")