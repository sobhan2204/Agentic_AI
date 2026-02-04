import asyncio
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
NOMINATIM_API = "https://nominatim.openstreetmap.org/search"

@mcp.tool()
async def get_current_weather(lat: float, lon: float) -> dict:
    """Get current weather using latitude and longitude."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                OPEN_METEO_FORECAST_API,
                params=params,
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"Weather fetch failed: {str(e)}"}

    current = data.get("current_weather")
    if not current:
        return {"error": "Weather data unavailable"}

    return {
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "wind_direction": current.get("winddirection"),
        "weather_code": current.get("weathercode"),
        "time": current.get("time"),
    }

@mcp.tool()
async def get_air_quality(lat: float, lon: float) -> dict:
    """Get latest air quality data for a location."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "pm2_5,pm10,carbon_monoxide,"
            "nitrogen_dioxide,sulphur_dioxide,ozone"
        ),
        "timezone": "auto",
        "models": "cams_global",
        "past_days": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                OPEN_METEO_AQI_API,
                params=params,
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"AQI fetch failed: {str(e)}"}

    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        return {"error": "AQI data unavailable"}

    times = hourly["time"]

    def val(key: str, i: int):
        series = hourly.get(key)
        return series[i] if series and len(series) > i else None

    latest_idx = None
    for i in range(len(times) - 1, -1, -1):
        if any(
            val(k, i) is not None
            for k in [
                "pm2_5", "pm10", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide", "ozone"
            ]
        ):
            latest_idx = i
            break

    if latest_idx is None:
        return {"error": "AQI data unavailable"}

    return {
        "pm2_5": val("pm2_5", latest_idx),
        "pm10": val("pm10", latest_idx),
        "co": val("carbon_monoxide", latest_idx),
        "no2": val("nitrogen_dioxide", latest_idx),
        "so2": val("sulphur_dioxide", latest_idx),
        "o3": val("ozone", latest_idx),
        "timestamp": times[latest_idx],
    }


@mcp.tool()
async def get_geo_details(place: str) -> dict:
    """Convert a place name into geo coordinates."""
    params = {
        "q": place,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "geo-weather-mcp/1.0"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                NOMINATIM_API,
                params=params,
                headers=headers,
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"Geocoding failed: {str(e)}"}

    if not data:
        return {"error": "Location not found"}

    loc = data[0]
    return {
        "place": place,
        "latitude": float(loc["lat"]),
        "longitude": float(loc["lon"]),
        "display_name": loc.get("display_name"),
        "country": loc.get("display_name", "").split(",")[-1].strip(),
    }

@mcp.tool()
async def get_environment_report(place: str) -> dict:
    """Get location, weather, and air quality for a place."""
    geo = await get_geo_details(place)
    if "error" in geo:
        return geo

    weather = await get_current_weather(
        geo["latitude"], geo["longitude"]
    )

    air_quality = await get_air_quality(
        geo["latitude"], geo["longitude"]
    )

    return {
        "location": geo,
        "weather": weather,
        "air_quality": air_quality,
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
