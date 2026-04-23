import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("spotify")

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_RECOMMENDATIONS_URL = "https://api.spotify.com/v1/recommendations"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Use conservative, broadly valid Spotify seed genres.
DEFAULT_MOOD_PROFILES: Dict[str, Dict[str, Any]] = {
    "happy": {
        "valence": 0.82,
        "energy": 0.72,
        "tempo": 120,
        "instrumentalness": 0.10,
        "seed_genres": ["pop", "dance", "electronic"],
    },
    "sad": {
        "valence": 0.20,
        "energy": 0.32,
        "tempo": 84,
        "instrumentalness": 0.35,
        "seed_genres": ["acoustic", "indie", "jazz"],
    },
    "focus": {
        "valence": 0.45,
        "energy": 0.45,
        "tempo": 95,
        "instrumentalness": 0.85,
        "seed_genres": ["ambient", "classical", "chill"],
    },
    "calm": {
        "valence": 0.48,
        "energy": 0.35,
        "tempo": 88,
        "instrumentalness": 0.70,
        "seed_genres": ["ambient", "acoustic", "jazz"],
    },
    "energetic": {
        "valence": 0.66,
        "energy": 0.88,
        "tempo": 132,
        "instrumentalness": 0.12,
        "seed_genres": ["edm", "electronic", "rock"],
    },
    "romantic": {
        "valence": 0.62,
        "energy": 0.46,
        "tempo": 96,
        "instrumentalness": 0.20,
        "seed_genres": ["r-n-b", "soul", "pop"],
    },
    "default": {
        "valence": 0.55,
        "energy": 0.55,
        "tempo": 110,
        "instrumentalness": 0.25,
        "seed_genres": ["pop", "indie", "rock"],
    },
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class SpotifyAuthManager:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    def _fetch_access_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment"
            )

        response = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=20,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Spotify token request failed: HTTP {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        token = payload.get("access_token", "")
        expires_in = int(payload.get("expires_in", 3600))
        if not token:
            raise RuntimeError("Spotify token response did not include access_token")

        # Refresh a bit early to avoid boundary expiry.
        self._access_token = token
        self._expires_at = time.time() + max(120, expires_in - 60)
        return token

    def get_access_token(self, force_refresh: bool = False) -> str:
        if (
            force_refresh
            or not self._access_token
            or time.time() >= self._expires_at
        ):
            return self._fetch_access_token()
        return self._access_token

    def authorized_get(self, url: str, params: Dict[str, Any]) -> requests.Response:
        # Retry once with forced refresh on 401.
        for attempt in range(2):
            token = self.get_access_token(force_refresh=(attempt == 1))
            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=25,
            )
            if response.status_code == 401 and attempt == 0:
                continue
            return response

        return response


def _infer_mood_heuristic(text: str) -> str:
    query = (text or "").lower()

    mapping = {
        "focus": ["focus", "study", "work", "concentrate", "deep work"],
        "sad": ["sad", "down", "heartbroken", "depressed", "cry"],
        "happy": ["happy", "joy", "cheerful", "uplift", "good mood"],
        "calm": ["calm", "relax", "chill", "sleep", "peaceful", "lofi"],
        "energetic": ["gym", "workout", "run", "hype", "party", "energy"],
        "romantic": ["love", "romantic", "date", "valentine"],
    }

    for mood, words in mapping.items():
        if any(word in query for word in words):
            return mood

    return "default"


def _infer_mood_with_llm(text: str) -> Optional[Dict[str, Any]]:
    groq_api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None

    system_prompt = (
        "You are a music mood classifier. "
        "Given a user request, infer the mood and return ONLY valid JSON with keys: "
        "mood, valence, energy, tempo, instrumentalness, seed_genres. "
        "valence/energy/instrumentalness must be floats between 0 and 1. "
        "tempo must be BPM between 60 and 180. "
        "seed_genres must be an array of 1-3 Spotify-friendly genre strings."
    )

    body = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "max_tokens": 250,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    }

    response = requests.post(
        GROQ_CHAT_URL,
        headers={
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=20,
    )

    if response.status_code != 200:
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def _normalize_profile(llm_profile: Optional[Dict[str, Any]], mood_text: str) -> Dict[str, Any]:
    inferred_mood = _infer_mood_heuristic(mood_text)
    profile = dict(DEFAULT_MOOD_PROFILES.get(inferred_mood, DEFAULT_MOOD_PROFILES["default"]))

    if llm_profile:
        llm_mood = str(llm_profile.get("mood", "")).strip().lower()
        if llm_mood in DEFAULT_MOOD_PROFILES:
            profile = dict(DEFAULT_MOOD_PROFILES[llm_mood])
            inferred_mood = llm_mood

        profile["valence"] = _clamp(_safe_float(llm_profile.get("valence"), profile["valence"]), 0.0, 1.0)
        profile["energy"] = _clamp(_safe_float(llm_profile.get("energy"), profile["energy"]), 0.0, 1.0)
        profile["instrumentalness"] = _clamp(
            _safe_float(llm_profile.get("instrumentalness"), profile["instrumentalness"]), 0.0, 1.0
        )
        profile["tempo"] = _clamp(_safe_float(llm_profile.get("tempo"), profile["tempo"]), 60.0, 180.0)

        seed_genres = llm_profile.get("seed_genres")
        if isinstance(seed_genres, list) and seed_genres:
            profile["seed_genres"] = [str(g).strip().lower() for g in seed_genres if str(g).strip()][:3]

    profile["mood"] = inferred_mood
    return profile


def _apply_time_of_day_bias(profile: Dict[str, Any], time_of_day: Optional[str]) -> Dict[str, Any]:
    adjusted = dict(profile)

    tod = (time_of_day or "").strip().lower()
    if not tod:
        hour = datetime.now().hour
        if 5 <= hour <= 11:
            tod = "morning"
        elif 12 <= hour <= 17:
            tod = "afternoon"
        elif 18 <= hour <= 22:
            tod = "evening"
        else:
            tod = "night"

    if tod == "morning":
        delta = 0.08
    elif tod == "afternoon":
        delta = 0.06
    elif tod == "evening":
        delta = -0.02
    else:
        delta = -0.12

    adjusted["energy"] = _clamp(_safe_float(adjusted.get("energy"), 0.55) + delta, 0.05, 0.95)
    adjusted["time_of_day"] = tod
    return adjusted


def _parse_json_maybe(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def _extract_previous_artists(payload: Dict[str, Any]) -> List[str]:
    artists: List[str] = []

    direct = payload.get("previous_artists")
    if isinstance(direct, list):
        artists.extend([str(a).strip() for a in direct if str(a).strip()])

    tracks = payload.get("previous_tracks")
    if isinstance(tracks, list):
        for item in tracks:
            if isinstance(item, dict):
                artist = str(item.get("artist", "")).strip()
                if artist:
                    artists.append(artist)

    # Keep order and deduplicate.
    deduped: List[str] = []
    seen = set()
    for artist in artists:
        key = artist.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(artist)

    return deduped[:3]


def _format_tracks(raw_tracks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []

    for track in raw_tracks[:5]:
        artists = track.get("artists") or []
        artist_name = ", ".join(a.get("name", "") for a in artists if isinstance(a, dict)).strip()

        formatted.append(
            {
                "name": str(track.get("name", "")).strip(),
                "artist": artist_name,
                "url": str((track.get("external_urls") or {}).get("spotify", "")).strip(),
                "preview": str(track.get("preview_url") or "").strip(),
            }
        )

    return formatted


def _resolve_artist_seed_ids(auth: SpotifyAuthManager, artists: List[str], market: str) -> List[str]:
    ids: List[str] = []

    for artist in artists[:2]:
        response = auth.authorized_get(
            SPOTIFY_SEARCH_URL,
            {
                "q": artist,
                "type": "artist",
                "limit": 1,
                "market": market,
            },
        )
        if response.status_code != 200:
            continue

        data = response.json()
        items = ((data.get("artists") or {}).get("items") or [])
        if not items:
            continue

        artist_id = items[0].get("id")
        if artist_id:
            ids.append(str(artist_id))

    return ids


def _recommendations_request(
    auth: SpotifyAuthManager,
    profile: Dict[str, Any],
    previous_artists: List[str],
    market: str,
) -> Dict[str, Any]:
    seed_genres = list(profile.get("seed_genres") or ["pop"])
    seed_artist_ids = _resolve_artist_seed_ids(auth, previous_artists, market) if previous_artists else []

    # Spotify supports up to 5 total seeds across genres/artists/tracks.
    remaining_slots = max(0, 5 - len(seed_artist_ids))
    seed_genres = seed_genres[:remaining_slots] if remaining_slots else []

    params: Dict[str, Any] = {
        "limit": 5,
        "market": market,
        "target_valence": round(_safe_float(profile.get("valence"), 0.55), 3),
        "target_energy": round(_safe_float(profile.get("energy"), 0.55), 3),
        "target_tempo": round(_safe_float(profile.get("tempo"), 110), 1),
        "target_instrumentalness": round(_safe_float(profile.get("instrumentalness"), 0.25), 3),
    }

    if seed_genres:
        params["seed_genres"] = ",".join(seed_genres)
    if seed_artist_ids:
        params["seed_artists"] = ",".join(seed_artist_ids)

    if "seed_genres" not in params and "seed_artists" not in params:
        params["seed_genres"] = "pop"

    response = auth.authorized_get(SPOTIFY_RECOMMENDATIONS_URL, params)
    if response.status_code != 200:
        return {
            "tracks": [],
            "error": f"Spotify recommendations failed: HTTP {response.status_code}: {response.text[:200]}",
        }

    payload = response.json()
    tracks = _format_tracks(payload.get("tracks") or [])
    return {"tracks": tracks}


def _search_fallback(auth: SpotifyAuthManager, mood: str, previous_artists: List[str], market: str) -> Dict[str, Any]:
    query_parts = [mood.strip() or "music"]
    if previous_artists:
        query_parts.append(previous_artists[0])

    response = auth.authorized_get(
        SPOTIFY_SEARCH_URL,
        {
            "q": " ".join(query_parts),
            "type": "track",
            "limit": 5,
            "market": market,
        },
    )

    if response.status_code != 200:
        return {
            "tracks": [],
            "error": f"Spotify search fallback failed: HTTP {response.status_code}: {response.text[:200]}",
        }

    payload = response.json()
    items = ((payload.get("tracks") or {}).get("items") or [])
    return {"tracks": _format_tracks(items)}


def get_recommendations_by_mood(mood: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = context or {}
    market = str(context.get("market", "US")).strip().upper() or "US"
    previous_artists = _extract_previous_artists(context)

    llm_profile = _infer_mood_with_llm(mood)
    profile = _normalize_profile(llm_profile, mood)
    profile = _apply_time_of_day_bias(profile, context.get("time_of_day"))

    try:
        auth = SpotifyAuthManager(
            client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
        )

        rec_result = _recommendations_request(auth, profile, previous_artists, market)
        tracks = rec_result.get("tracks", [])

        if tracks:
            return {
                "tracks": tracks,
                "mood": profile.get("mood", "default"),
                "features": {
                    "valence": profile.get("valence"),
                    "energy": profile.get("energy"),
                    "tempo": profile.get("tempo"),
                    "instrumentalness": profile.get("instrumentalness"),
                    "time_of_day": profile.get("time_of_day"),
                },
                "source": "spotify_recommendations",
            }

        fallback = _search_fallback(auth, mood, previous_artists, market)
        return {
            "tracks": fallback.get("tracks", []),
            "mood": profile.get("mood", "default"),
            "features": {
                "valence": profile.get("valence"),
                "energy": profile.get("energy"),
                "tempo": profile.get("tempo"),
                "instrumentalness": profile.get("instrumentalness"),
                "time_of_day": profile.get("time_of_day"),
            },
            "source": "spotify_search_fallback",
            "warning": rec_result.get("error") or fallback.get("error") or "No recommendations returned; used fallback.",
        }

    except Exception as exc:
        return {
            "tracks": [],
            "error": f"Spotify mood recommender failed: {type(exc).__name__}: {str(exc)[:300]}",
        }


def _merge_payload(mood: str, context_json: str) -> Dict[str, Any]:
    payload = _parse_json_maybe(mood)
    if not payload:
        payload = {"mood": mood}

    context = _parse_json_maybe(context_json)
    if context:
        payload.update(context)

    payload.setdefault("mood", mood)
    payload["mood"] = str(payload.get("mood", "")).strip()
    if not payload["mood"]:
        payload["mood"] = "music recommendations"

    return payload


@mcp.tool(name="spotify_mood_recommend")
def spotify_mood_recommend(mood: str, context_json: str = "") -> str:
    """
    Recommend tracks based on user mood.

    Inputs:
    - mood: either plain natural language ("I feel sad") or a JSON string
            like {"mood":"focus music","time_of_day":"night"}
    - context_json: optional JSON string for extra context
            like {"previous_artists":["Hans Zimmer"]}

    Returns:
    JSON string with shape:
    {
      "tracks": [
        {"name":"...", "artist":"...", "url":"...", "preview":"..."}
      ]
    }
    """
    payload = _merge_payload(mood, context_json)
    result = get_recommendations_by_mood(payload.get("mood", ""), context=payload)

    # Guarantee required output key.
    if "tracks" not in result:
        result["tracks"] = []

    return json.dumps(result, ensure_ascii=True)


if __name__ == "__main__":
    mcp.run(transport="stdio")
