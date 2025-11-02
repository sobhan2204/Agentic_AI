import os
import requests
import base64
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("Spotify Music Player")

from dotenv import load_dotenv
load_dotenv()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET") ## to retrive langchian API key
if not SPOTIFY_CLIENT_SECRET:
     raise ValueError(" SPOTIFY_CLIENT_SECRET not found in .env")
os.environ["SPOTIFY_CLIENT_SECRET"] = SPOTIFY_CLIENT_SECRET

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID") ## to retrive langchian API key
if not SPOTIFY_CLIENT_ID:
     raise ValueError(" SPOTIFY_CLIENT_ID not found in .env")
os.environ["SPOTIFY_CLIENT_ID"] = SPOTIFY_CLIENT_ID
# Spotify API Configuration
# SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
# print(SPOTIFY_CLIENT_ID)
# SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
# print(SPOTIFY_CLIENT_SECRET)
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')

class SpotifyAuth:
    def __init__(self):
        self.access_token = None
        self.token_type = "Bearer"
    
    def get_client_credentials_token(self) -> Optional[str]:
        """Get access token using Client Credentials flow (for search only)"""
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise ValueError("Spotify Client ID and Secret must be set in environment variables")
        
        # Encode credentials
        credentials = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
        
        headers = {
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {'grant_type': 'client_credentials'}
        
        response = requests.post('https://accounts.spotify.com/api/token', headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests"""
        if not self.access_token:
            self.get_client_credentials_token()
        
        return {
            'Authorization': f'{self.token_type} {self.access_token}',
            'Content-Type': 'application/json'
        }

# Initialize auth
auth = SpotifyAuth()

@mcp.tool()
def search_tracks(query: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for tracks on Spotify
    
    Args:
        query: Search query (song name, artist, album, etc.)
        limit: Number of results to return (max 50)
    
    Returns:
        Dictionary containing search results with track information
    """
    try:
        headers = auth.get_auth_headers()
        
        params = {
            'q': query,
            'type': 'track',
            'limit': min(limit, 50)
        }
        
        response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            tracks = []
            
            for track in data['tracks']['items']:
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'external_urls': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                }
                tracks.append(track_info)
            
            return {
                'success': True,
                'query': query,
                'total_results': data['tracks']['total'],
                'tracks': tracks
            }
        else:
            return {
                'success': False,
                'error': f"Search failed: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

@mcp.tool()
def search_artists(query: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for artists on Spotify
    
    Args:
        query: Artist name to search for
        limit: Number of results to return (max 50)
    
    Returns:
        Dictionary containing artist search results
    """
    try:
        headers = auth.get_auth_headers()
        
        params = {
            'q': query,
            'type': 'artist',
            'limit': min(limit, 50)
        }
        
        response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            artists = []
            
            for artist in data['artists']['items']:
                artist_info = {
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres'],
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total'],
                    'external_urls': artist['external_urls']['spotify'],
                    'images': artist['images']
                }
                artists.append(artist_info)
            
            return {
                'success': True,
                'query': query,
                'total_results': data['artists']['total'],
                'artists': artists
            }
        else:
            return {
                'success': False,
                'error': f"Search failed: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

@mcp.tool()
def get_track_details(track_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific track
    
    Args:
        track_id: Spotify track ID
    
    Returns:
        Dictionary containing detailed track information
    """
    try:
        headers = auth.get_auth_headers()
        
        response = requests.get(f'https://api.spotify.com/v1/tracks/{track_id}', headers=headers)
        
        if response.status_code == 200:
            track = response.json()
            
            track_details = {
                #'id': track['id'],
                'name': track['name'],
                'artists': [{'name': artist['name'], 'id': artist['id']} for artist in track['artists']],
                'album': {
                    'name': track['album']['name'],
                    'id': track['album']['id'],
                    'release_date': track['album']['release_date'],
                    'images': track['album']['images']
                },
                'duration_ms': track['duration_ms'],
                #'explicit': track['explicit'],
                'popularity': track['popularity'],
                #'preview_url': track['preview_url'],
                'external_urls': track['external_urls']['spotify'],
                'audio_features': None  # Will be filled if we fetch audio features
            }
            
            return {
                'success': True,
                'track': track_details
            }
        else:
            return {
                'success': False,
                'error': f"Failed to get track details: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

@mcp.tool()
def get_artist_top_tracks(artist_id: str, country: str = 'US') -> Dict[str, Any]:
    """
    Get an artist's top tracks
    
    Args:
        artist_id: Spotify artist ID
        country: Country code for market (default: US)
    
    Returns:
        Dictionary containing artist's top tracks
    """
    try:
        headers = auth.get_auth_headers()
        
        params = {'market': country}
        response = requests.get(f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks', 
                              headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            tracks = []
            
            for track in data['tracks']:
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'preview_url': track['preview_url'],
                    'external_urls': track['external_urls']['spotify']
                }
                tracks.append(track_info)
            
            return {
                'success': True,
                'artist_id': artist_id,
                'top_tracks': tracks
            }
        else:
            return {
                'success': False,
                'error': f"Failed to get top tracks: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }
@mcp.tool()
def create_playlist_url(track_ids: List[str], playlist_name: str = "My Playlist") -> Dict[str, Any]:
    """
    Create a Spotify playlist URL with specified tracks
    Note: This creates a URL that users can use to create a playlist, 
    as creating playlists requires user authentication
    
    Args:
        track_ids: List of Spotify track IDs
        playlist_name: Name for the playlist
    
    Returns:
        Dictionary containing playlist creation URL and track information
    """
    try:
        if not track_ids:
            return {
                'success': False,
                'error': 'No track IDs provided'
            }
        
        # Create Spotify URIs from track IDs
        track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
        
        # Create a URL that opens Spotify with these tracks
        # Users can then save as a playlist
        spotify_urls = [f"https://open.spotify.com/track/{track_id}" for track_id in track_ids]
        
        return {
            'success': True,
            'playlist_name': playlist_name,
            'track_count': len(track_ids),
            'track_uris': track_uris,
            'track_urls': spotify_urls,
            'message': 'Open these URLs in Spotify to play the tracks. You can create a playlist by adding them to your library.'
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

@mcp.tool()
def get_audio_features(track_id: str) -> Dict[str, Any]:
    """
    Get audio features for a track (danceability, energy, etc.)
    
    Args:
        track_id: Spotify track ID
    
    Returns:
        Dictionary containing audio features
    """
    try:
        headers = auth.get_auth_headers()
        
        response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
        
        if response.status_code == 200:
            features = response.json()
            
            return {
                'success': True,
                'track_id': track_id,
                'audio_features': {
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'valence': features['valence'],  # positiveness
                    'tempo': features['tempo'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'speechiness': features['speechiness'],
                    'liveness': features['liveness'],
                    'loudness': features['loudness'],
                    'key': features['key'],
                    'mode': features['mode'],
                    'time_signature': features['time_signature']
                }
            }
        else:
            return {
                'success': False,
                'error': f"Failed to get audio features: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

@mcp.tool()
def search_and_play(query: str, play_first: bool = True) -> Dict[str, Any]:
    """
    Search for tracks and return playable information
    
    Args:
        query: Search query for tracks
        play_first: Whether to return the first result for immediate playing
    
    Returns:
        Dictionary containing search results and play information
    """
    try:
        # Search for tracks
        search_results = search_tracks(query, limit=1 if play_first else 10)
        
        if not search_results['success'] or not search_results['tracks']:
            return {
                'success': False,
                'error': 'No tracks found for the given query'
            }
        
        if play_first:
            track = search_results['tracks'][0]
            return {
                'success': True,
                'action': 'play_track',
                'track': track,
                'spotify_url': track['external_urls'],
                'preview_url': track['preview_url'],
                'message': f"Playing: {track['name']} by {', '.join(track['artists'])}"
            }
        else:
            return {
                'success': True,
                'action': 'search_results',
                'results': search_results,
                'message': f"Found {len(search_results['tracks'])} tracks matching '{query}'"
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }

# FIXED TESTING CODE - Now uses real track IDs from search results
print("=== SPOTIFY MCP TOOLS  ===\n")

# Test 1: Search for tracks
print("1. Searching for 'Bohemian Rhapsody Queen'...")
result = search_tracks("Bohemian Rhapsody Queen", limit=5)
print(f"Search Success: {result['success']}")
if result['success']:
    print(f"Found {len(result['tracks'])} tracks:")
    for i, track in enumerate(result['tracks']):
        print(f"   {i+1}. {track['name']} by {', '.join(track['artists'])} (ID: {track['id']})")
print()

# Test 2: Search and play
print("2. Search and play 'Shape of You'...")
result2 = search_and_play("Shape of You", play_first=True)
print(f"Search and Play Success: {result2['success']}")
if result2['success']:
    print(f"Message: {result2['message']}")
    print(f"Spotify URL: {result2['spotify_url']}")
print()

# Test 3: Get audio features using REAL track ID from search results
print("3. Getting audio features...")
if result['success'] and result['tracks']:
    # Use the first track ID from our search results
    first_track = result['tracks'][0]
    real_track_id = first_track['id']
    
    print(f"Getting audio features for: {first_track['name']} (ID: {real_track_id})")
    result3 = get_audio_features(real_track_id)
    
    print(f"Audio Features Success: {result3['success']}")
    if result3['success']:
        features = result3['audio_features']
        print("Audio Features:")
        print(f"   - Danceability: {features['danceability']:.3f}")
        print(f"   - Energy: {features['energy']:.3f}")
        print(f"   - Valence (positiveness): {features['valence']:.3f}")
        print(f"   - Tempo: {features['tempo']:.1f} BPM")
        print(f"   - Acousticness: {features['acousticness']:.3f}")
        print(f"   - Instrumentalness: {features['instrumentalness']:.3f}")
    else:
        print(f"Error: {result3['error']}")
else:
    print("Cannot get audio features - no tracks found in search")
print()

# Test 4: Get track details using real track ID
print("4. Getting track details...")
if result['success'] and result['tracks']:
    track_details_result = get_track_details(real_track_id)
    print(f"Track Details Success: {track_details_result['success']}")
    if track_details_result['success']:
        track_info = track_details_result['track']
        print(f"Track Details for: {track_info['name']}")
        print(f"   - Album: {track_info['album']['name']}")
        print(f"   - Release Date: {track_info['album']['release_date']}")
        print(f"   - Popularity: {track_info['popularity']}")
        print(f"   - Duration: {track_info['duration_ms']/1000:.0f} seconds")

#print("\n=== ALL TESTS COMPLETED ===")

# Print the results in your original format
#print("\n=== ORIGINAL FORMAT RESULTS ===")
#print("result (search_tracks):", result, '\n')
#print("result2 (search_and_play):", result2, '\n')
if 'result3' in locals():
    print("result3 (get_audio_features):", result3, '\n')
else:
    print("result3: Could not get audio features - no search results available\n")
if __name__ == "__main__":
    mcp.run(transport="stdio")  # useful if we want to run the server in the terminal locally in this
    #we will get the input and output in the teminal itself