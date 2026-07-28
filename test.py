import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://api.freenewsapi.io/v1/news",
    headers={
        "x-api-key": os.getenv("FREENEWS_API_KEY").strip()
    },
    timeout=(10, 60)
)

print(response.status_code)
print(response.text[:1000])