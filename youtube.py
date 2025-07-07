from googleapiclient.discovery import build
import dotenv as env
import os
import requests
from typing import List, Dict, Any

# Load environment variables
env.load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_youtube_videos(api_key, query, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": API_KEY
    }

    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json()
        videos = []
        for item in results.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append((title, video_url))
        return videos
    else:
        print("Error:", response.status_code, response.text)
        return []

keyword = input("Enter a keyword to search for YouTube videos: ")
results = search_youtube_videos(API_KEY, keyword)

for i, (title, url) in enumerate(results, 1):
    print(f"{i}. {title}\n   {url}")
