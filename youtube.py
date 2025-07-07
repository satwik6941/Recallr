from googleapiclient.discovery import build
import dotenv as env
import os
import requests
from llama_index.llms.groq import Groq
from typing import List, Dict, Any
import asyncio

# Load environment variables
env.load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

# Validate API keys only when running as main script
def validate_api_keys():
    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable is required")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable is required")

def get_user_query():
    """Get user input for YouTube search"""
    return input("Enter a keyword to search for YouTube videos: ")

async def generate_search_keywords(user_query: str) -> List[str]:
    """Generate relevant keywords for YouTube search using Groq LLM"""
    try:
        groq_llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=360.0,
            temperature=0.1,
            max_tokens=1000,
        )
        
        prompt = f'''Based on this user query: "{user_query}"
        
Generate 5 relevant keywords for searching YouTube videos. Each keyword should be on a separate line.
The keywords can be single words or phrases that would help find the most relevant videos.

Format your response as:
1. keyword1
2. keyword2
3. keyword3
4. keyword4
5. keyword5

Only provide the keywords, no additional text.'''
        
        response = await groq_llm.acomplete(prompt)
        
        # Parse the response to extract keywords
        keywords = []
        lines = str(response).strip().split('\n')
        for line in lines:
            # Remove numbering and clean up
            cleaned = line.strip()
            if cleaned and any(char.isalpha() for char in cleaned):
                # Remove leading numbers and dots
                cleaned = cleaned.lstrip('0123456789. ')
                if cleaned:
                    keywords.append(cleaned)
        
        # If parsing fails, return the original query
        if not keywords:
            keywords = [user_query]
        
        return keywords[:5]  # Limit to 5 keywords
        
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return [user_query]  # Fallback to original query

def search_youtube_videos(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search YouTube videos using the YouTube API"""
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=max_results,
            order='relevance'
        ).execute()
        
        videos = []
        for item in search_response.get('items', []):
            video_info = {
                'id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                'channel': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            videos.append(video_info)
        
        return videos
        
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return []

def search_youtube_videos_requests(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Alternative method using requests (fallback)"""
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": "relevance",
            "key": API_KEY
        }

        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json()
            videos = []
            for item in results.get("items", []):
                video_info = {
                    'id': item["id"]["videoId"],
                    'title': item["snippet"]["title"],
                    'description': item["snippet"]["description"][:200] + '...' if len(item["snippet"]["description"]) > 200 else item["snippet"]["description"],
                    'channel': item["snippet"]["channelTitle"],
                    'published_at': item["snippet"]["publishedAt"],
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                videos.append(video_info)
            return videos
        else:
            print("Error:", response.status_code, response.text)
            return []
    except Exception as e:
        print(f"Error with requests method: {e}")
        return []

def display_videos(videos: List[Dict[str, Any]], query: str):
    """Display search results in a formatted way"""
    if not videos:
        print(f"No videos found for query: '{query}'")
        return
    
    print(f"\n{'='*60}")
    print(f"Search Results for: '{query}'")
    print(f"{'='*60}")
    
    for i, video in enumerate(videos, 1):
        print(f"\n{i}. {video['title']}")
        print(f"   Channel: {video['channel']}")
        print(f"   Published: {video['published_at'][:10]}")  # Show only date
        print(f"   Description: {video['description']}")
        print(f"   URL: {video['url']}")
        print("-" * 60)

async def search_youtube_for_query(user_query: str, max_results_per_keyword: int = 3) -> List[Dict[str, Any]]:
    """Main function to search YouTube videos for a given query. Can be called from other files.
    
    Args:
        user_query (str): The search query from the user
        max_results_per_keyword (int): Maximum results per keyword
        
    Returns:
        List[Dict[str, Any]]: List of unique video information dictionaries
    """
    try:
        print(f"🎥 YouTube Search: Processing query '{user_query}'...")
        
        # Generate keywords using Groq
        keywords = await generate_search_keywords(user_query)
        print(f"🔍 Generated keywords: {', '.join(keywords)}")
        
        # Search for videos using each keyword
        all_videos = []
        for keyword in keywords[:3]:  # Limit to 3 keywords for efficiency
            print(f"   Searching for: '{keyword}'...")
            
            # Try the Google API client first
            videos = search_youtube_videos(keyword, max_results=max_results_per_keyword)
            
            # If that fails, try the requests method
            if not videos:
                videos = search_youtube_videos_requests(keyword, max_results=max_results_per_keyword)
            
            if videos:
                all_videos.extend(videos)
                print(f"   Found {len(videos)} videos for '{keyword}'")
            else:
                print(f"   No videos found for: '{keyword}'")
        
        # Remove duplicates based on video ID
        unique_videos = []
        seen_ids = set()
        for video in all_videos:
            if video['id'] not in seen_ids:
                unique_videos.append(video)
                seen_ids.add(video['id'])
        
        print(f"✅ YouTube Search Complete: Found {len(unique_videos)} unique videos")
        return unique_videos
        
    except Exception as e:
        print(f"❌ Error in YouTube search: {e}")
        return []

def format_videos_for_llm(videos: List[Dict[str, Any]], query: str) -> str:
    """Format video search results for LLM consumption
    
    Args:
        videos (List[Dict[str, Any]]): List of video information
        query (str): Original search query
        
    Returns:
        str: Formatted string for LLM processing
    """
    if not videos:
        return f"No YouTube videos found for query: '{query}'"
    
    result = f"YouTube Video Results for '{query}':\n\n"
    
    for i, video in enumerate(videos[:8], 1):  # Limit to 8 videos to avoid token overflow
        result += f"{i}. **{video['title']}**\n"
        result += f"   Channel: {video['channel']}\n"
        result += f"   URL: {video['url']}\n"
        result += f"   Published: {video['published_at'][:10]}\n"
        result += f"   Description: {video['description']}\n\n"
    
    return result

async def process_youtube_query(user_query: str) -> str:
    """Complete YouTube processing function for external calls
    
    Args:
        user_query (str): The user's search query
        
    Returns:
        str: Formatted results ready for LLM consumption
    """
    try:
        # Search for videos
        videos = await search_youtube_for_query(user_query)
        
        # Format for LLM
        formatted_result = format_videos_for_llm(videos, user_query)
        
        return formatted_result
        
    except Exception as e:
        return f"Error processing YouTube query '{user_query}': {str(e)}"

async def main():
    """Main function to run the YouTube search standalone"""
    try:
        # Validate API keys when running as main script
        validate_api_keys()
        
        # Get user query
        user_query = get_user_query()
        
        if not user_query.strip():
            print("Please enter a valid search query.")
            return
        
        print(f"\nStarting YouTube search for: '{user_query}'...")
        
        # Use the new processing function
        videos = await search_youtube_for_query(user_query)
        
        # Display results
        if videos:
            display_videos(videos, user_query)
            print(f"\n{'='*60}")
            print(f"SUMMARY: Found {len(videos)} unique videos")
            print(f"{'='*60}")
        else:
            print("No videos found.")
        
    except KeyboardInterrupt:
        print("\nSearch cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
