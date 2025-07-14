import asyncio
import requests
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini

# Import the RAG pipeline from hybrid.py
from hybrid import search_documents_with_context

load_dotenv()

# Validate API keys
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is required")
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# YouTube API key is optional - will gracefully handle if missing
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    print("Warning: YOUTUBE_API_KEY not found. YouTube search functionality will be limited.")

# Initialize LLM for processing with better settings
groq_llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=360.0,
    temperature=0.1,  # Lower temperature for more consistent output
    max_tokens=1000,  # Limit response length
)

# Conversation context to maintain chat history
conversation_history = []

# Google Custom Search function
def google_search(query: str) -> str:
    """Search the web using Google Custom Search API
    
    Args:
        query (str): The search query
        
    Returns:
        str: Search results formatted as text
    """
    try:
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            'key': os.getenv("GOOGLE_API_KEY"),
            'cx': os.getenv("GOOGLE_CSE_ID"),
            'q': query,
            'num': 5  # Number of results to return
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'items' not in data:
            return f"No search results found for: {query}"
        
        # Format results
        results = []
        for item in data['items']:
            title = item.get('title', 'No title')
            link = item.get('link', 'No link')
            snippet = item.get('snippet', 'No description')
            
            results.append(f"Title: {title}\nURL: {link}\nDescription: {snippet}\n")
        
        return f"Web search results for '{query}':\n\n" + "\n".join(results)
        
    except requests.exceptions.RequestException as e:
        return f"Error performing web search: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# YouTube integration - using external youtube.py module
async def youtube_search_tool_function(query: str) -> str:
    """Search YouTube for educational videos using external youtube.py module"""
    try:
        # Import the youtube module functions
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from youtube import process_youtube_query
        
        # Process the query using the external module
        result = await process_youtube_query(query)
        
        return result
        
    except ImportError as e:
        return f"Error importing YouTube module: {str(e)}"
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

async def get_web_search_results(query: str) -> str:
    """Get web search results using direct Google search (bypassing agent for reliability)
    
    Args:
        query (str): The search query
        
    Returns:
        str: Processed web search results
    """
    try:
        # Get raw web search results directly
        web_results = google_search(query)
        
        # Use Groq LLM directly to process the results
        process_prompt = f"""Based on these web search results, provide a comprehensive answer to the query '{query}':

{web_results}

Please provide a clear, concise answer based on the search results."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        return str(groq_response)
    except Exception as e:
        return f"Error getting web search results: {str(e)}"

async def get_youtube_search_results(query: str) -> str:
    """Get YouTube search results using external youtube.py module
    
    Args:
        query (str): The search query
        
    Returns:
        str: Processed YouTube search results
    """
    try:
        print(f"Calling external YouTube module...")
        
        # Import and use the external youtube module
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from youtube import process_youtube_query
        
        # Get YouTube search results from external module
        youtube_results = await process_youtube_query(query)
        
        # Check if we got results
        if "No YouTube videos found" in youtube_results or "Error" in youtube_results:
            return youtube_results
        
        # Use Groq LLM to process and summarize the results while preserving URLs
        process_prompt = f"""Based on these YouTube search results, provide a summary of the available educational videos for the query '{query}':

{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning
4. Keep all the URLs exactly as provided in the original results

Please provide a clear summary highlighting the most relevant videos for learning."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        # Combine original results with AI summary, ensuring URLs are preserved
        final_result = f"{youtube_results}\n\n**AI Summary:**\n{str(groq_response)}\n\n**Direct Video Links:**\n"
        
        # Extract and add direct links from original results to ensure they're visible
        lines = youtube_results.split('\n')
        for line in lines:
            if 'URL:' in line or 'https://www.youtube.com/watch' in line:
                final_result += f"{line}\n"
        
        return final_result
        
    except Exception as e:
        return f"Error getting YouTube search results: {str(e)}"

async def synthesize_final_answer(query: str, rag_result: str, web_result: str, youtube_result: str = None) -> str:
    """Synthesize final answer from RAG, web search, and YouTube results"""
    try:
        if youtube_result:
            # Check if user is specifically asking for videos
            video_keywords = ['video', 'watch', 'tutorial', 'explain', 'show', 'demonstration', 'visual', 'youtube']
            should_recommend_videos = any(keyword in query.lower() for keyword in video_keywords)
            
            # Build conversation context for LLM
            conversation_context = ""
            if conversation_history:
                conversation_context = "\n💬 **Previous Conversation:**\n"
                for i, exchange in enumerate(conversation_history[-5:], 1):  # Last 5 exchanges for context
                    conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
                    conversation_context += f"   I responded: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}\n\n"
            
            synthesis_prompt = f"""
You are a friendly and knowledgeable tutor helping a student understand concepts. Think of yourself as talking to a real person who needs clear, helpful explanations.

{conversation_context}

📝 **Current Question:** "{query}"

I have information from three sources about this question:

📚 **From Course Materials:**
{rag_result}

🌐 **From Web Search:**
{web_result}

🎥 **Available YouTube Videos:**
{youtube_result}

Please respond naturally and conversationally, keeping in mind our previous conversation:

1. Answer the current question directly and clearly
2. Reference previous topics we discussed if relevant
3. Use simple, everyday language that's easy to understand
4. Combine information from the sources when helpful
5. Be concise and focused on what the student actually asked
6. Only recommend YouTube videos if the student specifically asked for videos, tutorials, visual explanations, or similar requests
7. If recommending videos, include the exact URLs and format as: "Video Title" - Watch here: [URL]

Remember:
- Be warm and encouraging in your tone
- Build on our previous conversation naturally
- Explain things step by step if needed
- Don't overwhelm with unnecessary information
- Make it feel like a continuing conversation with a friend

Please provide a helpful, human-like response:
"""
        else:
            # Build conversation context for LLM
            conversation_context = ""
            if conversation_history:
                conversation_context = "\n💬 **Previous Conversation:**\n"
                for i, exchange in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges for context
                    conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
                    conversation_context += f"   I responded: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}\n\n"
            
            synthesis_prompt = f"""
You are a friendly and knowledgeable tutor helping a student understand concepts. Think of yourself as talking to a real person who needs clear, helpful explanations.

{conversation_context}

📝 **Current Question:** "{query}"

I have information from two sources about this question:

📚 **From Course Materials:**
{rag_result}

🌐 **From Web Search:**
{web_result}

Please respond naturally and conversationally, keeping in mind our previous conversation:

1. Answer the current question directly and clearly
2. Reference previous topics we discussed if relevant
3. Use simple, everyday language that's easy to understand
4. Combine information from both sources when helpful
5. Be concise and focused on what the student actually asked
6. If the student is just checking their understanding or asking a doubt, simply confirm or clarify

Remember:
- Be warm and encouraging in your tone
- Build on our previous conversation naturally
- Explain things step by step if needed
- Don't overwhelm with unnecessary information
- Make it feel like a continuing conversation with a friend
- Only mention videos if the student specifically asked for visual explanations

Please provide a helpful, human-like response:
"""
        
        # Use Gemini for final synthesis
        gemini_llm = Gemini(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        final_response = await gemini_llm.acomplete(synthesis_prompt)
        
        # Update conversation history with final answer
        conversation_history.append({
            "user": query,
            "assistant": str(final_response)
        })
        
        # Keep only last 10 exchanges to prevent context overflow
        if len(conversation_history) > 10:
            conversation_history.pop(0)
            
        return str(final_response)
    except Exception as e:
        return f"Error synthesizing final answer: {str(e)}"

async def main():
    print("Enhanced Hybrid AI Assistant starting up...")
    print("Initializing RAG pipeline...")
    
    # Initialize RAG pipeline first by making a dummy call to trigger index loading
    try:
        # This will initialize all the indexes (vector, keyword, BM25) upfront
        await search_documents_with_context("initialization", [])
        print("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: RAG initialization failed: {e}")
        print("Continuing with web search and YouTube only...")
    
    print("🚀 System ready! Combining document search (RAG) with web search and YouTube videos.")
    print("The system will remember our conversation context.")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("Processing...")
            print("Searching documents...")
            
            # Get RAG results using the hybrid.py module
            rag_result = await search_documents_with_context(user_query, conversation_history)
            print("Document search complete")
            
            print("Searching web...")
            # Get web search results using Groq directly (more reliable)
            web_result = await get_web_search_results(user_query)
            print("Web search complete")
            
            print("Searching YouTube...")
            # Get YouTube search results
            youtube_result = await get_youtube_search_results(user_query)
            print("YouTube search complete")
            
            print("Synthesizing final answer...")
            # Synthesize final answer using Gemini with all three sources
            final_answer = await synthesize_final_answer(user_query, rag_result, web_result, youtube_result)
            
            print(f"\n{'='*50}")
            print("FINAL ANSWER:")
            print(f"{'='*50}")
            print(final_answer)
            print(f"{'='*50}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            # Fallback to just RAG search
            try:
                print("Trying fallback to document search only...")
                result = await search_documents_with_context(user_query, conversation_history)
                print(f"Fallback result: {result}")
            except Exception as fallback_error:
                print(f"All methods failed: {fallback_error}")

# Run the orchestrator
if __name__ == "__main__":
    asyncio.run(main())