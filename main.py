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
if not os.getenv("GEMINI_1_API_KEY"):
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

async def analyze_query_context_dependency(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Analyze if the query depends on conversation context and extract key information
    
    Args:
        query (str): The user's query
        conversation_history (List[Dict]): Previous conversation history
        
    Returns:
        Dict containing analysis results
    """
    try:
        # Check for context-dependent words/phrases
        context_indicators = [
            'it', 'this', 'that', 'they', 'them', 'these', 'those',
            'the above', 'previously', 'earlier', 'before', 'as mentioned',
            'like you said', 'from what you told', 'the one you mentioned',
            'explain more', 'tell me more', 'elaborate', 'expand on',
            'what about', 'how about', 'and also', 'additionally'
        ]
        
        needs_context = any(indicator in query.lower() for indicator in context_indicators)
        
        # Get recent topics from conversation history
        recent_topics = []
        if conversation_history and len(conversation_history) > 0:
            for exchange in conversation_history[-3:]:
                # Extract key topics from recent exchanges
                topics_prompt = f"""
Extract the main topics/concepts from this conversation exchange:
User: {exchange['user']}
Assistant: {exchange['assistant'][:200]}...

List the key topics/concepts (maximum 3):"""
                
                topics_response = await groq_llm.acomplete(topics_prompt)
                topics = str(topics_response).strip().split('\n')
                recent_topics.extend([topic.strip('- ').strip() for topic in topics if topic.strip()])
        
        return {
            'needs_context': needs_context,
            'recent_topics': recent_topics[:5],  # Keep top 5 recent topics
            'context_indicators_found': [indicator for indicator in context_indicators if indicator in query.lower()]
        }
        
    except Exception as e:
        return {
            'needs_context': False,
            'recent_topics': [],
            'context_indicators_found': [],
            'error': str(e)
        }

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

async def get_web_search_results(query: str, conversation_history: List[Dict] = None) -> str:
    """Get web search results using direct Google search with conversation context
    
    Args:
        query (str): The search query
        conversation_history (List[Dict]): Previous conversation history for context
        
    Returns:
        str: Processed web search results
    """
    try:
        # Build context-aware query for web search
        if conversation_history and len(conversation_history) > 0:
            # Get recent conversation context
            recent_context = ""
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                recent_context += f"User asked: {exchange['user']}\nAssistant answered: {exchange['assistant'][:300]}...\n\n"
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{recent_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Get raw web search results directly
        web_results = google_search(search_query)
        
        # Use Groq LLM directly to process the results with context
        if conversation_history and len(conversation_history) > 0:
            process_prompt = f"""You are answering a follow-up question in an ongoing conversation.

Recent conversation context:
{recent_context}

Current question: "{query}"
Search query used: "{search_query}"

Web search results:
{web_results}

Please provide a comprehensive answer that:
1. Considers the conversation context
2. Addresses the current question directly
3. Resolves any references to previous topics
4. Uses the web search results to provide accurate information

Answer:"""
        else:
            process_prompt = f"""Based on these web search results, provide a comprehensive answer to the query '{query}':

{web_results}

Please provide a clear, concise answer based on the search results."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        return str(groq_response)
    except Exception as e:
        return f"Error getting web search results: {str(e)}"

async def get_youtube_search_results(query: str, conversation_history: List[Dict] = None) -> str:
    """Get YouTube search results using external youtube.py module with conversation context
    
    Args:
        query (str): The search query
        conversation_history (List[Dict]): Previous conversation history for context
        
    Returns:
        str: Processed YouTube search results
    """
    try:
        print(f"Calling external YouTube module...")
        
        # Build context-aware query for YouTube search
        if conversation_history and len(conversation_history) > 0:
            # Get recent conversation context
            recent_context = ""
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                recent_context += f"User asked: {exchange['user']}\nAssistant answered: {exchange['assistant'][:300]}...\n\n"
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{recent_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable for YouTube videos. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Import and use the external youtube module
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from youtube import process_youtube_query
        
        # Get YouTube search results from external module
        youtube_results = await process_youtube_query(search_query)
        
        # Check if we got results
        if "No YouTube videos found" in youtube_results or "Error" in youtube_results:
            return youtube_results
        
        # Use Groq LLM to process and summarize the results while preserving URLs
        if conversation_history and len(conversation_history) > 0:
            process_prompt = f"""You are helping with a follow-up question in an ongoing conversation.

Recent conversation context:
{recent_context}

Current question: "{query}"
Search query used: "{search_query}"

YouTube search results:
{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning, considering the conversation context
4. Keep all the URLs exactly as provided in the original results
5. Address how these videos relate to the current question and previous conversation

Please provide a clear summary highlighting the most relevant videos for learning:"""
        else:
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

This is a continuing conversation. Here's our recent chat history:
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

1. **IMPORTANT**: If the user's question contains pronouns like "it", "this", "that", "they", etc., make sure you understand what they're referring to from our previous conversation
2. Answer the current question directly and clearly, building on what we've already discussed
3. Reference previous topics we discussed when relevant (e.g., "As we discussed earlier about...")
4. Use simple, everyday language that's easy to understand
5. Combine information from the sources when helpful
6. Be concise and focused on what the student actually asked
7. Only recommend YouTube videos if the student specifically asked for videos, tutorials, visual explanations, or similar requests
8. If recommending videos, include the exact URLs and format as: "Video Title" - Watch here: [URL]
9. **Make connections**: If this question relates to something we talked about before, explicitly mention that connection

Remember:
- Be warm and encouraging in your tone
- This is a continuing conversation, not a standalone question
- Build on our previous conversation naturally
- Explain things step by step if needed
- Don't overwhelm with unnecessary information
- Make it feel like a continuing conversation with a friend who remembers what we've talked about

Please provide a helpful, human-like response that shows you understand the context of our conversation:
"""
        else:
            # Build conversation context for LLM
            conversation_context = ""
            if conversation_history:
                conversation_context = "\n💬 **Previous Conversation:**\n"
                for i, exchange in enumerate(conversation_history[-5:], 1):  # Last 5 exchanges for context
                    conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
                    conversation_context += f"   I responded: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}\n\n"
            
            synthesis_prompt = f"""
You are a friendly and knowledgeable tutor helping a student understand concepts. Think of yourself as talking to a real person who needs clear, helpful explanations.

This is a continuing conversation. Here's our recent chat history:
{conversation_context}

📝 **Current Question:** "{query}"

I have information from two sources about this question:

📚 **From Course Materials:**
{rag_result}

🌐 **From Web Search:**
{web_result}

🎥 **Available YouTube Videos:**
{youtube_result}

Please respond naturally and conversationally, keeping in mind our previous conversation:

1. **IMPORTANT**: If the user's question contains pronouns like "it", "this", "that", "they", etc., make sure you understand what they're referring to from our previous conversation
2. Answer the current question directly and clearly, building on what we've already discussed
3. Reference previous topics we discussed when relevant (e.g., "As we discussed earlier about...")
4. Use simple, everyday language that's easy to understand
5. Combine information from both sources when helpful
6. Be concise and focused on what the student actually asked
7. If the student is just checking their understanding or asking a doubt, simply confirm or clarify
8. **Make connections**: If this question relates to something we talked about before, explicitly mention that connection

Remember:
- Be warm and encouraging in your tone
- This is a continuing conversation, not a standalone question
- Build on our previous conversation naturally
- Explain things step by step if needed
- Don't overwhelm with unnecessary information
- Make it feel like a continuing conversation with a friend who remembers what we've talked about
- Only mention videos and video links, if the student specifically asked for visual explanations

Please provide a helpful, human-like response that shows you understand the context of our conversation:
"""
        
        # Use Gemini for final synthesis
        gemini_llm = Gemini(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_1_API_KEY")
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
        
        # Debug: Show conversation history size
        print(f"💭 Conversation history: {len(conversation_history)} exchanges stored")
            
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
    print("The system will remember our conversation context and resolve references like 'it', 'this', etc.")
    print("\n💡 Tips:")
    print("- Ask follow-up questions using 'it', 'this', 'that' to test context resolution")
    print("- Type 'summary' to see conversation history")
    print("- The system will automatically detect when you're referring to previous topics")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit'/'summary' for options): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() == 'summary':
                print_conversation_summary()
                continue
                
            print("Processing...")
            
            # Analyze if the query needs conversation context
            context_analysis = await analyze_query_context_dependency(user_query, conversation_history)
            if context_analysis.get('needs_context'):
                print(f"🔄 Detected context dependency: {', '.join(context_analysis.get('context_indicators_found', []))}")
                if context_analysis.get('recent_topics'):
                    print(f"📝 Recent topics: {', '.join(context_analysis['recent_topics'][:3])}")
            
            print("Searching documents...")
            
            # Get RAG results using the hybrid.py module
            rag_result = await search_documents_with_context(user_query, conversation_history)
            print("Document search complete")
            
            print("Searching web...")
            # Get web search results using Groq directly (more reliable) with conversation context
            web_result = await get_web_search_results(user_query, conversation_history)
            print("Web search complete")
            
            print("Searching YouTube...")
            # Get YouTube search results with conversation context
            youtube_result = await get_youtube_search_results(user_query, conversation_history)
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

def print_conversation_summary():
    """Print a summary of the current conversation"""
    if not conversation_history:
        print("🔄 No conversation history yet.")
        return
    
    print(f"\n📖 **Conversation Summary** ({len(conversation_history)} exchanges):")
    for i, exchange in enumerate(conversation_history[-3:], 1):  # Show last 3
        print(f"  {i}. User: {exchange['user'][:60]}{'...' if len(exchange['user']) > 60 else ''}")
        print(f"     Bot: {exchange['assistant'][:80]}{'...' if len(exchange['assistant']) > 80 else ''}")
    print()

# Run the orchestrator
if __name__ == "__main__":
    asyncio.run(main())