import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

# Import the RAG pipeline and search functions from hybrid.py
from hybrid import (
    search_documents_with_context, 
    analyze_query_context_dependency,
    get_web_search_results,
    get_youtube_search_results
)

load_dotenv()

# Validate API keys
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# YouTube API key is optional - will gracefully handle if missing
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    print("Warning: YOUTUBE_API_KEY not found. YouTube search functionality will be limited.")

# Conversation context to maintain chat history
conversation_history = []

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
                for i, exchange in enumerate(conversation_history[-10:], 1):  # Last 10 exchanges for context
                    conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
                    conversation_context += f"   I responded: {exchange['assistant'][:2000]}{'...' if len(exchange['assistant']) > 2000 else ''}\n\n"
            
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
                for i, exchange in enumerate(conversation_history[-10:], 1):  # Last 10 exchanges for context
                    conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
                    conversation_context += f"   I responded: {exchange['assistant'][:2000]}{'...' if len(exchange['assistant']) > 2000 else ''}\n\n"
            
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
        
        # Debug: Show conversation history size
        print(f"💭 Conversation history: {len(conversation_history)} exchanges stored")
            
        return str(final_response)
    except Exception as e:
        return f"Error synthesizing final answer: {str(e)}"

async def main():
    print("Enhanced Hybrid AI Assistant starting up...")
    print("Initializing the pipeline...")
    
    # Initialize RAG pipeline first by making a dummy call to trigger index loading
    try:
        # This will initialize all the indexes (vector, keyword, BM25) upfront
        await search_documents_with_context("initialization", [])
        print("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: RAG initialization failed: {e}")
        print("Continuing with web search and YouTube only...")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit'/'summary' for options): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Happy Learning!")
                break
            elif user_query.lower() == 'summary':
                await print_conversation_summary()
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
            print("Happy learning!")
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

async def print_conversation_summary():
    """Print an AI-generated summary of the current conversation"""
    if not conversation_history:
        print("🔄 No conversation history yet.")
        return
    
    try:
        # Import groq_llm from hybrid
        from hybrid import get_groq_llm
        
        # Build the conversation context for the LLM
        conversation_text = ""
        for i, exchange in enumerate(conversation_history, 1):
            conversation_text += f"Exchange {i}:\n"
            conversation_text += f"User: {exchange['user']}\n"
            conversation_text += f"Assistant: {exchange['assistant']}\n\n"
        
        # Simple system prompt for summarization
        summary_prompt = f"""Please provide a clear and concise summary of this conversation between a student and an AI tutor. 
        Focus on the main topics discussed, key questions asked, and important concepts covered.

Conversation:
{conversation_text}

Summary:"""
        
        print("🤖 Generating conversation summary...")
        
        # Use Groq LLM to generate the summary
        summary_response = await get_groq_llm().acomplete(summary_prompt)
        
        print(f"\n📖 **Conversation Summary** ({len(conversation_history)} exchanges):")
        print("=" * 60)
        print(str(summary_response))
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to the original format if LLM fails
        print(f"\n📖 **Conversation Summary** ({len(conversation_history)} exchanges):")
        for i, exchange in enumerate(conversation_history[-10:], 1):  # Show last 10
            print(f"  {i}. User: {exchange['user'][:60]}{'...' if len(exchange['user']) > 60 else ''}")
            print(f"     Bot: {exchange['assistant'][:800]}{'...' if len(exchange['assistant']) > 800 else ''}")
        print()

if __name__ == "__main__":
    asyncio.run(main())