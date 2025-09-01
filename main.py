import asyncio
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

from hybrid import (
    search_documents_with_context, 
    analyze_query_context_dependency,
    get_web_search_results,
    get_youtube_search_results
)    

from code_search import chat_with_gemini, add_user_message, add_ai_message, save_conversation_to_file
import time
import os
from datetime import datetime

load_dotenv()

# Validate API keys
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# YouTube API key is optional - will gracefully handle if missing
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    print("Warning: YOUTUBE_API_KEY not found. YouTube search functionality will be limited.")

# Conversation history file path
CONVERSATION_FILE = "conversation_history.txt"

def save_conversation_history():
    """Save the entire conversation history to a text file"""
    try:
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Recallr Conversation History ===\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total exchanges: {len(conversation_history)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, exchange in enumerate(conversation_history, 1):
                f.write(f"Exchange {i}:\n")
                f.write(f"Timestamp: {exchange.get('timestamp', 'N/A')}\n")
                f.write(f"User: {exchange['user']}\n")
                f.write(f"Assistant: {exchange['assistant']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"💾 Conversation saved to {CONVERSATION_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving conversation: {str(e)}")

def load_conversation_history():
    """Load conversation history from file if it exists"""
    global conversation_history
    try:
        if os.path.exists(CONVERSATION_FILE):
            # For now, we'll start fresh each session
            # You can implement parsing logic here if needed
            print(f"📁 Found existing conversation file: {CONVERSATION_FILE}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error loading conversation: {str(e)}")
        return False

def add_to_conversation_history(user_query: str, assistant_response: str):
    """Add an exchange to conversation history and save to file"""
    exchange = {
        "user": user_query,
        "assistant": assistant_response,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    conversation_history.append(exchange)
    
    # Save to file after each exchange
    save_conversation_history()

async def analyze_query_routing(query: str) -> Dict[str, Any]:
    """Use orchestrator LLM to analyze query and determine routing strategy"""
    try:
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n💬 **Recent Conversation Context:**\n"
            for i, exchange in enumerate(conversation_history, 1):  # Last 3 exchanges for context
                conversation_context += f"{i}. User: \"{exchange['user']}\"\n"
                conversation_context += f"   Assistant: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}\n\n"

        routing_prompt = f"""
You are an intelligent query router for an AI academic assistant. Analyze the user's query and determine the best routing strategy.

{conversation_context}

Current user query: "{query}"

Available routing options:
1. CODE_SEARCH - For programming, coding, software development, debugging, algorithms, data structures, specific programming languages, frameworks, libraries, APIs, etc.
2. ACADEMIC_RAG - For academic subjects, study materials, course content, general knowledge, research topics, etc.

Instructions:
- Analyze the query carefully considering the conversation context
- If the query is about programming, coding, software development, debugging, or any technical coding topic, route to CODE_SEARCH
- If the query is about academic subjects, study materials, or general knowledge, route to ACADEMIC_RAG
- Consider context - if previous messages were about coding and current query uses pronouns like "it", "this", "that", it might be coding-related

Respond with ONLY a JSON object in this exact format:
{{
    "routing": "CODE_SEARCH_OR_ACADEMIC_RAG",
    "confidence": 0.8,
    "reasoning": "brief explanation of why this routing was chosen"
}}

Where routing should be either "CODE_SEARCH" or "ACADEMIC_RAG".
"""

        # Use Gemini orchestrator for routing decision
        gemini_llm = GoogleGenAI(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        routing_response = await gemini_llm.acomplete(routing_prompt)
        routing_text = str(routing_response).strip()
        
        # Parse JSON response
        try:
            # Extract JSON from response if it's wrapped in other text
            import json
            if '{' in routing_text and '}' in routing_text:
                json_start = routing_text.find('{')
                json_end = routing_text.rfind('}') + 1
                json_text = routing_text[json_start:json_end]
                routing_data = json.loads(json_text)
                
                return {
                    'routing': routing_data.get('routing', 'ACADEMIC_RAG'),
                    'confidence': routing_data.get('confidence', 0.5),
                    'reasoning': routing_data.get('reasoning', 'Default routing'),
                    'raw_response': routing_text
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    'routing': 'ACADEMIC_RAG',
                    'confidence': 0.5,
                    'reasoning': 'JSON parsing failed, defaulting to academic RAG',
                    'raw_response': routing_text
                }
        except json.JSONDecodeError:
            return {
                'routing': 'ACADEMIC_RAG',
                'confidence': 0.5,
                'reasoning': 'JSON parsing error, defaulting to academic RAG',
                'raw_response': routing_text
            }
            
    except Exception as e:
        # Fallback to academic RAG if routing fails
        return {
            'routing': 'ACADEMIC_RAG',
            'confidence': 0.5,
            'reasoning': f'Routing analysis failed: {str(e)}',
            'raw_response': ''
        }

async def code_search_answer(query: str) -> str:
    """Handle coding-related queries using code_search.py"""
    try:
        print("🔍 Detected coding query - routing to specialized code assistant...")
        
        # Add user message to code search context
        add_user_message(query)
        
        # Get responses from both models instead of just one
        print("🤖 Getting responses from both Mistral and Gemini...")
        from code_search import get_dual_responses, save_dual_responses_to_file
        
        responses = get_dual_responses(query)
        
        # Add primary response to code search context
        add_ai_message(responses["primary"])
        
        # Save both responses to the same file
        print("💾 Saving dual responses to file...")
        save_dual_responses_to_file(query, responses)
        
        # Wait a moment for file to be written
        time.sleep(0.5)
        
        # Read the output file to get the complete conversation
        output_content = ""
        output_file_path = "code_results_answer.txt"
        
        # Try to read the file with multiple attempts
        for attempt in range(3):
            try:
                if os.path.exists(output_file_path):
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        output_content = f.read()
                    print("✅ Successfully read output file")
                    break
                else:
                    print(f"⚠️ Attempt {attempt + 1}: Output file not found, waiting...")
                    time.sleep(1)
            except Exception as file_error:
                print(f"⚠️ Attempt {attempt + 1}: Error reading file: {file_error}")
                time.sleep(1)
        
        # If file reading failed, use direct response
        if not output_content:
            print("⚠️ Could not read output file, using direct response")
            output_content = f"USER: {query}\\n\\nASSISTANT: {responses['primary']}"
        
        # Analyze the code search results with the orchestrator LLM
        analysis_prompt = f"""You are an expert coding assistant who has 20+ years of experience in analyzing and providing coding assistance results.

Original user query: "{query}"

Code search assistant response and conversation log (includes both Mistral and Gemini responses):
{output_content}

Please provide a final, polished answer that:
1. Directly addresses the user's coding question
2. Maintains the conversational tone from our ongoing session
3. Synthesizes the best parts from both AI responses when available
4. Includes any relevant code examples or explanations from the code search results
5. Builds on our previous conversation if relevant
6. Is clear, concise, and helpful for a student learning to code
7. If both responses are available, combine their strengths into a comprehensive answer

Remember: This is part of an ongoing conversation with a student. Be encouraging and educational."""
        
        # Use Gemini orchestrator to analyze and refine the response
        gemini_llm = GoogleGenAI(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        print("🤖 Orchestrator analyzing both AI responses...")
        final_response = await gemini_llm.acomplete(analysis_prompt)
        
        return str(final_response)
        
    except Exception as e:
        return f"Error in code search: {str(e)}. Please try rephrasing your coding question."

# Conversation context to maintain chat history
conversation_history = []

def save_conversation_history():
    """Save the entire conversation history to a text file"""
    try:
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Recallr Conversation History ===\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total exchanges: {len(conversation_history)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, exchange in enumerate(conversation_history, 1):
                f.write(f"Exchange {i}:\n")
                f.write(f"Timestamp: {exchange.get('timestamp', 'N/A')}\n")
                f.write(f"User: {exchange['user']}\n")
                f.write(f"Assistant: {exchange['assistant']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"💾 Conversation saved to {CONVERSATION_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving conversation: {str(e)}")

def load_conversation_history():
    """Load conversation history from file if it exists"""
    global conversation_history
    try:
        if os.path.exists(CONVERSATION_FILE):
            # For now, we'll start fresh each session
            # You can implement parsing logic here if needed
            print(f"📁 Found existing conversation file: {CONVERSATION_FILE}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error loading conversation: {str(e)}")
        return False

def add_to_conversation_history(user_query: str, assistant_response: str):
    """Add an exchange to conversation history and save to file"""
    exchange = {
        "user": user_query,
        "assistant": assistant_response,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    conversation_history.append(exchange)
    
    # Save to file after each exchange
    save_conversation_history()

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
                for i, exchange in enumerate(conversation_history, 1):  
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
                for i, exchange in enumerate(conversation_history, 1):
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
        gemini_llm = GoogleGenAI(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        final_response = await gemini_llm.acomplete(synthesis_prompt)
        
        # Debug: Show conversation history size
        print(f"💭 Conversation history: {len(conversation_history)} exchanges stored")
            
        return str(final_response)
    except Exception as e:
        return f"Error synthesizing final answer: {str(e)}"

async def main():
    print("Recallr -Your AI Academic Assistant is starting up...")
    print("Initializing the pipeline...")
    
    # Load existing conversation history
    load_conversation_history()
    
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
                
            print("🧠 Analyzing query for optimal routing...")
            
            # Use orchestrator LLM to determine routing strategy
            routing_analysis = await analyze_query_routing(user_query)
            
            print(f"🎯 Routing decision: {routing_analysis['routing']} (confidence: {routing_analysis['confidence']:.2f})")
            print(f"💡 Reasoning: {routing_analysis['reasoning']}")
            
            if routing_analysis['routing'] == 'CODE_SEARCH':
                # Route to specialized code search assistant
                print("🔍 Routing to specialized coding assistant...")
                final_answer = await code_search_answer(user_query)
                
                # Add to conversation history and save to file
                add_to_conversation_history(user_query, final_answer)
                
            else:
                # Route to academic RAG pipeline (existing flow)
                print("📚 Routing to academic knowledge pipeline...")
                
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
                
                # Add to conversation history and save to file
                add_to_conversation_history(user_query, final_answer)
            
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
        for i, exchange in enumerate(conversation_history, 1):  # Show all exchanges
            print(f"  {i}. User: {exchange['user'][:60]}{'...' if len(exchange['user']) > 60 else ''}")
            print(f"     Bot: {exchange['assistant'][:800]}{'...' if len(exchange['assistant']) > 800 else ''}")
        print()

if __name__ == "__main__":
    asyncio.run(main())