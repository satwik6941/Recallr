import asyncio
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

from hybrid import (
    search_documents_with_context, 
    analyze_query_context_dependency,
    get_web_search_results,
    get_youtube_search_results
)    

from code_search import add_user_message as code_add_user_message, add_ai_message as code_add_ai_message, get_dual_responses as code_get_dual_responses, save_dual_responses_to_file as code_save_dual_responses_to_file
from math_search import add_user_message as math_add_user_message, add_ai_message as math_add_ai_message, get_dual_responses as math_get_dual_responses, save_dual_responses_to_file as math_save_dual_responses_to_file
from doc_processing import get_system_prompt_with_caching, has_pdf_collection_changed
import time
import os
from datetime import datetime
from pathlib import Path

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

# Global variable to store the academic system prompt
ACADEMIC_SYSTEM_PROMPT = None

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
You are an intelligent query router for an AI academic assistant that serves students from kindergarten to M-Tech level. Analyze the user's query and determine the best routing strategy.

{conversation_context}

Current user query: "{query}"

**Available routing options:**

1. **MATH_SEARCH** - For ALL mathematics-related queries across ALL educational levels:
   **Elementary (K-5):** Basic arithmetic, counting, shapes, simple addition/subtraction
   **Middle School (6-8):** Fractions, decimals, basic algebra, geometry, percentages
   **High School (9-12):** Advanced algebra, trigonometry, calculus, statistics, coordinate geometry
   **Engineering (B-Tech/M-Tech):** Advanced calculus, differential equations, linear algebra, discrete math, numerical methods, Applied Mathematics, Statistics and Probability, Operations Research

   **Math Keywords to detect:** numbers, equations, solve, calculate, formula, theorem, proof, derivative, integral, matrix, probability, statistics, geometry, algebra, calculus, trigonometry, arithmetic, mathematical, math problem, step-by-step solution

2. **CODE_SEARCH** - For ALL programming and computer science queries:
   **Beginner:** Scratch, basic programming concepts, logic building
   **School Level:** Python basics, simple algorithms, basic coding
   **Engineering:** Advanced programming, data structures, algorithms, software development, debugging, frameworks, APIs, databases, web development, machine learning code
   
   **Programming Keywords to detect:** code, programming, python, java, javascript, C++, algorithm, function, variable, loop, array, debugging, software, app, website, database, API, framework, git, coding

3. **ACADEMIC_RAG** - For ALL other academic subjects and general knowledge:
   **All Levels:** Science (physics, chemistry, biology), social studies, history, geography, literature, languages, engineering subjects (non-coding), research topics, study materials, course content, general knowledge
   
   **Academic Keywords to detect:** science, physics, chemistry, biology, history, geography, literature, essay, theory, concept, explain, definition, study, course, subject

**SMART ROUTING RULES:**

**Priority System:**
1. **Mathematics FIRST:** If query relates to numbers, mathematical operations, mathematical concepts (basic to advanced), equations, formulas, or asks for calculations/mathematical solutions or doubts/queries → MATH_SEARCH
2. **Programming SECOND:** If query relates to coding, programming languages, software development, or technical implementation or any kinds of doubts/queries → CODE_SEARCH
3. **Academic THIRD:** All other educational content, theories, concepts, general knowledge → ACADEMIC_RAG

**Level-Adaptive Detection:**
- **Simple Math:** "What is 2+2?" or "How to add fractions?" → MATH_SEARCH
- **Advanced Math:** "Solve differential equation" or "Find derivative of sin(x)" → MATH_SEARCH
- **Basic Programming:** "How to print in Python?" → CODE_SEARCH
- **Advanced Programming:** "Implement binary search tree" → CODE_SEARCH
- **Science Concepts:** "What is photosynthesis?" → ACADEMIC_RAG
- **Engineering Theory:** "Explain thermodynamics" → ACADEMIC_RAG

**Context Consideration:**
- If previous conversation was about math and current query uses pronouns ("solve this", "what about it"), likely MATH_SEARCH
- If previous conversation was about coding and current query references ("debug this", "how to fix it"), likely CODE_SEARCH
- Consider educational level from context

**Edge Cases:**
- "Mathematical algorithms" → Focus on implementation = CODE_SEARCH, Focus on theory = MATH_SEARCH
- "Statistics in Python" → Implementation = CODE_SEARCH, Mathematical concepts = MATH_SEARCH
- "Physics equations" → MATH_SEARCH (if solving), ACADEMIC_RAG (if explaining concepts)

Respond with ONLY a JSON object in this exact format:
{{
    "routing": "MATH_SEARCH_OR_CODE_SEARCH_OR_ACADEMIC_RAG",
    "confidence": 0.85,
    "reasoning": "brief explanation including detected educational level and key indicators"
}}

Where routing should be either "MATH_SEARCH" or "CODE_SEARCH" or "ACADEMIC_RAG".
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

async def math_search_answer(query: str) -> str:
    """Handle math-related queries using math_search.py"""
    try:
        print("� Detected math query - routing to specialized mathematics assistant...")
        
        # Add user message to math search context
        math_add_user_message(query)
        
        # Get responses from both models instead of just one
        print("🤖 Getting responses from both Mistral and Gemini...")
        
        responses = math_get_dual_responses(query)
        
        # Add primary response to math search context
        math_add_ai_message(responses["primary"])
        
        # Save both responses to the same file
        print("💾 Saving dual responses to file...")
        math_save_dual_responses_to_file(query, responses)
        
        # Wait a moment for file to be written
        time.sleep(0.5)
        
        # Read the output file to get the complete conversation
        output_content = ""
        output_file_path = "math_results_answer.txt"
        
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
        analysis_prompt = f"""You are an expert mathematics tutor with 20+ years of experience in teaching and analyzing mathematical concepts. You have extraordinary teaching methods, numerous publications, and prestigious awards in mathematics education.

Original user query: "{query}"

Math search assistant response and conversation log (includes both Mistral and Gemini responses):
{output_content}

Please provide a final, polished answer that:
1. **Directly addresses the user's mathematics question** with step-by-step solutions
2. **Maintains the conversational tone** from our ongoing session
3. **Synthesizes the best parts** from both AI responses when available
4. **Breaks down complex problems** into 3-5 manageable steps with clear explanations
5. **Provides real-world examples** that students can relate to (sports, social media, daily life)
6. **Connects mathematical concepts** to previously discussed topics when relevant
7. **Uses simple, encouraging language** before introducing mathematical terminology
8. **Shows the reasoning** behind each mathematical step, not just the procedure
9. **Includes visual descriptions** of graphs, patterns, or geometric concepts when helpful
10. **Builds confidence** through supportive, student-friendly explanations

STUDENT-FOCUSED APPROACH:
- Start with a relatable real-world example when possible
- Explain WHY each step is necessary, not just HOW to do it
- Use encouraging language like "Great question!", "Let's tackle this together"
- Break complex concepts into digestible pieces
- Connect new learning to previous mathematical knowledge
- Show multiple solution methods when applicable
- Verify answers and explain if the result makes sense

MATHEMATICAL COMMUNICATION STYLE:
- Use clear, step-by-step formatting
- Explain mathematical reasoning and logic
- Provide context for when and why to use specific methods
- Include encouraging remarks about the learning process
- Make abstract concepts concrete through examples
- Build mathematical confidence and understanding

Remember: This is part of an ongoing conversation with a student learning mathematics. Be encouraging, educational, and focus on building deep mathematical understanding rather than just providing answers."""
        
        # Use Gemini orchestrator to analyze and refine the response
        gemini_llm = GoogleGenAI(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        print("🤖 Orchestrator analyzing both AI responses...")
        final_response = await gemini_llm.acomplete(analysis_prompt)
        
        return str(final_response)
        
    except Exception as e:
        return f"Error in math search: {str(e)}. Please try rephrasing your mathematics question."

async def code_search_answer(query: str) -> str:
    """Handle coding-related queries using code_search.py"""
    try:
        print("🔍 Detected coding query - routing to specialized code assistant...")
        
        # Add user message to code search context
        code_add_user_message(query)
        
        # Get responses from both models instead of just one
        print("🤖 Getting responses from both Mistral and Gemini...")
        
        responses = code_get_dual_responses(query)
        
        # Add primary response to code search context
        code_add_ai_message(responses["primary"])
        
        # Save both responses to the same file
        print("💾 Saving dual responses to file...")
        code_save_dual_responses_to_file(query, responses)
        
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

async def refresh_academic_system_prompt():
    """Manually refresh the academic system prompt by re-checking PDF collection"""
    global ACADEMIC_SYSTEM_PROMPT
    
    try:
        print("🔄 Refreshing Academic System Prompt...")
        data_path = Path('data')
        ACADEMIC_SYSTEM_PROMPT = get_system_prompt_with_caching(data_path)
        print("✅ Academic System Prompt refreshed successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Error refreshing Academic System Prompt: {e}")
        return False

async def check_and_update_documents():
    """Check for document changes and update system prompt if needed"""
    global ACADEMIC_SYSTEM_PROMPT
    
    try:
        data_path = Path('data')
        
        # Check if PDF collection has changed
        if has_pdf_collection_changed(data_path):
            print("📚 Document collection updated - refreshing academic knowledge...")
            
            # Get current PDF files for comparison
            from doc_processing import get_current_pdf_files, load_cached_pdf_list
            current_pdfs = get_current_pdf_files(data_path)
            cached_pdfs = load_cached_pdf_list()
            
            # Show what changed
            added = set(current_pdfs) - set(cached_pdfs) if cached_pdfs else set(current_pdfs)
            removed = set(cached_pdfs) - set(current_pdfs) if cached_pdfs else set()
            
            change_info = []
            if added:
                change_info.append(f"Added: {len(added)} file(s)")
            if removed:
                change_info.append(f"Removed: {len(removed)} file(s)")
            
            if change_info:
                print(f"📋 Changes detected: {', '.join(change_info)}")
            
            # Update the system prompt
            ACADEMIC_SYSTEM_PROMPT = get_system_prompt_with_caching(data_path)
            
            print("✅ Academic knowledge updated! Your conversation will now reflect the latest documents.")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error checking document changes: {e}")
        return False

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
    global ACADEMIC_SYSTEM_PROMPT
    
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
{ACADEMIC_SYSTEM_PROMPT if ACADEMIC_SYSTEM_PROMPT else "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."}

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
7. If the student is just checking their understanding or asking a doubt, simply confirm or clarify
8. **Make connections**: If this question relates to something we talked about before, explicitly mention that connection

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
{ACADEMIC_SYSTEM_PROMPT if ACADEMIC_SYSTEM_PROMPT else "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."}

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
    global ACADEMIC_SYSTEM_PROMPT
    
    print("Recallr -Your AI Academic Assistant is starting up...")
    print("Initializing the pipeline...")
    
    # Load existing conversation history
    load_conversation_history()
    
    # Initialize Academic System Prompt with smart caching
    print("📚 Initializing Academic System Prompt...")
    try:
        data_path = Path('data')
        ACADEMIC_SYSTEM_PROMPT = get_system_prompt_with_caching(data_path)
        print("✅ Academic System Prompt initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Academic System Prompt initialization failed: {e}")
        ACADEMIC_SYSTEM_PROMPT = "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."
        print("Using default academic system prompt...")
    
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
            user_query = input("\nEnter your query (or 'quit'/'summary'/'refresh' for options): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Happy Learning!")
                break
            elif user_query.lower() == 'summary':
                await print_conversation_summary()
                continue
            elif user_query.lower() in ['refresh', 'reload']:
                await refresh_academic_system_prompt()
                continue
                
            # ✨ AUTO-CHECK: Check for document changes before processing each query
            documents_updated = await check_and_update_documents()
            
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
                
            elif routing_analysis['routing'] == 'MATH_SEARCH':
                # Route to specialized math search assistant
                print("🔢 Routing to specialized mathematics assistant...")
                final_answer = await math_search_answer(user_query)
                
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