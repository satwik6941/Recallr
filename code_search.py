import asyncio
import os
import dotenv as env
from typing import List, Dict, Any
import json
import groq as Groq
from mistralai import Mistral
from openai import OpenAI
from tavily import TavilyClient

env.load_dotenv()

# Initialize OpenAI and Tavily clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Message history for conversation context
messages_context = []

def add_user_message(user_prompt: str):
    """Add user message to conversation history"""
    messages_context.append({
        "role": "user", 
        "content": user_prompt,
        "timestamp": len(messages_context) + 1
    })

def add_ai_message(ai_response: str):
    """Add AI response to conversation history"""
    messages_context.append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": len(messages_context) + 1
    })

def get_conversation_context() -> str:
    """Build conversation context for the AI"""
    if not messages_context:
        return ""
    
    context = "Previous conversation context:\n"
    recent_messages = messages_context
    
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        context += f"{role}: {content}\n"
    
    return context + "\n"

def analyze_query_context_dependency(query: str) -> Dict[str, Any]:
    """Analyze if the query depends on conversation context"""
    context_indicators = [
        'it', 'this', 'that', 'they', 'them', 'these', 'those',
        'the above', 'previously', 'earlier', 'before', 'as mentioned',
        'like you said', 'from what you told', 'the one you mentioned',
        'explain more', 'tell me more', 'elaborate', 'expand on',
        'what about', 'how about', 'and also', 'additionally'
    ]
    
    needs_context = any(indicator in query.lower() for indicator in context_indicators)
    found_indicators = [indicator for indicator in context_indicators if indicator in query.lower()]
    
    return {
        'needs_context': needs_context,
        'context_indicators_found': found_indicators
    }

def chat_with_mistral(user_prompt: str) -> str:
    """Simple Mistral chat without web search or tool calling"""
    try:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            print("Mistral API key not found, falling back to OpenAI")
            return chat_with_openai(user_prompt)
        
        # Analyze context and build conversation context
        context_analysis = analyze_query_context_dependency(user_prompt)
        conversation_context = get_conversation_context()
        
        # Build enhanced prompt with context (same logic as OpenAI)
        if context_analysis['needs_context'] and conversation_context:
            enhanced_prompt = f"""
{conversation_context}

Current user query: "{user_prompt}"

Please provide a comprehensive answer that builds on our previous conversation.
"""
        else:
            enhanced_prompt = user_prompt
        
        # Initialize Mistral client
        try:
            mistral_client = Mistral(api_key=mistral_api_key)
            print("🤖 Using Mistral Codestral...")
        except Exception as init_error:
            print(f"Failed to initialize Mistral client: {init_error}")
            return chat_with_openai(user_prompt)
        
        # Mistral-specific system instruction (no web search references)
        mistral_system_instruction = f'''
You are Mistral Codestral, an expert coding assistant with 20+ years of deep knowledge in programming languages, frameworks, and software development best practices.

{conversation_context}

CORE CAPABILITIES:
1. **Context Awareness**: If the user's question contains pronouns (it, this, that, they, etc.) or refers to previous topics, understand what they're referring to from our conversation history.
2. **Code-First Approach**: Always provide practical, executable code examples with detailed explanations.
3. **Best Practices**: Suggest modern coding patterns, optimization techniques, and industry standards.
4. **Problem Solving**: Break down complex coding problems into manageable steps.
5. **Multiple Solutions**: When applicable, provide different approaches (beginner vs advanced, different paradigms).
6. **Debugging Help**: Identify potential issues and provide solutions for common coding problems.
7. **Clear Documentation**: Include comments and explanations within code snippets.
8. **Conversation Flow**: Reference previous topics when relevant and build naturally on our conversation.
9. **Encouraging Tone**: Be supportive and help users learn through examples.

SPECIALIZATIONS:
- Algorithm design and data structures
- API development and integration  
- Database design and optimization
- Frontend and backend development
- Code review and refactoring
- Performance optimization
- Security best practices

Remember: This is a continuing conversation, not a standalone question. Focus on providing accurate, well-structured code solutions.
'''
        
        # Prepare messages for Mistral (simple chat without tools)
        messages = [
            {
                "role": "system",
                "content": mistral_system_instruction
            },
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]
        
        # Make API call to Mistral with comprehensive error handling
        try:
            response = mistral_client.chat.complete(
                model="codestral-latest",
                messages=messages
            )
            
            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                mistral_response = choice.message.content
                
                if mistral_response and len(mistral_response.strip()) > 0:
                    return mistral_response
                else:
                    print("Empty response from Mistral, falling back to OpenAI")
                    return chat_with_openai(user_prompt)
            else:
                print("No valid response from Mistral, falling back to OpenAI")
                return chat_with_openai(user_prompt)
                
        except Exception as mistral_error:
            error_msg = str(mistral_error).lower()
            
            # Handle specific Mistral error types gracefully - never break the code
            if any(code in error_msg for code in ["400", "401", "402", "403", "404", "429", "500", "502", "503"]):
                print(f"Mistral API error ({mistral_error}), falling back to OpenAI")
                return chat_with_openai(user_prompt)
            elif any(issue in error_msg for issue in ["timeout", "connection", "network", "ssl"]):
                print(f"Mistral connection error, falling back to OpenAI")
                return chat_with_openai(user_prompt)
            elif "rate limit" in error_msg or "quota" in error_msg:
                print(f"Mistral rate limit exceeded, falling back to OpenAI")
                return chat_with_openai(user_prompt)
            else:
                print(f"Unexpected Mistral error ({mistral_error}), falling back to OpenAI")
                return chat_with_openai(user_prompt)
        
    except Exception as general_error:
        print(f"General error in Mistral setup ({general_error}), falling back to OpenAI")
        return chat_with_openai(user_prompt)

def chat_with_openai(user_prompt: str) -> str:
    """Main chat function using OpenAI with conversation context and web search via Tavily"""
    try:
        # Analyze if query needs context
        context_analysis = analyze_query_context_dependency(user_prompt)

        # Build the enhanced query with conversation context
        conversation_context = get_conversation_context()

        if context_analysis['needs_context'] and conversation_context:
            enhanced_prompt = f"""
{conversation_context}

Current user query: "{user_prompt}"

Please provide a comprehensive answer that builds on our previous conversation.
"""
        else:
            enhanced_prompt = user_prompt

        # Perform Tavily web search for current information
        search_context = ""
        try:
            search_results = tavily_client.search(
                query=user_prompt,
                search_depth="basic",
                max_results=3
            )

            if search_results and "results" in search_results:
                search_context = "\n\nWeb Search Results:\n"
                for idx, result in enumerate(search_results["results"], 1):
                    search_context += f"{idx}. {result.get('title', 'No title')}\n"
                    search_context += f"   URL: {result.get('url', 'No URL')}\n"
                    search_context += f"   Content: {result.get('content', 'No content')[:200]}...\n"
        except Exception as search_error:
            print(f"Web search error: {search_error}")

        system_instruction = f'''You are an expert coding assistant with real-time web search capabilities and 20+ years of hands-on experience.

{conversation_context}

CORE CAPABILITIES & INSTRUCTIONS:
1. **Context Awareness**: If the user's question contains pronouns (it, this, that, they, etc.) or refers to previous topics, understand what they're referring to from our conversation history.
2. **Start with Examples**: Always begin by explaining the user query with a simple real-life example, then provide the solution.
3. **Real-Time Information**: Use the provided web search results to find the most current and relevant information from:
    - Stack Overflow, GitHub, official documentation
    - GeeksforGeeks, CodeProject, programming blogs
    - Community forums and Q&A sites
    - Latest framework updates and version-specific information
4. **Current Information**: Verify information is up-to-date using the search results, especially for:
    - API changes and deprecations
    - New library versions and features
    - Best practices and security updates
    - Framework-specific solutions
5. **Clear Formatting**: Provide well-formatted code snippets with explanations.
6. **Simple Explanations**: Break down complex concepts with examples.
7. **Conversation Flow**: Reference previous topics when relevant and build naturally on our conversation.
8. **Encouraging Tone**: Be warm and supportive.
9. **Source Citations**: When using web search results, mention the source or reference with URLs.

Remember: This is a continuing conversation, not a standalone question. Use the provided search results to provide the most accurate and current information.
'''

        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": enhanced_prompt + search_context}
            ]
        )

        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "Sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        return f"Error: {str(e)}"

def get_dual_responses(user_prompt: str) -> Dict[str, str]:
    """Get responses from both Mistral and OpenAI models"""
    responses = {
        "mistral": None,
        "openai": None,
        "primary": None  # The response that will be added to conversation context
    }

    try:
        # Try Mistral first
        print("🤖 Getting response from Mistral Codestral...")
        mistral_response = chat_with_mistral(user_prompt)

        if mistral_response and "Error:" not in mistral_response and len(mistral_response.strip()) > 0:
            responses["mistral"] = mistral_response
            responses["primary"] = mistral_response  # Use Mistral as primary if successful
            print("✅ Mistral response obtained")
        else:
            print("❌ Mistral response failed or empty")
    except Exception as e:
        print(f"❌ Mistral error: {e}")

    try:
        # Always get OpenAI response as well
        print("🌐 Getting response from OpenAI with web search...")
        openai_response = chat_with_openai(user_prompt)

        if openai_response and "Error:" not in openai_response and len(openai_response.strip()) > 0:
            responses["openai"] = openai_response
            print("✅ OpenAI response obtained")

            # If Mistral failed, use OpenAI as primary
            if not responses["primary"]:
                responses["primary"] = openai_response
        else:
            print("❌ OpenAI response failed or empty")
    except Exception as e:
        print(f"❌ OpenAI error: {e}")

    # Fallback if both failed
    if not responses["primary"]:
        responses["primary"] = "Sorry, both AI models are currently unavailable. Please try again later."

    return responses

def save_dual_responses_to_file(user_query: str, responses: Dict[str, str]):
    """Save conversation with both Mistral and OpenAI responses"""
    try:
        output_filename = "code_results_answer.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== DUAL AI CODE SEARCH CONVERSATION LOG ===\n\n")

            # Write full conversation history
            exchange_count = 0
            for i in range(0, len(messages_context), 2):
                if i+1 < len(messages_context):
                    exchange_count += 1
                    f.write(f"[Exchange {exchange_count}]\n")
                    f.write(f"USER: {messages_context[i]['content']}\n\n")
                    f.write(f"PRIMARY RESPONSE: {messages_context[i+1]['content']}\n\n")
                    f.write("-" * 60 + "\n\n")

            # Add current exchange with both responses
            if responses.get("mistral") or responses.get("openai"):
                f.write(f"[Current Exchange - Dual Responses]\n")
                f.write(f"USER: {user_query}\n\n")

                if responses.get("mistral"):
                    f.write("🤖 MISTRAL RESPONSE:\n")
                    f.write(f"{responses['mistral']}\n\n")
                    f.write("-" * 40 + "\n\n")

                if responses.get("openai"):
                    f.write("🌐 OPENAI RESPONSE:\n")
                    f.write(f"{responses['openai']}\n\n")
                    f.write("-" * 40 + "\n\n")

            f.write("=" * 60 + "\n")
            f.write("END OF CONVERSATION\n")

        print(f"✅ Dual responses saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving dual responses: {e}")

def save_conversation_to_file(user_query: str, ai_response: str):
    """Save the conversation exchange to a file"""
    try:
        output_filename = "code_results_answer.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== CODE SEARCH CONVERSATION LOG ===\n\n")
            
            # Write full conversation history
            exchange_count = 0
            for i in range(0, len(messages_context), 2):
                if i+1 < len(messages_context):
                    exchange_count += 1
                    f.write(f"[Exchange {exchange_count}]\n")
                    f.write(f"USER: {messages_context[i]['content']}\n\n")
                    f.write(f"ASSISTANT: {messages_context[i+1]['content']}\n\n")
                    f.write("-" * 60 + "\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("END OF CONVERSATION\n")
        
        print(f"✅ Conversation saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving conversation: {e}")

def clear_conversation_history():
    """Clear the conversation history"""
    global messages_context
    messages_context.clear()
    print("🧹 Conversation history cleared!")

def print_conversation_summary():
    """Print a summary of the whole conversation using Groq"""
    if not messages_context:
        print("🔄 No conversation history yet.")
        return
    
    try:
        # Initialize Groq client (without messages in constructor)
        groq_client = Groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Build conversation text for the prompt
        conversation_text = ""
        for i, msg in enumerate(messages_context):
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # Create the messages for the API call
        summary_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide a clear and detailed summary of the whole conversation between the user and assistant. Make sure that all topics, concepts and conversations are covered"
            },
            {
                "role": "user", 
                "content": f"Please summarize this conversation:\n\n{conversation_text}"
            }
        ]
        
        print("🤖 Generating conversation summary...")
        
        # Make the API call
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=summary_messages,
            temperature=0.1
        )
        
        # Display the summary
        summary = completion.choices[0].message.content
        print(f"\n Conversation Summary ({len(messages_context)//2} exchanges):")
        print()
        print(summary)
    except Exception as e:
        print(f"Error generating summary: {e}")

def main():
    """Main conversation loop"""
    print("📝 Commands: 'summary' | 'clear' | 'quit'")
    print("-" * 60)
    
    while True:
        try:
            user_prompt = input("\n💬 Enter your coding question: ")
            
            # Handle commands
            if user_prompt.strip().lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye! Happy coding!")
                print("👋 Goodbye! Happy Learning!")
                break
            elif user_prompt.strip().lower() == "summary":
                print_conversation_summary()
                continue
            elif user_prompt.strip().lower() == "clear":
                clear_conversation_history()
                continue
            elif not user_prompt.strip():
                print(" Please enter a valid question.")
                continue
            
            # Add user message to conversation
            add_user_message(user_prompt)
            
            # Analyze context dependency
            context_analysis = analyze_query_context_dependency(user_prompt)
            if context_analysis['needs_context'] and len(messages_context) > 1:
                print(f"🔍 Detected context dependency: {', '.join(context_analysis['context_indicators_found'])}")
            
            print("🔄 Getting responses from both Mistral and OpenAI...")
            
            try:
                # Get responses from both AI models
                responses = get_dual_responses(user_prompt)
                
                # Add primary response to conversation context
                add_ai_message(responses["primary"])
                
                # Display responses
                print(f"\n{'='*60}")
                
                if responses.get("mistral"):
                    print("🤖 MISTRAL: ")
                    print(f"{'='*60}")
                    print(responses["mistral"])
                    print(f"{'='*60}\n")
                
                if responses.get("openai"):
                    print("🌐 OPENAI: ")
                    print(f"{'='*60}")
                    print(responses["openai"])
                    print(f"{'='*60}")
                
                # Save dual responses to file
                save_dual_responses_to_file(user_prompt, responses)
                
                print(f"💾 Dual responses saved | Conversation: {len(messages_context)//2} exchanges stored")
                
            except Exception as e:
                print(f"Error getting response: {e}")
                # Remove the user message if we couldn't get a response
                messages_context.pop()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Happy coding!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()