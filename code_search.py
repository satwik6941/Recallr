import asyncio
from google import genai
from google.genai import types
import os
import dotenv as env
from typing import List, Dict, Any
import json
import groq as Groq

env.load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_2_API_KEY"))

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

def chat_with_gemini(user_prompt: str) -> str:
    """Main chat function using Gemini with conversation context and web search"""
    try:
        # Analyze if query needs context
        context_analysis = analyze_query_context_dependency(user_prompt)
        
        # Build the enhanced query with conversation context
        conversation_context = get_conversation_context()
        
        # Setup grounding tool for web search
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        if context_analysis['needs_context'] and conversation_context:
            enhanced_prompt = f"""
{conversation_context}

Current user query: "{user_prompt}"

Please provide a comprehensive answer that builds on our previous conversation.
"""
        else:
            enhanced_prompt = user_prompt
        
        # Configure Gemini response with web search
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2000,
            system_instruction=f'''
You are an expert coding assistant with 20+ years of hands-on experience. 

{conversation_context}

INSTRUCTIONS:
1. **Context Awareness**: If the user's question contains pronouns (it, this, that, they, etc.) or refers to previous topics, understand what they're referring to from our conversation history.
2. **Start with Examples**: Always begin by explaining the user query with a simple real-life example, then provide the solution.
3. **Web Search**: Use web search to find the most current and relevant information from:
    - Stack Overflow, GitHub, official documentation
    - GeeksforGeeks, CodeProject, programming blogs
    - Community forums and Q&A sites
4. **Clear Formatting**: Provide well-formatted code snippets with explanations.
5. **Simple Explanations**: Break down complex concepts with examples.
6. **Conversation Flow**: Reference previous topics when relevant and build naturally on our conversation.
7. **Encouraging Tone**: Be warm and supportive.

Remember: This is a continuing conversation, not a standalone question.
'''
        )
        
        # Get response from Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=enhanced_prompt,
            config=config
        )
        
        if response and response.text:
            return response.text
        else:
            return "Sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        return f"Error: {str(e)}"

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
            temperature=0.1,
            max_tokens=2000
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
            
            print("🔄 Processing your query with conversation context and web search...")
            
            try:
                # Get AI response
                answer = chat_with_gemini(user_prompt)
                
                # Add AI response to conversation
                add_ai_message(answer)
                
                # Display response
                print(f"\n{'='*60}")
                print("🤖 ASSISTANT:")
                print(f"{'='*60}")
                print(answer)
                print(f"{'='*60}")
                
                # Save conversation to file
                save_conversation_to_file(user_prompt, answer)
                
                print(f"💾 Conversation: {len(messages_context)//2} exchanges stored")
                
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