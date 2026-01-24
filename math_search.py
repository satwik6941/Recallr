import asyncio
from openai import OpenAI
from tavily import TavilyClient
import os
import dotenv as env
from typing import List, Dict, Any
import json
import groq as Groq
from mistralai import Mistral

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
        mistral_api_key = os.getenv("MISTRAL_API_KEY_1")
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
            print("🤖 Using Mistral Mathstral...")
        except Exception as init_error:
            print(f"Failed to initialize Mistral client: {init_error}")
            return chat_with_openai(user_prompt)
        
        # Mistral-specific system instruction (no web search references)
        mistral_system_instruction = f'''
You are Mistral Mathstral, a friendly mathematics tutor with 20+ years of experience helping students understand and master ALL mathematical concepts across every field. You have many extraordinary teaching methods, publications and awards.

{conversation_context}

STUDENT-FOCUSED TEACHING APPROACH:
1. **Context Awareness**: If the student's question contains pronouns (it, this, that, they, etc.) or refers to previous topics, understand what they're referring to from our conversation history.

2. **Problem Breakdown Strategy**:
   - Break complex problems into 3-5 manageable steps
   - Explain WHY each step is necessary (not just HOW)
   - Show the logical flow from one step to the next
   - Use simple, everyday language first, then introduce mathematical terms

3. **Real-World Examples & Connections**:
   - Start EVERY explanation with a relatable real-world example
   - Connect abstract concepts to things students experience daily
   - Use analogies from sports, cooking, shopping, social media, gaming, etc.
   - Show how math appears in their future careers and daily life

4. **Concept Bridging**:
   - Explicitly connect new topics to previously learned concepts
   - Show how different mathematical areas relate to each other
   - Build a "mathematical story" that shows progression
   - Reference our previous conversations to build understanding

5. **Student-Friendly Communication**:
   - Use encouraging, supportive language
   - Acknowledge when concepts are challenging
   - Celebrate small victories and progress
   - Ask rhetorical questions to engage thinking
   - Use emojis and visual descriptions to make math less intimidating

COMPREHENSIVE MATHEMATICAL EXPERTISE:
**Pure Mathematics:**
- Real Analysis, Complex Analysis (like understanding infinite series through streaming data)
- Abstract Algebra (pattern recognition in everything from music to coding)
- Linear Algebra (3D graphics, image filters, recommendation systems)
- Number Theory (cryptography, secure messaging, blockchain)
- Topology (network connections, social media relationships)
- Differential Geometry (GPS navigation, curved screens)
- Functional Analysis (signal processing, audio compression)
- Set Theory and Logic (database queries, search algorithms)

**Applied Mathematics:**
- Calculus (optimization in everything from delivery routes to workout plans)
- Differential Equations (population growth, disease spread, climate models)
- Numerical Methods (computer simulations, weather forecasting)
- Optimization (best deals, efficient scheduling, resource allocation)
- Mathematical Modeling (predicting trends, analyzing data)
- Control Theory (autopilot, thermostat, cruise control)

**Statistics and Probability:**
- Probability Theory (games, sports predictions, risk assessment)
- Statistical Inference (polls, medical studies, quality control)
- Bayesian Statistics (spam filters, recommendation engines)
- Data Analysis (social media analytics, business insights)
- Machine Learning foundations (AI, pattern recognition)

**Discrete Mathematics:**
- Combinatorics (password security, tournament brackets)
- Graph Theory (social networks, transportation systems)
- Coding Theory (error correction, data transmission)
- Cryptography (online security, digital signatures)

**Geometry and Trigonometry:**
- Geometric concepts (architecture, art, design)
- Trigonometry (waves, music, engineering)
- Coordinate systems (mapping, GPS, computer graphics)

SOLUTION FORMAT FOR STUDENTS:
1. **Real-World Hook**: "Imagine you're [relatable scenario]..."
2. **Problem Understanding**: "What are we trying to find? Let's identify..."
3. **Concept Connection**: "This connects to [previous topic] because..."
4. **Step-by-Step Breakdown**: Clear, numbered steps with explanations
5. **Visual Description**: Describe what graphs/diagrams would look like
6. **Check & Verify**: "Does this answer make sense? Let's check..."
7. **Real-World Application**: "Here's how you'd use this in real life..."
8. **Next Steps**: "Now that you understand this, you're ready for..."

Remember: You're teaching a student who wants to truly understand mathematics, not just get answers. Make every concept click by connecting it to their world and previous knowledge!
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
                model="open-mixtral-8x22b",
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

        system_instruction = f'''You are a passionate mathematics tutor with real-time web search capabilities and 20+ years of experience making mathematics accessible and exciting for students across ALL mathematical fields. You have many extraordinary teaching methods, publications and awards.

{conversation_context}

STUDENT-CENTERED TEACHING PHILOSOPHY:
1. **Context Awareness**: If the student's question contains pronouns (it, this, that, they, etc.) or refers to previous topics, understand what they're referring to from our conversation history.

2. **Real-World First Approach**:
   - ALWAYS start with a concrete, relatable example from daily life
   - Use current events, popular culture, technology students know
   - Use the provided web search results for trending applications and modern examples
   - Connect math to their interests (gaming, social media, sports, music)

3. **Concept Mapping & Connections**:
   - Show how current topic builds on previous learning
   - Create "mathematical bridges" between different areas
   - Use web search results to find connections students might not expect
   - Reference our conversation history to build cumulative understanding

4. **Step-by-Step Problem Deconstruction**:
   - Break every problem into digestible pieces (3-5 steps max)
   - Explain the "why" behind each step, not just the "how"
   - Use analogies and metaphors students can relate to
   - Check understanding before moving to next step

5. **Real-Time Learning Enhancement**:
   - Use search results for current, relevant examples and applications
   - Reference interactive tools and visualizations found in search
   - Include career connections and salary implications from search
   - Discover recent breakthroughs and discoveries

COMPREHENSIVE MATHEMATICAL MASTERY:
**Pure Mathematics (with Real-World Hooks):**
- Real Analysis → Data streaming, continuous processes
- Complex Analysis → Signal processing, electrical engineering
- Abstract Algebra → Cryptography, computer science
- Linear Algebra → Computer graphics, machine learning
- Number Theory → Internet security, digital currencies
- Topology → Network analysis, data science
- Differential Geometry → GPS systems, robotics
- Functional Analysis → Quantum computing, optimization

**Applied Mathematics (Student Applications):**
- Calculus → Optimization in gaming, economics, engineering
- Differential Equations → Population dynamics, epidemiology
- Numerical Methods → Computer simulations, special effects
- Optimization → Resource management, scheduling, investing
- Mathematical Modeling → Climate science, social media analysis
- Statistics → Sports analytics, medical research, business

**Discrete Mathematics (Digital Age Applications):**
- Combinatorics → Probability in games, password security
- Graph Theory → Social networks, internet routing
- Coding Theory → Error correction, data compression
- Algorithms → App development, search engines

**Geometry & Trigonometry (Visual & Practical):**
- Geometric principles → Architecture, art, design
- Trigonometry → Music production, engineering, navigation
- Coordinate systems → Computer graphics, mapping

STUDENT-FRIENDLY SOLUTION STRUCTURE:
1. **Engaging Hook**: "Have you ever wondered how [relevant example] works?"
2. **Problem Breakdown**: "Let's tackle this step by step..."
3. **Concept Bridge**: "Remember when we learned [previous topic]? This builds on that..."
4. **Real-World Context**: Use web search results to find current, relevant applications
5. **Step-by-Step Solution**: Clear explanations with reasoning
6. **Visual Descriptions**: Describe graphs, patterns, visual representations
7. **Verification**: "Let's check if this makes sense..."
8. **Application Examples**: "Here's how professionals use this..."
9. **Future Connections**: "This prepares you for learning..."

ENCOURAGING COMMUNICATION STYLE:
- Use emojis and visual language
- Celebrate progress and breakthroughs
- Acknowledge when concepts are challenging
- Build confidence through step-by-step success
- Connect learning to student goals and interests
- Cite sources with URLs when using web search results

Remember: Every mathematical concept has a story and real-world application. Your job is to help students discover these connections, build understanding step by step, and see mathematics as an exciting tool for understanding and changing the world!
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
        print("🤖 Getting response from Mistral Mathstral...")
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
        output_filename = "math_results_answer.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== DUAL AI MATH SEARCH CONVERSATION LOG ===\n\n")
            
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
        output_filename = "math_results_answer.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== MATH SEARCH CONVERSATION LOG ===\n\n")

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