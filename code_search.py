import asyncio
from google import genai
from google.genai import types
import os
import dotenv as env
from typing import List, Dict, Any
import asyncio
import os
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import json

env.load_dotenv()

# Conversation context to maintain chat history
conversation_history = []

client = genai.Client(api_key=os.getenv("GEMINI_2_API_KEY"))

def analyze_query_context_dependency(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
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
                recent_topics.append(exchange.get('topic', ''))
        
        return {
            'needs_context': needs_context,
            'recent_topics': [topic for topic in recent_topics if topic],
            'context_indicators_found': [indicator for indicator in context_indicators if indicator in query.lower()]
        }
        
    except Exception as e:
        return {
            'needs_context': False,
            'recent_topics': [],
            'context_indicators_found': [],
            'error': str(e)
        }

def build_context_aware_query(user_query: str, conversation_history: List[Dict]) -> str:
    """Build a context-aware query that includes conversation history"""
    
    if not conversation_history:
        return user_query
    
    # Build context from recent conversation
    context_parts = []
    for exchange in conversation_history[-3:]:  # Last 3 exchanges
        context_parts.append(f"Previous Q: {exchange['user']}")
        context_parts.append(f"Previous A: {exchange['assistant'][:200]}...")
    
    context_str = "\n".join(context_parts)
    
    enhanced_query = f"""
Previous Conversation Context:
{context_str}

Current Question: {user_query}

Please provide a comprehensive answer that considers the conversation context and resolves any references to previous topics.
"""
    
    return enhanced_query

def build_conversation_context_for_system(conversation_history: List[Dict]) -> str:
    """Build conversation context string for system instruction"""
    
    if not conversation_history:
        return ""
    
    conversation_context = "\n💬 **Previous Conversation:**\n"
    for i, exchange in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
        conversation_context += f"{i}. Student asked: \"{exchange['user']}\"\n"
        conversation_context += f"   I responded: {exchange['assistant'][:200]}{'...' if len(exchange['assistant']) > 200 else ''}\n\n"
    
    return conversation_context

def process_query_with_context(user_query: str) -> str:
    """Process the query with conversation context using Gemini"""
    
    # Analyze context dependency
    context_analysis = analyze_query_context_dependency(user_query, conversation_history)
    
    if context_analysis.get('needs_context') and conversation_history:
        print(f"Detected context dependency: {', '.join(context_analysis.get('context_indicators_found', []))}")
        enhanced_query = build_context_aware_query(user_query, conversation_history)
    else:
        enhanced_query = user_query
    
    # Setup grounding tool for web search
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    
    # Build conversation context for system instruction
    conversation_context = build_conversation_context_for_system(conversation_history)
    
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        system_instruction=f'''
You are an expert and helpful coding assistant who has 20+ years of hands-on experience in every programming language and has a proven track record of solving complex coding problems and projects.

{conversation_context}

Your task is to help the user with their coding problems and doubts. Provide them with the best possible solution by explaining to the user in simple terms and in a concise manner.

IMPORTANT CONTEXT HANDLING:
1. **If the user's question contains pronouns like "it", "this", "that", "they", etc., make sure you understand what they're referring to from our previous conversation**
2. **Reference previous topics we discussed when relevant (e.g., "As we discussed earlier about...")**
3. **Build on our previous conversation naturally**
4. **Make connections**: If this question relates to something we talked about before, explicitly mention that connection

Keep the user query as the context and search for relevant code snippets, examples, and explanations from the web.
Primarily search for answers in the following resources and websites:
1. https://stackoverflow.com/
2. https://www.quora.com/topic/Computer-Programming
3. https://stackexchange.com/
4. https://www.reddit.com/r/programming/
5. https://www.geeksforgeeks.org/
6. https://www.codeproject.com/
7. https://coderanch.com/
8. https://developers.google.com/community/
9. All open source code repositories like GitHub, GitLab, Bitbucket, etc.
10. All official documentation of programming languages, frameworks, and libraries.
11. All relevant blogs and articles related to programming.

Secondarily, then search the whole web for the best answers, solutions and explanations.

Then combine the results and provide a comprehensive answer to the user query.

OUTPUT:
IMPORTANT THING: When you start generating the content, always start by explaining the user query with a simple real-life example and then provide the solution such that the user can connect the dots (understand the problem and solution).

1. **Acknowledge the conversation context** if this is a follow-up question
2. Provide a concise, on-point answer and clear answer to the user query.
3. If the answer involves any code snippets, provide the answer of the user, then display the code snippets in a well-formatted manner and give a short and easy-to-understand explanation of the code.
4. If the answer involves any complex concepts, provide a simple and easy-to-understand explanation of the concept and try to explain it with examples in simple terms.
5. If the user query seems like a doubt or checking their understanding, provide a clear and concise answer to the doubt and explain the concept in simple terms.
6. **Reference previous conversation topics when relevant**

Remember: This is a continuing conversation, not a standalone question. Be warm and encouraging in your tone.
'''
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=enhanced_query,
            config=config
        )
        
        # Extract the main topic for conversation history
        topic = user_query[:50] + "..." if len(user_query) > 50 else user_query
        
        # Update conversation history
        conversation_history.append({
            "user": user_query,
            "assistant": response.text,
            "topic": topic
        })
        
        # Keep only last 10 exchanges to prevent context overflow
        if len(conversation_history) > 10:
            conversation_history.pop(0)
        
        # Save Gemini's answer to file
        output_filename = "code_results_answer.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== GEMINI'S ANSWER FOR: '{user_query}' ===\n\n")
            f.write(response.text)
            f.write("\n\n" + "="*60 + "\n\n")
        
        print(f"✅ Gemini's answer saved to '{output_filename}'")
        
        # Now get additional answers using crawl4ai
        asyncio.run(get_additional_answers_with_crawl4ai(user_query, output_filename))
        
        return response.text
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

async def get_additional_answers_with_crawl4ai(user_query: str, output_filename: str):
    """Get additional answers using crawl4ai and append to the same file"""
    
    print("\n🔍 Getting additional answers using crawl4ai...")
    
    # Get top websites from Gemini
    top_urls = await get_top_websites_from_gemini(user_query)
    
    if not top_urls:
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("=== CRAWL4AI RESULTS ===\n\n")
            f.write("No relevant websites found for additional crawling.\n\n")
        return
    
    # Append crawl4ai results header
    with open(output_filename, "a", encoding="utf-8") as f:
        f.write("=== CRAWL4AI ADDITIONAL ANSWERS ===\n\n")
        f.write(f"Found {len(top_urls)} websites to crawl for additional information...\n\n")
    
    # Crawl websites and append results
    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(top_urls[:5], 1):  # Limit to 5 URLs for performance
            print(f"Crawling [{i}/5]: {url}")
            try:
                crawler_config = CrawlerRunConfig(
                    cache_mode="BYPASS",
                    timeout=30
                )
                result = await crawler.arun(url=url, config=crawler_config)
                
                with open(output_filename, "a", encoding="utf-8") as f:
                    f.write(f"--- Website {i}: {url} ---\n")
                    
                    if result.success and result.markdown.fit_markdown:
                        content = result.markdown.fit_markdown[:3000]  # Limit content
                        f.write(f"✅ Successfully crawled\n\n{content}\n\n")
                        print(f"✅ Successfully crawled {url}")
                    else:
                        error_msg = result.error_message if result.error_message else "No substantial content found"
                        f.write(f"❌ Failed to crawl: {error_msg}\n\n")
                        print(f"❌ Failed to crawl {url}")
                        
            except Exception as e:
                with open(output_filename, "a", encoding="utf-8") as f:
                    f.write(f"❌ Error crawling {url}: {str(e)}\n\n")
                print(f"❌ Error crawling {url}: {str(e)}")
    
    # Add final footer
    with open(output_filename, "a", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("END OF RESULTS\n")
    
    print(f"✅ All results saved to '{output_filename}'")

def print_conversation_summary():
    """Print a summary of the current conversation"""
    if not conversation_history:
        print("No conversation history yet.")
        return
    
    print(f" **Conversation Summary** ({len(conversation_history)} exchanges):")
    for i, exchange in enumerate(conversation_history[-5:], 1):  # Show last 5
        print(f"  {i}. User: {exchange['user'][:60]}{'...' if len(exchange['user']) > 60 else ''}")
        print(f"     Bot: {exchange['assistant'][:80]}{'...' if len(exchange['assistant']) > 80 else ''}")
    print()

def clear_conversation_history():
    """Clear the conversation history"""
    global conversation_history
    conversation_history.clear()
    print("🧹 Conversation history cleared!")

async def get_top_websites_from_gemini(user_query: str) -> list[str]:
    try:
        prompt = f"""
        Based on the following query, provide a list of up to 10 highly relevant and authoritative websites or resources (URLs only) where one could find comprehensive information.
        Format your response as a JSON array of strings, where each string is a valid URL.
        Only include the URLs, no other text or explanation.

        Query: "{user_query}"

        Example output:
        ["https://www.example.com/resource1", "https://www.anothersite.org/info"]
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response and response.text:
            try:
                urls = json.loads(response.text)
                # Filter to ensure they are valid-looking URLs and limit to 10
                valid_urls = [url for url in urls if url.startswith("http")][:10]
                return valid_urls
            except json.JSONDecodeError:
                print(f"Gemini did not return valid JSON for website list: {response.text}")
                return []
        else:
            print("Gemini did not provide website suggestions.")
            return []
    except Exception as e:
        print(f"Error getting websites from Gemini: {e}")
        return []

def main():
    print("Enhanced Code Search Assistant with Conversation Context")
    print("The system will remember our conversation and resolve references like 'it', 'this', etc.")
    print("\n Tips:")
    print("- Ask follow-up questions using 'it', 'this', 'that' to test context resolution")
    print("- Type 'summary' to see conversation history")
    print("- Type 'clear' to clear conversation history")
    print("- The system will automatically detect when you're referring to previous topics")
    print("- Type 'quit' to exit")
    
    while True:
        try:
            user_query = input("\nEnter your coding question (or 'quit'/'summary'/'clear'): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! Happy coding!")
                break
            elif user_query.lower() == 'summary':
                print_conversation_summary()
                continue
            elif user_query.lower() == 'clear':
                clear_conversation_history()
                continue
            elif not user_query.strip():
                print("Please enter a valid question.")
                continue
            
            print("🔍 Processing your query with conversation context...")
            
            # Analyze context dependency before processing
            context_analysis = analyze_query_context_dependency(user_query, conversation_history)
            if context_analysis.get('needs_context') and conversation_history:
                print(f"Detected context dependency: {', '.join(context_analysis.get('context_indicators_found', []))}")
                if context_analysis.get('recent_topics'):
                    print(f"Recent topics: {', '.join(context_analysis['recent_topics'][:3])}")
            
            # Process query with context
            result = process_query_with_context(user_query)
            
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(result)
            print(f"{'='*60}")
            print(f"Conversation history: {len(conversation_history)} exchanges stored")
            
        except KeyboardInterrupt:
            print("\n Goodbye! Happy coding!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()