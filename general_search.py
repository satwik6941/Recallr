import os
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

# Initialize OpenAI and Tavily clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def answer_query_with_google_search(query: str) -> str:
    """Answer query using Tavily web search and OpenAI for processing"""
    try:
        # Perform Tavily search
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        # Format search results for context
        search_context = ""
        if search_results and "results" in search_results:
            for idx, result in enumerate(search_results["results"], 1):
                search_context += f"\n{idx}. {result.get('title', 'No title')}\n"
                search_context += f"   URL: {result.get('url', 'No URL')}\n"
                search_context += f"   Content: {result.get('content', 'No content')}\n"

        # Use OpenAI to process and answer based on search results
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant with 20+ years of experience in answering any kind of question by using web search to provide relevant, concise, and accurate information.
                    Your task is to answer the user query using the provided web search results from multiple sources (websites, blog pages, research papers websites etc).

                    Follow these guidelines while answering:
                    1. Analyze the web search results provided to gather information.
                    2. Provide concise and accurate answers based on the information gathered from web search.
                    3. Cite your sources in the answer with URLs.
                    4. If the information is not sufficient in the search results, supplement with your knowledge but clearly indicate this.
                    5. IMPORTANT: Always explain the answer to the user query in a simple and easy to understand manner.
                    6. Always ask a follow up question to engage the user and clarify their doubts or queries.

                    Examples:
                    - User: I want to know about latest advancements in AI.
                        Assistant: (Analyzes web search results to find recent articles and papers on AI advancements. Then summarizes the findings in simple terms, gives detailed explanations and cites sources with URLs.)
                    - User: I am doing a project on Robotics and I need recent publications on ZMP (zero mobility point)?
                        Assistant: (Uses web search results to find recent publications and articles on ZMP from research paper websites, academic blog sites, etc. Then summarizes the findings, gives detailed explanations, explains the concept in simple terms and cites sources with URLs.)
                    """
                },
                {
                    "role": "user",
                    "content": f"User question: {query}\n\nWeb search results:\n{search_context}\n\nPlease provide a comprehensive answer based on these search results."
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error performing web search: {str(e)}"

