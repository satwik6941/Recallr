import asyncio
# from crawl4ai import *
# from crawl4ai import Crawl4AI
from google import genai
from google.genai import types
import os
import dotenv as env

env.load_dotenv()

user_query = input("Enter your question: ")

client = genai.Client(api_key=os.getenv("GEMINI_2_API_KEY"))
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool],
    system_instruction=f'''
    You are an expert and helpful coding assistant who has 20+ years of hands on experience in each and every programming language and have a proven track record of solving complex coding problems and complex projects.
    Your task is to help the user with their coding problems and doubts. Provide them with the best possible solution by explain the user in simple terms and concise manner.
    Here is the user query: {user_query}

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

    Secondarily, then search the whole web for the best answers, solutions and explanations

    Then combine the results and provide a comprehensive answer to the user query.

    OUTPUT:
    IMPORTANT THING: When you start generating the content, always start by explaining the user query with a simple real life example and then provide the solution such that the user can connect the dots (understand the problem and solution).
    1. Provide a concise, on the point answer and clear answer to the user query.
    2. If the answer involves any code snippets, provide the answer of the user, then display the code snippets in a well formatted manner and give a short and easy to understand explanation of the code.
    3. If the answer involves any complex concepts, provide a simple and easy to understand explanation of the concept and try to explain it with examples in simple terms.
    4. If the user query seems like a doubt or checking his understanding, provide a clear and concise answer to the doubt and explain the concept in simple terms.
'''
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=user_query,
    config=config
)

print(response.text)

# # Initialize crawler and Gemini LLM
# crawler = Crawl4AI()
# llm = genai.GenerativeModel("gemini-2.0-flash")

# # Step 1: Crawl the Stack Overflow page
# url = "https://stackoverflow.com/questions/123456/how-do-i-resolve-git-merge-conflicts"
# html_data = crawler.crawl(url)  # This returns structured HTML metadata

# # Step 2: Extract question and top answer
# question_text = html_data.get("question", {}).get("text", "No question found")
# top_answer = html_data.get("answers", [{}])[0].get("text", "No answer found")

# # Step 3: Enrich with Gemini
# enrich_prompt = f"""Here's a Stack Overflow question and its top answer:
# Question: {question_text}
# Answer: {top_answer}
# Can you summarize the solution, clarify any ambiguities, and suggest best practices for resolving merge conflicts?"""

# response = llm.generate_content(enrich_prompt)

# # Step 4: Display results
# print("❓ Question:", question_text)
# print("✅ Top Answer:", top_answer[:500], "...")  # Truncate long answers for display
# print("🧠 Gemini Summary:", response.text)
