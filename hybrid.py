from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import asyncio
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Validate API keys
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is required")
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/e5-large-v2", 
    cache_folder="./cache",
    device="cuda"  # Use CUDA for GPU acceleration
)
Settings.llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize LLM for agent
Settings.llm1 = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=360.0,
)

# Try to load existing storage, otherwise create new indexes
storage_dir = "storage"
try:
    print("Trying to load existing indexes from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    vector_index = load_index_from_storage(storage_context, index_id="vector")
    keyword_index = load_index_from_storage(storage_context, index_id="keyword")
    print("Loaded existing indexes from storage.")
    
    # Still need to load documents for BM25 (not stored)
    documents = SimpleDirectoryReader("data").load_data()
    parser = SimpleNodeParser.from_defaults(
        chunk_size=1000,
        chunk_overlap=200,
    )
    nodes = parser.get_nodes_from_documents(documents)
    
except Exception as e:
    print(f"Could not load from storage ({e}), creating new indexes...")
    
    print("Loading documents...")
    try:
        documents = SimpleDirectoryReader("data").load_data()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        raise

    # Parse documents into nodes for BM25 with proper chunking
    print("Parsing documents into nodes with chunking...")
    parser = SimpleNodeParser.from_defaults(
        chunk_size=1000,  # Smaller chunks for better retrieval
        chunk_overlap=200,  # Some overlap to maintain context
    )
    nodes = parser.get_nodes_from_documents(documents)

    print("Creating indexes...")
    # Create vector store index from nodes (with chunking)
    vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
    
    # Create keyword table index from nodes
    keyword_index = SimpleKeywordTableIndex(nodes)
    
    # Save indexes to storage
    print("Saving indexes to storage...")
    vector_index.set_index_id("vector")
    keyword_index.set_index_id("keyword")
    vector_index.storage_context.persist(persist_dir=storage_dir)
    keyword_index.storage_context.persist(persist_dir=storage_dir)

print("Creating retrievers...")
# Create retrievers with smaller top_k to reduce context size
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=3)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

# Define a custom hybrid retriever class
class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self._retrievers = retrievers
        super().__init__()

    def _retrieve(self, query_bundle):
        # Retrieve results from each retriever
        all_results = []
        for retriever in self._retrievers:
            try:
                results = retriever.retrieve(query_bundle)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Retriever failed: {e}")
                continue

        # Create a dictionary to store unique nodes and their highest scores
        unique_nodes = {}
        for res in all_results:
            node_id = res.node.node_id
            if node_id not in unique_nodes:
                unique_nodes[node_id] = res
            else:
                # If node already exists, update with the higher score
                if hasattr(res, 'score') and hasattr(unique_nodes[node_id], 'score'):
                    if res.score and (not unique_nodes[node_id].score or res.score > unique_nodes[node_id].score):
                        unique_nodes[node_id] = res
        
        # Return the unique nodes as a list
        return list(unique_nodes.values())

# Instantiate the hybrid retriever
hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever, bm25_retriever])

# Create hybrid query engine
hybrid_query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=Settings.llm,
)

print("Creating query engine...")

# Conversation context to maintain chat history
conversation_history = []

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
            'num': 3  # Number of results to return
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

# Create FunctionTool objects
google_search_tool = FunctionTool.from_defaults(
    fn=google_search,
    name="google_search",
    description="Search the web always using Google Custom Search API. Use this always for current events, general knowledge, or any web search queries."
)

# Create ReActAgent with function calling capabilities
agent = ReActAgent.from_tools(
    [google_search_tool],
    llm=Settings.llm1,
    verbose=True,
    system_prompt="""You are a helpful AI assistant and expert in explain concepts, answering questions, and providing information for students.
    You have been give access to the google search tool to find information on the web. 

For each user query:
- Analyze what the user is asking for
- Search the web using the google_search tool always when the user asks for information.
- Return the results in a clear,simple and concise manner
- Always provide helpful and detailed responses

Be conversational and helpful!"""
)

async def search_documents_with_context(query: str) -> str:
    """Search through documents using hybrid retrieval with conversation context
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        str: The response from the document search
    """
    try:
        # Build context-aware query
        if conversation_history:
            # Include recent conversation history for context
            recent_history = conversation_history[-4:]  # Last 4 exchanges
            context_prompt = f"""
Previous conversation:
{chr(10).join([f"User: {h['user']}" + chr(10) + f"Assistant: {h['assistant']}" for h in recent_history])}

Current question: {query}

Please answer the current question, considering the conversation context above.
"""
        else:
            context_prompt = query
            
        response = await hybrid_query_engine.aquery(context_prompt)
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

async def get_web_search_results(query: str) -> str:
    """Get web search results using Groq LLM and Google Custom Search
    
    Args:
        query (str): The search query
        
    Returns:
        str: Processed web search results
    """
    try:
        # Get raw web search results
        web_results = google_search(query)
        
        # Use Groq LLM to process and summarize the web results
        groq_response = await agent.achat(f"Based on these web search results, provide a comprehensive answer to the query '{query}':\n\n{web_results}")
        
        return str(groq_response)
    except Exception as e:
        return f"Error getting web search results: {str(e)}"

async def synthesize_final_answer(query: str, rag_result: str, web_result: str) -> str:
    """Synthesize final answer from RAG and web search results"""
    try:
        synthesis_prompt = f"""
You are an expert and helpful AI assistant in helping students understanding concepts. I have two sources of information to answer the user's query: "{query}"

Source 1 - Document Search (RAG):
{rag_result}

Source 2 - Web Search:
{web_result}

Please provide a comprehensive, accurate answer by:
1. Combining information from both sources
2. Highlighting any complementary information
3. Noting any contradictions and explaining them
4. Providing a well-structured, coherent response
5. Citing which source provided specific information when relevant
6. IMPORTANT: Explain the answer in a way that is easy to understand for students and use simple terms

Give me the best possible answer using both sources of information.
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
            
        return str(final_response)
    except Exception as e:
        return f"Error synthesizing final answer: {str(e)}"

async def main():
    print("Hybrid AI Assistant ready! Combining document search (RAG) with web search.")
    print("The system will remember our conversation context.")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("Processing...")
            print("🔍 Searching documents...")
            
            # Get RAG results using Gemini
            rag_result = await search_documents_with_context(user_query)
            print("✅ Document search complete")
            
            print("🌐 Searching web...")
            # Get web search results using Groq
            web_result = await get_web_search_results(user_query)
            print("✅ Web search complete")
            
            print("🤖 Synthesizing final answer...")
            # Synthesize final answer using Gemini
            final_answer = await synthesize_final_answer(user_query, rag_result, web_result)
            
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
                result = await search_documents_with_context(user_query)
                print(f"Fallback result: {result}")
            except Exception as fallback_error:
                print(f"All methods failed: {fallback_error}")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())