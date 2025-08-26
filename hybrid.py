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
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
import os
import requests
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Validate API keys
if not os.getenv("GEMINI_1_API_KEY"):
    raise ValueError("GEMINI_1_API_KEY environment variable is required")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is required")

groq_llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=360.0,
    temperature=0.1,  # Lower 
    max_tokens=2000,  # Limit response length
)

# Settings control global defaults - these will be set when RAG is initialized
def initialize_gemini_settings():
    """Initialize Gemini settings for RAG"""
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=os.getenv("GEMINI_1_API_KEY")
    )

    Settings.llm = GoogleGenAI(
        model="models/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_1_API_KEY")
    )

# Global variables to store the initialized components
vector_index = None
keyword_index = None
hybrid_query_engine = None
nodes = None

# Message management functions for cleaner conversation handling
def add_user_message(messages: List[Dict], user_prompt: str):
    """Add user message to conversation messages"""
    messages.append({
        "role": "user",
        "content": user_prompt
    })

def add_ai_message(messages: List[Dict], ai_response: str):
    """Add AI response to conversation messages"""
    messages.append({
        "role": "assistant", 
        "content": ai_response
    })

def get_conversation_context_from_messages(messages: List[Dict], limit: int = 12) -> str:
    """Build conversation context string from messages list
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        limit: Maximum number of recent messages to include
    """
    if not messages:
        return ""
    
    # Get recent messages (limit to avoid context overflow)
    recent_messages = messages[-limit:] if len(messages) > limit else messages
    
    context = "Previous conversation context:\n"
    for msg in recent_messages:
        role = msg["role"].title()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        context += f"{role}: {content}\n"
    
    return context + "\n"

def analyze_query_context_dependency_simple(query: str, messages: List[Dict] = None) -> Dict[str, Any]:
    """Analyze if the query depends on conversation context using simple message structure"""
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
        'context_indicators_found': found_indicators,
        'has_conversation': bool(messages and len(messages) > 0)
    }

async def search_documents_with_messages(query: str, messages: List[Dict] = None) -> str:
    """Search through documents using hybrid retrieval with message-based conversation context
    
    Args:
        query (str): The search query to find relevant documents
        messages (List[Dict]): Message list with 'role' and 'content' for context
        
    Returns:
        str: The response from the document search
    """
    global hybrid_query_engine
    
    # Initialize if not already done
    if hybrid_query_engine is None:
        initialize_rag_pipeline()
    
    try:
        # Build context-aware query using message structure
        if messages and len(messages) > 0:
            # Get conversation context from messages
            conversation_context = get_conversation_context_from_messages(messages, limit=8)
            
            # First, let's resolve any references in the current query
            resolution_prompt = f"""
Based on this recent conversation:
{conversation_context}

The user is now asking: "{query}"

If this question contains pronouns (it, this, that, they, etc.) or refers to concepts from the previous conversation, please reformulate the query to be more specific and complete for document search. Include the specific topics, concepts, or terms being referenced.

If the question is already clear and specific, return it as is.

Reformulated query:"""
            
            resolved_response = await Settings.llm.acomplete(resolution_prompt)
            resolved_query = str(resolved_response).strip()
            
            # Use the resolved query if it's meaningful, otherwise use original
            search_query = resolved_query if len(resolved_query) > 10 and "reformulated" not in resolved_query.lower() else query
            
            # Now build the context for the final response
            context_prompt = f"""
{conversation_context}

Current question: {query}
Search query used: {search_query}

Please answer the current question considering:
1. The conversation context above
2. How this question relates to previous topics discussed
3. Any references to earlier concepts or answers
4. Provide a comprehensive answer that builds on our conversation

Answer the question: {query}
"""
        else:
            context_prompt = query
            
        response = await hybrid_query_engine.aquery(context_prompt)
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

async def get_web_search_with_messages(query: str, messages: List[Dict] = None) -> str:
    """Get web search results using message-based conversation context
    
    Args:
        query (str): The search query
        messages (List[Dict]): Message list for context
        
    Returns:
        str: Processed web search results
    """
    try:
        # Build context-aware query for web search using messages
        if messages and len(messages) > 0:
            # Get recent conversation context from messages
            conversation_context = get_conversation_context_from_messages(messages, limit=6)
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{conversation_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Get raw web search results directly
        web_results = google_search(search_query)
        
        # Use Groq LLM directly to process the results with context
        if messages and len(messages) > 0:
            conversation_context = get_conversation_context_from_messages(messages, limit=6)
            process_prompt = f"""You are answering a follow-up question in an ongoing conversation.

Recent conversation context:
{conversation_context}

Current question: "{query}"
Search query used: "{search_query}"

Web search results:
{web_results}

Please provide a comprehensive answer that:
1. Considers the conversation context
2. Addresses the current question directly
3. Resolves any references to previous topics
4. Uses the web search results to provide accurate information

Answer:"""
        else:
            process_prompt = f"""Based on these web search results, provide a comprehensive answer to the query '{query}':

{web_results}

Please provide a clear, concise answer based on the search results."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        return str(groq_response)
    except Exception as e:
        return f"Error getting web search results: {str(e)}"

async def get_youtube_search_with_messages(query: str, messages: List[Dict] = None) -> str:
    """Get YouTube search results using message-based conversation context
    
    Args:
        query (str): The search query
        messages (List[Dict]): Message list for context
        
    Returns:
        str: Processed YouTube search results
    """
    try:
        print(f"Calling external YouTube module...")
        
        # Build context-aware query for YouTube search using messages
        if messages and len(messages) > 0:
            # Get recent conversation context from messages
            conversation_context = get_conversation_context_from_messages(messages, limit=6)
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{conversation_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable for YouTube videos. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Import and use the external youtube module
        sys.path.append(os.path.dirname(__file__))
        
        from youtube import process_youtube_query
        
        # Get YouTube search results from external module
        youtube_results = await process_youtube_query(search_query)
        
        # Check if we got results
        if "No YouTube videos found" in youtube_results or "Error" in youtube_results:
            return youtube_results
        
        # Use Groq LLM to process and summarize the results while preserving URLs
        if messages and len(messages) > 0:
            conversation_context = get_conversation_context_from_messages(messages, limit=6)
            process_prompt = f"""You are helping with a follow-up question in an ongoing conversation.

Recent conversation context:
{conversation_context}

Current question: "{query}"
Search query used: "{search_query}"

YouTube search results:
{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning, considering the conversation context
4. Keep all the URLs exactly as provided in the original results
5. Address how these videos relate to the current question and previous conversation

Please provide a clear summary highlighting the most relevant videos for learning:"""
        else:
            process_prompt = f"""Based on these YouTube search results, provide a summary of the available educational videos for the query '{query}':

{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning
4. Keep all the URLs exactly as provided in the original results

Please provide a clear summary highlighting the most relevant videos for learning."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        # Combine original results with AI summary, ensuring URLs are preserved
        final_result = f"{youtube_results}\n\n**AI Summary:**\n{str(groq_response)}\n\n**Direct Video Links:**\n"
        
        # Extract and add direct links from original results to ensure they're visible
        lines = youtube_results.split('\n')
        for line in lines:
            if 'URL:' in line or 'https://www.youtube.com/watch' in line:
                final_result += f"{line}\n"
        
        return final_result
        
    except Exception as e:
        return f"Error getting YouTube search results: {str(e)}"

# Legacy function for backward compatibility
async def analyze_query_context_dependency(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
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
            for exchange in conversation_history:
                # Extract key topics from recent exchanges
                topics_prompt = f"""
Extract the main topics/concepts from this conversation exchange:
User: {exchange['user']}
Assistant: {exchange['assistant'][:200]}...

List the key topics/concepts (maximum 3):"""
                
                topics_response = await groq_llm.acomplete(topics_prompt)
                topics = str(topics_response).strip().split('\n')
                recent_topics.extend([topic.strip('- ').strip() for topic in topics if topic.strip()])
        
        return {
            'needs_context': needs_context,
            'recent_topics': recent_topics,  # Keep top 5 recent topics
            'context_indicators_found': [indicator for indicator in context_indicators if indicator in query.lower()]
        }
        
    except Exception as e:
        return {
            'needs_context': False,
            'recent_topics': [],
            'context_indicators_found': [],
            'error': str(e)
        }

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
            'num': 5  # Number of results to return
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

async def get_web_search_results(query: str, conversation_history: List[Dict] = None) -> str:
    """Get web search results using direct Google search with conversation context
    
    Args:
        query (str): The search query
        conversation_history (List[Dict]): Previous conversation history for context
        
    Returns:
        str: Processed web search results
    """
    try:
        # Build context-aware query for web search
        if conversation_history and len(conversation_history) > 0:
            # Get recent conversation context
            recent_context = ""
            for exchange in conversation_history:  # Last 3 exchanges
                recent_context += f"User asked: {exchange['user']}\nAssistant answered: {exchange['assistant'][:300]}...\n\n"
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{recent_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Get raw web search results directly
        web_results = google_search(search_query)
        
        # Use Groq LLM directly to process the results with context
        if conversation_history and len(conversation_history) > 0:
            process_prompt = f"""You are answering a follow-up question in an ongoing conversation.

Recent conversation context:
{recent_context}

Current question: "{query}"
Search query used: "{search_query}"

Web search results:
{web_results}

Please provide a comprehensive answer that:
1. Considers the conversation context
2. Addresses the current question directly
3. Resolves any references to previous topics
4. Uses the web search results to provide accurate information

Answer:"""
        else:
            process_prompt = f"""Based on these web search results, provide a comprehensive answer to the query '{query}':

{web_results}

Please provide a clear, concise answer based on the search results."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        return str(groq_response)
    except Exception as e:
        return f"Error getting web search results: {str(e)}"

async def get_youtube_search_results(query: str, conversation_history: List[Dict] = None) -> str:
    """Get YouTube search results using external youtube.py module with conversation context
    
    Args:
        query (str): The search query
        conversation_history (List[Dict]): Previous conversation history for context
        
    Returns:
        str: Processed YouTube search results
    """
    try:
        print(f"Calling external YouTube module...")
        
        # Build context-aware query for YouTube search
        if conversation_history and len(conversation_history) > 0:
            # Get recent conversation context
            recent_context = ""
            for exchange in conversation_history: 
                recent_context += f"User asked: {exchange['user']}\nAssistant answered: {exchange['assistant'][:300]}...\n\n"
            
            # Create an enhanced query that resolves references
            context_prompt = f"""
Based on this recent conversation:
{recent_context}

The user is now asking: "{query}"

If the user's question contains pronouns like "it", "this", "that", "they", etc., or refers to something from the previous conversation, please rephrase the query to be more specific and searchable for YouTube videos. Otherwise, keep the original query.

Provide only the reformulated search query:"""
            
            reformulated_response = await groq_llm.acomplete(context_prompt)
            search_query = str(reformulated_response).strip()
            
            # If the reformulated query is too short or unclear, use original
            if len(search_query) < 10 or "reformulated" in search_query.lower():
                search_query = query
        else:
            search_query = query
        
        # Import and use the external youtube module
        sys.path.append(os.path.dirname(__file__))
        
        from youtube import process_youtube_query
        
        # Get YouTube search results from external module
        youtube_results = await process_youtube_query(search_query)
        
        # Check if we got results
        if "No YouTube videos found" in youtube_results or "Error" in youtube_results:
            return youtube_results
        
        # Use Groq LLM to process and summarize the results while preserving URLs
        if conversation_history and len(conversation_history) > 0:
            process_prompt = f"""You are helping with a follow-up question in an ongoing conversation.

Recent conversation context:
{recent_context}

Current question: "{query}"
Search query used: "{search_query}"

YouTube search results:
{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning, considering the conversation context
4. Keep all the URLs exactly as provided in the original results
5. Address how these videos relate to the current question and previous conversation

Please provide a clear summary highlighting the most relevant videos for learning:"""
        else:
            process_prompt = f"""Based on these YouTube search results, provide a summary of the available educational videos for the query '{query}':

{youtube_results}

IMPORTANT: 
1. Always include the exact YouTube URLs from the original results
2. Format each video recommendation as: "Video Title" - URL: [exact URL]
3. Recommend the top 3 videos for learning
4. Keep all the URLs exactly as provided in the original results

Please provide a clear summary highlighting the most relevant videos for learning."""
        
        groq_response = await groq_llm.acomplete(process_prompt)
        
        # Combine original results with AI summary, ensuring URLs are preserved
        final_result = f"{youtube_results}\n\n**AI Summary:**\n{str(groq_response)}\n\n**Direct Video Links:**\n"
        
        # Extract and add direct links from original results to ensure they're visible
        lines = youtube_results.split('\n')
        for line in lines:
            if 'URL:' in line or 'https://www.youtube.com/watch' in line:
                final_result += f"{line}\n"
        
        return final_result
        
    except Exception as e:
        return f"Error getting YouTube search results: {str(e)}"

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with hybrid retrieval"""
    global vector_index, keyword_index, hybrid_query_engine, nodes
    
    # Initialize Gemini settings first
    initialize_gemini_settings()
    
    # Try to load existing storage, otherwise create new indexes
    storage_dir = "storage"
    
    # First, always load documents as they're needed for BM25
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
    
    # Try to load existing indexes
    try:
        print("Trying to load existing indexes from storage...")
        
        # Check if storage directory exists and has the required files
        if os.path.exists(storage_dir) and os.path.exists(os.path.join(storage_dir, "index_store.json")):
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            
            # Try to load with specific index IDs first
            vector_index = None
            keyword_index = None
            
            try:
                vector_index = load_index_from_storage(storage_context, index_id="vector")
                print("✅ Vector index loaded successfully")
            except Exception as vector_error:
                print(f"⚠️ Failed to load vector index with ID: {vector_error}")
                try:
                    # Try loading without index ID (fallback)
                    vector_index = load_index_from_storage(storage_context)
                    print("✅ Vector index loaded without ID")
                except Exception as fallback_error:
                    print(f"❌ Failed to load vector index completely: {fallback_error}")
                    vector_index = None
            
            try:
                keyword_index = load_index_from_storage(storage_context, index_id="keyword")
                print("✅ Keyword index loaded successfully")
            except Exception as keyword_error:
                print(f"⚠️ Failed to load keyword index: {keyword_error}")
                keyword_index = None
            
            # If either index failed to load, recreate the missing ones
            if vector_index is None or keyword_index is None:
                print("🔄 Recreating missing indexes...")
                if vector_index is None:
                    print("Creating new vector index...")
                    vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
                if keyword_index is None:
                    print("Creating new keyword index...")
                    keyword_index = SimpleKeywordTableIndex(nodes)
                
                # Save the newly created indexes
                print("💾 Saving reconstructed indexes...")
                try:
                    vector_index.set_index_id("vector")
                    keyword_index.set_index_id("keyword")
                    vector_index.storage_context.persist(persist_dir=storage_dir)
                    keyword_index.storage_context.persist(persist_dir=storage_dir)
                    print("✅ Indexes saved successfully")
                except Exception as save_error:
                    print(f"⚠️ Warning: Could not save indexes: {save_error}")
            else:
                print("✅ All indexes loaded successfully from storage")
        else:
            raise Exception("Storage directory or index_store.json not found")
        
    except Exception as e:
        print(f"Could not load from storage ({e}), creating new indexes...")
        
        print("Creating new indexes from scratch...")
        # Create vector store index from nodes (with chunking)
        vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
        
        # Create keyword table index from nodes
        keyword_index = SimpleKeywordTableIndex(nodes)
        
        # Save indexes to storage with improved error handling
        print("Saving new indexes to storage...")
        try:
            # Ensure storage directory exists
            os.makedirs(storage_dir, exist_ok=True)
            
            # Set index IDs and persist
            vector_index.set_index_id("vector")
            keyword_index.set_index_id("keyword")
            
            # Save each index separately for better error handling
            vector_index.storage_context.persist(persist_dir=storage_dir)
            keyword_index.storage_context.persist(persist_dir=storage_dir)
            print("Successfully saved indexes to storage.")
        except Exception as save_error:
            print(f"Warning: Could not save indexes to storage: {save_error}")
            print("Indexes created in memory, will need to recreate on next run.")

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

    print("RAG pipeline initialized successfully!")
    return hybrid_query_engine

async def search_documents_with_context(query: str, conversation_history: List[Dict] = None) -> str:
    """Search through documents using hybrid retrieval with conversation context
    
    Args:
        query (str): The search query to find relevant documents
        conversation_history (List[Dict]): Previous conversation history for context
        
    Returns:
        str: The response from the document search
    """
    global hybrid_query_engine
    
    # Initialize if not already done
    if hybrid_query_engine is None:
        initialize_rag_pipeline()
    
    try:
        # Build context-aware query
        if conversation_history and len(conversation_history) > 0:
            # Include recent conversation history for context resolution
            recent_history = conversation_history 
            
            # First, let's resolve any references in the current query
            context_for_resolution = ""
            for h in recent_history:
                context_for_resolution += f"User: {h['user']}\nAssistant: {h['assistant'][:400]}...\n\n"
            
            # Use LLM to resolve references and create a better search query
            resolution_prompt = f"""
Based on this recent conversation:
{context_for_resolution}

The user is now asking: "{query}"

If this question contains pronouns (it, this, that, they, etc.) or refers to concepts from the previous conversation, please reformulate the query to be more specific and complete for document search. Include the specific topics, concepts, or terms being referenced.

If the question is already clear and specific, return it as is.

Reformulated query:"""
            
            resolved_response = await Settings.llm.acomplete(resolution_prompt)
            resolved_query = str(resolved_response).strip()
            
            # Use the resolved query if it's meaningful, otherwise use original
            search_query = resolved_query if len(resolved_query) > 10 and "reformulated" not in resolved_query.lower() else query
            
            # Now build the context for the final response
            context_prompt = f"""
Previous conversation context:
{chr(10).join([f"User: {h['user']}" + chr(10) + f"Assistant: {h['assistant'][:300]}..." for h in recent_history])}

Current question: {query}
Search query used: {search_query}

Please answer the current question considering:
1. The conversation context above
2. How this question relates to previous topics discussed
3. Any references to earlier concepts or answers
4. Provide a comprehensive answer that builds on our conversation

Answer the question: {query}
"""
        else:
            context_prompt = query
            
        response = await hybrid_query_engine.aquery(context_prompt)
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

def get_rag_query_engine():
    """Get the initialized RAG query engine"""
    global hybrid_query_engine
    
    if hybrid_query_engine is None:
        initialize_rag_pipeline()
    
    return hybrid_query_engine

# High-level convenience functions following the provided pattern
async def search_with_messages(query: str, messages: List[Dict]) -> str:
    """High-level search function that combines RAG, web search and YouTube
    
    Args:
        query: User's search query
        messages: Conversation messages list with 'role' and 'content'
    
    Returns:
        Comprehensive search results
    """
    try:
        print("🔍 Searching documents...")
        rag_result = await search_documents_with_messages(query, messages)
        
        print("🌐 Searching web...")
        web_result = await get_web_search_with_messages(query, messages)
        
        print("🎥 Searching YouTube...")
        youtube_result = await get_youtube_search_with_messages(query, messages)
        
        # Combine results
        combined_result = f"""
📚 **Document Search Results:**
{rag_result}

🌐 **Web Search Results:**
{web_result}

🎥 **YouTube Results:**
{youtube_result}
"""
        return combined_result
        
    except Exception as e:
        return f"Error in search_with_messages: {str(e)}"

def get_groq_llm():
    """Get the Groq LLM instance for external use"""
    return groq_llm

# Initialize the pipeline when the module is imported
if __name__ == "__main__":
    # Only initialize if run directly
    initialize_rag_pipeline()
    print("RAG pipeline ready for use!")