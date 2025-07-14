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
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Validate API keys
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

# Global variables to store the initialized components
vector_index = None
keyword_index = None
hybrid_query_engine = None
nodes = None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with hybrid retrieval"""
    global vector_index, keyword_index, hybrid_query_engine, nodes
    
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
            
            # Try to load with specific index IDs
            try:
                vector_index = load_index_from_storage(storage_context, index_id="vector")
                keyword_index = load_index_from_storage(storage_context, index_id="keyword")
                print("Loaded existing indexes from storage.")
            except Exception as index_error:
                print(f"Failed to load specific indexes: {index_error}")
                # Try loading without index IDs (fallback)
                vector_index = load_index_from_storage(storage_context)
                keyword_index = SimpleKeywordTableIndex(nodes)  # Recreate keyword index
                print("Loaded vector index, recreated keyword index.")
        else:
            raise Exception("Storage directory or index_store.json not found")
        
    except Exception as e:
        print(f"Could not load from storage ({e}), creating new indexes...")
        
        print("Creating indexes...")
        # Create vector store index from nodes (with chunking)
        vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
        
        # Create keyword table index from nodes
        keyword_index = SimpleKeywordTableIndex(nodes)
        
        # Save indexes to storage with improved error handling
        print("Saving indexes to storage...")
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

def get_rag_query_engine():
    """Get the initialized RAG query engine"""
    global hybrid_query_engine
    
    if hybrid_query_engine is None:
        initialize_rag_pipeline()
    
    return hybrid_query_engine

# Initialize the pipeline when the module is imported
if __name__ == "__main__":
    # Only initialize if run directly
    initialize_rag_pipeline()
    print("RAG pipeline ready for use!")