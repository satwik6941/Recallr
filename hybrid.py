from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import asyncio
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Validate API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is required")

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5", 
    cache_folder="./cache"
)
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=360.0,
)

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

print("Creating retrievers...")
# Create retrievers with smaller top_k to reduce context size
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

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

async def search_documents(query: str) -> str:
    """Search through documents using hybrid retrieval (vector + keyword + BM25)
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        str: The response from the document search
    """
    try:
        response = await hybrid_query_engine.aquery(query)
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

async def main():
    """Main function to run the agent"""
    print("Document search ready! You can ask questions about documents.")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("Processing...")
            
            # Use direct search to avoid workflow errors
            result = await search_documents(user_query)
            print(f"\nResponse: {result}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            # Final fallback
            try:
                print("Trying fallback search...")
                # Try sync version as final fallback
                response = hybrid_query_engine.query(user_query)
                print(f"Fallback result: {response}")
            except Exception as fallback_error:
                print(f"All methods failed: {fallback_error}")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())