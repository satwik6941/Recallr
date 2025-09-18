import fitz
from transformers import AutoTokenizer
import os, hashlib, json
from pathlib import Path
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

file_path = Path('data')

# Cache file paths
PDF_LIST_CACHE = "pdf_cache.json"
SYSTEM_PROMPT_CACHE = "system_prompt_cache.txt"

groq_llm = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_current_pdf_files(file_path):
    """Get list of current PDF files in the directory"""
    if not file_path.exists():
        return []
    
    pdf_files = []
    for file in os.listdir(file_path):
        if file.endswith(".pdf"):
            pdf_files.append(file)
    
    return sorted(pdf_files)  # Sort for consistent comparison

def load_cached_pdf_list():
    """Load the cached PDF list from file"""
    try:
        if os.path.exists(PDF_LIST_CACHE):
            with open(PDF_LIST_CACHE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading PDF cache: {e}")
    return []

def save_pdf_list_cache(pdf_list):
    """Save the current PDF list to cache"""
    try:
        with open(PDF_LIST_CACHE, 'w', encoding='utf-8') as f:
            json.dump(pdf_list, f, indent=2)
    except Exception as e:
        print(f"Error saving PDF cache: {e}")

def load_cached_system_prompt():
    """Load the cached system prompt from file"""
    try:
        if os.path.exists(SYSTEM_PROMPT_CACHE):
            with open(SYSTEM_PROMPT_CACHE, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"Error loading system prompt cache: {e}")
    return None

def save_system_prompt_cache(system_prompt):
    """Save the system prompt to cache"""
    try:
        with open(SYSTEM_PROMPT_CACHE, 'w', encoding='utf-8') as f:
            f.write(system_prompt)
    except Exception as e:
        print(f"Error saving system prompt cache: {e}")

def has_pdf_collection_changed(file_path):
    """Check if PDF collection has changed compared to cache"""
    current_pdfs = get_current_pdf_files(file_path)
    cached_pdfs = load_cached_pdf_list()
    
    return current_pdfs != cached_pdfs

def extract_text(file_path):
    """Extract text from various file formats including PDF"""    
    # Convert string path to Path object if needed
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        # Load PDF using PyMuPDF
        load_pdf = fitz.open(str(file_path))
        # Extract text from each page
        documents = [page.get_text() for page in load_pdf]
        # Combine all pages into one text
        full_text = "\n".join(documents)
        load_pdf.close()  # Close the PDF file
        return full_text
    else:
        # For other file types, you can add more extraction logic here
        # For now, try to read as plain text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            return ""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def profile_documents(file_path):
    profile = {}
    for file in os.listdir(file_path):
        if not file.endswith(".pdf"): continue
        text = extract_text(os.path.join(file_path, file))
        tokens = tokenizer.encode(text, truncation=False)
        headers = extract_headers(text)
        profile[file] = {
            "tokens": len(tokens),
            "headers": headers,
            "text": text
        }
    return profile

def extract_headers(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if line.isupper() or line.startswith("Q.")]

def chunk_text(text, chunk_size=500, overlap=50):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def select_best_chunk(chunks):
    return chunks[0]  # or use density heuristics

def generate_system_prompt_from_chunk(chunk):
    prompt = f"""
You are an academic assistant. Analyze this document chunk and create a detailed system prompt that an LLM can use to answer student queries based on the document content.

Document chunk:
\"\"\"{chunk[:2000]}\"\"\"

Analyse the chunks and create a comprehensive system prompt that includes all the summary of the topics which will be covered in the document.
Start the system prompt with "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards." and ensure it is concise yet informative.
Return only the system prompt text, no other formatting.
"""
    try:
        response = groq_llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating system prompt: {e}")
        return "You are an academic assistant. Help students with their queries based on the provided documents."

def get_system_prompt_with_caching(file_path):
    """
    Main function to get system prompt with smart caching.
    Only processes documents if PDF collection has changed.
    """
    # Check if PDF collection has changed
    if has_pdf_collection_changed(file_path):
        print("📚 PDF collection changed - processing documents...")
        
        # Get current PDF list
        current_pdfs = get_current_pdf_files(file_path)
        
        if not current_pdfs:
            print("⚠️ No PDF files found in data directory!")
            default_prompt = "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."
            save_system_prompt_cache(default_prompt)
            save_pdf_list_cache([])
            return default_prompt
        
        # Process documents and generate new system prompt
        system_prompt = process_all_documents(file_path)
        
        # Update caches
        save_system_prompt_cache(system_prompt)
        save_pdf_list_cache(current_pdfs)
        
        print(f"✅ Processed {len(current_pdfs)} PDF files and updated caches")
        return system_prompt
    
    else:
        print("📋 PDF collection unchanged - using cached system prompt...")
        cached_prompt = load_cached_system_prompt()
        
        if cached_prompt:
            print("✅ Loaded system prompt from cache")
            return cached_prompt
        else:
            print("⚠️ Cache miss - processing documents...")
            # Cache doesn't exist, process documents
            return get_system_prompt_with_caching(file_path)

def process_all_documents(file_path):
    """Process all PDF files and generate a comprehensive system prompt"""
    all_chunks = []
    
    # Check if directory exists
    if not file_path.exists():
        print(f"Directory {file_path} does not exist!")
        return "You are an academic assistant."
    
    # Process each PDF file
    for file in os.listdir(file_path):
        if not file.endswith(".pdf"): 
            continue
        
        print(f"Processing {file}...")
        text = extract_text(os.path.join(file_path, file))
        if text.strip():  # Only process if text was extracted
            chunks = chunk_text(text)
            if chunks:
                best_chunk = select_best_chunk(chunks)
                all_chunks.append(best_chunk)
    
    if not all_chunks:
        print("No PDF files found or no text extracted!")
        return "You are an academic assistant."
    
    # Combine chunks and generate system prompt
    combined_content = "\n\n".join(all_chunks[:3])  # Use first 3 chunks to avoid token limit
    return generate_system_prompt_from_chunk(combined_content)

# Main execution - now uses caching
if __name__ == "__main__":
    system_prompt = get_system_prompt_with_caching(file_path)
    print(system_prompt)