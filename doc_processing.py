import fitz
from transformers import AutoTokenizer
import os, hashlib, json
from pathlib import Path
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def is_quiet_mode():
    """Check if we're in quiet mode for cleaner output"""
    return os.getenv('RECALLR_QUIET_MODE', '0') == '1'

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
    """Split text into overlapping chunks of specified token size"""
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def calculate_chunk_quality_score(chunk):
    """Calculate a quality score for a chunk based on various factors"""
    score = 0
    
    # Length score (prefer medium-length chunks)
    word_count = len(chunk.split())
    if 50 <= word_count <= 200:
        score += 2
    elif word_count >= 30:
        score += 1
    
    # Content richness (prefer chunks with diverse vocabulary)
    unique_words = len(set(chunk.lower().split()))
    total_words = len(chunk.split())
    if total_words > 0:
        diversity_ratio = unique_words / total_words
        score += diversity_ratio * 3
    
    # Keyword relevance (prefer chunks with academic keywords)
    academic_keywords = [
        'definition', 'concept', 'theory', 'principle', 'method', 'process',
        'analysis', 'research', 'study', 'conclusion', 'result', 'figure',
        'table', 'chapter', 'section', 'important', 'significant', 'key'
    ]
    
    chunk_lower = chunk.lower()
    keyword_count = sum(1 for keyword in academic_keywords if keyword in chunk_lower)
    score += keyword_count * 0.5
    
    # Structure indicators (prefer chunks with good structure)
    structure_indicators = ['.', ':', ';', '\n', '•', '-', '1.', '2.', '3.']
    structure_score = sum(1 for indicator in structure_indicators if indicator in chunk)
    score += min(structure_score * 0.1, 2)  # Cap at 2 points
    
    # Penalize chunks that are too short or mostly whitespace/special characters
    if len(chunk.strip()) < 50:
        score -= 3
    
    # Penalize chunks with too many special characters
    special_char_ratio = sum(1 for char in chunk if not char.isalnum() and char != ' ') / len(chunk)
    if special_char_ratio > 0.3:
        score -= 2
    
    return max(0, score)  # Ensure score is never negative

def select_best_chunks(chunks, num_chunks=3):
    """Select the best chunks based on quality scoring"""
    if not chunks:
        return []
    
    if len(chunks) <= num_chunks:
        return chunks
    
    # Calculate scores for all chunks
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        score = calculate_chunk_quality_score(chunk)
        chunk_scores.append((score, i, chunk))
    
    # Sort by score (descending) and select top chunks
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    best_chunks = [chunk for score, idx, chunk in chunk_scores[:num_chunks]]
    
    return best_chunks

def generate_system_prompt_from_chunks(pdf_chunks_dict):
    """Generate system prompt from multiple PDF chunks"""
    
    # Prepare the content for the prompt
    content_summary = []
    for pdf_name, chunks in pdf_chunks_dict.items():
        content_summary.append(f"\n=== Content from {pdf_name} ===")
        for i, chunk in enumerate(chunks, 1):
            # Truncate each chunk to avoid token limits
            truncated_chunk = chunk[:800] if len(chunk) > 800 else chunk
            content_summary.append(f"\nChunk {i}:\n{truncated_chunk}")
    
    combined_content = "\n".join(content_summary)
    
    prompt = f"""
You are an academic assistant. Analyze these document chunks from multiple PDF files and create a comprehensive system prompt that an LLM can use to answer student queries based on the document content.

Document chunks from PDFs:
\"\"\"{combined_content[:6000]}\"\"\"

Analyze all the chunks and create a comprehensive system prompt that includes:
1. A summary of all the topics covered across the documents
2. The specific domains/subjects these documents relate to
3. The level of detail and complexity found in the content
4. Key concepts, processes, or methodologies mentioned

Start the system prompt with "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards." 

The system prompt should be detailed enough to guide the LLM to provide accurate, contextual responses based on the document content, but concise enough to be practical.

Return only the system prompt text, no other formatting or explanations.
"""
    
    try:
        response = groq_llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=3000,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating system prompt: {e}")
        return "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards. Help students with their queries based on the provided documents."

def get_system_prompt_with_caching(file_path):
    """
    Main function to get system prompt with smart caching.
    Only processes documents if PDF collection has changed.
    """
    # Check if PDF collection has changed
    if has_pdf_collection_changed(file_path):
        if not is_quiet_mode():
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
        
        if not is_quiet_mode():
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
    """Process all PDF files and generate a comprehensive system prompt from best chunks"""
    pdf_chunks_dict = {}
    
    # Check if directory exists
    if not file_path.exists():
        print(f"Directory {file_path} does not exist!")
        return "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."
    
    # Process each PDF file
    pdf_files = [f for f in os.listdir(file_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        if not is_quiet_mode():
            print("No PDF files found!")
        return "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."
    
    if not is_quiet_mode():
        print(f"Found {len(pdf_files)} PDF files to process...")
    
    for file in pdf_files:
        if not is_quiet_mode():
            print(f"📄 Processing {file}...")
        
        text = extract_text(os.path.join(file_path, file))
        
        if not text.strip():
            if not is_quiet_mode():
                print(f"⚠️ No text extracted from {file}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        if not is_quiet_mode():
            print(f"   Created {len(chunks)} chunks")
        
        if chunks:
            # Select 3 best chunks from this PDF
            best_chunks = select_best_chunks(chunks, num_chunks=3)
            pdf_chunks_dict[file] = best_chunks
            if not is_quiet_mode():
                print(f"   Selected {len(best_chunks)} best chunks")
        else:
            if not is_quiet_mode():
                print(f"   No valid chunks created from {file}")
    
    if not pdf_chunks_dict:
        if not is_quiet_mode():
            print("No valid chunks extracted from any PDF!")
        return "You are an expert AI powered academic assistant with over 20+ years of experience, who has multiple achievements, publications and awards."
    
    # Generate system prompt from all selected chunks
    total_chunks = sum(len(chunks) for chunks in pdf_chunks_dict.values())
    if not is_quiet_mode():
        print(f"🤖 Generating system prompt from {total_chunks} selected chunks across {len(pdf_chunks_dict)} PDFs...")
    
    return generate_system_prompt_from_chunks(pdf_chunks_dict)

# Main execution - now uses caching
if __name__ == "__main__":
    system_prompt = get_system_prompt_with_caching(file_path)
    print("\n" + "="*50)
    print("GENERATED SYSTEM PROMPT:")
    print("="*50)
    print(system_prompt)