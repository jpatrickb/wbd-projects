from llama_cpp import Llama
import regex as re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create embedding model
emb_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the Llama model
model_path = 'models/llama-2-13b-chat.Q4_K_S.gguf'

llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, 
    n_gpu_layers=30,
    n_ctx=4096, # Context window
)


# Function to process the book
def process_book(bookname='bertrand.txt'):
    # Read in the book
    with open(bookname, 'r') as f:
        book = f.read()

    # Split into paragraphs
    paragraphs = book.split('\n\n')

    # Initialize chapter and page variables to track
    chapter_num = None
    chapter_name = None
    page_number = 8
    processed = []

    i = 0
    while i < len(paragraphs):
        par = paragraphs[i].strip()
        # print(par)

        # Check for chapter headers
        chapter_match = re.match(r"^CHAPTER\s+([IVXLCDM]+)$", par)
        if chapter_match:
            chapter_num = chapter_match.group(1)  # Roman numeral
            chapter_name = paragraphs[i + 1].strip()  # Chapter name, two lines down
            i += 2  # Skip to the next block of lines after the chapter heading
            continue

        # Check for page headers
        page_match = re.match(r"^\s*(\d+)\s*(.*)$", par)
        if page_match:
            page_number = page_match.group(1)
            page_heading = page_match.group(2).strip()
            # print(page_number)
            i += 1
            continue

        page_match_end = re.match(r"^(.*)\s+(\d+)$", par)
        if page_match_end:
            page_number = page_match_end.group(2)
            page_heading = page_match_end.group(1).strip()
            # print(page_number)
            i += 1
            continue

        # Skip empty lines
        if not par:
            i += 1
            continue

        # Save the paragraph
        full_par = """
        Chapter number: {}
        Chapter name: {}
        Page number: {}
        {}
        """.format(chapter_num, chapter_name, page_number, par)
        processed.append(full_par)

        # Increment
        i += 1


    # Save processed book to a file
    with open("processed_book.txt", 'w') as f:
        for par in processed:
            f.write(par)

    # Process the chunks
    chunk_vectors = np.array([emb_model.encode(chunk) for chunk in process_book()])
    np.save('chunk_vectors.npy', chunk_vectors)

    # Create the Faiss index
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(chunk_vectors)

    faiss.write_index(index, 'vectorstore.index')

    return chunk_vectors, index, processed

def get_chunk_index():
    try:
        chunk_vectors = np.load('chunk_vectors.npy')
        index = faiss.read_index('vectorstore.index')
        with open('processed_book.txt', 'r') as f:
            processed_book = f.readlines()
    except FileNotFoundError:
        chunk_vectors, index, processed_book = process_book()
    return chunk_vectors, index, processed_book

def retrieve_relevant_chunks(query, n_relevant=5):
    chunk_vectors, index, processed_book = get_chunk_index()
    query_vector = emb_model.encode(query)
    _, idxs = index.search(query_vector.reshape(1, -1), n_relevant)
    return [processed_book[i] for i in idxs[0]]

def generate_response(query):
    updated_prompt = f"""
    Answer the following question: "{query}"
    Please also cite your source, including page number and chapter.
    """
    relevant_chunks = retrieve_relevant_chunks(query)
    return llm.generate_response(query, relevant_chunks)