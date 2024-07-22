import os
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama
from typing import List, Tuple

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_directory() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

def load_faiss_index(project_directory: str) -> faiss.Index:
    index_file = os.path.join(project_directory, "00_data", "document_embeddings.index")
    index = faiss.read_index(index_file)
    logger.info(f"Loaded FAISS index from: {index_file}")
    return index

def load_chunk(chunk_file: str) -> str:
    with open(chunk_file, "r", encoding="utf-8") as file:
        return file.read()

def load_text_chunks(project_directory: str) -> List[str]:
    chunks_dir = os.path.join(project_directory, "00_data", "chunks")
    chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")])
    
    with ThreadPoolExecutor() as executor:
        chunks = list(executor.map(load_chunk, chunk_files))
    
    logger.info(f"Loaded {len(chunks)} text chunks from: {chunks_dir}")
    return chunks

# Load the model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
logger.info(f"Loaded sentence transformer model: {model}")

def preprocess_query(query: str) -> str:
    query = re.sub(r'[^a-zA-Z\s]', '', query.lower())
    tokens = word_tokenize(query)
    stop_words = set(stopwords.words('english'))
    preprocessed_query = ' '.join(token for token in tokens if token not in stop_words)
    logger.debug(f"Preprocessed query: {preprocessed_query}")
    return preprocessed_query

def create_query_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
    query_embedding = model.encode(query, show_progress_bar=False)
    logger.debug(f"Created query embedding: {query_embedding}")
    return query_embedding

def semantic_search(query_embedding: np.ndarray, index: faiss.Index, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(np.array([query_embedding]), top_k)
    logger.debug(f"Semantic search results - Indices: {I[0]}, Scores: {D[0]}")
    return I[0], D[0]

def save_ranked_results(indices: np.ndarray, scores: np.ndarray, chunks: List[str], project_directory: str, session_id: str) -> None:
    results_file = os.path.join(project_directory, "00_data", f"ranked_results_{session_id}.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        for idx, score in zip(indices, scores):
            f.write(f"Chunk Index: {idx}\nScore: {score}\nText: {chunks[idx]}\n\n")
    logger.info(f"Saved ranked results to: {results_file}")

def clean_up_ranked_results_files(project_directory: str, threshold_minutes: int = 30) -> None:
    results_dir = os.path.join(project_directory, "00_data")
    current_time = time.time()
    for file_name in os.listdir(results_dir):
        if file_name.startswith("ranked_results_") and file_name.endswith(".txt"):
            file_path = os.path.join(results_dir, file_name)
            if (current_time - os.path.getmtime(file_path)) / 60 > threshold_minutes:
                os.remove(file_path)
                logger.info(f"Deleted ranked results file: {file_path}")

def generate_response(query: str, relevant_chunks: List[str], top_k: int = 1) -> str:
    top_k_chunks = relevant_chunks[:top_k]
    limited_chunks = [' '.join(chunk.split()[:2500]) for chunk in top_k_chunks]

    prompt = (
        f"As an AI assistant, your task is to provide a helpful response to the user's query based on the relevant excerpts provided below. Follow these instructions:\n"
        f"1. Carefully review the user's query and understand the specific information they are seeking.\n"
        f"2. Examine the relevant excerpts and identify the information that directly addresses the user's query.\n"
        f"3. Formulate a concise and targeted response that answers the user's query using only the information from the provided excerpts.\n"
        f"4. If the user's query cannot be satisfactorily answered using the provided excerpts, politely inform the user that the required information is not available in the given context.\n"
        f"\nUser Query: {query}\n"
        f"Relevant Excerpts:\n"
        f"{' '.join(limited_chunks)}\n"
        f"Assistant's Response:"
    )

    logger.debug(f"Generated prompt:\n{prompt}")

    try:
        llm = Ollama(model="llama3")
        response = llm.invoke(prompt)
    except Exception as e:
        if "404" in str(e):
            try:
                os.system("ollama pull llama3")
                llm = Ollama(model="llama3")
                response = llm.invoke(prompt)
            except Exception as inner_e:
                logger.error(f"An error occurred while generating the response after pulling the model: {inner_e}")
                return "An error occurred while generating the response. Please try again later."
        else:
            logger.error(f"An error occurred while generating the response: {e}")
            return "An error occurred while generating the response. Please try again later."

    logger.debug(f"Generated response:\n{response}")
    return response

def process_query(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], project_directory: str, session_id: str) -> str:
    preprocessed_query = preprocess_query(query)
    query_embedding = create_query_embedding(preprocessed_query, model)
    similar_chunk_indices, similar_chunk_scores = semantic_search(query_embedding, index, top_k=5)
    save_ranked_results(similar_chunk_indices, similar_chunk_scores, chunks, project_directory, session_id)

    relevant_chunks = [chunks[idx] for idx in similar_chunk_indices]
    logger.debug(f"Relevant chunks:\n{relevant_chunks}")
    response = generate_response(query, relevant_chunks, top_k=1)
    return response

def main():
    st.set_page_config(page_title="WSU Survey Assistant", page_icon=":memo:")
    st.title("WSU Survey Assistant")

    project_directory = get_project_directory()
    image_path = os.path.join(project_directory, "00_data", "LOGO.png")
    st.sidebar.image(image_path, use_column_width=True)

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
        logger.info(f"New session ID: {st.session_state['session_id']}")

    query = st.text_input("Enter your query:")

    if query:
        logger.info(f"User query: {query}")
        response = process_query(query, model, index, chunks, project_directory, st.session_state['session_id'])
        st.write("Assistant's Response:")
        st.write(response)

    clean_up_ranked_results_files(project_directory, threshold_minutes=30)

if __name__ == "__main__":
    project_directory = get_project_directory()
    index = load_faiss_index(project_directory)
    chunks = load_text_chunks(project_directory)
    main()
# Run the Streamlit app using the command: cd 02_index_emb_user_query , streamlit run 01_user_query.py , ollama run llama3
