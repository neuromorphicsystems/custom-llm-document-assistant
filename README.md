# custom-llm-document-assistant
This project implements a document processing and querying system using a custom Language Model (LLM) and semantic search. It allows users to load PDF documents, process them into chunks, create embeddings, and then query the processed information using a streamlit-based user interface.


## Overview

This project implements an advanced document processing and querying system. It allows users to load PDF documents, process them into manageable chunks, create embeddings for efficient searching, and then query the processed information using a custom Language Model (LLM) through a user-friendly interface.

## Features

- PDF Document Processing: Extracts text from PDF files and splits it into chunks.
- Text Embedding Creation: Generates embeddings using SentenceTransformers.
- Efficient Similarity Search: Utilizes FAISS index for fast retrieval of relevant text chunks.
- Interactive User Interface: Streamlit-based UI for easy querying.
- Custom LLM Integration: Uses Ollama to run a local LLM for generating responses.

## Project Structure

- `00_data/`: Stores input PDFs, processed chunks, and embeddings.
- `01_load_pdf.py`: Handles PDF loading and text chunking.
- `02_embedding_creation.py`: Creates embeddings from processed text chunks.
- `01_user_query.py`: Streamlit app for user interaction and query processing.

## Installation

1. Clone the repository:
2. Install required dependencies, or user docker image
3. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/).

## Usage

1. Place your PDF document in the `00_data/` directory.

2. Process the PDF and create chunks: python 01_load_pdf.py

3. Generate embeddings: python 02_embedding_creation.py

4. Launch the Streamlit app: streamlit run 01_user_query.py

5. In a separate terminal, run the Ollama model: ollama run llama3
  
6. Open your web browser and go to the URL provided by Streamlit to use the query interface.

## Configuration

You can customize various aspects of the system by modifying parameters in the scripts:

- `01_load_pdf.py`: Adjust `max_chunk_length` to change text chunk sizes.
- `02_embedding_creation.py`: Modify the `Config` class to change embedding model, batch size, or index type.
- `01_user_query.py`: Adjust the number of relevant chunks or modify the LLM prompt.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
