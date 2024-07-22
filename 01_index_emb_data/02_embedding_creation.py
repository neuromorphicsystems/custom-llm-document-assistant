import os
import numpy as np
import faiss
import logging
from tqdm import tqdm
import importlib.metadata
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from typing import List, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logging.error(f"Failed to import necessary modules for SentenceTransformer: {str(e)}")
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    batch_size: int = 32
    num_workers: int = 4
    use_gpu: bool = torch.cuda.is_available()
    index_type: str = "FlatL2"

def log_version_info():
    try:
        st_version = importlib.metadata.version("sentence-transformers")
        faiss_version = importlib.metadata.version("faiss-cpu")  # or faiss-gpu if you're using GPU
        torch_version = torch.__version__
        logger.info(f"sentence-transformers version: {st_version}")
        logger.info(f"faiss version: {faiss_version}")
        logger.info(f"torch version: {torch_version}")
    except importlib.metadata.PackageNotFoundError as e:
        logger.error(f"Package not found: {str(e)}")

def get_project_directory():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    logger.info(f"Project Directory: {project_directory}")
    return project_directory

def load_text_chunks(project_directory: str) -> List[str]:
    chunks_dir = os.path.join(project_directory, "00_data", "chunks")
    chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")])
    
    chunks = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks", unit="file"):
        try:
            with open(chunk_file, "r", encoding="utf-8") as file:
                chunks.append(file.read())
        except Exception as e:
            logger.error(f"Error loading chunk from {chunk_file}: {str(e)}")
    
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def create_embeddings_batch(model: SentenceTransformer, chunks: List[str]) -> np.ndarray:
    return model.encode(chunks, show_progress_bar=False)

def create_embeddings(chunks: List[str], config: Config) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("SentenceTransformer is not available. Please ensure all dependencies are installed.")
    
    model = SentenceTransformer(config.model_name)
    if config.use_gpu:
        model = model.to(torch.device("cuda"))
    
    embeddings = []
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []
        for i in range(0, len(chunks), config.batch_size):
            batch = chunks[i:i+config.batch_size]
            futures.append(executor.submit(create_embeddings_batch, model, batch))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings", unit="batch"):
            embeddings.extend(future.result())
    
    return np.array(embeddings).astype('float32')

def create_faiss_index(embeddings: np.ndarray, config: Config) -> faiss.Index:
    embedding_dim = embeddings.shape[1]
    
    if config.index_type == "FlatL2":
        index = faiss.IndexFlatL2(embedding_dim)
    elif config.index_type == "IVFFlat":
        nlist = min(int(np.sqrt(len(embeddings))), 256)  # number of clusters
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    else:
        raise ValueError(f"Unsupported index type: {config.index_type}")
    
    if config.use_gpu and hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    return index

def store_embeddings_in_faiss(embeddings: np.ndarray, project_directory: str, config: Config):
    index = create_faiss_index(embeddings, config)
    
    index_file = os.path.join(project_directory, "00_data", f"document_embeddings_{config.index_type}.index")
    if config.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_file)
    logger.info(f"FAISS index saved to: {index_file}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and store document embeddings")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-MiniLM-L6-v2", help="Name of the sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--index-type", choices=["FlatL2", "IVFFlat"], default="FlatL2", help="Type of FAISS index to use")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = Config(
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu=not args.no_gpu and torch.cuda.is_available(),
        index_type=args.index_type
    )
    
    log_version_info()
    project_directory = get_project_directory()
    
    try:
        text_chunks = load_text_chunks(project_directory)
        embeddings = create_embeddings(text_chunks, config)
        store_embeddings_in_faiss(embeddings, project_directory, config)
    except Exception as e:
        logger.error(f"An error occurred during the embedding process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
