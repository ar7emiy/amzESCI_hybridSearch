import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    This function builds a FAISS index using the product embeddings for fast similarity search.

    Arguments:
    - embeddings: Embeddings for the products.

    Returns:
    - A FAISS index based on the provided embeddings.
    """

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search_index(index, query_embeddings, k=10):
    """
    Perform search on the FAISS index to find the top-k closest products for each query.

    Arguments:
    - index: Pre-built FAISS index for the products.
    - query_embeddings: Embeddings of the queries.
    - k: Number of closest products to retrieve.

    Returns:
    - Distances and indices of the top-k nearest products.
    """

    print("Searching index...")
    distances, indices = index.search(query_embeddings, k)
    return distances, indices