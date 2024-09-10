import numpy as np
from rank_bm25 import BM25Okapi
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def preprocess_text(text):
    return ''.join(char.lower() for char in text if char.isalnum() or char.isspace())

def process_batch(args):
    bm25, batch = args
    return {query: bm25.get_scores(preprocess_text(query).split()) for query in batch}

def precompute_bm25_scores_parallel(bm25, queries, batch_size=1000):
    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_batch, [(bm25, batch) for batch in batches]), 
                            total=len(batches), desc="Computing BM25 scores"))
    
    scores = {}
    for batch_result in results:
        scores.update(batch_result)
    
    return scores

def build_bm25_index(corpus):
    """
    Builds a BM25 index from a corpus of product descriptions.
    
    This method preprocesses the text by lowercasing and removing punctuation, then tokenizes
    the corpus into individual terms. It uses the BM25Okapi algorithm to build the index.
    
    Arguments:
    - corpus: List of product descriptions or texts for indexing.
    
    Returns:
    - BM25Okapi object representing the index.
    """
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]
    tokenized_corpus = [doc.split() for doc in preprocessed_corpus]
    return BM25Okapi(tokenized_corpus)

def process_hybrid_query(args):
    query, query_embedding, bm25_scores, faiss_index, products, alpha, k = args
    bm25_scores_for_query = bm25_scores[query]
    semantic_distances, semantic_indices = faiss_index.search(query_embedding.reshape(1, -1), k)
    semantic_scores = 1 - (semantic_distances[0] / np.max(semantic_distances[0]))
    combined_scores = alpha * bm25_scores_for_query[semantic_indices[0]] + (1 - alpha) * semantic_scores
    sorted_indices = np.argsort(combined_scores)[::-1]
    return [products[semantic_indices[0][i]]['product_id'] for i in sorted_indices]

def hybrid_search_parallel(queries, bm25_scores, faiss_index, product_embeddings, query_embeddings, products, alpha=0.5, k=10):
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_hybrid_query, 
                                      [(query, query_embedding, bm25_scores, faiss_index, products, alpha, k) 
                                       for query, query_embedding in zip(queries, query_embeddings)]), 
                            total=len(queries), desc="Performing hybrid search"))
    return results