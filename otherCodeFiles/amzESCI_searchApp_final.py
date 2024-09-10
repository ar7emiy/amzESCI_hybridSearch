import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_and_preprocess_data(sample_size=1, random_state=42):
    """
    Loads the shopping queries and product datasets, filters by conditions, merges the datasets, and optionally samples the data.
    We filter to use the small version of dataset with US locale only, as well as random (non-stratified) sampling for faster embedding and indexing.
    
    P.S. removing special characters and puntuation lead to worse semantic search results
    Using PCA to reduce dimentions also lead to worse semantic results
    Loading chunks disabled as there is no memory contraint assumed
    
    Parameters:
    sample_size (float): The fraction of data to sample. Default is 1.0 (no sampling).
    random_state (int): Seed for reproducibility of sampling. Default is 42.

    Returns:
    pd.DataFrame: Preprocessed dataframe.
    """
        
    base_url = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
    
    print("Loading examples dataset...")
    df_examples = pd.read_parquet(f'{base_url}/shopping_queries_dataset_examples.parquet')
    
    print("Loading products dataset...")
    df_products = pd.read_parquet(f'{base_url}/shopping_queries_dataset_products.parquet')
    
    print("Filtering data...")
    df_examples = df_examples[(df_examples['small_version'] == 1) & (df_examples['product_locale'] == 'us')]
    df_products = df_products[df_products['product_locale'] == 'us']
    
    print("Merging datasets...")
    df = pd.merge(
        df_examples,
        df_products,
        how='left',
        on=['product_locale', 'product_id']
    )
    
    if sample_size < 1.0:
        print(f"Sampling {sample_size*100}% of queries...")
        unique_queries = df['query_id'].unique()
        np.random.seed(random_state)
        sampled_queries = np.random.choice(unique_queries, size=int(len(unique_queries) * sample_size), replace=False)
        df = df[df['query_id'].isin(sampled_queries)]
    
    return df

def process_product_batch(batch, model, max_length=512):
    texts = [
        f"{row['product_title']} {row['product_description']} {row['product_bullet_point']}"[:max_length]
        for _, row in batch.iterrows()
    ]
    return model.encode(texts, show_progress_bar=False)

def process_dataframe(df, model, batch_size=32):
    """
    This function encodes both the queries and products into embeddings using a pre-trained SentenceTransformer model.
    Each product is represented by a combination of product title, description, and bullet points.

    Arguments:
    - df: DataFrame containing merged query-product data.
    - model: Pre-trained SentenceTransformer model for encoding.
    - batch_size: Number of examples to encode at once to manage memory.

    Returns:
    - Encoded query embeddings, product embeddings, and their corresponding unique entries.
    """

    unique_queries = df['query'].unique()
    unique_products = df[['product_id', 'product_title', 'product_description', 'product_bullet_point']].drop_duplicates('product_id')
    
    print("Encoding queries...")
    query_embeddings = model.encode(unique_queries.tolist(), batch_size=batch_size, show_progress_bar=True)
    
    print("Encoding products...")
    num_products = len(unique_products)
    product_embeddings = []
    
    for i in tqdm(range(0, num_products, batch_size), desc="Processing products"):
        batch = unique_products.iloc[i:i+batch_size]
        batch_embeddings = process_product_batch(batch, model)
        product_embeddings.extend(batch_embeddings)
    
    product_embeddings = np.array(product_embeddings)
    
    return query_embeddings, product_embeddings, unique_queries, unique_products


########################
#FAISS INDEXING & SEARCH
########################

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




###################
# EVALUATION BLOCK
###################

def calculate_mrr(relevance_scores):
    """Mean Reciprocal Rank: evaluates the first relevant item rank."""

    reciprocal_ranks = []
    for scores in relevance_scores:
        rank = next((i + 1 for i, s in enumerate(scores) if s > 0), 0)
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)
    return np.mean(reciprocal_ranks)

def calculate_hits_at_n(relevance_scores, n):
    """Hits@N: checks if at least one relevant item appears in the top N results."""

    hits = [1 if sum(scores[:n]) > 0 else 0 for scores in relevance_scores]
    return np.mean(hits)

def calculate_ndcg(relevance_scores, k=10):
    """Normalized Discounted Cumulative Gain: measures ranking quality, penalizing lower-ranked relevant results."""

    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k) / dcg_max

    scores = [ndcg_at_k(r, k) for r in relevance_scores]
    return np.mean(scores)

def calculate_map(relevance_scores):
    """Mean Average Precision: evaluates precision averaged across different recall levels."""

    aps = []
    for scores in relevance_scores:
        precision_at_k = [sum(scores[:k+1]) / (k+1) for k in range(len(scores))]
        ap = sum([p * r for p, r in zip(precision_at_k, scores)]) / sum(scores) if sum(scores) > 0 else 0
        aps.append(ap)
    return np.mean(aps)

def calculate_precision_at_k(relevance_scores, k):
    """Precision@K"""

    return np.mean([sum(scores[:k]) / k for scores in relevance_scores])

def calculate_recall_at_k(relevance_scores, k):
    """Recall@k"""

    return np.mean([sum(scores[:k]) / sum(scores) if sum(scores) > 0 else 0 for scores in relevance_scores])

def evaluate_rankings(df, unique_queries, unique_products, search_results):
    """
    Evaluate the search rankings using various metrics.

    Parameters:
    df (pd.DataFrame): The dataframe containing query and product information.
    unique_queries (np.ndarray): Array of unique queries.
    unique_products (pd.DataFrame): Dataframe of unique products.
    search_results (list): List of search results for each query.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    print("Evaluating rankings...")
    relevance_scores = []
    debug_info = []
    
    # Create a mapping from product ID to index
    product_id_to_idx = {pid: idx for idx, pid in enumerate(unique_products['product_id'])}
    
    # Iterate over each query and its corresponding search results
    for query_idx, (query, rankings) in enumerate(zip(unique_queries, search_results)):
        scores = []
        debug_query = []
        
        # Filter the dataframe for the current query
        query_df = df[df['query'] == query]
        
        # Get the set of relevant product IDs for the current query
        relevant_product_ids = set(query_df[query_df['esci_label'].isin(['E', 'S'])]['product_id'])
        
        # Iterate over the ranked results
        for rank, result in enumerate(rankings):
            if isinstance(result, (int, np.integer)):
                predicted_product_id = unique_products.iloc[result]['product_id']
            else:
                predicted_product_id = result
            
            predicted_product = unique_products.iloc[product_id_to_idx[predicted_product_id]]
            predicted_product_title = predicted_product['product_title']
            
            # Determine if the predicted product is relevant
            score = 1 if predicted_product_id in relevant_product_ids else 0
            scores.append(score)
            
            # Add debugging information
            debug_query.append(f"Rank {rank + 1}: ID {predicted_product_id}, Title: {predicted_product_title[:50]}..., Relevant: {'Yes' if score == 1 else 'No'}")
        
        relevance_scores.append(scores)
        debug_info.append((query, debug_query))
    
    # Calculate evaluation metrics
    mrr = calculate_mrr(relevance_scores)
    hits_at_1 = calculate_hits_at_n(relevance_scores, 1)
    hits_at_5 = calculate_hits_at_n(relevance_scores, 5)
    hits_at_10 = calculate_hits_at_n(relevance_scores, 10)
    ndcg = calculate_ndcg(relevance_scores)
    map_score = calculate_map(relevance_scores)
    precision_at_5 = calculate_precision_at_k(relevance_scores, 5)
    recall_at_10 = calculate_recall_at_k(relevance_scores, 10)
    
    # Print debugging information for the first few queries
    print("\nDebugging Information:")
    for query, debug in debug_info[:5]:
        print(f"\nQuery: {query}")
        for line in debug[:10]:
            print(line)
    
    # Return the evaluation metrics
    return {
        "mrr": mrr,
        "hits@1": hits_at_1,
        "hits@5": hits_at_5,
        "hits@10": hits_at_10,
        "ndcg@10": ndcg,
        "map": map_score,
        "precision@5": precision_at_5,
        "recall@10": recall_at_10
    }

def inspect_data(df, unique_products, unique_queries, product_embeddings, query_embeddings, indices):
    result = "\nData Inspection:\n"
    result += f"Total judgements: {len(df)}\n"
    result += f"Unique products: {len(unique_products)}\n"
    result += f"Unique queries: {len(unique_queries)}\n"
    result += f"Product embeddings shape: {product_embeddings.shape}\n"
    result += f"Query embeddings shape: {query_embeddings.shape}\n"
    
    result += "\nDistribution of ESCI labels:\n"
    result += df['esci_label'].value_counts(normalize=True).to_string()
    
    result += "\n\nSample query and its top 5 results:\n"
    sample_query_idx = 0
    sample_query = unique_queries[sample_query_idx]
    result += f"Query: {sample_query}\n"
    
    top_5_indices = indices[sample_query_idx][:5]
    result += "Top 5 results:\n"
    for i, idx in enumerate(top_5_indices, 1):
        product = unique_products.iloc[idx]
        result += f"{i}. {product['product_title']} (Product ID: {product['product_id']})\n"
    
    print(result) 
    return result



#########################
# HYBRID SEARCH FUNCTIONS
#########################

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

#############################################
# saving for reusability and extra testing
#############################################

def save_checkpoint(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def analyze_vector_index(embeddings, labels, n_samples=1000):
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    #was not useful for inferencing how to improve performance, but lets one see the lack of singificant clustering in the vector indexes
    #there is some general shape, likely product category group based, all of which is expected since its an amazon dataset

    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("2D Visualization of Embeddings")
    plt.savefig("embeddings_visualization.png")
    plt.close()
    
    distances = cosine_similarity(embeddings)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    return {
        "avg_distance": avg_distance,
        "std_distance": std_distance,
        "visualization": "embeddings_visualization.png"
    }


def main():
    sample_size = 0.01  # adjust this based on your memory constraints
    folder_path = 'allminilml6_hybridtest/'
    dt_size = '1_1pct_'
    model_name = "all-MiniLM-L6-v2"
    
    os.makedirs(folder_path, exist_ok=True)

    full_df_path = os.path.join(folder_path, f"{dt_size}full_df.pkl")
    embeddings_path = os.path.join(folder_path, f"{dt_size}embeddings.pkl")
    faiss_index_path = os.path.join(folder_path, f"{dt_size}faiss_index.pkl")
    bm25_index_path = os.path.join(folder_path, f"{dt_size}bm25_index.pkl")
    bm25_scores_path = os.path.join(folder_path, f"{dt_size}bm25_scores.pkl")

    try:
        if os.path.exists(full_df_path):
            print("Loading preprocessed data from checkpoint...")
            df = load_checkpoint(full_df_path)
        else:
            print("Preprocessing data...")
            df = load_and_preprocess_data(sample_size=sample_size)
            save_checkpoint(df, full_df_path)

        if os.path.exists(embeddings_path):
            print("Loading embeddings from checkpoint...")
            embeddings = load_checkpoint(embeddings_path)
            query_embeddings = embeddings['query']
            product_embeddings = embeddings['product']
            unique_queries = embeddings['unique_queries']
            unique_products = embeddings['unique_products']
        else:
            print("Generating embeddings...")
            model = SentenceTransformer(model_name)
            query_embeddings, product_embeddings, unique_queries, unique_products = process_dataframe(df, model)
            embeddings = {
                'query': query_embeddings,
                'product': product_embeddings,
                'unique_queries': unique_queries,
                'unique_products': unique_products
            }
            save_checkpoint(embeddings, embeddings_path)

        if os.path.exists(faiss_index_path):
            print("Loading FAISS index from checkpoint...")
            product_index = load_checkpoint(faiss_index_path)
        else:
            print("Building FAISS index...")
            product_index = build_faiss_index(product_embeddings)
            save_checkpoint(product_index, faiss_index_path)

        if os.path.exists(bm25_index_path):
            print("Loading BM25 index from checkpoint...")
            bm25 = load_checkpoint(bm25_index_path)
        else:
            print("Building BM25 index...")
            product_texts = unique_products.apply(lambda row: f"{row['product_title']} {row['product_description']} {row['product_bullet_point']}", axis=1).tolist()
            bm25 = build_bm25_index(product_texts)
            save_checkpoint(bm25, bm25_index_path)

        if os.path.exists(bm25_scores_path):
            print("Loading pre-computed BM25 scores from checkpoint...")
            bm25_scores = load_checkpoint(bm25_scores_path)
        else:
            print("Pre-computing BM25 scores...")
            bm25_scores = precompute_bm25_scores_parallel(bm25, unique_queries, batch_size=1000)
            save_checkpoint(bm25_scores, bm25_scores_path)

        print("Performing hybrid search...")
        hybrid_results = hybrid_search_parallel(unique_queries, bm25_scores, product_index, 
                                                product_embeddings, query_embeddings, 
                                                unique_products.to_dict('records'))
        print("Performing semantic search...")
        semantic_distances, semantic_indices = search_index(product_index, query_embeddings, k=10)


        inspection_results = inspect_data(df, unique_products, unique_queries, product_embeddings, query_embeddings, semantic_indices)

        print("Evaluating semantic search rankings...")
        semantic_evaluation = evaluate_rankings(df, unique_queries, unique_products, semantic_indices)

        print("Evaluating hybrid search rankings...")
        hybrid_evaluation = evaluate_rankings(df, unique_queries, unique_products, hybrid_results)

        print("\nSemantic Search Results:")
        for metric, score in semantic_evaluation.items():
            print(f"{metric}: {score:.4f}")

        print("\nHybrid Search Results:")
        for metric, score in hybrid_evaluation.items():
            print(f"{metric}: {score:.4f}")

        print("Analyzing vector index...")
        label_encoder = {label: i for i, label in enumerate(['E', 'S', 'C', 'I'])}
        encoded_labels = [label_encoder[label] for label in df['esci_label']]
        index_analysis = analyze_vector_index(product_embeddings, encoded_labels)
        print("Vector Index Analysis:", index_analysis)

        results_path = os.path.join(folder_path, f"{dt_size}results_receipt.txt")
        with open(results_path, 'w') as f:
            f.write(f"Sample Size: {sample_size * 100}%\n\n")
            f.write("Data Inspection Results:\n")
            f.write(inspection_results)
            
            f.write("\n\nSemantic Search Evaluation Results:\n")
            for metric, score in semantic_evaluation.items():
                f.write(f"{metric}: {score:.4f}\n")
            
            f.write("\n\nHybrid Search Evaluation Results:\n")
            for metric, score in hybrid_evaluation.items():
                f.write(f"{metric}: {score:.4f}\n")
            
            f.write("\nVector Index Analysis:\n")
            f.write(str(index_analysis))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()