import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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



def analyze_vector_index(embeddings, labels, n_samples=1000):
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
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