import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_and_preprocess_data(sample_size=1, random_state=42):
    """
    Loads the shopping queries and product datasets, filters by conditions, merges the datasets, and optionally samples the data.
    We filter to use the small version of dataset with US locale only, as well as random (non-stratified) sampling for faster embedding and indexing.
    
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