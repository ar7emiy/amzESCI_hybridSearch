import os
import traceback
from sentence_transformers import SentenceTransformer

# Import all necessary functions from our package
from final import (
    load_and_preprocess_data,
    process_dataframe,
    build_faiss_index,
    search_index,
    evaluate_rankings,
    inspect_data,
    analyze_vector_index,
    build_bm25_index,
    precompute_bm25_scores_parallel,
    hybrid_search_parallel,
    save_checkpoint,
    load_checkpoint
)

def main():
    sample_size = 0.01  # adjust this based on your memory constraints
    folder_path = 'allminilml6_hybridtest/'
    dt_size = '1pct_'
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
        traceback.print_exc()

if __name__ == "__main__":
    main()