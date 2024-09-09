# Amazon ESCI Search Application

This application implements a semantic and hybrid search system for the Amazon ESCI (Exact, Substitute, Complement, Irrelevant) dataset. It utilizes FAISS for efficient similarity search and combines it with BM25 for a hybrid approach.

## Setup and Installation

1. **Prerequisites**
   - Python 3.7+
   - pip (Python package installer)

2. **Required Libraries**
   Install the following libraries using pip:
```
pip install pandas numpy sentence-transformers faiss-cpu scikit-learn matplotlib rank_bm25
```

3. **Download Files**
- Download the `amzESCI_searchApp_final.py` script
- Download the `allminilmL6_hybrid_1pct` folder containing pre-processed data and checkpoints

4. **File Structure**
Ensure your directory structure looks like this:

your_project_folder/
├── amzESCI_searchApp_final.py
└── allminilmL6_hybrid_1pct/
├── 1pct_full_df.pkl
├── 1pct_embeddings.pkl
├── 1pct_faiss_index.pkl
└── 1pct_bm25_index.pkl

## Usage

1. **Default Run (1% Sample Size)**
To run the script with the default 1% sample size:

This will use the pre-processed data in the `allminilmL6_hybrid_1pct` folder.

2. **Custom Sample Size**
If you want to run the script with a different sample size:
- Open `amzESCI_searchApp_final.py` in a text editor
- Locate the `main()` function
- Modify the `sample_size` variable (e.g., `sample_size = 0.05` for 5%)
- Save the file and run it as in step 1

Note: Running with a different sample size will require reprocessing the data, which may take longer and require more computational resources.

## Features

- Data preprocessing and sampling
- Semantic search using FAISS
- Hybrid search combining FAISS and BM25
- Evaluation metrics including MRR, Hits@N, NDCG, MAP, Precision, and Recall
- Data inspection and visualization

## Output

The script will generate:
- Console output with search performance metrics
- A `results_receipt.txt` file in the `allminilmL6_hybrid_1pct` folder containing detailed results
- An `embeddings_visualization.png` file showing a 2D visualization of the embeddings

## Customization

- To use a different pre-trained model, modify the `model_name` variable in the `main()` function
- Adjust the `k` parameter in `search_index()` and `hybrid_search()` functions to change the number of results returned

## Troubleshooting

If you encounter any errors:
- Ensure all required libraries are installed
- Check that the file structure is correct
- Verify that you have sufficient disk space and RAM, if possible use GPU if using sample size over 15%.

For any persistent issues, refer to the error message and traceback printed in the console.

--------------------------------------------
#Overall review of the takehome assessment:
---------------------------------------------

Initial Data Exploration and Preprocessing:
I began by loading and exploring the Amazon ESCI dataset, which proved to be quite large and challenging to process on a 16GB CPU. To manage this, I implemented a sampling strategy, allowing me to work with a smaller subset of the data (initially 10%, later reduced to 1% for quicker iterations). I also focused on the US locale to simplify the problem space.
Feature Engineering and Embedding:
I experimented with various text preprocessing techniques, including removing special characters and punctuation. However, I found that these preprocessing steps actually led to worse semantic search results, so I opted for minimal preprocessing. I used the SentenceTransformer library with the "all-MiniLM-L6-v2" model to generate embeddings for both queries and products.
Dimensionality Reduction Attempts:
In an effort to improve performance and reduce memory usage, I attempted to use PCA for dimensionality reduction on the embeddings. However, this also led to decreased semantic search performance, so I decided to keep the full-dimensional embeddings.
Indexing and Search Implementation:
I implemented a FAISS index for efficient similarity search on the product embeddings. This allowed for fast retrieval of the most similar products for each query based on cosine similarity.
Evaluation Metrics:
To comprehensively assess the performance of the search system, I implemented multiple evaluation metrics:

Mean Reciprocal Rank (MRR)
Hits@N (for N=1, 5, and 10)
Normalized Discounted Cumulative Gain (NDCG)
Mean Average Precision (MAP)
Precision@5
Recall@10
These metrics provided a multifaceted view of the search performance, allowing me to make informed decisions about improvements.


Hybrid Search Implementation:
To improve upon the pure semantic search, I implemented a hybrid search approach combining FAISS with BM25. This involved creating a BM25 index for the products and combining the BM25 scores with the semantic similarity scores. The hybrid approach showed improvements over pure semantic search across most metrics.
Reranking Considerations:
I explored various reranking options to further improve search results. This included considering Learning to Rank (LTR) approaches, specifically looking into LightGBM's LGBMRanker for pairwise ranking. I also considered BERT-based reranking but ultimately decided against it due to the computational constraints of the 16GB CPU.
Performance Optimization and Reusability:
Given the computational constraints, I implemented checkpointing mechanisms to save intermediate results (preprocessed data, embeddings, FAISS index, BM25 index). This allowed for quicker iterations and experimentation without having to reprocess the entire dataset each time.
Analytical Steps:
Throughout the process, I conducted various analytical steps:

Inspected the distribution of ESCI labels in the dataset
Analyzed the performance differences between semantic and hybrid search
Visualized the embeddings using PCA and t-SNE to understand the distribution of products in the embedding space
Examined sample queries and their top results to get a qualitative sense of search performance


Challenges and Solutions:
The main challenge was working with such a large dataset on limited hardware. I addressed this by:

Implementing sampling to work with a smaller subset of data
Using efficient indexing methods (FAISS) for fast similarity search
Implementing checkpointing to save and load intermediate results
Carefully considering the trade-offs between model complexity and performance


Final Implementation:
The final implementation includes:

A data loading and preprocessing pipeline with sampling options
Embedding generation using SentenceTransformer
FAISS indexing for semantic search
BM25 indexing for text-based search
A hybrid search combining semantic and BM25 scores
Comprehensive evaluation metrics
Data inspection and visualization tools
