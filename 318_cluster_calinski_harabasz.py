import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import numpy as np

def determine_optimal_clusters(data, min_k=2, max_k=10):
    ch_scores = []

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)
        ch_index = calinski_harabasz_score(data, labels)
        ch_scores.append((k, ch_index))
        print(f"Calinski-Harabasz Index for k = {k}: {ch_index}")

    optimal_k = max(ch_scores, key=lambda x: x[1])[0]  # Higher Calinski-Harabasz Index is better
    print("Optimal number of clusters:", optimal_k)
    return optimal_k

def main():
    parser = argparse.ArgumentParser(description="Cluster data and assign clusters using Calinski-Harabasz Index.")
    parser.add_argument("--input-file", required=True, help="Path to the input Excel file.")
    args = parser.parse_args()

    # Load Excel file
    df = pd.read_excel(args.input_file)

    # only keep rows with unique columns of 'park_day' and 'avg_wait_this_day'
    df = df.drop_duplicates(subset=['park_day', 'avg_wait_this_day'])

    if 'avg_wait_this_day' not in df.columns:
        raise ValueError("The input file must contain an 'avg_wait_this_day' column.")

    # Handle missing values
    if df['avg_wait_this_day'].isna().any():
        print("Warning: Missing values detected in 'avg_wait_this_day'. Filling with column mean.")
        #df['avg_wait_this_day'].fillna(df['avg_wait_this_day'].mean(), inplace=True)
        df = df.copy()
        df['avg_wait_this_day'] = df['avg_wait_this_day'].fillna(df['avg_wait_this_day'].mean())
        
	# round avg_wait_this_day to the nearest whole integer
    df['avg_wait_this_day'] = df['avg_wait_this_day'].round(0)

    # Extract the data for clustering
    data = df[['avg_wait_this_day']].to_numpy()

    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(data)

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(data)

    # Save the clustered data
    output_file = args.input_file.replace('.xlsx', '_clustered.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

if __name__ == "__main__":
    main()
