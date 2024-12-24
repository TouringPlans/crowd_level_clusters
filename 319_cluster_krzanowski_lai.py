import argparse
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def calculate_krzanowski_lai(data, labels, k):
    # Calculate the total within-cluster sum of squares (Wk)
    clusters = np.unique(labels)
    Wk = 0
    for cluster in clusters:
        points_in_cluster = data[labels == cluster]
        centroid = np.mean(points_in_cluster, axis=0)
        Wk += np.sum((points_in_cluster - centroid) ** 2)
    return Wk

def determine_optimal_clusters(data, min_k=2, max_k=10):
    Wk_values = []
    KL_scores = []

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)
        Wk = calculate_krzanowski_lai(data, labels, k)
        Wk_values.append(Wk)
        
        if len(Wk_values) > 2:
            Wk_prev = Wk_values[-2]
            Wk_next = Wk_values[-1]
            KL = ((Wk_prev - Wk_next) / Wk_values[-3]) if Wk_values[-3] != 0 else 0
            KL_scores.append((k - 1, KL))
            print(f"Krzanowski-Lai Index for k = {k - 1}: {round(KL,4)}")

    optimal_k = max(KL_scores, key=lambda x: x[1])[0] if KL_scores else min_k
    print("Optimal number of clusters:", optimal_k)
    return optimal_k

def main():
    parser = argparse.ArgumentParser(description="Cluster data and assign clusters using Krzanowski-Lai Index.")
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
        df = df.copy()
        df['avg_wait_this_day'] = df['avg_wait_this_day'].fillna(df['avg_wait_this_day'].mean())

    # round avg_wait_this_day to the nearest whole integer
    df['avg_wait_this_day'] = df['avg_wait_this_day'].round(0)
	
	# Extract the data for clustering
    data = df[['avg_wait_this_day']].to_numpy()
    
	# Get unique values and their counts
    unique_values, counts = np.unique(data, return_counts=True)

	# Display frequencies
    for value, count in zip(unique_values, counts):
        print(f"Value {value} appears {count} times.")

    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(data)
    
    print("Exiting early")
    exit(0)

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(data)

    # Save the clustered data
    output_file = args.input_file.replace('.xlsx', '_clustered.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

if __name__ == "__main__":
    main()
