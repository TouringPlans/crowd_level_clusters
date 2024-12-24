import argparse
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np

def determine_optimal_clusters(data, min_k=2, max_k=10):
    bic_scores = []

    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bic = gmm.bic(data)  # Bayesian Information Criterion
        bic_scores.append((k, bic))
        print(f"BIC for k = {k}: {bic}")

    optimal_k = min(bic_scores, key=lambda x: x[1])[0]  # Lower BIC is better
    return optimal_k

def main():
    parser = argparse.ArgumentParser(description="Cluster data and assign clusters using Gaussian Mixture Models.")
    parser.add_argument("--input-file", required=True, help="Path to the input Excel file.")
    args = parser.parse_args()

    # Load Excel file
    df = pd.read_excel(args.input_file)

    if 'avg_wait_this_day' not in df.columns:
        raise ValueError("The input file must contain an 'avg_wait_this_day' column.")
    
	# only keep rows with unique columns of 'park_day' and 'avg_wait_this_day'
    df = df.drop_duplicates(subset=['park_day', 'avg_wait_this_day'])

    # Handle missing values
    if df['avg_wait_this_day'].isna().any():
        print("Warning: Missing values detected in 'avg_wait_this_day'. Filling with column mean.")
        df = df.copy()
        df['avg_wait_this_day'] = df['avg_wait_this_day'].fillna(df['avg_wait_this_day'].mean())

    # Extract the data for clustering
    data = df[['avg_wait_this_day']].to_numpy()

    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(data)

    # Perform clustering with the optimal number of clusters
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm.fit(data)
    df['cluster'] = gmm.predict(data)

    # Calculate and display cluster frequencies, percentages, and average 'avg_wait_this_day'
    cluster_counts = df['cluster'].value_counts().sort_index()
    total_points = len(df)
    print("Cluster Frequencies, Percentages, and Average 'avg_wait_this_day':")
    for cluster, count in cluster_counts.items():
        percentage = (count / total_points) * 100
        avg_wait = df[df['cluster'] == cluster]['avg_wait_this_day'].mean()
        print(f"Cluster {cluster}: {count} points ({percentage:.2f}%), Average 'avg_wait_this_day': {avg_wait:.2f}")

    # Save the clustered data
    output_file = args.input_file.replace('.xlsx', '_clustered.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

if __name__ == "__main__":
    main()
