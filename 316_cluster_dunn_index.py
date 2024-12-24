import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_dunn_index(data, labels):
    # Convert labels to a numpy array
    labels = np.array(labels)

    # Calculate intra-cluster distances
    intra_cluster_distances = []
    for cluster in set(labels):
        points_in_cluster = data[labels == cluster]
        if len(points_in_cluster) > 1:
            distances = pairwise_distances(points_in_cluster, metric='euclidean')
            intra_cluster_distances.append(np.min(distances[np.triu_indices_from(distances, k=1)]))
        else:
            intra_cluster_distances.append(0)

    # Calculate inter-cluster distances
    inter_cluster_distances = []
    cluster_centers = []
    for cluster in set(labels):
        points_in_cluster = data[labels == cluster]
        cluster_centers.append(np.mean(points_in_cluster, axis=0))

    for i, center_i in enumerate(cluster_centers):
        for j, center_j in enumerate(cluster_centers):
            if i >= j:
                continue
            distance = np.linalg.norm(center_i - center_j)
            inter_cluster_distances.append(distance)

    # Calculate Dunn's Index
    dunn_index = min(inter_cluster_distances) / max(intra_cluster_distances) if intra_cluster_distances else 0
    return dunn_index

def determine_optimal_clusters(tmp_data, min_k=3, max_k=10):
    dunn_indices = []
    
	# print some information about 'tmp_data'
    #print("Data shape:", tmp_data.shape)
    
	# 'tmp_data' is a numpy array with duplicate values that we need to make unique.
    # store the unique values in 'data' and remove duplicates from 'tmp_data'
    data = np.unique(tmp_data, axis=0)

	# print some information about 'data'
    #print("Data shape after removing duplicates:", data.shape)

    for k in range(min_k, max_k + 1):
        print("Calculating Dunn's Index for k =", k)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)
        dunn_index = calculate_dunn_index(data, labels)
        dunn_indices.append((k, dunn_index))
        print("Dunn's Index for k = ", k, " is ", dunn_index)

    optimal_k = max(dunn_indices, key=lambda x: x[1])[0]
    

    print("Optimal number of clusters: ", optimal_k)
    return optimal_k

def main():
    parser = argparse.ArgumentParser(description="Cluster data and assign clusters using Dunn's Validity Index.")
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
    
    print("Exiting early.")
    return

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(data)

    # Save the clustered data
    output_file = args.input_file.replace('.xlsx', '_clustered.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

if __name__ == "__main__":
    main()
