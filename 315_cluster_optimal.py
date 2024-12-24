import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import matplotlib.pyplot as plt
#
# Also try: 
# Dunn's Validity Index
# Davies-Bouldin Index
# Calinski-Harabasz Index
# Measures of Krzanowski and Lai

# declare the standard deviation limit to be 0.20
std_dev_limit = 0.20
min_average_wait_time_for_std_dev_test = 20

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Determine optimal number of clusters based on date and average posted wait time.")
    parser.add_argument('--input-file', required=True, help='Path to the input Excel file')
    parser.add_argument('--output-file', required=True, help='Path to the output Excel file')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory for charts and reports')
    parser.add_argument('--max-clusters', required=True, help='Maximum number of clusters to try')
    parser.add_argument('--min-clusters', required=True, help='Minimum number of clusters to try')
    return parser.parse_args()

def load_data(file_path):
    print("Loading data from " + file_path)
    # Load the Excel file
    df = pd.read_excel(file_path)
    return df

def calculate_average_wait(df):
    # Calculate the average estimated wait for each park day
    print("Calculating average wait time for each day...")
    avg_wait_per_day = df.groupby('park_day')['estimated_wait'].mean().reset_index()

    # print the first few rows of the dataframe
    print(avg_wait_per_day.head())

    # round avg_wait_this_day to the nearest whole number
    avg_wait_per_day['estimated_wait'] = avg_wait_per_day['estimated_wait'].round(0)
    
    avg_wait_per_day.columns = ['park_day', 'avg_wait_this_day']
    
    # Merge the calculated averages back into the original dataframe
    print("Merging the calculated averages back into the original dataframe...")
    df = pd.merge(df, avg_wait_per_day, on='park_day', how='left')
    
    return df, avg_wait_per_day

def determine_optimal_clusters(avg_wait_per_day, minN, maxN):
    # Ensure valid range for clusters
    if minN > maxN:
        minN, maxN = maxN, minN
    if minN < 3:
        minN = 3
    if maxN > 10:
        maxN = 10

    silhouette_scores = []
    K = range(minN, maxN + 1)
    best_k = minN
    best_score = -1

    for k in K:
        print(f"\nTrying clustering with {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=0)
        avg_wait_per_day['cluster_num'] = kmeans.fit_predict(avg_wait_per_day[['avg_wait_this_day']])

        # Calculate silhouette score
        score = silhouette_score(avg_wait_per_day[['avg_wait_this_day']], avg_wait_per_day['cluster_num'])
        silhouette_scores.append(score)

        # Summarize clusters for the current k
        cluster_summary = avg_wait_per_day.groupby('cluster_num')['avg_wait_this_day'].agg(['mean', 'std']).reset_index()
        cluster_summary.columns = ['Cluster', 'Average Wait Time', 'Standard Deviation']

        print(f"Summary for {k} clusters:")
        print(cluster_summary)

        # see if any of the clusters has a std_dev:avg_wait ratio above 0.20 and
        # and average wait time > 20 minutes. If so, do not consider this 
        # k for the best score
        is_valid_cluster = True
        for index, row in cluster_summary.iterrows():
            z = row['Standard Deviation'] / row['Average Wait Time']
            if z > std_dev_limit and row['Average Wait Time'] > min_average_wait_time_for_std_dev_test:
                is_valid_cluster = False
                print(f"Cluster {row['Cluster']:.0f} has a std_dev:avg_wait ratio of {z:.2f} which is above {std_dev_limit:.2f}. Skipping this k.")
                break
        
        # see if this is the best score seen so far
        if score > best_score and is_valid_cluster:
            best_score = score
            best_k = k

        

    # Plot silhouette scores for different values of k
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'b-', marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different k')
    plt.savefig(args.output_dir + "/silhouette_scores.png")
    
    return best_k

def perform_clustering(df, avg_wait_per_day, optimal_k):
    print("Performing clustering with optimal number of clusters [" + str(optimal_k) + "]...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    avg_wait_per_day['cluster_num'] = kmeans.fit_predict(avg_wait_per_day[['avg_wait_this_day']])
    df = pd.merge(df, avg_wait_per_day[['park_day', 'cluster_num']], on='park_day', how='left')
    df.rename(columns={'cluster_num': 'cluster_number'}, inplace=True)
    return df, kmeans

def summarize_clusters(avg_wait_per_day):
    print("Summarizing clusters...")
    cluster_summary = avg_wait_per_day.groupby('cluster_num')['avg_wait_this_day'].agg(['count', 'mean', 'std']).reset_index()
    cluster_summary.columns = ['cluster_num', 'num_days', 'avg_wait', 'std_dev']
    return cluster_summary

if __name__ == "__main__":
    args = parse_args()
    df = load_data(args.input_file)
    df, avg_wait_per_day = calculate_average_wait(df)
    optimal_k = determine_optimal_clusters(avg_wait_per_day, int(args.max_clusters), int(args.min_clusters))
    print(f"\nOptimal number of clusters determined: {optimal_k}")
    df, kmeans = perform_clustering(df, avg_wait_per_day, optimal_k)
    cluster_summary = summarize_clusters(avg_wait_per_day)
    print("\nFinal Cluster Summary:")
    print(cluster_summary)
    print("Saving updated data to [" + args.output_file + "]")
    df.to_excel(args.output_file, index=False)
