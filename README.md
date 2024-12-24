# Unsupervised Clustering Techniques in Python To Determine Theme Park Crowd Levels

TouringPlans created this repository to share different ideas for how to determine crowd levels at theme parks. It includes data for one attraction: The Amazing Adventures of Spider-Man at Universal Orlando's Islands of Adventure theme park.

The repository includes these sample programs that implement these clustering algorithms:

- 315_cluster_optimal.py - Silhouette Coefficients using iterative K-Means
- 316_cluster_dunn_index.py - Dunn's Index
- 317_cluster_davies_bouldin.py - The Davies-Bouldin index
- 317_cluster_calinski_harabasz.py - The Calinski-Harabasz method
- 319_cluster_krzanowski_lai.py - The iterative method of Krzanowski and Lai
- 902_cluster_gmm.py - Gaussian mixture models

## Installation and Configuration

I ran these programs on macOS 14.6.1 using the terminal app and python 3.12.3.
You'll need these python packages: argparse, matplotlib, numpy, pandas, and sklearn.

## Running the Code

Unzip the data file with this command:

unzip IA01_215_valid.xlsx.zip

(IA01 is TouringPlans' internal name for The Amazing Adventures of Spider-Man.)

Run the programs this way:

> $ python3 902_cluster_gmm.py --input-file=./IA01_215_valid.xlsx

> $ python3 319_cluster_krzanowski_lai.py --input-file=./IA01_215_valid.xlsx

> $ python3 318_cluster_calinski_harabasz.py --input-file=./IA01_215_valid.xlsx

> $ python3 317_cluster_davies_bouldin.py --input-file=./IA01_215_valid.xlsx

> $ python3 316_cluster_dunn_index.py --input-file=./IA01_215_valid.xlsx

Each of these will output various statistics about the clusters it has evaluated.

The program 315_cluster_optimal.py takes a few more command-line arguments and produces more output:

> $ python3 315_cluster_optimal.py --input-file=./IA01_215_valid.xlsx --output-file=IA01_clustered.xlsx --output-dir=./ --max-clusters=10 --min-clusters 3

Where:

> output-file is an output file that'll contain a new column with the crowd level for each date

> output-dir is a directory into which the program will place a chart of the silhouette scores

> min-clusters and --max-clusters are the minumum and maximum number of clusters to consider

## Getting Help

Comments, suggestions, and code improvements are welcome. Create a pull request if you do something near.

For questions that you think would be better done via email than github: len@touringplans.com.

## License

This software is made available via the GNU GLPv3 license. The usual caveats apply: For the love of God, don't use anything I've written for anything important. It's almost certainly wrong, and the best outcome you can hope for is minor catastrophic damage.


