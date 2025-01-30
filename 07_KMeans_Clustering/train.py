from src.KMeans import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def main():
    """
    Main function to perform K-Means clustering on synthetic data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic dataset with 3 centers
    X, y = make_blobs(n_samples=500, centers=3, n_features=2, shuffle=True, random_state=40)
    
    # Display dataset shape and number of unique clusters
    print(f"Dataset Shape: {X.shape}")
    num_clusters = len(np.unique(y))
    print(f"Number of Clusters: {num_clusters}")

    # Initialize and train the K-Means model
    kmeans = KMeans(K=num_clusters, epochs=150, plot=True)
    cluster_labels = kmeans.fit(X)

    # Plot the final clustering results
    kmeans._plot()

    # Display final cluster assignments
    print(f"Final Cluster Labels: {np.unique(cluster_labels)}")

if __name__ == '__main__':
    main()
