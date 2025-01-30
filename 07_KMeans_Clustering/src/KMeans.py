import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt

# Global Function
def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        x1: First point as a numpy array.
        x2: Second point as a numpy array.
        
    Returns:
        Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K: int = 3, epochs: int = 100, plot: bool = False):
        """
        Initializes the K-Means clustering model.
        
        Args:
            K: Number of clusters.
            epochs: Number of iterations.
            plot: Whether to plot the clustering process.
        """
        self.K = K
        self.epochs = epochs
        self.plot = plot
        self.centroids = None  # Stores centroids
        self.clusters = None   # Stores assigned clusters
        self.X = None
        self.n_samples = None
        self.n_features = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fits the K-Means model on the given dataset.
        
        Args:
            X: Input dataset (features only).
            
        Returns:
            Cluster labels for each data point.
        """
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape
        
        # Initialize centroids randomly from the dataset
        random_inds = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = self.X[random_inds]
        
        for _ in range(self.epochs):
            self.clusters = self._create_clusters()  # Assign points to clusters
            new_centroids = self._compute_new_centroids()  # Recalculate centroids
            
            # Check for convergence (if centroids don't change, stop training)
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
            if self.plot:
                self._plot()
        
        return self._get_cluster_labels()
    
    def _create_clusters(self):
        """
        Assigns each sample to the nearest centroid to form clusters.
        
        Returns:
            A list of cluster assignments (indices of data points in each cluster).
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            closest_centroid = self._find_closest_centroid(sample)
            clusters[closest_centroid].append(idx)
        return clusters

    def _find_closest_centroid(self, sample: np.ndarray) -> int:
        """
        Finds the closest centroid index for a given sample.
        
        Args:
            sample: A single data point.
            
        Returns:
            Index of the nearest centroid.
        """
        distances = [euclidean_distance(sample, centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def _compute_new_centroids(self) -> np.ndarray:
        """
        Computes new centroids as the mean of assigned clusters.
        Returns:
            New centroid positions as a numpy array.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:  # Avoid division by zero
                centroids[cluster_idx] = np.mean(self.X[cluster], axis=0)
        return centroids
    
    def _get_cluster_labels(self) -> np.ndarray:
        """
        Assigns labels to each sample based on the cluster assignment.
        Returns:
            Array of cluster labels for each data point.
        """
        labels = np.empty(self.n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def _plot(self):
        """
        Plots the clustered data points along with centroids (only for 2D data).
        """
        if self.n_features != 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cluster in enumerate(self.clusters):
            points = self.X[cluster].T
            ax.scatter(*points)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker="x", color="black", linewidth=2)
        plt.show()
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts the cluster labels for new data points.
        Args:
            X: New data points.
        Returns:
            Predicted cluster labels for each data point.
        """
        X = np.array(X)
        return np.array([self._find_closest_centroid(sample) for sample in X])
