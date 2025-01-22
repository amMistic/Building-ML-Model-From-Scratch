import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from collections import Counter

class KNN:
    def __init__(self, K: int):
        self.K = K
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        Fit K-Nearest Neighbours model.

        Args:
            X: Input data (features) as a Pandas DataFrame or NumPy array.
            y: Target values as a Pandas Series or NumPy array.
        """
        self.X_train = np.array(X)  # Convert to NumPy array for consistency
        self.y_train = np.array(y)  # Convert to NumPy array for consistency
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the Euclidean Distance between two data points.

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            The Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict the target value for each sample in the input data.

        Args:
            X: Input data (features) as a Pandas DataFrame or NumPy array.

        Returns:
            NumPy array of predicted target values.
        """
        X = np.array(X)  # Ensure input is a NumPy array
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x: np.ndarray) -> Union[int, float]:
        """
        Predict the target value for a single sample.

        Args:
            x: A single sample as a NumPy array.

        Returns:
            Predicted target value (most common among K nearest neighbors).
        """
        # Calculate distances to all training samples
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of the K nearest neighbors
        k_nearest_inds = np.argsort(distances)[:self.K]
        
        # Get the labels of the K nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_inds]
        
        # Return the most common label
        counter = Counter(k_nearest_labels)
        return counter.most_common(1)[0][0]
