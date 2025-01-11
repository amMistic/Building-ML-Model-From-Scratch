import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.costs: List[float] = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation"""
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features: int) -> None:
        """Initialize weights and bias"""
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
    def compute_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss"""
        m = len(y)
        cost = -(1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return float(cost)
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias"""
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = float((1/m) * np.sum(y_pred - y))
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model
        
        Parameters:
        X: np.ndarray, shape (m_samples, n_features)
        y: np.ndarray, shape (m_samples,) or (m_samples, 1)
        """
        m, n = X.shape
        self.initialize_parameters(n)
        
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        
        for i in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            cost = self.compute_cost(y, y_pred)
            self.costs.append(cost)
            
            dw, db = self.compute_gradients(X, y, y_pred)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1
        
        Parameters:
        X: np.ndarray, shape (m_samples, n_features)
        
        Returns:
        np.ndarray: Probability predictions, shape (m_samples, 1)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels
        
        Parameters:
        X: np.ndarray, shape (m_samples, n_features)
        threshold: float, classification threshold
        
        Returns:
        np.ndarray: Binary predictions, shape (m_samples, 1)
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot decision boundary for 2D data
        
        Parameters:
        X: np.ndarray, shape (m_samples, 2)
        y: np.ndarray, shape (m_samples,) or (m_samples, 1)
        """
        if X.shape[1] != 2:
            raise ValueError("This function only works for 2D data")
            
        plt.figure(figsize=(10, 8))
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.show()
        
    def plot_cost_history(self) -> None:
        """Plot the cost history during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost History During Training')
        plt.show()