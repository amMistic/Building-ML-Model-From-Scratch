import numpy as np
import pandas as pd
from typing import Union

class SVM:
    def __init__(self, 
                 learning_rate: float = 0.001,
                 lambda_param: float = 0.01,
                 epochs: int = 1000,
                 kernel: str = 'linear',
                 degree: int = 2,
                 gamma: float = 1.0,
                 coef0: float = 0.01):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.kernel = kernel.lower()
        self.degree = degree
        self.epochs = epochs
        self.gamma = gamma
        self.coef0 = coef0
        
        self.kernel_matrix = None
        self.alpha = None
        self.bias = None
        self.X = None
        self.y = None
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]):
        '''
        Fit the model over the training data
        
        Args:
            X: Input dataset containing independent features
            y: Labels corresponding to training samples in X
            
        '''
        # Ensure inputs are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get number of samples and features
        n_samples, n_features = X.shape
        
        # Update labels (y should be in range of (-1, 1))
        y_ = np.where(y <= 0, -1, 1) 
        self.y = y_
        self.X = X
        
        # Initialize parameters
        self.alpha = np.zeros(n_samples)
        self.bias = 0.0
        
        # Kernel matrix
        self.kernel_matrix = np.zeros((n_samples, n_samples))  # Fixing the shape
        for i in range(n_samples):
            for j in range(n_samples):  # Kernel matrix should be square
                self.kernel_matrix[i, j] = self._kernel_function(X[i], X[j])
        
        # Gradient Descent on Lagrange multipliers to update alpha and bias
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y_[i] * (np.sum(self.alpha * y_ * self.kernel_matrix[i]) + self.bias) >= 1
                if not condition:
                    self.alpha[i] += self.learning_rate * (
                        1 - y_[i] * np.sum(self.alpha * y_ * self.kernel_matrix[i]) - self.lambda_param * self.alpha[i]
                    )
                    self.bias += self.learning_rate * y_[i]

    def _kernel_function(self, x1, x2):
        '''
        Access the kernel function depending on the user-defined type
        '''
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
        else:
            raise ValueError(f'Unsupported kernel type: {self.kernel}')
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        '''
        Predict the target label using kernel trick on input data

        Args:
            X: Dataset containing independent variables
        
        Returns:
            np.ndarray: Predicted labels
        '''   
        X = np.array(X)
        y_pred = []
        for x in X:
            decision = np.sum(self.alpha * self.y * 
                              np.array([self._kernel_function(x, xi) for xi in self.X])) + self.bias
            y_pred.append(np.sign(decision))
        
        return np.array(y_pred)
