import numpy as np 
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class LinearRegression:
    def __init__(self, 
                learning_rate: float = 0.0001, 
                epochs: int = 100,
                batch_size: int = 32):
        '''
        Initialized parameters and given arguments
        
        Args:
            learning_rate : Determined the step rate while computing gradient of parameters
            epochs : Number of time model train on same data
            batch_size : Number of training samples in each mini-batch. Default value
        '''
        
        # Parameters
        self.weights = None
        self.bias = None
        
        # History
        self.history_loss = []
        
        # Hyperparameters
        self.learning_rate = learning_rate  
        self.epochs = epochs 
        self.batch_size = batch_size 
        
    def _initialize_parameters(self, n_features : int):
        '''
        Initializing parameters for like self.weigths and self.bias shape 
        
        Args:
            n_features : Number of uniques features 
        '''
        self.weights = np.random.randn(n_features) * 0.001
        self.bias = 0.0
    
    def _compute_prediction(self, X : np.ndarray) -> np.ndarray:
        '''
        Computes the target values for given input stream
        
        Args:
            X : Input feature of shape (n_samples, n_features)
        
        Returns:
            Predicted values of input features of shape (n_samples,)
        '''
        return np.dot(X, self.weights) + self.bias

    def _compute_gradient_descent(self,
                                  X : np.ndarray,
                                  y : np.ndarray,
                                  y_pred : np.ndarray) -> Tuple[np.ndarray, float]:
        '''
        Compute Gradient Descent to minimized the cost/loss function and estimating most appropriate values of Weights and bias
        
        Args:
            X : Numpy array of features 
            y : Numpy series consisting actual value of target feature
            y_pred : Numpy series consisting predicted value for target feature
        
        Returns:
            Tuple of gradient_weights and gradient bias 
        '''
        # total number of training samples
        m = len(y)
        
        # compute error
        error = y_pred - y
        
        # compute gradient weights
        gradient_weights = (1/m) * np.dot(X.T, error)
        
        # compute gradient bias
        gradient_bias = (1/m) * np.sum(error)
        
        return gradient_weights, gradient_bias
    
    def _compute_error(self, y : np.ndarray, y_pred : np.ndarray) -> float:
        '''
        Compute the error between the predicted value and actual value of target feature.
        
        Args:
            y_pred : Predict values of target feature
            y : actual values of target features
            
        Returns:
            Mean Sqaure Error Cost
        '''
        m = len(y)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost
    
    def fit(self, 
            X: np.ndarray, 
            y : np.ndarray,
            X_val : Optional[np.ndarray] = None,
            y_val : Optional[np.ndarray] = None):
        '''
        Try to fit and train the model using mini-batch gradient descent
        
        Args:
            X : Input Feautres
            y : Target Feature
            X_val: Optional validation features
            y_val: Optional validation target values
        '''
        # Normalize input features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std  # Feature scaling
        
        # Convert into Numpy array
        X = np.array(X)
        y = np.array(y)
        
        # number of features
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n_features=n_features)
        
        # Train model
        for epoch in range(self.epochs):
            # shuffle the mini-batches
            Indices = np.random.permutation(n_samples)
            
            for start_index in range(0, n_samples, self.batch_size):
                batch_ind = Indices[start_index : start_index + self.batch_size]
                
                # get the data
                X_batch = X[batch_ind]
                y_batch = y[batch_ind]
                
                # compute prediction
                y_batch_pred = self._compute_prediction(X_batch)
                
                # compute gradient of parameters
                gradient_weights, gradient_bias = self._compute_gradient_descent(X_batch, y_batch, y_batch_pred)
                
                # update parameters
                self.weights -= self.learning_rate * np.clip(gradient_weights, -1, 1)
                self.bias -= self.learning_rate * np.clip(gradient_bias, -1, 1)
            
            y_pred = self._compute_prediction(X)
            cost = self._compute_error(y, y_pred)
            self.history_loss.append(cost)
            
            if epoch % 100 == 0:
                print(f'Epoch:{epoch} Cost: {cost:.4f}')        

                # If validation data is provided, compute validation cost
                if X_val is not None and y_val is not None:
                    y_val_pred = self._compute_prediction(X_val)
                    val_cost = self._compute_error(y_val, y_val_pred)
                    print(f"Validation Cost = {val_cost:.4f}")
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        '''
        Predict the values of target feature for corresponding training samples
        
        Args:
            X : Input numpy array features on which prediction will makes.
        
        Returns:
            Numpy Series of predicted values
        '''
        return self._compute_prediction(X)

    def plot_loss(self):
        '''
        Visualized the trend of loss with number of epochs.
        '''
        plt.figure(figsize=(10,16))
        plt.plot(range(self.epochs), self.history_loss)
        plt.title('Loss v/s Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
    