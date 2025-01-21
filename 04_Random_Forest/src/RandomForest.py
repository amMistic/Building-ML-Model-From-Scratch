import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple, Optional, Union
from .DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees : int = 100, max_depth : int = 10, min_saamples_split : int = 2, n_features : int = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_saamples_split
        self.n_features = n_features
        self.trees = []    
        
        
    def fit(self, X : pd.DataFrame, y : pd.DataFrame, X_val : Optional[pd.DataFrame] = None, y_val : Optional[pd.DataFrame] = None) -> None:
        '''
        Fit the Random Forest Classifier model.
        Build `n_estimator` Decision Tree, used DT , predict the target feature value.
        
        Args:
            X : Dataframe consist full of Independent variables
            y : DataFrame or Series of target variable vlaues
            X_val : Validation Dataset consisting independent variables
            y_val : Validaton Dataset consist target variable values
            
        '''
        
        # build the bootstrap 
        self.bootstrap_dataset = self._get_bootstrap_dataset(X, y)
        
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.n_features)
            X_samples, y_samples = self._get_bootstrap_dataset(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree) 
            
    
    def _get_bootstrap_dataset(self, X : pd.DataFrame, y : pd.DataFrame) -> pd.DataFrame : 
        
        '''
        Build the bootstrap dataset from the original dataset.
        Select the random training samples from the original dataset and also allow the repetate ones, untill 
        lenght of bootstrap dataset is equal to original dataset
        
        Args: 
            X : Original Dataset with all features
            y : Original Series with target feature values
        
        Returns:
            Bootstrapped Dataset.
        '''
        
        # get the total number of training samples
        n_samples = X.shape[0]
        
        # random indexs
        inds = np.random.choice(n_samples, n_samples, replace=True)
        
        # return the bootstrap dataset
        return X[inds], y[inds]        

    def predict(self, X : Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        '''
        Predict for the given Input Dataframe full of features(independent variables)
        Pass the inputs throught each tree
        Return the most common answer or prediction.
        
        Args:
            X  : Pandas DataFrame or NumpySeries 
        
        Returns:
            Return numpy array consisting all the predicted values
        '''
        # pass the inputs throught each tree and predict the value 
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        #
        tree_predicts = np.swapaxes(predictions, 0, 1)
        
        # return the prediction with most votes
        return np.array([self._most_common(pred) for pred in tree_predicts])

    
    def _most_common(self, y : np.array):
        '''
        Find the prediction which is most common among all the trees
         
        Args: 
            y : numpy array full of prediction of various DT

        Returns:
            The prediction which is most common will considered as 
            final predictions
        '''
        
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    