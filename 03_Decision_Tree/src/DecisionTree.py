import numpy as np
import pandas as pd
from src.Node import Node
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, Tuple, Union

class DTClassifier:
    def __init__(self,
                max_depth : int = 100,
                min_sample_splits : int = 2,
                n_features : int = None):
        '''
        Initialized stopping criteria parameters
        
        Args: 
            max_depth : Represent maximum depth decision tree can have
            min_sample_splits : Represent minimum samples required to split
            n_features : Represent number of features decision tree build on.
        '''
        # Parameters
        self.max_depth = max_depth
        self.min_sample_splits = min_sample_splits
        self.n_features = n_features
        
        #
        self.root = None
    
    
    def fit(self, X : np.ndarray, y : np.ndarray):
        ''''
        Fit the decision tree model on training data for classification
        
        Args:
            X : Numpy Array of features
            y : Numpy Series containing labels 
        '''
        
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._build_tree(X,y)
        
    
    def _build_tree(self, X : np.ndarray, y : np.ndarray, depth : int = 0):
        '''
        Build Decision Tree 
        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # checking Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_sample_splits or n_labels == 1):
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)
            
        # getting the random features 
        feats_indx = np.random.choice(n_feats, self.n_features, replace=False)
        
        # get the best split features and best split threshold
        best_features, best_threshold = self._get_best_splits(X, y, feats_indx)
        
        # split the data
        left_indx, right_indx = self._splits(X[:, best_features], best_threshold)
        left_child = self._build_tree(X[left_indx, :], y[left_indx], depth= depth + 1)
        right_child = self._build_tree(X[right_indx,:], y[right_indx], depth= depth + 1)
        return Node(feature=best_features, left = left_child, right= right_child, threshold=best_threshold)
        
    
    def _most_common(self, y : np.ndarray) -> Union[int, float]:
        '''
        Check for most common label within the series
        
        Args:
            y : Numpy series containing lables
        
        Return:
            Most common label in `y`
        '''
        counter = Counter(y)
        value = counter.most_common(1)[0][0] # `.most_common()` --> Return the list of tuple. 
        return value             # Hence, to access the `key`, most_common[0][0].
    
    def _get_best_splits(self, X : np.ndarray, y : np.ndarray, features_indx : np.ndarray) -> Tuple[float, int]:
        '''
        Get the best split features and best split threshold
        
        Args: 
            X : Numpy Array containing features
            y : Numpy series containing labels
            features_indx : Numpy array containg randomly selected feature indices  
        '''
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature_ind in features_indx:
            X_column = X[:, feature_ind]
            threshold = np.unique(X_column)
            
            for thre in threshold:
                # calculate information gain
                gain = self._information_gain(y, X_column, thre)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_ind
                    split_threshold = thre
        
        return split_idx, split_threshold

    
    def _information_gain(self, y : np.ndarray, X_column : np.ndarray, thre : float) -> float:
        '''
        Calculat the information gain
        
        Args:
            y : Numpy Series full of labels
            X_column : Numpy series containing segment of features
            thre : Threshold on which split would occur 
        
        Return:
            Float Values represent the total information gain
        '''
        
        # parent entropy
        parent_entropy = self._entropy(y)
        
        # Determine the left and right child indices
        left_child_ind, right_child_ind = self._splits(X_column, thre)
        if len(left_child_ind) == 0 or len(right_child_ind) == 0:
            return 0

        # weightage average 
        n_samples = len(y)
        n_samples_left_child, n_samples_right_child = len(left_child_ind), len(right_child_ind)
        
        # child entropy
        left_child_entropy = self._entropy(y[left_child_ind])
        right_child_entropy = self._entropy(y[right_child_ind])
        
        # Total children entropy
        child_entropy = (n_samples_left_child / n_samples) * left_child_entropy + (n_samples_right_child / n_samples) * right_child_entropy
        
        # Total Information gain 
        IG = parent_entropy - child_entropy
        return IG
                    
                    
    def _splits(self, X_column : np.ndarray, thre : float) -> Tuple[np.ndarray, np.ndarray] :
        '''
        Split the X_column based on the given threshold
        
        Args:
            X_column : Numpy series of features
            Threshold : Splitting Threshold
            
        Returns:
            Tuple of numpy array containing the indicies of left and rigth child
        '''
        left_child_ind = np.argwhere(X_column <= thre).flatten()
        right_child_ind = np.argwhere(X_column > thre).flatten()
        return left_child_ind, right_child_ind
        
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates the entropy of a label distribution.
        
        Args:
            y: Numpy series containing labels
            
        Return:
            Float value represent entropy of lable distribution
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])
        
        
    def predict(self, X : np.ndarray):
        '''
        Predict the lables of the input array
        '''
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node : Node) -> Union[int, float]:
        '''
        Traverse the tree from root node to leaf node
        
        Args:
            X : Input numpy array containing features
        
        Return:
            Predict labels for each corresponding training sample
        '''
        # check if not or not
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)