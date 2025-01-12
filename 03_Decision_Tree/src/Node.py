import numpy as np 
import pandas as pd
from typing import Optional, Union

 
class Node:
    def __init__(self, 
                feature : int = None,
                left : Optional['Node'] = None,
                right : Optional['Node'] = None,
                threshold : Optional[int] = None,
                *, value : Optional[Union[int, float]] = None):
        '''
        Initialized node 
        
        Args:
            feature : Feature Node take on among
            left : Represent left child of current node
            right : Represent right child of current node
            threshold : Threshold on which training samples splits into.
            value : Represent value of current node
        '''
        
        self.feature =  feature
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value
    
    def is_leaf_node(self):
        '''
        Check whether node is leaf node or parent node
        '''
        return self.value is not None