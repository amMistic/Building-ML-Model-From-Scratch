import numpy as np

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    X : array-like
        Features dataset
    y : array-like
        Target dataset
    test_size : float
        Should be between 0.0 and 1.0 and represent the proportion of the 
        dataset to include in the test split
    random_state : int
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The split datasets
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get total number of examples
    n_samples = len(X)
    
    # Calculate number of test samples
    n_test = int(n_samples * test_size)
    
    # Create random indices for shuffling
    indices = np.random.permutation(n_samples)
    
    # Split indices into training and test
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# 9819225564 ->  "Rubhji"
# 8237745923 -> ""

# 97026166305