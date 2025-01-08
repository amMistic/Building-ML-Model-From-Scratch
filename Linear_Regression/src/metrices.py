import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics for regression models
    
    Parameters:
    y_true: Array of actual values
    y_pred: Array of predicted values
    
    Returns:
    dict: Dictionary containing different metrics
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R-squared (R²)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # Calculate Adjusted R-squared (assuming one feature for simplicity)
    n = len(y_true)
    p = 1  # number of features (modify as needed)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    accuracy_matrices = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adjusted_r2': adjusted_r2
    }
    
    print(f'The Accuracy Matrices : \n')
    for key, value in accuracy_matrices.items():
        print(f'{key} : {value}')
    print('\n\n')