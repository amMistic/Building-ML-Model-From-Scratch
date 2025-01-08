from src.linear_regression import LinearRegression
from src.utils import custom_train_test_split
import matplotlib.pyplot as plt 
import numpy as np

def main():
    '''
    Implementing Linear Regression model on custom dataset
    '''

    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 3)  # 100 samples, 3 features
    true_weights = [0.5, -0.2, 1.5]
    y = np.dot(X, true_weights) + 0.1 * np.random.randn(100)  # Add noise
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = custom_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = LinearRegression(
        learning_rate=0.01,
        epochs=10000,
        batch_size=32
    )
    
    # Train the model
    model.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Plot training history
    model.plot_loss()
    
    # Print final weights
    print("\nTrue weights:", true_weights)
    print("Learned weights:", model.weights.round(3))
    print("Learned bias:", round(model.bias, 3))
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    main()
    
    