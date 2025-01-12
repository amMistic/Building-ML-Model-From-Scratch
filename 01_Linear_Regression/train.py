from sklearn.model_selection import train_test_split
from src.linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

def mse(y_test, predictions):
    '''
    Compute Mean Square Error values 
    
    Args:
        y_test : Actual Numpy Series with true values
        predictions : Predicted values 
    
    Returns:
        A float compute MSE values
    '''
    return np.mean((y_test - predictions)**2)

def main():
    '''
    Implementing Linear Regression model on custom generated dataset
    '''
    # Initialized the training dataset
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    
    # Split the dataset into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # plots all datapoints
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
    plt.show()

    # Load model and fit it.
    reg = LinearRegression(learning_rate=0.1, epochs=1000)
    reg.fit(X_train,y_train)
    
    # make prediction on testing data to evaluate the model performance
    predictions = reg.predict(X_test)

    # Mean Square Error 
    mse_ = mse(y_test, predictions)
    print('The Mean Square Error: ',round(mse_, 2))
    
    # predict on original generated dataset
    y_pred_line = reg.predict(X)
    
    # plot the linear regression line on training samples
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()
    
if __name__ == '__main__':
    main()