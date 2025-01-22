import numpy as np
from src.KNN import KNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def main():
    # load the dataset
    data = load_iris()
    X = data.data
    y = data.target
    
    # splits the dataset into training and testing dataset 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # split the testing dataset into testing and validation dataset
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.4, random_state=42
    )
    
    # load the KNN model 
    model = KNN(K=5)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    print(f'The Accuracy of KNN on training dataset: {round(accuracy_score(y_train, y_train_pred), 2)}')
    
    # Evaluate the model 
    ## Testing on testing dataset
    y_test_pred = model.predict(X_test)
    print(f'The Accuracy of KNN on testing dataset: {round(accuracy_score(y_test, y_test_pred), 2)}')
    
    ## Testing on validation model
    y_val_pred = model.predict(X_val)
    print(f'The Accuracy of KNN on validation dataset: {round(accuracy_score(y_val, y_val_pred), 2)}')
    

if __name__ == '__main__':
    main()