from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from src.RandomForest import RandomForest

def main():
    # load the dataset
    data  = load_breast_cancer()
    X = data.data
    y = data.target
    
    # split the data set into traing and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    ) 
    
    # also create cross validation set
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.4, random_state=42
    )
    
    # load the random forest model
    model = RandomForest(n_trees=50)
    model.fit(X_train, y_train)
    
    # prediction on training dataset
    y_train_pred = model.predict(X_train)
    print(f'The Accuracy of RandomForest on training data: {round(accuracy_score(y_train, y_train_pred), 4)}')
    
    # Test the model on testing dataset
    y_test_pred = model.predict(X_test)
    print(f'The Accuracy of Random model on testing data : {round(accuracy_score(y_test, y_test_pred), 4)}')

    # Test the model on cross validation set
    y_val_pred = model.predict(X_val)
    print(f'The Accuracy of Random model on Cross data : {round(accuracy_score(y_val, y_val_pred), 4)}')

if __name__ == '__main__':
    main()