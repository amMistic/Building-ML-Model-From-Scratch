from sklearn.model_selection import train_test_split
from src.DecisionTree import DTClassifier
from sklearn import datasets
import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def main():
    # load dataset
    data = datasets.load_breast_cancer()
    
    # load dataset into feature-labels 
    X, y = data.data, data.target

    # split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # load the model (Decision Tree Classifier)
    clf = DTClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    
    # Evaluate the model on testing dataset
    predictions = clf.predict(X_test)
    
    # Compute the accuracy of the model
    acc = accuracy(y_test, predictions)
    print("The Accuracy of the Decision Tree Classifier: ",round(acc, 2))

if __name__ == '__main__':
    main()