import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.SVM import SVM  # Assuming your SVM code is in a file named svm.py

def main():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Use only two classes for binary classification
    X = X[y < 2]
    y = y[y < 2]

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the SVM model
    svm = SVM(kernel='polynomial', learning_rate=0.001, lambda_param=0.01, epochs=100, degree=3)

    # Train the SVM model
    print("Training the SVM model...")
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    print("Making predictions...")
    y_pred = svm.predict(X_test)

    # Convert predictions to binary labels (-1 -> 0, 1 -> 1)
    y_pred = np.where(y_pred <= 0, 0, 1)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
