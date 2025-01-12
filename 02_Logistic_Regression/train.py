from src.logistic_Regression import LogisticRegression
from src.metrics import ModelEvaluator 
import pandas as pd

def main():
    
    # load the file
    df = pd.read_csv('data\\dataset.csv')
    
    Y = df['Target'].values
    X = df.drop(columns=['Target']).values
    
    # split the dataset into training and testing dataset
    train, test = df[:80], df[80:]
    
    # split the dataset into feature and target dataset
    y_train = train['Target'].values
    y_test = test['Target'].values
    
    X_train = train.drop(columns=['Target']).values
    X_test = test.drop(columns=['Target']).values
    
    # load the logistic regression model
    model = LogisticRegression(learning_rate=0.1, epochs=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # plot the loss 
    model.plot_cost_history()
    
    # plot the decision boundary
    model.plot_decision_boundary(X,Y)
    
    # evaluate
    evaluator = ModelEvaluator()
    accuracy = evaluator.recall(y_test, y_pred)
    print(f'The Testing Accuracy of model is: {accuracy}')
    
if __name__ == '__main__':
    main()