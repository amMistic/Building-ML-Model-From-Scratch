import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        self.metrics: Dict = {}
        
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score"""
        return np.mean(y_true == y_pred)
    
    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score"""
        cm = self.confusion_matrix(y_true, y_pred)
        return cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    
    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score"""
        cm = self.confusion_matrix(y_true, y_pred)
        return cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    def roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve"""
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            cm = self.confusion_matrix(y_true, y_pred)
            tpr = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            fpr = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            
        return np.array(fpr_list), np.array(tpr_list), thresholds
    
    def auc_score(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Calculate AUC score"""
        return np.trapz(tpr, fpr)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate all metrics"""
        self.metrics['confusion_matrix'] = self.confusion_matrix(y_true, y_pred)
        self.metrics['accuracy'] = self.accuracy(y_true, y_pred)
        self.metrics['precision'] = self.precision(y_true, y_pred)
        self.metrics['recall'] = self.recall(y_true, y_pred)
        self.metrics['f1_score'] = self.f1_score(y_true, y_pred)
        
        fpr, tpr, thresholds = self.roc_curve(y_true, y_prob)
        self.metrics['roc_curve'] = (fpr, tpr, thresholds)
        self.metrics['auc_score'] = self.auc_score(fpr, tpr)
        
        return self.metrics
    
    def plot_roc_curve(self) -> None:
        """Plot ROC curve"""
        if 'roc_curve' not in self.metrics:
            raise ValueError("Run evaluate() first")
            
        fpr, tpr, _ = self.metrics['roc_curve']
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {self.metrics["auc_score"]:.3f}')
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()