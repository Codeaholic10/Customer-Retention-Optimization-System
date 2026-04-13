# Model Evaluation Module
"""
This module handles model evaluation and performance metrics.
"""
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Evaluate model performance using standard classification metrics.
    
    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional, needed for ROC-AUC)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics_dict = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    
    if y_prob is not None:
        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_prob)
        
    return metrics_dict


def generate_evaluation_report(model, X_test, y_test):
    """
    Generate a comprehensive evaluation report for the model.
    
    Args:
        model: Trained model (requires predict and optionally predict_proba methods)
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with full evaluation report metrics
    """
    y_pred = model.predict(X_test)
    y_prob = None
    
    if hasattr(model, 'predict_proba'):
        # Assuming binary classification, we take the probability of the positive class
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
            
    return evaluate_model(y_test, y_pred, y_prob)
