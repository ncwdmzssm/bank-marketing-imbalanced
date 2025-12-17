from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd

def evaluate(y_true, y_pred, name):
    """
    Evaluate model performance using metrics suitable for imbalanced datasets.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Estimated targets as returned by classifier.
        name: Name of the model (for identification in output).
    
    Returns:
        A dictionary containing Accuracy, F1-Score, Recall (Minority), and G-Mean.
    """
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary') # Assuming binary classification
    recall_minority = recall_score(y_true, y_pred, pos_label=1) # Assuming minority class is labeled as 1
    
    # Calculate G-Mean
    # G-Mean is the square root of the product of the sensitivity (recall) of each class
    # For binary classification: sqrt(recall_class_0 * recall_class_1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate recall (sensitivity) for each class
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    g_mean = np.sqrt(recall_class_0 * recall_class_1)
    
    # Create a result dictionary
    results = {
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'F1-Score': round(f1, 4),
        'Recall (Minority)': round(recall_minority, 4),
        'G-Mean': round(g_mean, 4)
    }

    print(f"--- {name} Evaluation Metrics ---")
    for metric, value in results.items():
        if metric != 'Model':
            print(f"{metric}: {value}")

    return results