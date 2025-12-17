from sklearn.ensemble import RandomForestClassifier

def run_baseline(X_train, y_train, X_test):
    """
    Train a baseline model (RandomForestClassifier with default parameters)
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
    
    Returns:
        base_model: The trained model
        base_pred: Predictions on the test set
    """
    # Initialize Random Forest Classifier with default parameters
    base_model = RandomForestClassifier(random_state=42)
    
    # Train the model
    base_model.fit(X_train, y_train)
    
    # Predict on the test set
    base_pred = base_model.predict(X_test)
    
    return base_model, base_pred