"""Tree pruning and ensemble prediction for model efficiency optimization."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, recall_score


class TreePruner:
    """
    K-Means based tree clustering and selection for ensemble pruning.
    
    Core idea:
    1. Extract prediction matrices from all trees on validation set
    2. Cluster trees using K-Means into K clusters
    3. Select the tree with highest F1 score from each cluster
    4. Compute weights for selected trees based on their performance
    """
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        """
        Initialize TreePruner.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters for K-Means (typically 5-20, depending on tree count)
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.selected_trees = []
        self.tree_weights = np.array([])
        self.clustering_model = None
        self.cluster_centers = None
    
    def fit(
        self,
        rf_model,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TreePruner:
        """
        Fit the tree pruner on validation data.
        
        Parameters
        ----------
        rf_model : RandomForestClassifier
            Trained random forest model with estimators_
        X_val : np.ndarray
            Validation feature matrix (n_samples, n_features)
        y_val : np.ndarray
            Validation labels (n_samples,)
        
        Returns
        -------
        self : TreePruner
            Fitted pruner instance
        """
        n_trees = len(rf_model.estimators_)
        
        # Step 1: Extract prediction matrix from all trees on validation set
        # Shape: (n_samples, n_trees)
        prediction_matrix = np.zeros((X_val.shape[0], n_trees), dtype=np.float32)
        
        for tree_idx, tree in enumerate(rf_model.estimators_):
            # Get predictions from individual tree
            tree_pred = tree.predict_proba(X_val)[:, 1]  # probability of positive class
            prediction_matrix[:, tree_idx] = tree_pred
        
        # Step 2: Perform K-Means clustering on trees (use tree predictions as features)
        # Each tree is represented by its prediction vector across validation samples
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(prediction_matrix.T)  # Cluster trees, not samples
        self.clustering_model = kmeans
        self.cluster_centers = kmeans.cluster_centers_
        
        # Step 3: Select best tree from each cluster based on Recall score
        selected_trees = []
        tree_recall_scores = []
        
        for cluster_id in range(self.n_clusters):
            # Get trees in this cluster
            cluster_tree_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_tree_indices) == 0:
                continue
            
            # Compute Recall score for each tree in this cluster
            best_recall = -1.0
            best_tree_idx = cluster_tree_indices[0]
            
            for tree_idx in cluster_tree_indices:
                tree_pred = prediction_matrix[:, tree_idx]
                # Convert probabilities to hard predictions (threshold at 0.5)
                tree_pred_binary = (tree_pred >= 0.5).astype(int)
                recall = recall_score(y_val, tree_pred_binary, zero_division=0)
                
                if recall > best_recall:
                    best_recall = recall
                    best_tree_idx = tree_idx
            
            selected_trees.append(best_tree_idx)
            tree_recall_scores.append(best_recall)
        
        self.selected_trees = selected_trees
        
        # Step 4: Compute weights for selected trees
        # Normalize Recall scores to weights (higher Recall -> higher weight)
        tree_recall_scores = np.array(tree_recall_scores)
        
        # Avoid division by zero; if all Recall scores are 0, use uniform weights
        if tree_recall_scores.sum() > 0:
            self.tree_weights = tree_recall_scores / tree_recall_scores.sum()
        else:
            self.tree_weights = np.ones(len(selected_trees)) / len(selected_trees)
        
        return self
    
    def get_selected_trees(self) -> list:
        """Return list of selected tree indices."""
        return self.selected_trees
    
    def get_tree_weights(self) -> np.ndarray:
        """Return weights for selected trees."""
        return self.tree_weights


def predict_ensemble(
    rf_model,
    X: np.ndarray,
    selected_trees: list,
    tree_weights: np.ndarray,
    use_proba: bool = True,
) -> np.ndarray:
    """
    Make predictions using weighted ensemble of selected trees.
    
    Logic: ∑(Tree_i(x) × Weight_i)
    
    Parameters
    ----------
    rf_model : RandomForestClassifier
        Trained random forest model
    X : np.ndarray
        Feature matrix to predict on (n_samples, n_features)
    selected_trees : list
        List of selected tree indices
    tree_weights : np.ndarray
        Weights for each selected tree (should sum to 1)
    use_proba : bool
        If True, use predicted probabilities; if False, use hard predictions
    
    Returns
    -------
    predictions : np.ndarray
        Weighted ensemble predictions (n_samples,)
    """
    n_samples = X.shape[0]
    weighted_proba = np.zeros(n_samples, dtype=np.float32)
    
    for tree_idx, weight in zip(selected_trees, tree_weights):
        tree = rf_model.estimators_[tree_idx]
        
        if use_proba:
            # Use probability of positive class
            tree_pred = tree.predict_proba(X)[:, 1]
        else:
            # Use hard predictions (0 or 1)
            tree_pred = tree.predict(X).astype(np.float32)
        
        weighted_proba += tree_pred * weight
    
    return weighted_proba


def predict_ensemble_binary(
    rf_model,
    X: np.ndarray,
    selected_trees: list,
    tree_weights: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Make binary predictions using weighted ensemble with threshold.
    
    Parameters
    ----------
    rf_model : RandomForestClassifier
        Trained random forest model
    X : np.ndarray
        Feature matrix to predict on
    selected_trees : list
        List of selected tree indices
    tree_weights : np.ndarray
        Weights for each selected tree
    threshold : float
        Decision threshold for binary classification (default 0.5)
    
    Returns
    -------
    predictions : np.ndarray
        Binary predictions (0 or 1)
    """
    weighted_proba = predict_ensemble(rf_model, X, selected_trees, tree_weights, use_proba=True)
    return (weighted_proba >= threshold).astype(int)
