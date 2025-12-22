"""Efficiency module for model pruning and ensemble optimization."""

from .models import TreePruner, predict_ensemble

__all__ = ["TreePruner", "predict_ensemble"]
