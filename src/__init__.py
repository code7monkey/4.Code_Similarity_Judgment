"""Initialise the GraphCodeBERT project package.

This package provides utilities for preprocessing code, generating
training pairs, defining datasets, loading a GraphCodeBERT-based model
and running training/inference routines.
"""

from .utils import set_seed, remove_extras
from .dataset import CodePairDataset, build_pairs_dataset
from .model import GraphCodeBERTClassifier
from .trainer import train_model, infer_models

__all__ = [
    'set_seed',
    'remove_extras',
    'CodePairDataset',
    'build_pairs_dataset',
    'GraphCodeBERTClassifier',
    'train_model',
    'infer_models',
]
