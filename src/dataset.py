"""Dataset utilities for GraphCodeBERT code similarity project."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import remove_extras


class CodePairDataset(Dataset):
    """PyTorch Dataset for pairs of code snippets with similarity labels."""

    def __init__(self, tokenizer, data: pd.DataFrame, max_length: int = 512, include_labels: bool = True) -> None:
        self.tokenizer = tokenizer
        self.data = data.reset_index(drop=True)
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data.iloc[idx]
        code1 = remove_extras(record['code1'])
        code2 = remove_extras(record['code2'])
        inputs = self.tokenizer(
            code1,
            code2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        # Remove batch dimension returned by tokenizer
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.include_labels:
            inputs['labels'] = torch.tensor(int(record['similar']), dtype=torch.long)
        return inputs


def build_pairs_dataset(codes: List[str], labels: Iterable[int]) -> pd.DataFrame:
    """Construct a DataFrame of code pairs and similarity labels.

    This helper takes two lists of equal length (codes and labels) and
    returns a DataFrame with columns ``code1``, ``code2`` and ``similar``.
    It pairs each code with every other code and assigns the provided
    labels. In practice you would implement a more sophisticated
    pairing strategy to generate positive and negative examples.

    Args:
        codes: List of code strings.
        labels: Iterable of labels (0 or 1) of the same length as codes.

    Returns:
        DataFrame with columns ``code1``, ``code2``, ``similar``.
    """
    data = {
        'code1': codes,
        'code2': codes,  # In this placeholder, pair code with itself
        'similar': list(labels),
    }
    return pd.DataFrame(data)
