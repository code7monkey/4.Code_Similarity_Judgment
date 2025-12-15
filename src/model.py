"""Model wrapper for GraphCodeBERT code similarity classification."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class GraphCodeBERTClassifier(nn.Module):
    """Wrap HuggingFace AutoModelForSequenceClassification for code pairs."""

    def __init__(self, model_name: str = 'microsoft/graphcodebert-base', num_labels: int = 2) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, **inputs):  # type: ignore[override]
        return self.model(**inputs)
