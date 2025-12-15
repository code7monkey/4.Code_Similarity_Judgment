"""Training and inference routines for GraphCodeBERT."""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW

from .dataset import CodePairDataset
from .model import GraphCodeBERTClassifier
from .utils import set_seed


def train_model(config: Dict) -> None:
    """Train one or more GraphCodeBERT models according to the config.

    The configuration dictionary should contain training hyperparameters,
    dataset paths and the list of model output paths. If more than
    one model path is provided, the dataset is split into equal parts
    and each model is trained on a different subset to create an
    ensemble.

    Args:
        config: Dictionary specifying data paths, model names, output
            paths and hyperparameters.
    """
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load training data
    train_df = pd.read_csv(config['data']['train_csv'])
    # Prepare tokenizer
    model_name = config['model'].get('model_name', 'microsoft/graphcodebert-base')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Determine number of models to train
    model_paths: List[str] = config['assets'].get('model_paths', ['assets/model1.pt'])
    num_models = len(model_paths)
    # Split dataset into equal parts for each model
    splits = torch.chunk(torch.arange(len(train_df)), num_models)
    batch_size = config['training'].get('batch_size', 8)
    lr = config['training'].get('lr', 2e-5)
    epochs = config['training'].get('epochs', 1)
    for model_idx, model_path in enumerate(model_paths):
        indices = splits[model_idx].tolist()
        subset_df = train_df.iloc[indices].reset_index(drop=True)
        dataset = CodePairDataset(tokenizer, subset_df, max_length=config['model'].get('max_length', 256))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = GraphCodeBERTClassifier(model_name=model_name, num_labels=config['model'].get('num_labels', 2)).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Model {model_idx+1}/{num_models}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)


def infer_models(config: Dict) -> None:
    """Run inference using one or more trained models and save predictions.

    Args:
        config: Dictionary specifying test data path, model paths, and
            output submission path.
    """
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df = pd.read_csv(config['data']['test_csv'])
    model_name = config['model'].get('model_name', 'microsoft/graphcodebert-base')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_paths: List[str] = config['assets'].get('model_paths', ['assets/model1.pt'])
    batch_size = config['inference'].get('batch_size', 8)
    # Prepare dataset
    dataset = CodePairDataset(tokenizer, test_df, max_length=config['model'].get('max_length', 256), include_labels=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    for model_path in model_paths:
        model = GraphCodeBERTClassifier(model_name=model_name, num_labels=config['model'].get('num_labels', 2)).to(device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        model.eval()
        probs: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**batch_inputs)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=-1)[:, 1]  # Probability of class 1
                probs.append(prob.cpu())
        all_probs.append(torch.cat(probs))
    # Average probabilities across models
    avg_probs = torch.stack(all_probs).mean(dim=0)
    # Apply threshold 0.5 to get predictions
    preds = (avg_probs >= 0.5).long().numpy()
    submission = pd.DataFrame({
        'similar': preds,
    })
    # If sample submission has an id column, merge
    sample_path = config['data'].get('sample_submission_csv')
    if sample_path and os.path.exists(sample_path):
        sample_sub = pd.read_csv(sample_path)
        if 'id' in sample_sub.columns:
            submission['id'] = sample_sub['id']
            submission = submission[['id', 'similar']]
    output_path = config['assets'].get('output_path', 'submission.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
