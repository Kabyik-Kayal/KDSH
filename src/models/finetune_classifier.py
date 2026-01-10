"""
Training utilities for TextPath consistency classifier.
Provides train_epoch and validate functions for model training.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def train_epoch(model, dataloader, optimizer, device, class_weights=None, novel_specific=False):
    """
    Train for one epoch.
    
    Args:
        model: TextPathClassifier or NovelSpecificClassifier
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        class_weights: Optional class weights for imbalanced data
        novel_specific: If True, use novel-specific routing
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if novel_specific and 'book_name' in batch:
            book_names = batch['book_name']
            logits = model.forward_grouped(input_ids, attention_mask, book_names)
        else:
            logits = model.forward(input_ids, attention_mask)
        
        loss = loss_fn(logits, labels)
        
        loss.backward()
        
        # Clip gradients - handle both model types
        if hasattr(model, 'model'):
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, device, novel_specific=False):
    """
    Validate model.
    
    Args:
        model: TextPathClassifier or NovelSpecificClassifier
        dataloader: Validation data loader
        device: Device to use
        novel_specific: If True, use novel-specific routing
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if novel_specific and 'book_name' in batch:
                book_names = batch['book_name']
                logits = model.forward_grouped(input_ids, attention_mask, book_names)
            else:
                logits = model.forward(input_ids, attention_mask)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=['Contradict', 'Consistent'],
        zero_division=0
    )
    
    return accuracy, report, all_probs


# Export only the training utilities
__all__ = ['train_epoch', 'validate']
