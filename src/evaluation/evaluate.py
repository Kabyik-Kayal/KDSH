"""
Evaluation utilities for TextPath consistency classification.
Provides functions for model evaluation, prediction, and metrics.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.data_processing.classification_dataset import ConsistencyDataset
from src.models.textpath_classifier import NovelSpecificClassifier


def evaluate_model(
    model: NovelSpecificClassifier,
    dataloader: DataLoader,
    device: str
) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained NovelSpecificClassifier
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with metrics: accuracy, f1, precision, recall, 
        confusion_matrix, and classification_report
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            book_names = batch['book_name']
            
            logits = model.forward_grouped(input_ids, attention_mask, book_names)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, 
        target_names=['Contradict', 'Consistent'],
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def predict_batch(
    model: NovelSpecificClassifier,
    dataloader: DataLoader,
    device: str
) -> Tuple[List[int], List[int], List[float]]:
    """
    Generate predictions for a batch of samples.
    
    Args:
        model: Trained NovelSpecificClassifier
        dataloader: DataLoader for prediction data
        device: Device to run prediction on
        
    Returns:
        Tuple of (ids, predictions, probabilities)
    """
    model.eval()
    
    all_ids = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            book_names = batch['book_name']
            ids = batch['id']
            
            logits = model.forward_grouped(input_ids, attention_mask, book_names)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            if torch.is_tensor(ids):
                all_ids.extend(ids.numpy())
            else:
                all_ids.extend(ids)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return all_ids, all_preds, all_probs


def save_predictions(
    ids: List[int],
    predictions: List[int],
    output_path: str,
    label_map: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Save predictions to CSV file.
    
    Args:
        ids: Sample IDs
        predictions: Predicted labels (0 or 1)
        output_path: Path to save CSV
        label_map: Optional mapping from int to string labels
        
    Returns:
        DataFrame with predictions
    """
    if label_map is None:
        label_map = {0: 'contradict', 1: 'consistent'}
    
    results = pd.DataFrame({
        'id': ids,
        'label': [label_map[p] for p in predictions]
    })
    results = results.sort_values('id').reset_index(drop=True)
    results.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved to {output_path}")
    print("\nPrediction distribution:")
    print(results['label'].value_counts())
    
    return results


def print_evaluation_report(metrics: Dict, title: str = "EVALUATION RESULTS"):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary from evaluate_model()
        title: Report title
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"            Contra  Consist")
    print(f"Actual Contra  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Consist {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    print("\nClassification Report:")
    report = metrics['classification_report']
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 52)
    for cls in ['Contradict', 'Consistent']:
        r = report[cls]
        print(f"{cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")


def run_full_evaluation(
    models_dir: str,
    train_csv: str,
    novels_dir: str,
    tokenizer_path: str,
    retrievers: Dict,
    device: str,
    max_tokens: int = 512,
    top_k_retrieval: int = 2,
    output_model_path: Optional[str] = None
) -> Dict:
    """
    Run complete evaluation pipeline on validation set.
    
    Args:
        models_dir: Directory containing trained models
        train_csv: Path to training CSV (will use 20% as validation)
        novels_dir: Directory containing novel text files
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to run on
        max_tokens: Maximum sequence length
        top_k_retrieval: Number of chunks to retrieve
        output_model_path: Optional path to load specific checkpoint
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("FULL EVALUATION")
    print("="*60)
    
    # Load validation dataset
    dataset = ConsistencyDataset(
        csv_path=train_csv,
        novel_dir=novels_dir,
        tokenizer_path=tokenizer_path,
        retriever=retrievers,
        max_tokens=max_tokens,
        mode='train',
        top_k_retrieval=top_k_retrieval
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"✅ Loaded {len(val_dataset)} validation samples")
    
    # Load model
    model = NovelSpecificClassifier(
        models_dir=models_dir,
        device=device,
        freeze_bdh=False
    )
    
    if output_model_path:
        model.load(output_model_path)
    
    model.eval()
    
    # Evaluate
    metrics = evaluate_model(model, val_loader, device)
    print_evaluation_report(metrics)
    
    return metrics


def run_test_prediction(
    models_dir: str,
    test_csv: str,
    novels_dir: str,
    tokenizer_path: str,
    retrievers: Dict,
    device: str,
    output_path: str,
    max_tokens: int = 512,
    top_k_retrieval: int = 2,
    output_model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate predictions for test set.
    
    Args:
        models_dir: Directory containing trained models
        test_csv: Path to test CSV
        novels_dir: Directory containing novel text files
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to run on
        output_path: Path to save predictions CSV
        max_tokens: Maximum sequence length
        top_k_retrieval: Number of chunks to retrieve
        output_model_path: Optional path to load specific checkpoint
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Load test dataset
    test_dataset = ConsistencyDataset(
        csv_path=test_csv,
        novel_dir=novels_dir,
        tokenizer_path=tokenizer_path,
        retriever=retrievers,
        max_tokens=max_tokens,
        mode='test',
        top_k_retrieval=top_k_retrieval
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=0
    )
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    # Load model
    model = NovelSpecificClassifier(
        models_dir=models_dir,
        device=device,
        freeze_bdh=False
    )
    
    if output_model_path:
        model.load(output_model_path)
    
    model.eval()
    
    # Generate predictions
    ids, preds, probs = predict_batch(model, test_loader, device)
    
    # Save results
    results = save_predictions(ids, preds, output_path)
    
    return results


# Export functions
__all__ = [
    'evaluate_model',
    'predict_batch',
    'save_predictions',
    'print_evaluation_report',
    'run_full_evaluation',
    'run_test_prediction'
]
