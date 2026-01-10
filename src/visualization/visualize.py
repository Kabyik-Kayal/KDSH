"""
Visualization utilities for TextPath classifier analysis.
Provides functions to visualize model behavior and activations.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tokenizers import Tokenizer
from tqdm import tqdm

from src.models.textpath_classifier import TextPathClassifier, NovelSpecificClassifier
from src.models.textpath import TextPath


def extract_embeddings(
    model: TextPathClassifier,
    texts: list,
    tokenizer: Tokenizer,
    device: str,
    max_tokens: int = 256
) -> np.ndarray:
    """
    Extract pooled embeddings for a list of texts.
    
    Returns:
        numpy array of shape (num_texts, embedding_dim)
    """
    model.eval()
    embeddings = []
    
    pad_token_id = tokenizer.token_to_id('<pad>') or 0
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer.encode(text)
            input_ids = encoding.ids[:max_tokens]
            
            # Pad
            attention_mask = [1] * len(input_ids)
            padding = max_tokens - len(input_ids)
            if padding > 0:
                input_ids = input_ids + [pad_token_id] * padding
                attention_mask = attention_mask + [0] * padding
            
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
            
            emb = model.get_embeddings(input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def visualize_embedding_space(
    embeddings: np.ndarray,
    labels: list,
    save_path: Path = None,
    title: str = "Embedding Space"
):
    """
    Visualize embeddings in 2D using PCA.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # Separate by label
    consistent_mask = np.array(labels) == 1
    contradict_mask = np.array(labels) == 0
    
    plt.scatter(
        embeddings_2d[consistent_mask, 0],
        embeddings_2d[consistent_mask, 1],
        c='green', alpha=0.7, label='Consistent', s=50
    )
    plt.scatter(
        embeddings_2d[contradict_mask, 0],
        embeddings_2d[contradict_mask, 1],
        c='red', alpha=0.7, label='Contradict', s=50
    )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Embedding visualization saved to {save_path}")
    
    plt.close()


def analyze_attention_patterns(
    model: TextPath,
    tokenizer: Tokenizer,
    texts: list,
    labels: list,
    device: str,
    save_path: Path = None
):
    """
    Analyze attention patterns for consistent vs contradict examples.
    """
    model.eval()
    
    # Hook to capture attention-like patterns
    attention_data = {'consistent': [], 'contradict': []}
    
    def get_attention_hook(label_name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_data[label_name].append(output[1].detach().cpu())
        return hook_fn
    
    for text, label in zip(texts, labels):
        label_name = 'consistent' if label == 1 else 'contradict'
        
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor([encoding.ids[:256]], dtype=torch.long).to(device)
        
        # Register hook
        if hasattr(model.bdh, 'linear_attn'):
            handle = model.bdh.linear_attn.register_forward_hook(
                get_attention_hook(label_name)
            )
        
            with torch.no_grad():
                _ = model(input_ids)
            
            handle.remove()
    
    # Visualize average attention patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (label_name, color) in enumerate([('consistent', 'green'), ('contradict', 'red')]):
        ax = axes[idx]
        if attention_data[label_name]:
            avg_attn = torch.stack(attention_data[label_name]).mean(dim=0)
            if avg_attn.dim() >= 2:
                sns.heatmap(avg_attn[:32, :32].numpy(), ax=ax, cmap='Blues')
        ax.set_title(f'{label_name.capitalize()} Examples')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Attention analysis saved to {save_path}")
    
    plt.close()


def plot_classification_confidence(
    probabilities: list,
    labels: list,
    predictions: list,
    save_path: Path = None
):
    """
    Plot model confidence for correct vs incorrect predictions.
    """
    correct_probs = []
    incorrect_probs = []
    
    for prob, label, pred in zip(probabilities, labels, predictions):
        # Convert prob to confidence in predicted class
        confidence = prob if pred == 1 else (1 - prob)
        
        if pred == label:
            correct_probs.append(confidence)
        else:
            incorrect_probs.append(confidence)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(correct_probs, bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect', color='red')
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Model Confidence: Correct vs Incorrect Predictions')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Confidence plot saved to {save_path}")
    
    plt.close()


def visualize_by_character(
    df,
    predictions: list,
    labels: list,
    save_path: Path = None
):
    """
    Visualize accuracy by character.
    """
    char_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        char = df.iloc[i]['char']
        char_stats[char]['total'] += 1
        if pred == label:
            char_stats[char]['correct'] += 1
    
    # Calculate accuracy per character
    chars = []
    accuracies = []
    counts = []
    
    for char, stats in char_stats.items():
        chars.append(char)
        accuracies.append(stats['correct'] / stats['total'])
        counts.append(stats['total'])
    
    # Sort by accuracy
    sorted_idx = np.argsort(accuracies)
    chars = [chars[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(chars, accuracies, color='steelblue')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'n={count}', va='center', fontsize=9)
    
    plt.xlabel('Accuracy')
    plt.title('Classification Accuracy by Character')
    plt.xlim(0, 1.2)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Character analysis saved to {save_path}")
    
    plt.close()


def visualize_by_book(
    df,
    predictions: list,
    labels: list,
    save_path: Path = None
):
    """
    Visualize accuracy by book.
    """
    book_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        book = df.iloc[i]['book_name']
        book_stats[book]['total'] += 1
        if pred == label:
            book_stats[book]['correct'] += 1
    
    books = list(book_stats.keys())
    accuracies = [book_stats[b]['correct'] / book_stats[b]['total'] for b in books]
    counts = [book_stats[b]['total'] for b in books]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(books, accuracies, color=['steelblue', 'coral'])
    
    for bar, count, acc in zip(bars, counts, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}\n(n={count})', ha='center', fontsize=10)
    
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Book')
    plt.ylim(0, 1.2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Book analysis saved to {save_path}")
    
    plt.close()


# Export functions
__all__ = [
    'extract_embeddings',
    'visualize_embedding_space',
    'analyze_attention_patterns',
    'plot_classification_confidence',
    'visualize_by_character',
    'visualize_by_book'
]


def run_all_visualizations():
    """Main execution function for generating all visualizations."""
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.data_processing.classification_dataset import ConsistencyDataset
    from src.data_processing.retrieval import PathwayNovelRetriever
    
    # 1. Setup paths and config
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    models_dir = ROOT / 'models'
    data_dir = ROOT / 'Dataset'
    viz_dir = ROOT / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    tokenizer_path = models_dir / 'custom_tokenizer.json'
    train_csv = data_dir / 'train.csv'
    novels_dir = data_dir / 'Books'
    
    print("="*60)
    print("🐉 TextPath Visualization Suite")
    print("="*60)
    
    # 2. Build retrievers and dataset
    print("\nBuilding retrievers...")
    retrievers = {}
    for novel_file in novels_dir.glob('*.txt'):
        retrievers[novel_file.stem] = PathwayNovelRetriever(
            novel_path=novel_file,
            chunk_size=200,
            overlap=50
        )
        
    dataset = ConsistencyDataset(
        csv_path=str(train_csv),
        novel_dir=str(novels_dir),
        tokenizer_path=str(tokenizer_path),
        retriever=retrievers,
        max_tokens=512,
        mode='train'
    )
    
    # 3. Load Model
    print("\nLoading models...")
    model = NovelSpecificClassifier(
        models_dir=str(models_dir),
        device=device
    )
    
    # Try to load best weights if available
    best_weights_path = models_dir / 'textpath_classifier_best.pt'
    if (models_dir / 'textpath_classifier_best_castaways.pt').exists():
        model.load(str(best_weights_path))
    
    model.eval()
    
    # 4. Generate Predictions & Embeddings
    print("\nProcessing data for visualizations...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    texts = []
    labels = []
    predictions = []
    probabilities = []
    all_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Inference")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            book_names = batch['book_name']
            
            # Forward
            logits = model.forward_grouped(input_ids, attention_mask, book_names)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Get internal TextPath model for embedding extraction
            novel_key = model._get_novel_key(book_names[0])
            classifier = model.classifiers[novel_key]
            emb = classifier.get_embeddings(input_ids, attention_mask)
            
            probabilities.append(probs[0, 1].item())
            predictions.append(preds[0].item())
            labels.append(batch_labels[0].item())
            all_embeddings.append(emb.cpu().numpy())
            
            # Get original backstory for text-based viz
            texts.append(dataset.df.iloc[i]['content'])
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # 5. Run Visualization Tasks
    print("\nGenerating plots...")
    
    # Embedding Space
    visualize_embedding_space(
        all_embeddings, labels, 
        save_path=viz_dir / 'consistency_embedding_space.png'
    )
    
    # Confidence
    plot_classification_confidence(
        probabilities, labels, predictions,
        save_path=viz_dir / 'prediction_confidence.png'
    )
    
    # By Character
    visualize_by_character(
        dataset.df, predictions, labels,
        save_path=viz_dir / 'accuracy_by_character.png'
    )
    
    # By Book
    visualize_by_book(
        dataset.df, predictions, labels,
        save_path=viz_dir / 'accuracy_by_book.png'
    )
    
    print(f"\n✨ All visualizations saved to {viz_dir}")


if __name__ == "__main__":
    try:
        run_all_visualizations()
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
