"""
Score all training examples using retrieval + consistency scorer.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "data_processing"))

from textpath import TextPath
from consistency_scorer import ConsistencyScorer
from retrieval import NovelRetriever


def score_train_set():
    """Score all training examples"""
    print("="*60)
    print("SCORING TRAIN SET")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    
    # Paths
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    train_csv = ROOT / "Dataset" / "train.csv"
    books_dir = ROOT / "Dataset" / "Books"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Create scorer
    scorer = ConsistencyScorer(model, tokenizer, device, max_novel_tokens=512)
    
    # Build retrievers for each novel
    print("\nBuilding retrievers...")
    retrievers = {
        "The Count of Monte Cristo": NovelRetriever(
            books_dir / "The Count of Monte Cristo.txt",
            chunk_size=400,
            overlap=100
        ),
        "In Search of the Castaways": NovelRetriever(
            books_dir / "In search of the castaways.txt",
            chunk_size=400,
            overlap=100
        ),
    }
    
    # Load train data
    print("\nLoading train data...")
    df = pd.read_csv(train_csv)
    print(f"Train examples: {len(df)}")
    
    # Score each example
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        book_name = row['book_name']
        backstory = row['content']
        label = row['label']
        
        # Map label
        if label == 'consistent':
            label_int = 1
        elif label == 'contradict':
            label_int = 0
        else:
            label_int = int(label)
        
        # Retrieve relevant chunks
        retriever = retrievers[book_name]
        chunks = retriever.retrieve(backstory, top_k=3)
        
        # Combine top chunks
        combined_novel = ' '.join([chunk for chunk, _ in chunks])
        
        # Score consistency
        scores = scorer.score_consistency(backstory, combined_novel)
        
        results.append({
            'id': row['id'],
            'book_name': book_name,
            'char': row['char'],
            'label': label_int,
            **scores
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = ROOT / "outputs" / "train_scores.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Scores saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for label_name, label_val in [("Consistent", 1), ("Contradict", 0)]:
        subset = results_df[results_df['label'] == label_val]
        print(f"\n{label_name} (n={len(subset)}):")
        print(f"  Delta mean: {subset['delta'].mean():.4f}")
        print(f"  Delta std:  {subset['delta'].std():.4f}")
        print(f"  PPL ratio mean: {subset['ppl_ratio'].mean():.3f}")
    
    return results_df


if __name__ == "__main__":
    score_train_set()

