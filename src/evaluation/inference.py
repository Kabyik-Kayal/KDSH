"""
Run inference on test.csv using the trained consistency classifier.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
import joblib

# Add paths
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "data_processing"))

from textpath import TextPath, TextPathConfig
from consistency_scorer import ConsistencyScorer
from retrieval import NovelRetriever


def run_inference():
    """Run inference on test set"""
    print("="*60)
    print("TEST SET INFERENCE")
    print("="*60)
    
    # Paths
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = ROOT / "Dataset"
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    classifier_path = ROOT / "models" / "consistency_classifier.pkl"
    output_path = ROOT / "results.csv"
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading TextPath model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (params: {sum(p.numel() for p in model.parameters()):,})")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer loaded (vocab: {tokenizer.get_vocab_size():,})")
    
    # Load classifier
    print("\nLoading classifier...")
    classifier = joblib.load(classifier_path)
    print("Classifier loaded")
    
    # Build retrievers
    print("\nBuilding retrievers...")
    books_dir = data_dir / "Books"
    
    retrievers = {}
    for novel_file in books_dir.glob("*.txt"):
        book_name = novel_file.stem
        retriever = NovelRetriever(novel_file, chunk_size=256)
        retrievers[book_name] = retriever
        print(f"  {book_name}: {len(retriever.chunks)} chunks")
    
    # Create scorer
    scorer = ConsistencyScorer(model, tokenizer, device, max_novel_tokens=512)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Test examples: {len(test_df)}\n")
    
    # Run inference
    results = []
    
    print("Running inference...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Scoring"):
        example_id = row['id']
        book_name = row['book_name']
        char = row['char']
        backstory = row['content']
        
        # Find matching retriever
        retriever = None
        for key, ret in retrievers.items():
            if key.lower() in book_name.lower() or book_name.lower() in key.lower():
                retriever = ret
                break
        
        if retriever is None:
            # Fallback: use first retriever
            retriever = list(retrievers.values())[0]
        
        # Retrieve relevant chunks
        chunks = retriever.retrieve(backstory, top_k=3)
        combined_novel = "\n".join([chunk for chunk, _ in chunks[:2]])  # Use top 2 chunks
        
        # Score consistency
        scores = scorer.score_consistency(backstory, combined_novel)
        
        # Prepare features for classifier
        features = [[
            scores['delta'],
            scores['ppl_ratio'],
            scores['baseline_loss'],
            scores['primed_loss'],
        ]]
        
        # Predict
        prediction = int(classifier.predict(features)[0])
        
        # Generate simple rationale
        if prediction == 1:
            if scores['delta'] < 0:
                rationale = "Backstory provides helpful context (negative delta)"
            else:
                rationale = "Backstory consistent with narrative structure"
        else:
            if scores['delta'] > 0.05:
                rationale = "Backstory creates misleading expectations (high positive delta)"
            else:
                rationale = "Inconsistent with character development or plot constraints"
        
        results.append({
            'id': example_id,
            'prediction': prediction,
            'rationale': rationale,
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f" Results saved to {output_path}")
    print(f"{'='*60}")
    
    # Summary
    print("\nPrediction distribution:")
    print(results_df['prediction'].value_counts().sort_index())
    print(f"\nConsistent: {(results_df['prediction'] == 1).sum()}")
    print(f"Contradict: {(results_df['prediction'] == 0).sum()}")
    
    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    run_inference()