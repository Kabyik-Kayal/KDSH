"""
End-to-End Pipeline for Narrative Consistency Detection
========================================================
Complete pipeline using Pathway framework for RAG and BDH for classification.
Uses novel-specific pretrained models that leverage BDH's unique properties:
- Hebbian learning: neurons that fire together, wire together
- Sparse activations: ~5% neurons active, creating monosemantic representations  
- Causal circuits: learned Gx = E @ Dx encodes narrative reasoning

Usage:
    python run_pipeline.py --mode pretrain  # Pretrain BDH on novels (BDH-native)
    python run_pipeline.py --mode train     # Train the classifier
    python run_pipeline.py --mode predict   # Generate predictions
    python run_pipeline.py --mode evaluate  # Evaluate on validation set
    python run_pipeline.py --mode full      # Train + Evaluate + Predict
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse

# Import modular components
from src.config import PipelineConfig, get_config
from src.data_processing import build_pathway_retrievers
from src.training import run_training, run_pretraining
from src.evaluation import run_full_evaluation, run_test_prediction


# ============================================================
# Pipeline Mode Runners
# ============================================================

def do_pretraining(config: PipelineConfig):
    """Run BDH-native pretraining on novels."""
    return run_pretraining(
        epochs=config.pretrain_epochs,
        batch_size=config.batch_size,
        verbose=True
    )


def do_training(config: PipelineConfig, retrievers: dict):
    """Train novel-specific classifiers."""
    return run_training(
        models_dir=str(config.models_dir),
        train_csv=str(config.train_csv),
        novels_dir=str(config.novels_dir),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        output_model=str(config.output_model),
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        freeze_bdh=config.freeze_bdh,
        unfreeze_after_epoch=config.unfreeze_after_epoch
    )


def do_prediction(config: PipelineConfig, retrievers: dict):
    """Generate predictions using novel-specific classifiers."""
    return run_test_prediction(
        models_dir=str(config.models_dir),
        test_csv=str(config.test_csv),
        novels_dir=str(config.novels_dir),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        output_path=str(config.output_predictions),
        max_tokens=config.max_tokens,
        top_k_retrieval=config.top_k_retrieval,
        output_model_path=str(config.output_model)
    )


def do_evaluation(config: PipelineConfig, retrievers: dict):
    """Evaluate novel-specific classifiers."""
    return run_full_evaluation(
        models_dir=str(config.models_dir),
        train_csv=str(config.train_csv),
        novels_dir=str(config.novels_dir),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        max_tokens=config.max_tokens,
        top_k_retrieval=config.top_k_retrieval,
        output_model_path=str(config.output_model)
    )


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Narrative Consistency Detection Pipeline (Pathway + BDH)'
    )
    parser.add_argument(
        '--mode', 
        choices=['pretrain', 'train', 'predict', 'evaluate', 'full'], 
        default='full',
        help='Pipeline mode: pretrain, train, predict, evaluate, or full'
    )
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=50,
        help='Number of epochs for BDH pretraining (default: 50)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs for classifier training (default: 15)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (default: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: 1e-4)'
    )
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override with command-line arguments
    if args.pretrain_epochs != 50:
        config.pretrain_epochs = args.pretrain_epochs
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # Print header
    print("="*60)
    print("🐉 Narrative Consistency Detection Pipeline")
    print("   Using Pathway Framework + Novel-Specific BDH")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Mode: {args.mode}")
    
    # Pretraining mode
    if args.mode == 'pretrain':
        do_pretraining(config)
        return
    
    # Build retrievers for all other modes
    retrievers = build_pathway_retrievers(
        novels_dir=config.novels_dir,
        chunk_size=config.chunk_size,
        overlap=config.overlap
    )
    
    # Execute requested mode
    if args.mode == 'train':
        do_training(config, retrievers)
    elif args.mode == 'predict':
        do_prediction(config, retrievers)
    elif args.mode == 'evaluate':
        do_evaluation(config, retrievers)
    else:  # full
        do_training(config, retrievers)
        do_evaluation(config, retrievers)
        do_prediction(config, retrievers)
    
    print("\n" + "="*60)
    print(" Pipeline Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
