"""
Training utilities for TextPath consistency classification.
Provides a high-level Trainer class and training configuration.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.data_processing.classification_dataset import ConsistencyDataset
from src.models.textpath_classifier import NovelSpecificClassifier
from src.models.finetune_classifier import train_epoch, validate


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Paths
    models_dir: str = ""
    train_csv: str = ""
    novels_dir: str = ""
    tokenizer_path: str = ""
    output_model: str = ""
    
    # Training hyperparameters
    batch_size: int = 4
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_tokens: int = 512
    
    # Freezing strategy
    freeze_bdh: bool = True
    unfreeze_after_epoch: int = 5
    unfreeze_lr_multiplier: float = 0.1
    
    # Class weights for imbalanced data
    class_weights: Tuple[float, float] = (1.7, 1.0)
    
    # Device
    device: str = field(default_factory=lambda: (
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    ))
    
    # Random seed
    seed: int = 42
    
    # Validation split
    val_ratio: float = 0.2


class Trainer:
    """
    High-level trainer for TextPath consistency classification.
    Handles dataset loading, training loop, and model checkpointing.
    """
    
    def __init__(self, config: TrainingConfig, retrievers: Dict):
        """
        Initialize trainer.
        
        Args:
            config: TrainingConfig instance
            retrievers: Dictionary of PathwayNovelRetriever instances
        """
        self.config = config
        self.retrievers = retrievers
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_val_acc = 0.0
        
    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and split dataset into train/val loaders."""
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60)
        
        dataset = ConsistencyDataset(
            csv_path=self.config.train_csv,
            novel_dir=self.config.novels_dir,
            tokenizer_path=self.config.tokenizer_path,
            retriever=self.retrievers,
            max_tokens=self.config.max_tokens,
            mode='train'
        )
        
        # Train/val split
        train_size = int((1 - self.config.val_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        print(f"✅ Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return self.train_loader, self.val_loader
    
    def setup_model(self) -> NovelSpecificClassifier:
        """Initialize the model and optimizer."""
        print("\n" + "="*60)
        print("Initializing Model")
        print("="*60)
        
        self.model = NovelSpecificClassifier(
            models_dir=self.config.models_dir,
            device=self.config.device,
            freeze_bdh=self.config.freeze_bdh
        )
        
        self.optimizer = AdamW(
            list(self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.epochs
        )
        
        return self.model
    
    def train(self) -> NovelSpecificClassifier:
        """
        Run the complete training loop.
        
        Returns:
            Trained model
        """
        if self.train_loader is None:
            self.setup_data()
        if self.model is None:
            self.setup_model()
        
        print("\n" + "="*60)
        print("Training Novel-Specific Classifiers")
        print("="*60)
        
        class_weights = torch.tensor(
            self.config.class_weights, 
            dtype=torch.float32
        )
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 40)
            
            # Unfreeze BDH after specified epoch
            if epoch == self.config.unfreeze_after_epoch and self.config.freeze_bdh:
                self._unfreeze_bdh()
            
            # Train one epoch
            train_loss, train_acc = train_epoch(
                self.model, 
                self.train_loader, 
                self.optimizer, 
                self.config.device, 
                class_weights,
                novel_specific=True
            )
            
            # Validate
            val_acc, val_report, _ = validate(
                self.model, 
                self.val_loader, 
                self.config.device,
                novel_specific=True
            )
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save(
                    self.config.output_model, 
                    self.optimizer, 
                    epoch, 
                    val_acc
                )
                print(f"✓ Saved best model")
        
        print(f"\n✅ Training complete! Best Val Acc: {self.best_val_acc:.4f}")
        return self.model
    
    def _unfreeze_bdh(self):
        """Unfreeze BDH layers with reduced learning rate."""
        print("🔓 Unfreezing BDH layers...")
        self.model.unfreeze_bdh_layers()
        
        self.optimizer = AdamW(
            list(self.model.parameters()), 
            lr=self.config.learning_rate * self.config.unfreeze_lr_multiplier,
            weight_decay=self.config.weight_decay
        )


def run_training(
    models_dir: str,
    train_csv: str,
    novels_dir: str,
    tokenizer_path: str,
    retrievers: Dict,
    device: str,
    output_model: str,
    batch_size: int = 4,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    max_tokens: int = 512,
    freeze_bdh: bool = True,
    unfreeze_after_epoch: int = 5
) -> NovelSpecificClassifier:
    """
    Convenience function for running training.
    
    Args:
        models_dir: Directory containing pretrained BDH models
        train_csv: Path to training CSV
        novels_dir: Directory with novel text files
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to train on
        output_model: Path to save best model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        max_tokens: Maximum sequence length
        freeze_bdh: Whether to freeze BDH initially
        unfreeze_after_epoch: Epoch to unfreeze BDH
        
    Returns:
        Trained NovelSpecificClassifier
    """
    config = TrainingConfig(
        models_dir=models_dir,
        train_csv=train_csv,
        novels_dir=novels_dir,
        tokenizer_path=tokenizer_path,
        output_model=output_model,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        freeze_bdh=freeze_bdh,
        unfreeze_after_epoch=unfreeze_after_epoch,
        device=device
    )
    
    trainer = Trainer(config, retrievers)
    return trainer.train()


# Export classes and functions
__all__ = [
    'TrainingConfig',
    'Trainer',
    'run_training'
]
