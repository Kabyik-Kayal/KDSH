"""
TextPath Classifier: Wrapper for managing BDH-based consistency classification.
Loads pretrained LM and adds classification head.
Supports novel-specific models for better performance.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List

from .textpath import TextPath, TextPathConfig

# Fix for loading old checkpoints saved with different module path
# This allows torch.load to find 'textpath' module
from src.models import textpath as _textpath_module
sys.modules['textpath'] = _textpath_module


class TextPathClassifier:
    """
    Wrapper class for TextPath with classification head.
    Loads pretrained LM weights and optionally freezes BDH layers.
    """
    
    def __init__(
        self,
        pretrained_lm_path: str,
        device: str = 'cuda',
        freeze_bdh: bool = False
    ):
        """
        Args:
            pretrained_lm_path: Path to pretrained TextPath LM checkpoint
            device: 'cuda', 'mps', or 'cpu'
            freeze_bdh: If True, only train classification head
        """
        self.device = device
        
        # Load pretrained LM
        print(f"Loading pretrained TextPath from {pretrained_lm_path}")
        checkpoint = torch.load(pretrained_lm_path, map_location=device, weights_only=False)
        
        # Extract config from checkpoint or use defaults
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if isinstance(saved_config, dict):
                config = TextPathConfig(**saved_config)
            else:
                config = saved_config
        else:
            config = TextPathConfig()
        
        # Enable classification mode
        config.classification_mode = True
        
        # Create model with classification head
        self.model = TextPath(config).to(device)
        
        # Load pretrained weights (BDH + embeddings)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out classifier_head weights if present (we want fresh ones)
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith('classifier_head')
        }
        
        self.model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)} pretrained weights")
        
        # Optionally freeze BDH layers
        if freeze_bdh:
            self._freeze_bdh_layers()
    
    def _freeze_bdh_layers(self):
        """Freeze all parameters except classification head"""
        for name, param in self.model.named_parameters():
            if 'classifier_head' not in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def unfreeze_bdh_layers(self):
        """Unfreeze BDH layers for full fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All parameters unfrozen for fine-tuning")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning classification logits"""
        logits, _ = self.model(input_ids, attention_mask)
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get pooled embeddings for contrastive learning"""
        embeddings, _ = self.model(input_ids, attention_mask, return_embeddings=True)
        return embeddings
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def save(self, path: str, optimizer=None, epoch: int = 0, val_acc: float = 0.0):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'val_acc': val_acc
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Model saved to {path}")


class NovelSpecificClassifier:
    """
    Manages multiple TextPath classifiers, one per novel.
    Routes samples to the appropriate model based on book_name.
    """
    
    NOVEL_MODEL_MAP = {
        'castaways': 'textpath_in_search_of_the_castaways.pt',
        'monte cristo': 'textpath_the_count_of_monte_cristo.pt',
    }
    
    def __init__(
        self,
        models_dir: str,
        device: str = 'cuda',
        freeze_bdh: bool = False
    ):
        """
        Args:
            models_dir: Directory containing novel-specific pretrained models
            device: 'cuda', 'mps', or 'cpu'
            freeze_bdh: If True, only train classification heads
        """
        self.device = device
        self.models_dir = Path(models_dir)
        self.classifiers: Dict[str, TextPathClassifier] = {}
        self.freeze_bdh = freeze_bdh
        
        print("="*60)
        print("Loading Novel-Specific Classifiers")
        print("="*60)
        
        # Load classifier for each novel
        for novel_key, model_file in self.NOVEL_MODEL_MAP.items():
            model_path = self.models_dir / model_file
            if model_path.exists():
                print(f"\n📚 Loading model for '{novel_key}'...")
                self.classifiers[novel_key] = TextPathClassifier(
                    pretrained_lm_path=str(model_path),
                    device=device,
                    freeze_bdh=freeze_bdh
                )
            else:
                print(f"⚠️ Model not found for '{novel_key}': {model_path}")
        
        print(f"\n✅ Loaded {len(self.classifiers)} novel-specific classifiers")
    
    def _get_novel_key(self, book_name: str) -> str:
        """Map book_name to classifier key"""
        book_lower = book_name.lower()
        if 'castaways' in book_lower:
            return 'castaways'
        elif 'monte cristo' in book_lower:
            return 'monte cristo'
        else:
            # Fallback to first available
            return list(self.classifiers.keys())[0]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        book_names: List[str]
    ) -> torch.Tensor:
        """
        Forward pass routing each sample to its novel-specific classifier.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            book_names: List of book names for each sample in batch
        
        Returns:
            logits: [batch_size, 2]
        """
        batch_size = input_ids.size(0)
        all_logits = []
        
        for i in range(batch_size):
            novel_key = self._get_novel_key(book_names[i])
            classifier = self.classifiers[novel_key]
            
            # Single sample forward
            sample_ids = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
            
            logits = classifier.forward(sample_ids, sample_mask)
            all_logits.append(logits)
        
        return torch.cat(all_logits, dim=0)
    
    def forward_grouped(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        book_names: List[str]
    ) -> torch.Tensor:
        """
        Optimized forward pass that groups samples by novel for batch processing.
        """
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, 2, device=self.device)
        
        # Group indices by novel
        novel_indices: Dict[str, List[int]] = {}
        for i, book_name in enumerate(book_names):
            novel_key = self._get_novel_key(book_name)
            if novel_key not in novel_indices:
                novel_indices[novel_key] = []
            novel_indices[novel_key].append(i)
        
        # Process each group
        for novel_key, indices in novel_indices.items():
            if novel_key not in self.classifiers:
                continue
            
            classifier = self.classifiers[novel_key]
            idx_tensor = torch.tensor(indices, device=self.device)
            
            # Gather samples for this novel
            group_ids = input_ids[idx_tensor]
            group_mask = attention_mask[idx_tensor] if attention_mask is not None else None
            
            # Batch forward
            group_logits = classifier.forward(group_ids, group_mask)
            
            # Scatter back to original positions
            logits[idx_tensor] = group_logits
        
        return logits
    
    def train(self):
        """Set all models to training mode"""
        for classifier in self.classifiers.values():
            classifier.train()
    
    def eval(self):
        """Set all models to evaluation mode"""
        for classifier in self.classifiers.values():
            classifier.eval()
    
    def unfreeze_bdh_layers(self):
        """Unfreeze BDH layers in all classifiers"""
        for classifier in self.classifiers.values():
            classifier.unfreeze_bdh_layers()
    
    def parameters(self):
        """Yield all parameters from all classifiers"""
        for classifier in self.classifiers.values():
            yield from classifier.model.parameters()
    
    def save(self, path: str, optimizer=None, epoch: int = 0, val_acc: float = 0.0):
        """Save all classifier checkpoints"""
        path = Path(path)
        
        for novel_key, classifier in self.classifiers.items():
            save_path = path.parent / f"{path.stem}_{novel_key.replace(' ', '_')}{path.suffix}"
            classifier.save(str(save_path), optimizer, epoch, val_acc)
    
    def load(self, path: str):
        """Load classifier checkpoints"""
        path = Path(path)
        
        for novel_key in self.classifiers.keys():
            load_path = path.parent / f"{path.stem}_{novel_key.replace(' ', '_')}{path.suffix}"
            if load_path.exists():
                checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
                self.classifiers[novel_key].model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint for '{novel_key}' from {load_path}")
