""
State Manager for BDH synaptic state extraction and injection.
Handles the "working memory" persistence for narrative consistency checking.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))
from textpath import TextPath


class StateManager:
    """
    Manages BDH synaptic state for consistency checking.
    
    Two operational modes:
    1. STATELESS (default): Concatenate backstory + novel
    2. STATEFUL: Extract/inject state between forward passes
    """
    
    def __init__(
        self,
        model: TextPath,
        mode: str = 'stateless'  # 'stateless' or 'stateful'
    ):
        self.model = model
        self.mode = mode
        self._saved_states = {}
        
        # Detect if LinearAttention is truly stateful
        self._is_stateful = self._check_statefulness()
        
        if mode == 'stateful' and not self._is_stateful:
            print("⚠️ Warning: Model appears stateless, falling back to concatenation mode")
            self.mode = 'stateless'
        
        print(f"✅ StateManager initialized (mode: {self.mode})")
    
    def _check_statefulness(self) -> bool:
        """
        Check if LinearAttention maintains state between forward passes.
        """
        with torch.no_grad():
            dummy_input = torch.randint(0, 100, (1, 10))
            
            # First pass
            out1, _ = self.model(dummy_input)
            
            # Second pass with same input
            out2, _ = self.model(dummy_input)
            
            # If outputs differ, model is stateful
            return not torch.allclose(out1, out2, atol=1e-6)
    
    def save_state(self, name: str = 'default') -> Dict[str, Any]:
        """
        Save current model state.
        
        Args:
            name: Identifier for this state snapshot
            
        Returns:
            Dictionary containing state information
        """
        state = {}
        
        if self._is_stateful:
            # Extract from LinearAttention
            linear_attn = self.model.bdh.linear_attn
            
            # Try various possible state storage locations
            if hasattr(linear_attn, 'state'):
                state['synaptic_state'] = linear_attn.state.clone() if linear_attn.state is not None else None
            
            if hasattr(linear_attn, 'kv_state'):
                state['kv_state'] = linear_attn.kv_state.clone() if linear_attn.kv_state is not None else None
            
            if hasattr(linear_attn, 'hidden_state'):
                state['hidden_state'] = linear_attn.hidden_state.clone() if linear_attn.hidden_state is not None else None
            
            # Fallback: save all buffers
            if not state:
                state['buffers'] = {
                    name: buffer.clone()
                    for name, buffer in linear_attn.named_buffers()
                }
        
        else:
            # For stateless mode, we don't actually save state
            # Just mark it as saved for API consistency
            state['mode'] = 'stateless'
            state['message'] = 'No state to save in stateless mode'
        
        self._saved_states[name] = state
        return state
    
    def load_state(self, name: str = 'default'):
        """
        Restore a previously saved state.
        
        Args:
            name: Identifier of state to restore
        """
        if name not in self._saved_states:
            raise ValueError(f"No saved state with name '{name}'")
        
        state = self._saved_states[name]
        
        if self._is_stateful:
            linear_attn = self.model.bdh.linear_attn
            
            # Restore state
            if 'synaptic_state' in state and state['synaptic_state'] is not None:
                if hasattr(linear_attn, 'state'):
                    linear_attn.state = state['synaptic_state']
            
            if 'kv_state' in state and state['kv_state'] is not None:
                if hasattr(linear_attn, 'kv_state'):
                    linear_attn.kv_state = state['kv_state']
            
            if 'hidden_state' in state and state['hidden_state'] is not None:
                if hasattr(linear_attn, 'hidden_state'):
                    linear_attn.hidden_state = state['hidden_state']
            
            if 'buffers' in state:
                for name, buffer in state['buffers'].items():
                    if hasattr(linear_attn, name):
                        getattr(linear_attn, name).copy_(buffer)
    
    def reset_state(self):
        """
        Reset model to initial state (clear all working memory).
        """
        if self._is_stateful:
            linear_attn = self.model.bdh.linear_attn
            
            # Reset all possible state locations
            if hasattr(linear_attn, 'state'):
                linear_attn.state = None
            
            if hasattr(linear_attn, 'kv_state'):
                linear_attn.kv_state = None
            
            if hasattr(linear_attn, 'hidden_state'):
                linear_attn.hidden_state = None
            
            # Clear any registered buffers that might hold state
            for name, buffer in linear_attn.named_buffers():
                if 'state' in name.lower() or 'cache' in name.lower():
                    buffer.zero_()
    
    def process_with_context(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        return_state: bool = False
    ) -> Dict[str, Any]:
        """
        Process target sequence with context (backstory priming).
        
        Args:
            context_ids: [batch, context_len] - backstory tokens
            target_ids: [batch, target_len] - novel tokens
            return_state: whether to return final state
            
        Returns:
            Dictionary with logits, loss, and optionally state
        """
        if self.mode == 'stateless':
            # Concatenate context + target
            combined = torch.cat([context_ids, target_ids], dim=1)
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.model(combined[:, :-1])
            
            # Compute loss only on target portion
            context_len = context_ids.shape[1]
            target_logits = logits[:, context_len:, :]
            target_labels = combined[:, context_len+1:]
            
            loss = torch.nn.functional.cross_entropy(
                target_logits.reshape(-1, self.model.config.vocab_size),
                target_labels.reshape(-1),
                reduction='mean'
            )
            
            result = {
                'logits': target_logits,
                'loss': loss.item(),
                'mode': 'concatenated'
            }
            
        else:
            # STATEFUL mode: process context first, then target
            
            # 1. Process context (primes the state)
            self.reset_state()
            with torch.no_grad():
                _ = self.model(context_ids)
            
            # 2. Save primed state
            primed_state = self.save_state('primed')
            
            # 3. Process target with primed state
            with torch.no_grad():
                target_logits, _ = self.model(target_ids[:, :-1])
            
            # 4. Compute loss
            loss = torch.nn.functional.cross_entropy(
                target_logits.reshape(-1, self.model.config.vocab_size),
                target_ids[:, 1:].reshape(-1),
                reduction='mean'
            )
            
            result = {
                'logits': target_logits,
                'loss': loss.item(),
                'mode': 'stateful',
                'state': primed_state if return_state else None
            }
        
        return result
    
    def compute_baseline_loss(
        self,
        target_ids: torch.Tensor
    ) -> float:
        """
        Compute baseline loss on target without context.
        
        Args:
            target_ids: [batch, seq_len] novel tokens
            
        Returns:
            Loss value
        """
        self.reset_state()
        
        with torch.no_grad():
            logits, _ = self.model(target_ids[:, :-1])
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.model.config.vocab_size),
            target_ids[:, 1:].reshape(-1),
            reduction='mean'
        )
        
        return loss.item()


def test_state_manager():
    """Test StateManager functionality"""
    print("="*60)
    print("TESTING STATE MANAGER")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create state manager
    manager = StateManager(model, mode='stateless')
    
    # Test data
    context_ids = torch.randint(0, config.vocab_size, (1, 20)).to(device)
    target_ids = torch.randint(0, config.vocab_size, (1, 30)).to(device)
    
    print("\n" + "="*60)
    print("TEST 1: Process with context")
    print("="*60)
    
    result = manager.process_with_context(context_ids, target_ids)
    print(f"Mode: {result['mode']}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Logits shape: {result['logits'].shape}")
    
    print("\n" + "="*60)
    print("TEST 2: Baseline (no context)")
    print("="*60)
    
    baseline_loss = manager.compute_baseline_loss(target_ids)
    print(f"Baseline loss: {baseline_loss:.4f}")
    
    print("\n" + "="*60)
    print("TEST 3: Compare losses")
    print("="*60)
    
    delta = result['loss'] - baseline_loss
    print(f"Primed loss:   {result['loss']:.4f}")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"Delta:         {delta:.4f}")
    
    if self.mode == 'stateless':
        print("\nNote: In stateless mode, context effect depends on")
        print("      semantic coherence between context and target.")
    
    print("\n" + "="*60)
    print("TEST 4: State save/load/reset")
    print("="*60)
    
    manager.save_state('checkpoint_1')
    print("✅ State saved as 'checkpoint_1'")
    
    manager.reset_state()
    print("✅ State reset")
    
    manager.load_state('checkpoint_1')
    print("✅ State restored from 'checkpoint_1'")
    
    print("\n✅ StateManager tests complete!")


if __name__ == "__main__":
    test_state_manager()

