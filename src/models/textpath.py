"""
TextPath: BDH adapted for long-form narrative text processing
Extends the educational BDH to handle variable-length sequences and state management
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add BDH educational repo to path
ROOT = Path(__file__).resolve().parents[2]
BDH_EDU_DIR = ROOT / "repos" / "bdh_educational"
sys.path.append(str(BDH_EDU_DIR))

from bdh import BDH, BDHParameters


@dataclass
class TextPathConfig:
    """Configuration for TextPath model - BDH-based language model
    
    BDH Architecture Properties:
    - Hebbian Learning: Synapses strengthen when neurons co-activate
    - Sparse Activations: Only ~5% neurons fire (monosemantic representations)
    - Causal Circuits: Gx = E @ Dx encodes "if A then B" reasoning
    """
    vocab_size: int = 16384          # From custom tokenizer
    max_seq_len: int = 4096          # Maximum sequence length
    n_heads: int = 8                 # Attention heads
    n_neurons: int = 4096            # BDH neurons (scale-free graph)
    d_model: int = 256               # Model dimension
    n_layers: int = 4                # Number of BDH layers
    dropout: float = 0.1
    use_rope: bool = True            # Rotary position encoding
    sparsity_target: float = 0.05    # 5% neuron activation target (BDH's natural operating point)
    classification_mode: bool = False  # Enable classification head


class TextPath(nn.Module):
    """
    BDH-based language model for narrative consistency detection.
    
    Key BDH Properties Leveraged:
    - HEBBIAN LEARNING: "Neurons that fire together, wire together"
      Training on sequential passages builds causal circuits encoding
      character relationships, plot events, and narrative logic.
      
    - SPARSE ACTIVATIONS (~5%): Each concept (character, location, event)
      activates distinct neuron groups, creating monosemantic representations
      that make contradictions detectable.
      
    - CAUSAL CIRCUITS (Gx = E @ Dx): Learned weights encode reasoning like
      "If Dantès mentioned → prison/escape concepts should activate"
      
    - DYNAMIC SYNAPTIC STATE: Edge weights update during inference,
      building context-specific working memory.
    """
    
    def __init__(self, config: TextPathConfig):
        super().__init__()
        self.config = config
        
        # Create BDH parameters matching the educational implementation
        self.bdh_params = BDHParameters(
            V=config.vocab_size,
            T=config.max_seq_len,
            H=config.n_heads,
            N=config.n_neurons,
            D=config.d_model,
            L=config.n_layers,
            dropout=config.dropout,
            use_rope=config.use_rope,
            use_abs_pos=not config.use_rope,  # Use one or the other
        )
        
        # Initialize BDH core
        self.bdh = BDH(self.bdh_params)
        
        # Classification head (if enabled)
        if config.classification_mode:
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 128),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, 2)  # Binary: [Contradict, Consistent]
            )
        
        print(f"✅ TextPath initialized")
        print(f"   Vocab: {config.vocab_size:,}")
        print(f"   Neurons: {config.n_neurons:,}")
        print(f"   Layers: {config.n_layers}")
        print(f"   Classification mode: {config.classification_mode}")
        print(f"   Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with optional state extraction.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] mask (1=attend, 0=ignore)
            return_state: whether to return internal state
            return_embeddings: if True, return pooled embeddings (for classification)
            
        Returns:
            If classification_mode:
                logits: [batch_size, 2] for binary classification
            Else:
                logits: [batch_size, seq_len, vocab_size]
            state: optional dict with internal state σ
        """
        # BDH forward pass
        bdh_out = self.bdh(input_ids)
        
        if isinstance(bdh_out, tuple):
            logits, internal_state = bdh_out
        else:
            logits = bdh_out
            internal_state = None
        
        # Classification mode: pool sequence and classify
        if self.config.classification_mode:
            # Get hidden states before final LM projection
            # Access the internal embeddings from BDH
            x = self.bdh.emb(input_ids)  # (B, L, D)
            
            # Pool sequence: mean over valid tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask  # (B, D)
            else:
                pooled = x.mean(dim=1)  # (B, D)
            
            if return_embeddings:
                return pooled, None
            
            # Classification logits
            cls_logits = self.classifier_head(pooled)  # (B, 2)
            
            state = None
            if return_state:
                state = self.extract_state()
            
            return cls_logits, state
        
        # Original LM mode
        state = None
        if return_state:
            state = self.extract_state()
        
        return logits, state
    
    def extract_state(self) -> dict:
        """
        Extract internal synaptic state σ from LinearAttention.
        This is the "working memory" that encodes narrative constraints.
        """
        state = {}
        
        # The state is stored in the LinearAttention module
        linear_attn = self.bdh.linear_attn
        
        # Try to extract internal state (implementation-specific)
        # This depends on how the educational BDH stores state
        if hasattr(linear_attn, 'state'):
            state['synaptic_weights'] = linear_attn.state
        elif hasattr(linear_attn, 'kv_state'):
            state['kv_state'] = linear_attn.kv_state
        else:
            # Fallback: extract from module parameters
            state['linear_attn_params'] = {
                name: param.clone().detach()
                for name, param in linear_attn.named_parameters()
            }
        
        return state
    
    def inject_state(self, state: dict):
        """
        Inject a previously saved state into the model.
        Used for: backstory → state_prime → measure novel surprise
        """
        linear_attn = self.bdh.linear_attn
        
        if 'synaptic_weights' in state:
            if hasattr(linear_attn, 'state'):
                linear_attn.state = state['synaptic_weights']
        elif 'kv_state' in state:
            if hasattr(linear_attn, 'kv_state'):
                linear_attn.kv_state = state['kv_state']
        elif 'linear_attn_params' in state:
            for name, param in state['linear_attn_params'].items():
                if hasattr(linear_attn, name):
                    getattr(linear_attn, name).data.copy_(param.data)
    
    def reset_state(self):
        """
        Reset internal state to initial conditions.
        Used before processing a new example.
        """
        linear_attn = self.bdh.linear_attn
        
        # Reset any stateful components
        if hasattr(linear_attn, 'state'):
            linear_attn.state = None
        if hasattr(linear_attn, 'kv_state'):
            linear_attn.kv_state = None
    
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute perplexity of a sequence.
        Used for consistency scoring: high perplexity = contradiction
        
        Args:
            input_ids: [batch_size, seq_len]
            target_ids: [batch_size, seq_len] (if None, use shifted input_ids)
            
        Returns:
            perplexity: scalar tensor
        """
        if target_ids is None:
            # Standard autoregressive: predict next token
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        
        logits, _ = self.forward(input_ids)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='mean'
        )
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss)
        
        return perplexity
    
    def calculate_conditional_loss(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor
    ) -> float:
        """
        Calculate cross-entropy loss on target tokens conditioned on context.
        Used for Perplexity Delta scoring in the Generative Reasoning approach.
        
        Theory:
        - If backstory (context) is consistent with novel (target), conditioning
          on it should REDUCE the model's loss on target tokens.
        - Delta = Loss(target | empty) - Loss(target | context)
        - Positive delta = Consistent backstory
        
        Args:
            context_ids: [batch_size, context_len] - context tokens (backstory)
            target_ids: [batch_size, target_len] - target tokens (novel chunk)
            
        Returns:
            Mean cross-entropy loss on target portion only (float)
        """
        self.eval()
        
        with torch.no_grad():
            # Handle empty context case
            if context_ids.numel() == 0 or context_ids.size(1) == 0:
                # No context - just compute loss on target
                logits, _ = self.forward(target_ids[:, :-1])
                targets = target_ids[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    reduction='mean'
                )
                return loss.item()
            
            # Concatenate context and target
            full_input = torch.cat([context_ids, target_ids], dim=1)
            
            # Forward pass (exclude last token for next-token prediction)
            logits, _ = self.forward(full_input[:, :-1])
            
            # Create targets (shifted by 1)
            targets = full_input[:, 1:]
            
            # Create mask: 1 for target tokens, 0 for context tokens
            # After shifting, target portion starts at (context_len - 1)
            context_len = context_ids.size(1)
            mask = torch.zeros_like(targets, dtype=torch.float, device=targets.device)
            mask[:, context_len - 1:] = 1.0
            
            # Compute per-token loss
            loss_unreduced = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                reduction='none'
            )
            loss_unreduced = loss_unreduced.view(targets.shape)
            
            # Average only over target tokens (masked mean)
            masked_loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1e-9)
            
            return masked_loss.item()


def test_textpath():
    """Test TextPath initialization and forward pass"""
    print("="*60)
    print("TESTING TEXTPATH MODEL")
    print("="*60)
    
    # Small config for testing
    config = TextPathConfig(
        vocab_size=1000,
        max_seq_len=128,
        n_heads=4,
        n_neurons=512,
        d_model=128,
        n_layers=2,
        dropout=0.0,
    )
    
    model = TextPath(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward without state
    logits, state = model.forward(input_ids, return_state=False)
    print(f"Logits shape: {logits.shape}")
    print(f"State returned: {state is not None}")
    
    # Forward with state extraction
    logits, state = model.forward(input_ids, return_state=True)
    print(f"\nState extraction:")
    print(f"  Keys: {list(state.keys())}")
    
    # Test perplexity computation
    perplexity = model.compute_perplexity(input_ids)
    print(f"\nPerplexity: {perplexity.item():.2f}")
    
    # Test state management
    print("\nTesting state management:")
    state_backup = model.extract_state()
    print(f"  State extracted: {len(state_backup)} entries")
    
    model.reset_state()
    print(f"  State reset")
    
    model.inject_state(state_backup)
    print(f"  State injected")
    
    print("\nTextPath tests passed!")


if __name__ == "__main__":
    test_textpath()
