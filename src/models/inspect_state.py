"""
Inspect what state LinearAttention maintains internally.
This determines how we'll extract/inject state for consistency checking.
"""

import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from textpath import TextPath, TextPathConfig


def inspect_linear_attention_state():
    """
    Examine LinearAttention module to find where state is stored
    """
    print("="*60)
    print("LINEAR ATTENTION STATE INSPECTION")
    print("="*60)
    
    # Load pretrained model
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    model = TextPath(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Access LinearAttention
    linear_attn = model.bdh.linear_attn
    
    print("\nLinearAttention module:")
    print(linear_attn)
    
    print("\n" + "="*60)
    print("ATTRIBUTES:")
    print("="*60)
    
    # List all attributes
    for name in dir(linear_attn):
        if not name.startswith('_'):
            attr = getattr(linear_attn, name)
            if isinstance(attr, torch.Tensor):
                print(f"  {name}: Tensor {attr.shape}")
            elif isinstance(attr, torch.nn.Parameter):
                print(f"  {name}: Parameter {attr.shape}")
            elif not callable(attr):
                print(f"  {name}: {type(attr).__name__}")
    
    print("\n" + "="*60)
    print("TESTING STATE BEHAVIOR:")
    print("="*60)
    
    # Create two sequences
    seq1 = torch.randint(0, config.vocab_size, (1, 32))
    seq2 = torch.randint(0, config.vocab_size, (1, 32))
    
    print("\nProcessing sequence 1...")
    with torch.no_grad():
        logits1_a, _ = model(seq1)
    
    # Check if any internal state changed
    print("Checking for stateful behavior...")
    
    # Process seq1 again - should give same result if stateless
    with torch.no_grad():
        logits1_b, _ = model(seq1)
    
    same = torch.allclose(logits1_a, logits1_b, atol=1e-6)
    print(f"  Same input → same output: {same}")
    
    if same:
        print("  → LinearAttention appears STATELESS (weights don't change during inference)")
    else:
        print("  → LinearAttention appears STATEFUL (has hidden state)")
    
    print("\n" + "="*60)
    print("IMPLICATIONS FOR CONSISTENCY CHECKING:")
    print("="*60)
    
    if same:
        print("""
Since LinearAttention is stateless during inference, the synaptic state σ
is NOT maintained between forward passes. This means:

STRATEGY:
1. Concatenate backstory + novel text into one sequence
2. Feed the combined sequence through the model
3. Compute loss only on the novel portion (ignore backstory tokens)
4. Compare: loss(backstory + novel) vs loss(novel alone)

If backstory is consistent:
  → It provides useful context → LOWER loss on novel

If backstory contradicts:
  → It provides misleading context → HIGHER loss on novel
        """)
    else:
        print("""
LinearAttention maintains state during inference. This means:

STRATEGY:
1. Feed backstory → state updates to σ_primed
2. Feed novel with σ_primed → measure loss
3. Compare with baseline (novel without backstory priming)
        """)
    
    return model, linear_attn


if __name__ == "__main__":
    inspect_linear_attention_state()

