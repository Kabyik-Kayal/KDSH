"""
Core consistency scoring logic using perplexity delta.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict
import torch
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from textpath import TextPath, TextPathConfig


class ConsistencyScorer:
    """
    Scores backstory-novel consistency using perplexity delta.
    
    Strategy:
    1. Compute loss on novel alone (baseline)
    2. Compute loss on [backstory + novel] (primed)
    3. Delta = primed_loss - baseline_loss
    
    Interpretation:
    - Negative delta (primed < baseline): backstory helps → CONSISTENT
    - Positive delta (primed > baseline): backstory misleads → CONTRADICT
    """
    
    def __init__(
        self,
        model: TextPath,
        tokenizer: Tokenizer,
        device: torch.device,
        max_novel_tokens: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_novel_tokens = max_novel_tokens
        self.model.eval()
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """Compute cross-entropy loss"""
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.model.config.vocab_size),
            target_ids.reshape(-1),
            reduction='mean'
        )
        
        return loss.item()
    
    def score_consistency(
        self,
        backstory: str,
        novel_chunk: str,
    ) -> Dict[str, float]:
        """
        Score consistency between backstory and novel chunk.
        
        Returns:
            dict with keys: baseline_loss, primed_loss, delta, perplexity_ratio
        """
        # Tokenize
        backstory_ids = self.tokenizer.encode(backstory).ids
        novel_ids = self.tokenizer.encode(novel_chunk).ids
        
        # Truncate novel if too long
        if len(novel_ids) > self.max_novel_tokens:
            novel_ids = novel_ids[:self.max_novel_tokens]
        
        # === Baseline: novel alone ===
        novel_tensor = torch.tensor([novel_ids], dtype=torch.long).to(self.device)
        baseline_loss = self.compute_loss(
            novel_tensor[:, :-1],  # input
            novel_tensor[:, 1:]    # target
        )
        
        # === Primed: backstory + novel ===
        combined_ids = backstory_ids + novel_ids
        combined_tensor = torch.tensor([combined_ids], dtype=torch.long).to(self.device)
        
        # Compute loss only on the novel portion
        backstory_len = len(backstory_ids)
        novel_start = backstory_len
        novel_end = len(combined_ids)
        
        # Full forward pass on combined sequence
        with torch.no_grad():
            logits, _ = self.model(combined_tensor[:, :-1])
        
        # Extract logits corresponding to novel tokens
        novel_logits = logits[:, novel_start:, :]  # [1, novel_len, vocab]
        novel_targets = combined_tensor[:, novel_start+1:]  # [1, novel_len]
        
        primed_loss = torch.nn.functional.cross_entropy(
            novel_logits.reshape(-1, self.model.config.vocab_size),
            novel_targets.reshape(-1),
            reduction='mean'
        ).item()
        
        # === Compute metrics ===
        delta = primed_loss - baseline_loss
        baseline_ppl = torch.exp(torch.tensor(baseline_loss)).item()
        primed_ppl = torch.exp(torch.tensor(primed_loss)).item()
        ppl_ratio = primed_ppl / baseline_ppl if baseline_ppl > 0 else 1.0
        
        return {
            'baseline_loss': baseline_loss,
            'primed_loss': primed_loss,
            'delta': delta,
            'baseline_ppl': baseline_ppl,
            'primed_ppl': primed_ppl,
            'ppl_ratio': ppl_ratio,
        }


def test_scorer():
    """Test the consistency scorer"""
    print("="*60)
    print("TESTING CONSISTENCY SCORER")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Create scorer
    scorer = ConsistencyScorer(model, tokenizer, device)
    
    # Test case 1: Consistent backstory
    print("\n" + "="*60)
    print("TEST 1: CONSISTENT BACKSTORY")
    print("="*60)
    
    backstory_consistent = """
    Edmond Dantès was a young sailor from Marseille. He was honest,
    hardworking, and loved by his friends. He was engaged to Mercedes.
    """
    
    novel_chunk = """
    Edmond Dantès arrived in Marseille aboard the Pharaon. His captain
    had died during the voyage, and Edmond had successfully brought the
    ship to port. He was eager to see Mercedes, his beloved fiancée.
    """
    
    scores = scorer.score_consistency(backstory_consistent, novel_chunk)
    
    print(f"\nBackstory (consistent): {backstory_consistent[:60]}...")
    print(f"Novel chunk: {novel_chunk[:60]}...")
    print(f"\nScores:")
    print(f"  Baseline loss:     {scores['baseline_loss']:.4f}")
    print(f"  Primed loss:       {scores['primed_loss']:.4f}")
    print(f"  Delta:             {scores['delta']:.4f}")
    print(f"  Baseline PPL:      {scores['baseline_ppl']:.2f}")
    print(f"  Primed PPL:        {scores['primed_ppl']:.2f}")
    print(f"  PPL ratio:         {scores['ppl_ratio']:.3f}")
    
    # Test case 2: Contradictory backstory
    print("\n" + "="*60)
    print("TEST 2: CONTRADICTORY BACKSTORY")
    print("="*60)
    
    backstory_contradict = """
    Edmond Dantès was a wealthy aristocrat from Paris. He was a lifelong
    enemy of Mercedes and had never worked as a sailor. He owned vast estates.
    """
    
    scores = scorer.score_consistency(backstory_contradict, novel_chunk)
    
    print(f"\nBackstory (contradict): {backstory_contradict[:60]}...")
    print(f"Novel chunk: {novel_chunk[:60]}...")
    print(f"\nScores:")
    print(f"  Baseline loss:     {scores['baseline_loss']:.4f}")
    print(f"  Primed loss:       {scores['primed_loss']:.4f}")
    print(f"  Delta:             {scores['delta']:.4f}")
    print(f"  Baseline PPL:      {scores['baseline_ppl']:.2f}")
    print(f"  Primed PPL:        {scores['primed_ppl']:.2f}")
    print(f"  PPL ratio:         {scores['ppl_ratio']:.3f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("Negative delta → backstory helps → CONSISTENT")
    print("Positive delta → backstory misleads → CONTRADICT")
    print("="*60)


if __name__ == "__main__":
    test_scorer()

