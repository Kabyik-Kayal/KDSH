from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from pathlib import Path
import json

def train_custom_tokenizer(
    novel_paths: list[Path],
    vocab_size: int = 16384,
    output_path: Path = Path("models/custom_tokenizer.json")
):
    """
    Train a BPE tokenizer specifically on the 19th-century novels
    """
    print(f"Training tokenizer with vocab_size={vocab_size}")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Use ByteLevel pre-tokenizer (like GPT-2)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        show_progress=True,
    )
    
    # Train on the novel files
    print(f"\nTraining on files:")
    for p in novel_paths:
        print(f"  - {p}")
    
    tokenizer.train(files=[str(p) for p in novel_paths], trainer=trainer)
    
    # Add post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Save tokenizer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    
    print(f"\n Tokenizer saved to {output_path}")
    
    # Test the tokenizer
    test_text = "Edmond Dantès was arrested and imprisoned in the Château d'If"
    encoded = tokenizer.encode(test_text)
    print(f"\n{'='*60}")
    print(f"Test encoding:")
    print(f"Text: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Token count: {len(encoded.ids)}")
    print(f"{'='*60}")
    
    return tokenizer


def analyze_character_names(tokenizer: Tokenizer):
    """
    Analyze how character names are tokenized
    """
    test_names = [
        # The Count of Monte Cristo characters
        "Dantès", "Edmond", "Danglars", "Fernand", "Mercedes", 
        "Villefort", "Noirtier", "Caderousse", "Morrel",
        "Monte Cristo", "Abbé Faria", "Haydée",
        
        # In Search of the Castaways characters
        "Glenarvan", "Paganel", "Jacques Paganel", "Thalcave", 
        "Ayrton", "Tom Ayrton", "Robert Grant", "Mary Grant",
        "Kai-Koumou",
    ]
    
    print("\n" + "="*70)
    print("CHARACTER NAME TOKENIZATION ANALYSIS")
    print("="*70)
    print(f"{'Character Name':<25} | {'Tokens':<30} | Token Count")
    print("-"*70)
    
    for name in test_names:
        encoded = tokenizer.encode(name)
        tokens_str = str(encoded.tokens[:5])  # Show first 5 tokens
        if len(encoded.tokens) > 5:
            tokens_str += "..."
        print(f"{name:<25} | {tokens_str:<30} | {len(encoded.ids)}")
    
    print("="*70)
    print("\n Ideal: Character names should be 1-2 tokens for BDH concept neurons")


def test_novel_encoding(tokenizer: Tokenizer, novel_path: Path, sample_size: int = 1000):
    """
    Test encoding efficiency on actual novel text
    """
    print(f"\n{'='*70}")
    print(f"Testing encoding on: {novel_path.name}")
    print("="*70)
    
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()[:sample_size]  # First 1000 chars
    
    encoded = tokenizer.encode(text)
    
    print(f"Sample text ({len(text)} chars):")
    print(text[:200] + "...")
    print(f"\nEncoded to {len(encoded.ids)} tokens")
    print(f"Compression ratio: {len(text) / len(encoded.ids):.2f} chars/token")
    print("="*70)


if __name__ == "__main__":
    data_dir = Path("Dataset/Books")
    novel_paths = [
        data_dir / "The Count of Monte Cristo.txt",
        data_dir / "In search of the castaways.txt"
    ]
    
    # Verify files exist
    missing_files = [p for p in novel_paths if not p.exists()]
    if missing_files:
        print("Missing files:")
        for p in missing_files:
            print(f"  - {p}")
        exit(1)
    
    print("="*70)
    print("CUSTOM TOKENIZER TRAINING FOR KDSH 2026 - TRACK B")
    print("="*70)
    
    # Train tokenizer
    tokenizer = train_custom_tokenizer(
        novel_paths=novel_paths,
        vocab_size=16384,  # 16K vocab should be enough for 2 novels
        output_path=Path("models/custom_tokenizer.json")
    )
    
    # Analyze character names
    analyze_character_names(tokenizer)
    
    # Test encoding efficiency
    for novel_path in novel_paths:
        test_novel_encoding(tokenizer, novel_path, sample_size=1000)
    
    print("Tokenizer training and analysis complete!")