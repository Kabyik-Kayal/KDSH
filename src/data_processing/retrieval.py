"""
Retrieve relevant novel chunks for a given backstory.
Uses simple TF-IDF for efficiency (can upgrade to embeddings later).
"""

from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class NovelRetriever:
    """
    Retrieves relevant chunks from novels given a backstory query.
    """
    
    def __init__(
        self,
        novel_path: Path,
        chunk_size: int = 500,  # words
        overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Load novel
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean Gutenberg
        text = self._clean_gutenberg(text)
        
        # Create overlapping chunks
        self.chunks = self._create_chunks(text)
        
        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        
        print(f"✅ Indexed {len(self.chunks)} chunks from {novel_path.name}")
    
    def _clean_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg headers"""
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
        ]
        
        start_idx = 0
        for marker in start_markers:
            if marker in text:
                start_idx = text.find(marker)
                start_idx = text.find('\n', start_idx) + 1
                break
        
        end_idx = len(text)
        for marker in end_markers:
            if marker in text:
                end_idx = text.find(marker)
                break
        
        return text[start_idx:end_idx].strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping word-based chunks"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) > self.chunk_size // 2:  # Skip tiny end chunks
                chunks.append(' '.join(chunk_words))
            i += (self.chunk_size - self.overlap)
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for query.
        
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.chunks[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results


def test_retrieval():
    """Test retrieval system"""
    print("="*60)
    print("TESTING RETRIEVAL SYSTEM")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    novel_path = ROOT / "Dataset" / "Books" / "The Count of Monte Cristo.txt"
    
    # Build retriever
    retriever = NovelRetriever(novel_path, chunk_size=300, overlap=50)
    
    # Test query
    backstory = """
    Edmond Dantès was imprisoned in the Château d'If after being falsely
    accused of treason. He spent many years in prison before escaping.
    """
    
    print(f"\nQuery (backstory):")
    print(backstory[:100] + "...")
    
    print(f"\nRetrieving top 3 relevant chunks...")
    results = retriever.retrieve(backstory, top_k=3)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- Chunk {i} (similarity: {score:.3f}) ---")
        print(chunk[:200] + "...")
    
    print("\n✅ Retrieval test complete")


if __name__ == "__main__":
    test_retrieval()

