"""
Visualization module for TextPath classifier analysis.
"""
from .visualize import (
    extract_embeddings,
    visualize_embedding_space,
    analyze_attention_patterns,
    plot_classification_confidence,
    visualize_by_character,
    visualize_by_book,
    run_all_visualizations
)

__all__ = [
    'extract_embeddings',
    'visualize_embedding_space',
    'analyze_attention_patterns',
    'plot_classification_confidence',
    'visualize_by_character',
    'visualize_by_book',
    'run_all_visualizations'
]
