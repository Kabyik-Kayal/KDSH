import pathway as pw
import os
from pathlib import Path

class NovelSchema(pw.Schema):
    """Schema for novel text data"""
    book_name: str
    content: str
    source_path: str

class BackstorySchema(pw.Schema):
    """Schema for backstory data"""
    id: int
    book_name: str
    char: str
    caption: str
    content: str
    label: int | None  # None for test set

def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg headers/footers
    """
    # Find start of actual content
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

def load_novels(data_dir: Path) -> pw.Table:
    """
    Load novel texts using Pathway
    """
    novels = []
    
    # Load The Count of Monte Cristo
    monte_cristo_path = data_dir / "The_Count_of_Monte_Cristo.txt"
    if monte_cristo_path.exists():
        with open(monte_cristo_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = clean_gutenberg_text(content)
            novels.append({
                'book_name': 'The Count of Monte Cristo',
                'content': content,
                'source_path': str(monte_cristo_path)
            })
    
    # Load In Search of the Castaways
    castaways_path = data_dir / "In_Search_of_the_Castaways.txt"
    if castaways_path.exists():
        with open(castaways_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = clean_gutenberg_text(content)
            novels.append({
                'book_name': 'In Search of the Castaways',
                'content': content,
                'source_path': str(castaways_path)
            })
    
    # Create Pathway table
    table = pw.debug.table_from_rows(
        schema=NovelSchema,
        rows=novels
    )
    
    return table

def load_backstories(csv_path: Path, has_labels: bool = True) -> pw.Table:
    """
    Load train or test backstories into a Pathway table.
    """
    import pandas as pd
    import math

    df = pd.read_csv(csv_path)

    # Ensure columns exist
    expected_cols = ["id", "book_name", "char", "caption", "content"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    if has_labels:
        if "label" not in df.columns:
            raise ValueError(f"Missing 'label' column in {csv_path} for train data")
    else:
        # For test set, create a None label column
        df["label"] = None

    # Map string labels to integers
    label_map = {
        "consistent": 1,
        "contradict": 0,
    }

    rows = []
    for _, row in df.iterrows():
        caption = row["caption"]
        # Replace NaN with empty string
        if isinstance(caption, float) and math.isnan(caption):
            caption = ""

        label = row["label"]
        # Map label to int or keep as None
        if has_labels and label is not None:
            if isinstance(label, str):
                label = label_map.get(label, label_map.get(label.lower()))
                if label is None:
                    raise ValueError(f"Unknown label value: {row['label']}")
            else:
                label = int(label)
        else:
            label = None

        rows.append(
            (
                int(row["id"]),
                str(row["book_name"]),
                str(row["char"]),
                str(caption),
                str(row["content"]),
                label,
            )
        )

    table = pw.debug.table_from_rows(
        schema=BackstorySchema,
        rows=rows,
    )

    return table

# Test the ingestion
if __name__ == "__main__":
    data_dir = Path("Dataset")
    
    print("Loading novels...")
    novels_table = load_novels(data_dir / "Books")
    
    print("Loading train backstories...")
    train_table = load_backstories(data_dir / "train.csv", has_labels=True)
    
    print("Loading test backstories...")
    test_table = load_backstories(data_dir / "test.csv", has_labels=False)
    
    print("Data ingestion successful!")
    print("Novels table schema:", novels_table.schema)
    print("Train backstories schema:", train_table.schema)
    print("Test backstories schema:", test_table.schema)

