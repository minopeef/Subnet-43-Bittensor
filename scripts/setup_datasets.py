"""
Download required datasets for local testing.
"""
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphite.data.dataset_utils import load_default_dataset

if __name__ == "__main__":
    print("Loading datasets...")
    load_default_dataset()
    print("✓ Datasets loaded successfully")