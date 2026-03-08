# conftest.py – pytest configuration
import sys
from pathlib import Path

# Add project root to path so imports work without installing the package
sys.path.insert(0, str(Path(__file__).parent))
