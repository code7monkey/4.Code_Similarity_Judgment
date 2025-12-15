"""Utility functions for GraphCodeBERT project."""

import os
import random
import re
from typing import List

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except ImportError:
        # PyTorch is optional
        pass


def remove_extras(code: str) -> str:
    """Remove comments and extra whitespace from code.

    This simple heuristic removes single-line comments and docstrings for
    Python code. For other languages you may need to adjust the regex.

    Args:
        code: Raw source code as a string.

    Returns:
        Cleaned code string with comments and excess whitespace removed.
    """
    # Remove Python single-line comments
    code = re.sub(r"#.*", "", code)
    # Remove triple-quoted docstrings
    code = re.sub(r"\"\"\".*?\"\"\"", "", code, flags=re.DOTALL)
    code = re.sub(r"'''(.*?)'''", "", code, flags=re.DOTALL)
    # Remove leading/trailing whitespace on each line and drop empty lines
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return '\n'.join(lines)
