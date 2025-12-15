"""
Common helper utilities.
"""

from typing import Any, List, Tuple

def as_list(x: Any):
    """Return `x` as a list if it is not already a list/tuple."""
    return x if isinstance(x, (list, tuple)) else [x]
