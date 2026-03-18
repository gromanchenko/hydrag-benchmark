"""Three-state augmentation cache — re-exported from hydrag-core.

The canonical implementation lives in hydrag.doc2query (hydrag-core v1.2.0+).
This module re-exports all public names for backward compatibility.
"""

from hydrag.doc2query import (  # noqa: F401
    AugmentationCache,
    CacheEntry,
)

__all__ = [
    "AugmentationCache",
    "CacheEntry",
]
