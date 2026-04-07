"""Hatchling metadata hook: strips YAML frontmatter from README.md before PyPI upload."""
import re
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    """Strip ``---...---`` YAML frontmatter from README.md for the PyPI long description."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        readme_path = Path(self.root) / "README.md"
        raw = readme_path.read_text(encoding="utf-8")
        # Strip leading YAML frontmatter block (--- ... ---) if present
        stripped = re.sub(r"\A---\n.*?\n---\n\n?", "", raw, flags=re.DOTALL)
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": stripped,
        }
