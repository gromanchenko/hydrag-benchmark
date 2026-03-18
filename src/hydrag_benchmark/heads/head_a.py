"""Head A — Graph/Index retrieval. Zero GPU. CPU-only.

Builds offline indexes from parsed corpus using deterministic static analysis:
1. Symbol Index: inverted index mapping symbol → chunk IDs
2. Call Graph: directed graph of import/call references
3. Reference Graph: bidirectional links from doc cross-references

Query-time: extract identifiers → lookup → 1-hop graph expand → rank by
centrality × term overlap.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .base import Chunk, ScoredChunk

# ── Symbol extraction patterns (best-effort regex, per §10 Non-Goals) ────────

_PYTHON_SYMBOLS = re.compile(
    r"(?:^|\s)(?:def|class)\s+([A-Za-z_]\w*)"
    r"|(?:^|\s)([A-Za-z_]\w*)\s*="
    r"|(?:^|\s)import\s+([A-Za-z_][\w.]*)"
    r"|(?:^|\s)from\s+([A-Za-z_][\w.]*)\s+import",
    re.MULTILINE,
)

_GO_SYMBOLS = re.compile(
    r"(?:^|\s)func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)"
    r"|(?:^|\s)type\s+([A-Za-z_]\w*)"
    r"|(?:^|\s)var\s+([A-Za-z_]\w*)"
    r"|(?:^|\s)const\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)

_C_SYMBOLS = re.compile(
    r"(?:^|\s)(?:void|int|char|float|double|long|unsigned|struct|enum)\s+\*?\s*([A-Za-z_]\w*)\s*\("
    r"|(?:^|\s)(?:struct|enum|typedef)\s+([A-Za-z_]\w*)"
    r"|#define\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)

_JS_TS_SYMBOLS = re.compile(
    r"(?:^|\s)(?:function|class|interface|type|enum)\s+([A-Za-z_$]\w*)"
    r"|(?:^|\s)(?:const|let|var|export)\s+([A-Za-z_$]\w*)\s*="
    r"|(?:^|\s)import\s+.*?from\s+['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)

_RUST_SYMBOLS = re.compile(
    r"(?:^|\s)(?:fn|struct|enum|trait|type|const|static|mod)\s+([A-Za-z_]\w*)"
    r"|(?:^|\s)use\s+([A-Za-z_][\w:]*)",
    re.MULTILINE,
)

_GENERIC_IDENTIFIER = re.compile(r"\b([A-Za-z_]\w{2,})\b")

_LANG_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": _PYTHON_SYMBOLS,
    ".go": _GO_SYMBOLS,
    ".c": _C_SYMBOLS,
    ".h": _C_SYMBOLS,
    ".cpp": _C_SYMBOLS,
    ".cc": _C_SYMBOLS,
    ".cxx": _C_SYMBOLS,
    ".hpp": _C_SYMBOLS,
    ".js": _JS_TS_SYMBOLS,
    ".jsx": _JS_TS_SYMBOLS,
    ".ts": _JS_TS_SYMBOLS,
    ".tsx": _JS_TS_SYMBOLS,
    ".rs": _RUST_SYMBOLS,
}

# Import patterns for call graph
_PYTHON_IMPORTS = re.compile(
    r"^(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)", re.MULTILINE
)
_PYTHON_CALLS = re.compile(r"([A-Za-z_]\w*)\s*\(")

# Doc cross-reference patterns
_DOC_XREF = re.compile(
    r"\[([^\]]+)\]\([^)]+\)"   # markdown links
    r"|@see\s+(\S+)"           # @see tags
    r"|See\s+also[:\s]+(\S+)"  # "See also" references
    r"|`([A-Za-z_]\w+)`",      # backtick references
    re.IGNORECASE,
)


def extract_symbols(text: str, extension: str) -> list[str]:
    """Extract identifiers from a code/doc chunk based on file extension."""
    pattern = _LANG_PATTERNS.get(extension, _GENERIC_IDENTIFIER)
    matches = pattern.findall(text)
    symbols: list[str] = []
    for m in matches:
        if isinstance(m, tuple):
            symbols.extend(s for s in m if s)
        elif m:
            symbols.append(m)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for s in symbols:
        s_lower = s.lower()
        if s_lower not in seen and len(s) > 1:
            seen.add(s_lower)
            result.append(s)
    return result


def extract_query_identifiers(query: str) -> list[str]:
    """Extract identifiers and keywords from a query string."""
    # Split on whitespace and punctuation, keep tokens that look like identifiers
    tokens = re.findall(r"[A-Za-z_]\w+", query)
    # Filter out common English stop words
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "that",
        "this", "these", "those", "what", "which", "who", "whom", "how",
        "when", "where", "why", "not", "no", "nor", "and", "but", "or",
        "if", "then", "else", "for", "with", "from", "into", "about",
        "between", "through", "during", "after", "before", "above", "below",
        "to", "of", "in", "on", "at", "by", "up", "out", "off", "over",
        "under", "again", "further", "once", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "any", "such", "only", "own",
        "same", "so", "than", "too", "very", "just",
    }
    return [t for t in tokens if t.lower() not in stop and len(t) > 1]


# ── Index structures ────────────────────────────────────────────────────────


@dataclass
class SymbolIndex:
    """Inverted index: symbol (lowercased) → set of chunk IDs."""

    _index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, symbol: str, chunk_id: str) -> None:
        self._index[symbol.lower()].add(chunk_id)

    def lookup(self, symbol: str) -> set[str]:
        return self._index.get(symbol.lower(), set())

    def __len__(self) -> int:
        return len(self._index)


@dataclass
class Graph:
    """Directed graph for call/reference relationships."""

    _edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, source: str, target: str) -> None:
        self._edges[source].add(target)
        self._reverse[target].add(source)

    def neighbors(self, node: str) -> set[str]:
        """1-hop neighbors (both directions)."""
        return self._edges.get(node, set()) | self._reverse.get(node, set())

    def in_degree(self, node: str) -> int:
        """Number of inbound references (graph centrality proxy)."""
        return len(self._reverse.get(node, set()))

    def nodes(self) -> set[str]:
        return set(self._edges.keys()) | set(self._reverse.keys())


# ── Head A ──────────────────────────────────────────────────────────────────


class HeadA:
    """Graph/Index retrieval head. Zero GPU cost at query time.

    Retrieves by structural proximity to query-extracted identifiers.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._symbol_index = SymbolIndex()
        self._call_graph = Graph()
        self._ref_graph = Graph()
        self._build_indexes(chunks)

    @property
    def name(self) -> str:
        return "head_a"

    @property
    def symbol_index(self) -> SymbolIndex:
        return self._symbol_index

    @property
    def call_graph(self) -> Graph:
        return self._call_graph

    @property
    def ref_graph(self) -> Graph:
        return self._ref_graph

    def _build_indexes(self, chunks: list[Chunk]) -> None:
        """Build symbol index, call graph, and reference graph from chunks."""
        for chunk in chunks:
            ext = _infer_extension(chunk.source)

            # Symbol index
            symbols = extract_symbols(chunk.text, ext)
            chunk.symbols = symbols
            for sym in symbols:
                self._symbol_index.add(sym, chunk.chunk_id)

            # Call graph (Python imports/calls)
            if ext == ".py":
                self._build_python_call_edges(chunk)

            # Reference graph (doc cross-references)
            if ext in {".md", ".yaml", ".yml", ".toml", ".rst", ".txt"}:
                self._build_ref_edges(chunk)

    def _build_python_call_edges(self, chunk: Chunk) -> None:
        """Extract import and call edges for Python code."""
        for match in _PYTHON_IMPORTS.finditer(chunk.text):
            from_mod = match.group(1)
            imports = match.group(2)
            if from_mod:
                self._call_graph.add_edge(chunk.chunk_id, from_mod)
            for imp in imports.split(","):
                name = imp.strip().split(" as ")[0].strip()
                if name:
                    self._call_graph.add_edge(chunk.chunk_id, name)

        for match in _PYTHON_CALLS.finditer(chunk.text):
            callee = match.group(1)
            if callee not in {"if", "for", "while", "with", "return", "print", "def", "class"}:
                # Link chunk to any chunk that defines this symbol
                targets = self._symbol_index.lookup(callee)
                for t in targets:
                    if t != chunk.chunk_id:
                        self._call_graph.add_edge(chunk.chunk_id, t)

    def _build_ref_edges(self, chunk: Chunk) -> None:
        """Extract cross-references from documentation chunks."""
        for match in _DOC_XREF.finditer(chunk.text):
            ref = next((g for g in match.groups() if g), None)
            if ref:
                self._ref_graph.add_edge(chunk.chunk_id, ref)

    def retrieve(self, query: str, n_results: int = 10) -> list[ScoredChunk]:
        """Query-time retrieval via symbol lookup + graph expansion."""
        identifiers = extract_query_identifiers(query)
        if not identifiers:
            return []

        # Step 1: Lookup matching symbols in inverted index
        candidate_scores: dict[str, float] = defaultdict(float)
        for ident in identifiers:
            for chunk_id in self._symbol_index.lookup(ident):
                candidate_scores[chunk_id] += 1.0

        # Step 2: Expand via 1-hop graph neighbors
        direct_hits = set(candidate_scores.keys())
        for chunk_id in direct_hits:
            for neighbor in self._call_graph.neighbors(chunk_id):
                if neighbor in self._chunks:
                    candidate_scores[neighbor] += 0.5
            for neighbor in self._ref_graph.neighbors(chunk_id):
                if neighbor in self._chunks:
                    candidate_scores[neighbor] += 0.3

        # Step 3: Rank by centrality × term overlap
        scored: list[tuple[str, float]] = []
        for chunk_id, base_score in candidate_scores.items():
            if chunk_id not in self._chunks:
                continue
            centrality = 1.0 + self._call_graph.in_degree(chunk_id) + self._ref_graph.in_degree(chunk_id)
            term_overlap = self._term_overlap(identifiers, self._chunks[chunk_id])
            final_score = base_score * centrality * (0.5 + term_overlap)
            scored.append((chunk_id, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for chunk_id, score in scored[:n_results]:
            results.append(ScoredChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                head_origin="head_a",
            ))
        return results

    @staticmethod
    def _term_overlap(identifiers: list[str], chunk: Chunk) -> float:
        """Fraction of query identifiers found in chunk text."""
        if not identifiers:
            return 0.0
        text_lower = chunk.text.lower()
        found = sum(1 for ident in identifiers if ident.lower() in text_lower)
        return found / len(identifiers)


def _infer_extension(source: str) -> str:
    """Infer file extension from source path."""
    dot = source.rfind(".")
    if dot == -1:
        return ""
    return source[dot:].lower()
