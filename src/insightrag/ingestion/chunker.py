"""Semantic chunker.

Naive fixed-size splitting destroys context. We use a recursive strategy that
prefers natural boundaries: paragraphs > sentences > words. Each chunk carries
section metadata so the retriever can use it for filtering/boosting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken

from insightrag.ingestion.parser import ParsedDocument

# Order matters: try larger separators first
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata, "chunk_id": self.chunk_id}


class SemanticChunker:
    """Recursive chunker using token counts (not character counts).

    Token counting matches what the embedding/LLM models actually see, which is more accurate
    than character heuristics — especially for financial docs heavy on numbers and tickers.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, encoding: str = "cl100k_base"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding)

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text, disallowed_special=()))

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the largest separator that produces small enough pieces."""
        if self._token_count(text) <= self.chunk_size:
            return [text]

        for i, sep in enumerate(separators):
            if sep in text:
                parts = text.split(sep)
                # Re-attach separator so we don't lose punctuation
                parts = [p + sep for p in parts[:-1]] + [parts[-1]]
                remaining = separators[i + 1 :]
                out: list[str] = []
                for part in parts:
                    if self._token_count(part) <= self.chunk_size:
                        out.append(part)
                    else:
                        out.extend(self._split_text(part, remaining))
                return self._merge_small(out)

        # No separator worked — hard split by tokens
        return self._token_split(text)

    def _token_split(self, text: str) -> list[str]:
        tokens = self.encoder.encode(text, disallowed_special=())
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), step):
            piece = tokens[start : start + self.chunk_size]
            chunks.append(self.encoder.decode(piece))
        return chunks

    def _merge_small(self, parts: list[str]) -> list[str]:
        """Merge adjacent small parts up to chunk_size, with overlap between merged chunks."""
        merged: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for part in parts:
            tcount = self._token_count(part)
            if buffer_tokens + tcount <= self.chunk_size:
                buffer.append(part)
                buffer_tokens += tcount
            else:
                if buffer:
                    merged.append("".join(buffer))
                # Start new buffer with overlap from previous
                if merged and self.chunk_overlap > 0:
                    overlap_text = self._tail(merged[-1], self.chunk_overlap)
                    buffer = [overlap_text, part]
                    buffer_tokens = self._token_count(overlap_text) + tcount
                else:
                    buffer = [part]
                    buffer_tokens = tcount

        if buffer:
            merged.append("".join(buffer))
        return merged

    def _tail(self, text: str, n_tokens: int) -> str:
        tokens = self.encoder.encode(text, disallowed_special=())
        return self.encoder.decode(tokens[-n_tokens:])

    def chunk_document(self, doc: ParsedDocument) -> Iterable[Chunk]:
        """Chunk a parsed SEC filing, preserving section provenance in metadata."""
        for section in doc.sections:
            pieces = self._split_text(section.text, SEPARATORS)
            for i, piece in enumerate(pieces):
                text = re.sub(r"\s+", " ", piece).strip()
                if len(text) < 50:  # drop trivially small chunks
                    continue
                yield Chunk(
                    text=text,
                    metadata={
                        "ticker": doc.ticker,
                        "filing_date": doc.filing_date,
                        "accession_number": doc.accession_number,
                        "section": section.name,
                        "section_order": section.order,
                        "chunk_index": i,
                    },
                    chunk_id=f"{doc.ticker}_{doc.accession_number}_{section.name}_{i}",
                )
