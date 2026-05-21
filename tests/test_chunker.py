from insightrag.ingestion.chunker import SemanticChunker
from insightrag.ingestion.parser import ParsedDocument, ParsedSection


def test_chunker_respects_size():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    long_text = ("Apple reported strong revenue growth in the most recent quarter. " * 50).strip()
    doc = ParsedDocument(
        ticker="AAPL", filing_date="2023-10-30", accession_number="000",
        sections=[ParsedSection(name="mda", text=long_text, order=0)],
    )
    chunks = list(chunker.chunk_document(doc))
    assert len(chunks) > 1
    for c in chunks:
        # Allow some slack — recursive splitter may slightly exceed on edge cases
        assert chunker._token_count(c.text) <= chunker.chunk_size + 10


def test_chunker_metadata_preserved():
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
    doc = ParsedDocument(
        ticker="TSLA", filing_date="2024-01-29", accession_number="abc",
        sections=[ParsedSection(name="risk_factors", text="A " * 500, order=1)],
    )
    chunks = list(chunker.chunk_document(doc))
    assert all(c.metadata["ticker"] == "TSLA" for c in chunks)
    assert all(c.metadata["section"] == "risk_factors" for c in chunks)
    assert all(c.metadata["filing_date"] == "2024-01-29" for c in chunks)


def test_chunker_drops_tiny_chunks(sample_chunk_text):
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    doc = ParsedDocument(
        ticker="AAPL", filing_date="2023-10-30", accession_number="x",
        sections=[
            ParsedSection(name="mda", text=sample_chunk_text, order=0),
            ParsedSection(name="risk_factors", text="tiny", order=1),  # dropped
        ],
    )
    chunks = list(chunker.chunk_document(doc))
    assert all(len(c.text) >= 50 for c in chunks)
    sections = {c.metadata["section"] for c in chunks}
    assert "risk_factors" not in sections


def test_chunker_rejects_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=100, chunk_overlap=100)
