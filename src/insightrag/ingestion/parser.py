"""SEC 10-K filing parser.

Downloads filings from SEC EDGAR and extracts structured sections.
10-Ks have a well-known structure (Item 1, Item 1A, Item 7, etc.) which we
preserve as section metadata for better retrieval grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger
from sec_edgar_downloader import Downloader

# 10-K canonical sections — used to tag chunks with section metadata.
# Allows retrieval to filter "only show me Risk Factors" or boost specific sections.
ITEM_PATTERNS: dict[str, re.Pattern[str]] = {
    "business": re.compile(r"item\s+1[\.\s]+business", re.I),
    "risk_factors": re.compile(r"item\s+1a[\.\s]+risk\s+factors", re.I),
    "properties": re.compile(r"item\s+2[\.\s]+properties", re.I),
    "legal_proceedings": re.compile(r"item\s+3[\.\s]+legal\s+proceedings", re.I),
    "mda": re.compile(r"item\s+7[\.\s]+management", re.I),
    "financial_statements": re.compile(r"item\s+8[\.\s]+financial\s+statements", re.I),
    "controls": re.compile(r"item\s+9a[\.\s]+controls", re.I),
}


@dataclass
class ParsedSection:
    name: str
    text: str
    order: int


@dataclass
class ParsedDocument:
    ticker: str
    filing_date: str
    accession_number: str
    sections: list[ParsedSection] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"## {s.name}\n{s.text}" for s in self.sections)


class SECFilingParser:
    """Downloads and parses SEC 10-K filings."""

    def __init__(self, download_dir: Path, user_agent: str = "InsightRAG research@example.com"):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        # SEC requires a User-Agent identifying who is making the request
        self.downloader = Downloader("InsightRAG", user_agent, str(download_dir))

    def download(self, ticker: str, limit: int = 1) -> list[Path]:
        """Download the most recent N 10-K filings for a ticker."""
        logger.info(f"Downloading {limit} 10-K(s) for {ticker}")
        self.downloader.get("10-K", ticker, limit=limit)

        filing_dir = self.download_dir / "sec-edgar-filings" / ticker / "10-K"
        if not filing_dir.exists():
            logger.warning(f"No filings found for {ticker}")
            return []

        return sorted(filing_dir.glob("*/full-submission.txt"))

    def parse(self, filing_path: Path, ticker: str) -> ParsedDocument:
        """Parse a downloaded 10-K filing into structured sections."""
        logger.info(f"Parsing {filing_path}")
        raw = filing_path.read_text(encoding="utf-8", errors="ignore")

        # SEC filings are SGML envelopes — extract the main HTML document
        html = self._extract_primary_html(raw)
        soup = BeautifulSoup(html, "lxml")

        # Remove scripts/styles/tables of small numeric data (we keep textual content)
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = self._normalize_whitespace(text)

        sections = list(self._split_sections(text))
        accession = filing_path.parent.name

        return ParsedDocument(
            ticker=ticker,
            filing_date=self._extract_filing_date(raw),
            accession_number=accession,
            sections=sections,
        )

    @staticmethod
    def _extract_primary_html(raw: str) -> str:
        """Extract the main 10-K HTML from the SGML envelope."""
        # Find the first <DOCUMENT> block of type 10-K
        match = re.search(
            r"<DOCUMENT>.*?<TYPE>10-K.*?<TEXT>(.*?)</TEXT>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1) if match else raw

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_filing_date(raw: str) -> str:
        match = re.search(r"FILED AS OF DATE:\s*(\d{8})", raw)
        if match:
            d = match.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return "unknown"

    @staticmethod
    def _split_sections(text: str) -> Iterator[ParsedSection]:
        """Split full text into 10-K canonical sections using regex anchors."""
        # Find all section start positions
        anchors: list[tuple[int, str]] = []
        for name, pattern in ITEM_PATTERNS.items():
            for match in pattern.finditer(text):
                anchors.append((match.start(), name))
        anchors.sort()

        if not anchors:
            yield ParsedSection(name="full_document", text=text, order=0)
            return

        # De-duplicate: 10-Ks often have a Table of Contents that lists items
        # before they actually appear. We take the LAST occurrence of each section name,
        # which is almost always the real one.
        last_seen: dict[str, int] = {}
        for pos, name in anchors:
            last_seen[name] = pos
        ordered = sorted(last_seen.items(), key=lambda x: x[1])

        for i, (name, start) in enumerate(ordered):
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
            section_text = text[start:end].strip()
            if len(section_text) > 200:  # skip near-empty hits
                yield ParsedSection(name=name, text=section_text, order=i)
