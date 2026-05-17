#!/usr/bin/env python3
"""Bulk-ingest multiple companies' 10-K filings.

Usage:
    python scripts/bulk_ingest.py --tickers AAPL MSFT GOOGL TSLA NVDA --limit 2
    python scripts/bulk_ingest.py --tickers-file scripts/sp500_top50.txt --limit 1

Runs ingestion calls against a running InsightRAG API. Uses async concurrency with a
bounded semaphore so we don't overwhelm SEC EDGAR (their rate limit is ~10 req/sec
per IP — we stay well under that).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx


async def ingest_one(
    client: httpx.AsyncClient,
    base_url: str,
    ticker: str,
    limit: int,
    sem: asyncio.Semaphore,
) -> tuple[str, bool, str]:
    async with sem:
        try:
            resp = await client.post(
                f"{base_url}/v1/ingest",
                json={"ticker": ticker, "limit": limit},
                timeout=300,  # ingestion is slow (parse + embed)
            )
            resp.raise_for_status()
            body = resp.json()
            return ticker, True, f"{body['filings_ingested']} filings, {body['total_chunks']} chunks"
        except httpx.HTTPStatusError as e:
            return ticker, False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return ticker, False, f"{type(e).__name__}: {e}"


async def main(tickers: list[str], base_url: str, limit: int, concurrency: int) -> int:
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [ingest_one(client, base_url, t, limit, sem) for t in tickers]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - started
    succeeded = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - succeeded

    print(f"\n{'Ticker':<8} {'Status':<8} Detail")
    print("-" * 70)
    for ticker, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"{ticker:<8} {status:<8} {detail}")

    print(f"\nDone in {elapsed:.1f}s — {succeeded} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT")
    src.add_argument("--tickers-file", type=Path, help="File with one ticker per line")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--limit", type=int, default=1, help="Filings per ticker")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Max concurrent ingestions (keep low: each one is CPU-heavy)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tickers_file:
        tickers = [t.strip().upper() for t in args.tickers_file.read_text().splitlines() if t.strip()]
    else:
        tickers = [t.upper() for t in args.tickers]

    print(f"Ingesting {len(tickers)} tickers into {args.base_url} "
          f"(limit={args.limit}, concurrency={args.concurrency})")
    sys.exit(asyncio.run(main(tickers, args.base_url, args.limit, args.concurrency)))
