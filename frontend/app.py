"""InsightRAG — Streamlit chat frontend.

A thin client over the InsightRAG API: ask natural-language questions about SEC
10-K filings and get cited, grounded answers. Talks to the FastAPI backend over HTTP
(configurable via INSIGHTRAG_API_URL), so it works the same locally and in the cloud.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("INSIGHTRAG_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("INSIGHTRAG_API_KEY", "")
REQUEST_TIMEOUT = 120

# Sent on write endpoints when the backend enforces an API key.
AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="InsightRAG", page_icon="📊", layout="centered")


# ───────────────────────────── helpers ─────────────────────────────
def api_healthy() -> bool:
    try:
        r = requests.get(f"{API_URL}/healthz", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_query(question: str, ticker: str | None, top_k: int) -> dict:
    payload: dict = {"question": question, "top_k": top_k}
    if ticker:
        payload["ticker"] = ticker.upper()
    r = requests.post(
        f"{API_URL}/v1/query", json=payload, headers=AUTH_HEADERS, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def run_ingest(ticker: str, limit: int) -> dict:
    r = requests.post(
        f"{API_URL}/v1/ingest",
        json={"ticker": ticker.upper(), "limit": limit},
        headers=AUTH_HEADERS,
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} citations"):
        for c in citations:
            st.markdown(
                f"**[{c['index']}] {c.get('ticker', '?')}** · "
                f"_{c.get('section', '?')}_ · {c.get('filing_date', '?')}"
            )
            st.caption(c.get("text_preview", ""))


def render_metrics(metrics: dict) -> None:
    if not metrics:
        return
    cols = st.columns(4)
    cols[0].metric("Retrieval", f"{metrics.get('retrieval_latency_ms', 0):.0f} ms")
    cols[1].metric("Rerank", f"{metrics.get('rerank_latency_ms', 0):.0f} ms")
    cols[2].metric("Generation", f"{metrics.get('generation_latency_ms', 0):.0f} ms")
    cols[3].metric("Total", f"{metrics.get('total_latency_ms', 0):.0f} ms")


# ───────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"API: `{API_URL}`")
    st.write("🟢 Connected" if api_healthy() else "🔴 API unreachable")

    ticker = st.text_input("Filter by ticker", value="AAPL", help="Leave blank to search all")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("📥 Ingest a filing")
    st.caption("Download & index a company's latest 10-K from SEC EDGAR.")
    ingest_ticker = st.text_input("Ticker to ingest", value="AAPL", key="ingest_ticker")
    ingest_limit = st.number_input("How many filings", min_value=1, max_value=5, value=1)
    if st.button("Ingest", use_container_width=True):
        with st.spinner(f"Ingesting {ingest_ticker.upper()} — this can take a minute…"):
            try:
                res = run_ingest(ingest_ticker, int(ingest_limit))
                st.success(
                    f"Indexed {res['filings_ingested']} filing(s), "
                    f"{res['total_chunks']} chunks for {res['ticker']}."
                )
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")


# ───────────────────────────── main ────────────────────────────────
st.title("📊 InsightRAG")
st.caption("Cited answers over SEC 10-K filings — hybrid retrieval + LLM generation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations(msg.get("citations", []))
            render_metrics(msg.get("metrics", {}))

# Example prompts when empty
if not st.session_state.messages:
    st.info(
        "Try: *“What were the company's total net sales?”* or "
        "*“Summarize the principal risk factors.”*  "
        "Set the ticker in the sidebar first (and ingest it if needed)."
    )

if prompt := st.chat_input("Ask about a 10-K filing…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                data = run_query(prompt, ticker, top_k)
                answer = data.get("answer", "_No answer returned._")
                st.markdown(answer)
                render_citations(data.get("citations", []))
                render_metrics(data.get("metrics", {}))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": data.get("citations", []),
                        "metrics": data.get("metrics", {}),
                    }
                )
            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text if e.response is not None else str(e)
                st.error(f"Query failed: {detail or e}")
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")
