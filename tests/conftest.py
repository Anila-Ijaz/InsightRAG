"""Pytest configuration and shared fixtures."""
import os

import pytest

# Force test settings before any imports trigger Settings()
os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("OPENAI_API_KEY", "test-key-do-not-use")
os.environ.setdefault("ENABLE_PII_REDACTION", "false")


@pytest.fixture
def sample_chunk_text() -> str:
    return (
        "Net sales increased 8% year-over-year to $383.3 billion in fiscal 2023, "
        "driven by strong growth in Services and the iPhone product category. "
        "Gross margin improved to 44.1% from 43.3% in the prior year."
    )
