"""Structured JSON logging with loguru.

In production we emit JSON logs that get shipped to Loki/CloudWatch/Datadog.
In dev we emit pretty-printed colorized logs to stderr.
"""
from __future__ import annotations

import sys

from loguru import logger

from insightrag.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logger.remove()

    if settings.environment == "dev":
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            colorize=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            serialize=True,  # JSON output
        )
