from __future__ import annotations

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def log_step(name: str) -> None:
    console.rule(f"[bold cyan]{name}")
