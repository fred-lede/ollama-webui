from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolPolicy:
    enabled_tools: set[str] = field(default_factory=lambda: {"datetime", "calculator", "fetch_url", "web_search"})
    trusted_domains: tuple[str, ...] = (
        "ollama.com",
        "github.com",
        "docs.ollama.com",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "wsj.com",
        "nytimes.com",
        "theverge.com",
        "techcrunch.com",
        "arstechnica.com",
        "wired.com",
        "zdnet.com",
    )
    tool_timeout_seconds: int = 20
    max_retries: int = 1

    def is_tool_allowed(self, name: str) -> bool:
        return name in self.enabled_tools
