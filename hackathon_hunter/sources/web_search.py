from __future__ import annotations

from hackathon_hunter.sources.base import SourceAdapter, SourceResult


class WebSearchSource(SourceAdapter):
    name = "web_search"

    def collect(self) -> SourceResult:
        return SourceResult(
            source=self.name,
            errors=[
                "Live web search collection is not implemented in this offline MVP. "
                "Use --mock or import evidence-backed JSON into data/processed."
            ],
        )
