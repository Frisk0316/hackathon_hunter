from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hackathon_hunter.models import Hackathon


@dataclass
class SourceResult:
    source: str
    hackathons: list[Hackathon] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SourceAdapter(ABC):
    name: str

    @abstractmethod
    def collect(self) -> SourceResult:
        """Collect normalized hackathon records."""
