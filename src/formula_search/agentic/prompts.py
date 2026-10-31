"""Select frozen instructions and identify their exact UTF-8 content."""

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Literal

from .prompt_history import v1, v2, v3, v4

Strategy = Literal["depth-first", "beam"]
PROMPT_VERSION = "structured-v4"
PROMPT_VERSIONS = (
    "discovery-v1",
    "exploratory-v2",
    "exploratory-v3",
    "structured-v4",
)
_CATALOG = MappingProxyType(
    {
        "discovery-v1": (v1.DEPTH_FIRST, v1.BEAM),
        "exploratory-v2": (v2.DEPTH_FIRST, v2.BEAM),
        "exploratory-v3": (v3.DEPTH_FIRST, v3.BEAM),
        "structured-v4": (v4.DEPTH_FIRST, v4.BEAM),
    }
)


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    strategy: Strategy
    instructions: str

    @property
    def sha256(self) -> str:
        """Hash static instructions, not graph-dependent decision context."""
        return sha256(self.instructions.encode("utf-8")).hexdigest()

    @property
    def metadata(self) -> dict[str, str]:
        return {"prompt_version": self.version, "prompt_sha256": self.sha256}


def get_prompt(
    strategy: Strategy, version: str = PROMPT_VERSION
) -> PromptTemplate:
    if strategy not in ("depth-first", "beam"):
        raise ValueError(f"Unknown agentic strategy: {strategy}")
    if version not in _CATALOG:
        raise ValueError(f"Unknown prompt version: {version}")
    instructions = _CATALOG[version][0 if strategy == "depth-first" else 1]
    return PromptTemplate(version, strategy, instructions)
