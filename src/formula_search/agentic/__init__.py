"""LLM-selected depth-first and beam proof search."""

from .beam import BeamSession
from .depth_first import DepthFirstSession
from .fireworks import Fireworks

# Existing callers retain depth-first semantics.
Session = DepthFirstSession

__all__ = ["BeamSession", "DepthFirstSession", "Fireworks", "Session"]
