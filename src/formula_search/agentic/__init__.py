"""LLM-selected depth-first and beam proof search."""

from .beam import BeamSession
from .depth_first import DepthFirstSession
from .fireworks import Fireworks
from .model import IncompleteResponse, Model
from .openrouter import OpenRouter
from .providers import create_model
from .selection import DecisionModel

# Existing callers retain depth-first semantics.
Session = DepthFirstSession

__all__ = [
    "BeamSession",
    "DepthFirstSession",
    "DecisionModel",
    "Fireworks",
    "IncompleteResponse",
    "Model",
    "OpenRouter",
    "Session",
    "create_model",
]
