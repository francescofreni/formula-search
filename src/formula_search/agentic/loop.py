"""Compatibility imports for the original depth-first session."""

from .common import _state_key as _state_key
from .depth_first import DepthFirstSession

Session = DepthFirstSession
