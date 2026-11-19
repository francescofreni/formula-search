"""Fireworks Chat Completions adapter; existing imports remain supported."""

from .chat_completions import ChatCompletionsModel
from .model import IncompleteResponse as IncompleteResponse


class Fireworks(ChatCompletionsModel):
    """Key: explicit argument, FIREWORKS_API_KEY, or its local key file."""

    provider = "Fireworks"
    endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
    model_env = "FIREWORKS_MODEL"
    key_env = "FIREWORKS_API_KEY"
    key_directory = "fireworksai"
