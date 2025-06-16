"""
Nexus Forge - AI-Powered One-Shot App Builder

A revolutionary platform that coordinates multiple AI models to build complete applications
from natural language descriptions.
"""

__version__ = "1.0.0"
__author__ = "Nexus Forge Team"
__description__ = "AI-Powered One-Shot App Builder with Multi-Agent Orchestration"

from .agents import StarriOrchestrator
from .core.exceptions import NexusForgeError

__all__ = [
    "NexusForgeError",
    "StarriOrchestrator",
]
