"""
Jules Autonomous Coding Agent Module

This module provides high-level access to Jules' autonomous coding capabilities.
"""

from .coding_agent import (
    CodingTaskResult,
    JulesAgent,
    JulesAgentConfig,
    quick_feature,
    quick_fix,
)

__all__ = [
    "JulesAgent",
    "JulesAgentConfig",
    "CodingTaskResult",
    "quick_feature",
    "quick_fix",
]
