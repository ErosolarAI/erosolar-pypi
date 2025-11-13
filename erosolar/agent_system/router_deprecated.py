"""
DEPRECATED: This module contains the old intent-based routing system.

The intent-based routing system (IntentRouter, GraphRouter, ThreeTierRouter) has been
replaced with an embeddings-based system that uses vector similarity search and
model-driven tool selection.

New system components:
- EmbeddingsRouter: Pure embeddings-based tool discovery
- ModelDrivenToolSelector: LLM decides which tools to use
- EmbeddingsReActAgent: Complete agent with iterative discovery

For migration, use:
    from agent_system import EmbeddingsReActAgent
    from embeddings_config import create_embeddings_based_agent

    agent = create_embeddings_based_agent()

This file is kept for backward compatibility with existing tests only.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """
    Route decision data class (kept for backward compatibility).

    In the new embeddings-based system, tool discovery and selection
    are separated into ToolDiscoveryResult and ModelToolDecision.
    """
    tier: str
    tool_sequence: List[str]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentRouter:
    """
    DEPRECATED: Pattern-based intent router.

    This class has been removed. Use EmbeddingsRouter instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "IntentRouter is deprecated. Use EmbeddingsRouter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "IntentRouter has been removed. "
            "Use EmbeddingsRouter from agent_system.embeddings_router instead."
        )


class GraphRouter:
    """
    DEPRECATED: Workflow-based router.

    This class has been removed. Workflow support can be added to the
    embeddings-based system if needed.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GraphRouter is deprecated and has been removed.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "GraphRouter has been removed. "
            "Use EmbeddingsRouter from agent_system.embeddings_router instead."
        )


class SemanticRouter:
    """
    DEPRECATED: Basic semantic router.

    This class has been removed. Use EmbeddingsRouter instead,
    which provides more advanced semantic search capabilities.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SemanticRouter is deprecated. Use EmbeddingsRouter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "SemanticRouter has been removed. "
            "Use EmbeddingsRouter from agent_system.embeddings_router instead."
        )


class EnhancedSemanticRouter:
    """
    DEPRECATED: Enhanced semantic router.

    This class has been removed. Use EmbeddingsRouter instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "EnhancedSemanticRouter is deprecated. Use EmbeddingsRouter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "EnhancedSemanticRouter has been removed. "
            "Use EmbeddingsRouter from agent_system.embeddings_router instead."
        )


class ThreeTierRouter:
    """
    DEPRECATED: Three-tier coordinator router.

    This class has been removed. The new embeddings-based system
    uses a simpler architecture with EmbeddingsRouter and
    ModelDrivenToolSelector.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ThreeTierRouter is deprecated and has been removed.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "ThreeTierRouter has been removed. "
            "Use EmbeddingsRouter and ModelDrivenToolSelector from "
            "agent_system.embeddings_router instead."
        )
