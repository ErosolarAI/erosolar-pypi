"""
DEPRECATED: Intent-based routing has been replaced with embeddings-based routing.

This module re-exports deprecated classes for backward compatibility only.
All new code should use the embeddings-based system.

Migration guide:
--------------
OLD (Intent-based):
    from agent_system.router import IntentRouter, ThreeTierRouter
    router = ThreeTierRouter(intent_router, graph_router, semantic_router)
    decision = router.route(query)

NEW (Embeddings-based):
    from agent_system import EmbeddingsReActAgent
    from embeddings_config import create_embeddings_based_agent

    agent = create_embeddings_based_agent()
    result = agent.run(query)

Key differences:
- No pattern matching - uses embeddings for tool discovery
- Model decides which tools to use (not automatic)
- Supports iterative tool discovery if needed
- More scalable to large tool collections
"""

# Re-export deprecated classes with warnings
from .router_deprecated import (
    RouteDecision,
    IntentRouter,
    GraphRouter,
    SemanticRouter,
    EnhancedSemanticRouter,
    ThreeTierRouter,
)

__all__ = [
    "RouteDecision",
    "IntentRouter",
    "GraphRouter",
    "SemanticRouter",
    "EnhancedSemanticRouter",
    "ThreeTierRouter",
]
