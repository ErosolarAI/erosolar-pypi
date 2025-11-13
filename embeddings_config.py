"""Configuration for embeddings-based agent system with model-driven tool selection."""
from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

from agent_system import (
    EmbeddingsReActAgent,
    EmbeddingsRouter,
    EnhancedToolRegistry,
    ModelDrivenToolSelector,
)
from agent_system.tools.open_metro_weather import build_open_metro_weather_tool
from agent_system.tools.tavily import build_tavily_tools

logger = logging.getLogger(__name__)


def create_embeddings_based_agent(
    openai_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    initial_tool_count: int = 5,
    max_tool_count: int = 20,
    max_discovery_iterations: int = 2,
) -> EmbeddingsReActAgent:
    """
    Create an embeddings-based ReAct agent with model-driven tool selection.

    This replaces the old intent-based routing system with:
    1. Embeddings-based tool discovery (no pattern matching)
    2. Model-driven tool selection (LLM decides which tools to use)
    3. Iterative tool discovery (expand search if needed)

    Args:
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        embedding_model: Embedding model to use
        initial_tool_count: Number of tools to discover initially
        max_tool_count: Maximum tools to discover across iterations
        max_discovery_iterations: Max tool discovery expansion iterations

    Returns:
        EmbeddingsReActAgent instance
    """
    # Get API key
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass openai_api_key parameter."
        )

    # Create OpenAI clients
    embedding_client = OpenAI(api_key=api_key)
    llm_client = OpenAI(api_key=api_key)

    # Create enhanced tool registry
    logger.info("Creating enhanced tool registry...")
    registry = EnhancedToolRegistry(
        use_hierarchical=False,  # Use standard vector store for now
        embedding_model=embedding_model,
        embedding_batch_size=100,
    )

    # Register tools
    logger.info("Registering tools...")

    # Tavily tools (search and extract)
    for tool in build_tavily_tools():
        registry.register_tool(tool)
        logger.info(f"Registered tool: {tool.name}")

    # Weather tool
    for tool in build_open_metro_weather_tool():
        registry.register_tool(tool)
        logger.info(f"Registered tool: {tool.name}")

    # Configure embedding client
    logger.info(f"Configuring embeddings with model: {embedding_model}")
    registry.set_embedding_client(embedding_client, embedding_model)

    # Generate embeddings for all tools
    logger.info("Generating embeddings for all tools...")
    registry.ensure_embeddings()
    logger.info(f"Embeddings generated for {registry._stats['embedded_tools']} tools")

    # Create agent with lower similarity threshold
    logger.info("Creating embeddings-based ReAct agent...")
    agent = EmbeddingsReActAgent(
        registry=registry,
        embedding_client=embedding_client,
        llm_client=llm_client,
        embedding_model=embedding_model,
        initial_tool_count=initial_tool_count,
        max_tool_count=max_tool_count,
        max_discovery_iterations=max_discovery_iterations,
    )

    # Lower the minimum similarity threshold for better recall
    if hasattr(agent, 'router'):
        agent.router.min_similarity = 0.15
        logger.info("Set router minimum similarity threshold to 0.15")

    logger.info("Agent created successfully!")
    logger.info(f"  - Initial tool discovery: top {initial_tool_count}")
    logger.info(f"  - Maximum tools: {max_tool_count}")
    logger.info(f"  - Max discovery iterations: {max_discovery_iterations}")

    return agent


def get_agent_stats(agent: EmbeddingsReActAgent) -> dict:
    """Get comprehensive statistics from the agent."""
    return agent.get_stats()


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Create agent
        agent = create_embeddings_based_agent()

        # Test query
        test_query = "What's the weather like in San Francisco?"
        print(f"\nTest query: {test_query}\n")

        # Run agent
        result = agent.run(test_query)

        # Print results
        print("\n=== Discovery Result ===")
        discovery = result.get("discovery_result")
        if discovery:
            print(f"Tier: {discovery.tier}")
            print(f"Tools discovered: {len(discovery.tools)}")
            for tool, score in discovery.tools:
                print(f"  - {tool.name}: {score:.3f}")

        print("\n=== Selection Decision ===")
        decision = result.get("selection_decision")
        if decision:
            print(f"Selected tools: {decision.selected_tools}")
            print(f"Needs more tools: {decision.needs_more_tools}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"Confidence: {decision.confidence:.3f}")

        print("\n=== Tool Outputs ===")
        outputs = result.get("tool_outputs", [])
        print(f"Executed {len(outputs)} tools:")
        for output in outputs:
            print(f"  - {output.get('tool')}: {'Success' if output.get('success') else 'Failed'}")

        print("\n=== Final Response ===")
        response = result.get("final_response", {})
        print(response.get("content", "No content"))

        print("\n=== Statistics ===")
        stats = get_agent_stats(agent)
        print(f"Agent stats: {stats}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
