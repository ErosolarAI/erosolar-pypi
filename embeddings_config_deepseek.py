"""Configuration for embeddings-based agent system with DeepSeek Reasoner."""
from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

from agent_system import (
    EmbeddingsReActAgent,
    EnhancedToolRegistry,
)
from agent_system.tools.open_metro_weather import build_open_metro_weather_tool
from agent_system.tools.tavily import build_tavily_tools

logger = logging.getLogger(__name__)


def create_erosolar_agent(
    openai_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    deepseek_base_url: str = "https://api.deepseek.com",
    initial_tool_count: int = 5,
    max_tool_count: int = 20,
    max_discovery_iterations: int = 2,
    min_similarity: float = 0.15,
) -> EmbeddingsReActAgent:
    """
    Create Erosolar agent with DeepSeek Reasoner for inference.

    Uses:
    - OpenAI for embeddings (text-embedding-3-small)
    - DeepSeek for reasoning (deepseek-reasoner)
    - DeepSeek for tool selection (deepseek-chat)

    Args:
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        deepseek_api_key: DeepSeek API key (or use DEEPSEEK_API_KEY env var)
        embedding_model: Embedding model to use
        deepseek_base_url: DeepSeek API base URL
        initial_tool_count: Number of tools to discover initially
        max_tool_count: Maximum tools to discover across iterations
        max_discovery_iterations: Max tool discovery expansion iterations
        min_similarity: Minimum similarity threshold for tool discovery

    Returns:
        EmbeddingsReActAgent instance configured with DeepSeek
    """
    # Get API keys
    openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    deepseek_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")

    if not openai_key:
        raise ValueError(
            "OpenAI API key required for embeddings. "
            "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
        )

    if not deepseek_key:
        raise ValueError(
            "DeepSeek API key required for reasoning. "
            "Set DEEPSEEK_API_KEY environment variable or pass deepseek_api_key parameter."
        )

    # Create clients
    openai_client = OpenAI(api_key=openai_key)
    deepseek_client = OpenAI(
        api_key=deepseek_key,
        base_url=deepseek_base_url
    )

    logger.info("Creating Erosolar agent with DeepSeek Reasoner...")

    # Create enhanced tool registry
    logger.info("Creating enhanced tool registry...")
    registry = EnhancedToolRegistry(
        use_hierarchical=False,  # Use standard vector store
        embedding_model=embedding_model,
        embedding_batch_size=100,
    )

    # Register tools
    logger.info("Registering tools...")

    # Tavily tools (search and extract)
    for tool in build_tavily_tools():
        registry.register_tool(tool)
        logger.info(f"  Registered tool: {tool.name}")

    # Weather tool
    for tool in build_open_metro_weather_tool():
        registry.register_tool(tool)
        logger.info(f"  Registered tool: {tool.name}")

    # Configure embedding client (using OpenAI)
    logger.info(f"Configuring embeddings with model: {embedding_model}")
    registry.set_embedding_client(openai_client, embedding_model)

    # Generate embeddings for all tools
    logger.info("Generating embeddings for all tools...")
    registry.ensure_embeddings()
    logger.info(f"  Embeddings generated for {registry._stats['embedded_tools']} tools")

    # Create agent with DeepSeek for reasoning
    logger.info("Creating Erosolar agent with DeepSeek Reasoner...")
    agent = EmbeddingsReActAgent(
        registry=registry,
        embedding_client=openai_client,  # OpenAI for embeddings
        llm_client=deepseek_client,  # DeepSeek for reasoning
        embedding_model=embedding_model,
        initial_tool_count=initial_tool_count,
        max_tool_count=max_tool_count,
        max_discovery_iterations=max_discovery_iterations,
    )

    # Set minimum similarity threshold
    agent.router.min_similarity = min_similarity
    logger.info(f"  Set minimum similarity threshold: {min_similarity}")

    logger.info("✓ Erosolar agent created successfully!")
    logger.info(f"  - Initial tool discovery: top {initial_tool_count}")
    logger.info(f"  - Maximum tools: {max_tool_count}")
    logger.info(f"  - Max discovery iterations: {max_discovery_iterations}")
    logger.info(f"  - Reasoning model: deepseek-reasoner")
    logger.info(f"  - Tool selection model: deepseek-chat")
    logger.info(f"  - Embedding model: {embedding_model}")

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
        # Create Erosolar agent with DeepSeek
        agent = create_erosolar_agent()

        # Test query
        test_query = "What's the weather like in San Francisco?"
        print(f"\nTest query: {test_query}\n")

        # Run agent
        print("Running agent...")
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
        content = response.get("content", "No content")
        reasoning = response.get("reasoning_content")

        print(f"Content: {content[:300]}...")
        if reasoning:
            print(f"\nReasoning: {reasoning[:300]}...")

        print("\n=== Statistics ===")
        stats = get_agent_stats(agent)
        print(f"Agent stats: {stats}")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
