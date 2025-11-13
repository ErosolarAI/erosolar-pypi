"""Embeddings-first routing system with model-driven tool selection and iterative discovery."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToolDiscoveryResult:
    """Result from tool discovery via embeddings search."""
    tools: List[Tuple[Any, float]]  # (tool, similarity_score)
    query_embedding: np.ndarray
    search_radius: float
    tier: int  # 1 = first search, 2 = expanded search, etc.
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelToolDecision:
    """Model's decision about which tools to use."""
    selected_tools: List[str]  # Tool names the model wants to use
    needs_more_tools: bool  # Whether model wants to see more tool options
    reasoning: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingsRouter:
    """
    Embeddings-first router that discovers tools through vector similarity.

    Features:
    - Pure embeddings-based tool discovery (no pattern matching)
    - Iterative tool expansion when initial results insufficient
    - Configurable search depth and expansion strategy
    - Similarity scoring for ranking
    """

    def __init__(
        self,
        registry: Any,  # EnhancedToolRegistry
        embedding_client: Any,
        model: str = "text-embedding-3-small",
        initial_top_k: int = 5,
        max_top_k: int = 20,
        min_similarity: float = 0.15,  # Lower threshold for better recall
        expansion_factor: float = 1.5,
    ):
        """
        Initialize embeddings router.

        Args:
            registry: Tool registry with embedding support
            embedding_client: OpenAI client for embedding generation
            model: Embedding model to use
            initial_top_k: Number of tools to retrieve initially
            max_top_k: Maximum tools to retrieve across all iterations
            min_similarity: Minimum similarity threshold
            expansion_factor: How much to expand search in each iteration
        """
        self.registry = registry
        self.embedding_client = embedding_client
        self.model = model
        self.initial_top_k = initial_top_k
        self.max_top_k = max_top_k
        self.min_similarity = min_similarity
        self.expansion_factor = expansion_factor

        # Configure registry with embedding client
        if hasattr(registry, 'set_embedding_client'):
            registry.set_embedding_client(embedding_client, model)

        # Statistics
        self.stats = {
            "total_queries": 0,
            "tier1_sufficient": 0,
            "tier2_expansions": 0,
            "tier3_plus_expansions": 0,
            "no_tools_found": 0,
            "avg_similarity_score": 0.0,
        }

    def discover_tools(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_categories: Optional[List[str]] = None,
    ) -> ToolDiscoveryResult:
        """
        Discover tools using embeddings-based search.

        Args:
            query: User query to search for relevant tools
            top_k: Override number of tools to retrieve
            filter_categories: Optional category filtering

        Returns:
            ToolDiscoveryResult with discovered tools and metadata
        """
        start_time = time.perf_counter()
        self.stats["total_queries"] += 1

        k = top_k if top_k is not None else self.initial_top_k

        try:
            # Generate query embedding
            response = self.embedding_client.embeddings.create(
                model=self.model,
                input=query,
            )
            query_embedding = np.array(response.data[0].embedding)

            # Perform semantic search
            logger.info(f"Calling semantic_search with top_k={k}, threshold={self.min_similarity}")
            results = self.registry.semantic_search(
                query=query,
                top_k=k,
                filter_categories=filter_categories,
                similarity_threshold=self.min_similarity,
            )

            logger.info(f"Semantic search returned {len(results)} results for query: {query[:50]}")
            if results:
                logger.info(f"Top result: {results[0][0].name if hasattr(results[0][0], 'name') else results[0][0]} (score: {results[0][1]:.3f})")
            else:
                logger.warning(f"No results from semantic_search! Query: {query[:50]}, threshold: {self.min_similarity}")
                logger.warning("Falling back to returning ALL tools")
                # Fallback: Return all available tools with a neutral score
                all_tools = self.registry.list_tools()
                results = [(tool, 0.5) for tool in all_tools]
                logger.info(f"Fallback returned {len(results)} tools: {[t.name for t in all_tools]}")

            if results and results[0][1] > 0:  # Only track real semantic matches
                avg_score = sum(score for _, score in results) / len(results)
                self.stats["avg_similarity_score"] = (
                    self.stats["avg_similarity_score"] * 0.9 + avg_score * 0.1
                )
                self.stats["tier1_sufficient"] += 1
            elif not results:
                self.stats["no_tools_found"] += 1

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            reasoning = self._build_reasoning(results, query, tier=1)

            return ToolDiscoveryResult(
                tools=results,
                query_embedding=query_embedding,
                search_radius=1.0,
                tier=1,
                reasoning=reasoning,
                metadata={
                    "latency_ms": elapsed_ms,
                    "top_k": k,
                    "query_length": len(query),
                    "filter_categories": filter_categories,
                }
            )

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return ToolDiscoveryResult(
                tools=[],
                query_embedding=np.array([]),
                search_radius=1.0,
                tier=1,
                reasoning=f"Discovery failed: {str(e)}",
                metadata={"error": str(e)}
            )

    def expand_tool_search(
        self,
        previous_result: ToolDiscoveryResult,
        query: str,
        expansion_multiplier: float = 1.5,
    ) -> ToolDiscoveryResult:
        """
        Expand tool search to find less similar but potentially relevant tools.

        Args:
            previous_result: Previous discovery result to expand from
            query: Original query
            expansion_multiplier: How much to expand the search

        Returns:
            ToolDiscoveryResult with expanded tool set
        """
        start_time = time.perf_counter()

        # Calculate new top_k (expand search breadth)
        current_k = len(previous_result.tools)
        new_k = min(
            int(current_k * expansion_multiplier),
            self.max_top_k
        )

        # Also lower similarity threshold to catch more tools
        new_threshold = max(
            self.min_similarity * 0.8,  # Lower by 20%
            0.1  # But never below 0.1
        )

        new_tier = previous_result.tier + 1

        try:
            # Perform expanded search
            results = self.registry.semantic_search(
                query=query,
                top_k=new_k,
                similarity_threshold=new_threshold,
            )

            # Filter out tools already in previous results
            previous_tool_names = {
                tool.name for tool, _ in previous_result.tools
            }
            new_results = [
                (tool, score) for tool, score in results
                if tool.name not in previous_tool_names
            ]

            # Combine with previous results
            all_results = previous_result.tools + new_results

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update stats
            if new_tier == 2:
                self.stats["tier2_expansions"] += 1
            else:
                self.stats["tier3_plus_expansions"] += 1

            reasoning = self._build_reasoning(
                all_results,
                query,
                tier=new_tier,
                expanded_count=len(new_results)
            )

            return ToolDiscoveryResult(
                tools=all_results,
                query_embedding=previous_result.query_embedding,
                search_radius=previous_result.search_radius * expansion_multiplier,
                tier=new_tier,
                reasoning=reasoning,
                metadata={
                    "latency_ms": elapsed_ms,
                    "new_tools": len(new_results),
                    "total_tools": len(all_results),
                    "threshold": new_threshold,
                    "expansion_multiplier": expansion_multiplier,
                }
            )

        except Exception as e:
            logger.error(f"Tool expansion failed: {e}")
            # Return previous result if expansion fails
            return previous_result

    def _build_reasoning(
        self,
        results: List[Tuple[Any, float]],
        query: str,
        tier: int,
        expanded_count: int = 0
    ) -> str:
        """Build human-readable reasoning string."""
        if not results:
            return f"No tools found via embeddings search (tier {tier})"

        top_score = results[0][1]

        if tier == 1:
            if len(results) == 1:
                return f"Found 1 tool via embeddings (similarity: {top_score:.3f})"
            return f"Found {len(results)} tools via embeddings (top similarity: {top_score:.3f})"
        else:
            return (
                f"Expanded search (tier {tier}): "
                f"found {expanded_count} additional tools, "
                f"{len(results)} total (top similarity: {top_score:.3f})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        total = self.stats["total_queries"]
        if total == 0:
            return {**self.stats, "tier1_percentage": 0.0, "expansion_rate": 0.0}

        return {
            **self.stats,
            "tier1_percentage": self.stats["tier1_sufficient"] / total,
            "expansion_rate": (
                self.stats["tier2_expansions"] +
                self.stats["tier3_plus_expansions"]
            ) / total,
            "model": self.model,
            "initial_top_k": self.initial_top_k,
            "max_top_k": self.max_top_k,
        }


class ModelDrivenToolSelector:
    """
    Lets the LLM decide which tools to use from discovered options.

    Instead of automatically executing all discovered tools, this presents
    them to the model and lets it decide which ones are actually needed.
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize model-driven tool selector.

        Args:
            llm_client: OpenAI-compatible client for LLM calls
            system_prompt: Optional system prompt override
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt or self._default_system_prompt()

        self.stats = {
            "total_selections": 0,
            "avg_tools_selected": 0.0,
            "avg_tools_presented": 0.0,
            "expansion_requests": 0,
        }

    def _default_system_prompt(self) -> str:
        return """You are a tool selection assistant. Your job is to analyze the user's query and available tools, then decide which tools (if any) are needed to answer the query.

For each decision, respond with a JSON object:
{
    "selected_tools": ["tool1", "tool2"],  // List of tool names to use (can be empty)
    "needs_more_tools": false,  // Set to true if you need to see more tool options
    "reasoning": "explanation of your decision",
    "confidence": 0.85  // Your confidence in this decision (0-1)
}

Guidelines:
- Only select tools that are truly necessary for answering the query
- If the query can be answered without tools, return an empty selected_tools list
- If none of the available tools seem relevant, request more tools
- Be selective - don't use tools just because they're available
- Consider the tool descriptions and capabilities carefully"""

    def select_tools(
        self,
        query: str,
        available_tools: List[Tuple[Any, float]],  # (tool, similarity_score)
        max_selections: int = 3,
    ) -> ModelToolDecision:
        """
        Let the model decide which tools to use.

        Args:
            query: User query
            available_tools: List of (tool, score) tuples from discovery
            max_selections: Maximum number of tools model can select

        Returns:
            ModelToolDecision with selected tools and reasoning
        """
        self.stats["total_selections"] += 1
        self.stats["avg_tools_presented"] = (
            self.stats["avg_tools_presented"] * 0.9 +
            len(available_tools) * 0.1
        )

        try:
            # Build tool presentation for the model
            tools_text = self._format_tools_for_model(available_tools)

            # Create prompt
            user_message = f"""Query: {query}

Available tools:
{tools_text}

Decide which tools (if any) should be used to answer this query. Maximum selections: {max_selections}"""

            # Call model
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",  # Fast model for tool selection
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent decisions
            )

            # Parse response
            import json
            decision_data = json.loads(response.choices[0].message.content)

            selected_tools = decision_data.get("selected_tools", [])
            needs_more = decision_data.get("needs_more_tools", False)
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            confidence = float(decision_data.get("confidence", 0.5))

            # Validate tool names
            available_tool_names = {tool.name for tool, _ in available_tools}
            valid_selections = [
                t for t in selected_tools
                if t in available_tool_names
            ]

            # Update stats
            self.stats["avg_tools_selected"] = (
                self.stats["avg_tools_selected"] * 0.9 +
                len(valid_selections) * 0.1
            )
            if needs_more:
                self.stats["expansion_requests"] += 1

            return ModelToolDecision(
                selected_tools=valid_selections,
                needs_more_tools=needs_more,
                reasoning=reasoning,
                confidence=confidence,
                metadata={
                    "tools_presented": len(available_tools),
                    "invalid_selections": len(selected_tools) - len(valid_selections),
                }
            )

        except Exception as e:
            logger.error(f"Model tool selection failed: {e}")
            # Fallback: select top tool
            if available_tools:
                top_tool = available_tools[0][0].name
                return ModelToolDecision(
                    selected_tools=[top_tool],
                    needs_more_tools=False,
                    reasoning=f"Selection failed, using top tool as fallback: {str(e)}",
                    confidence=0.3,
                    metadata={"error": str(e), "fallback": True}
                )
            else:
                return ModelToolDecision(
                    selected_tools=[],
                    needs_more_tools=False,
                    reasoning=f"Selection failed and no tools available: {str(e)}",
                    confidence=0.0,
                    metadata={"error": str(e)}
                )

    def _format_tools_for_model(
        self,
        tools: List[Tuple[Any, float]]
    ) -> str:
        """Format tools for presentation to the model."""
        lines = []
        for i, (tool, score) in enumerate(tools, 1):
            categories = tool.metadata.get("categories", [])
            cat_str = f" [{', '.join(categories)}]" if categories else ""

            lines.append(
                f"{i}. {tool.name}{cat_str} (similarity: {score:.3f})\n"
                f"   Description: {tool.description}"
            )

        return "\n\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            **self.stats,
            "selection_rate": (
                self.stats["avg_tools_selected"] /
                max(self.stats["avg_tools_presented"], 1)
            ),
        }
