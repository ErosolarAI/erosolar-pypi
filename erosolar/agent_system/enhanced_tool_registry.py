"""Enhanced MCP tool registry with scalable embedding support for massive tool collections."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tool_registry import MCPTool, WorkflowDefinition, WorkflowStep, ToolCache, ToolExecutionError
from .vector_store import VectorStore, HierarchicalVectorStore

logger = logging.getLogger(__name__)


class EnhancedToolRegistry:
    """
    Enhanced MCP tool registry with scalable embedding support.

    Features:
    - Efficient vector storage for hundreds of thousands of tools
    - Batch embedding generation
    - Multiple embedding models support
    - Hierarchical search for massive scale
    - Lazy loading and caching
    """

    def __init__(
        self,
        cache: Optional[ToolCache] = None,
        use_hierarchical: bool = False,
        n_clusters: int = 100,
        cache_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",  # Default to smaller, cheaper model
        embedding_batch_size: int = 100,
    ):
        # Tool storage
        self._tools: Dict[str, MCPTool] = {}
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._lock = threading.Lock()

        # Caching
        self._cache = cache or ToolCache()

        # Embedding configuration
        self._embedding_client = None
        self._embedding_model = embedding_model
        self._embedding_batch_size = embedding_batch_size
        self._embedding_dimension = self._get_embedding_dimension(embedding_model)

        # Vector storage
        cache_path = os.path.join(cache_dir, "vectors") if cache_dir else None
        if use_hierarchical:
            self._vector_store = HierarchicalVectorStore(
                dimension=self._embedding_dimension,
                n_clusters=n_clusters,
                metric="cosine",
                cache_dir=cache_path,
            )
        else:
            self._vector_store = VectorStore(
                dimension=self._embedding_dimension,
                metric="cosine",
                cache_dir=cache_path,
            )

        # Stats and metadata
        self._stats = {
            "total_tools": 0,
            "embedded_tools": 0,
            "embedding_generation_time_ms": 0,
            "last_embedding_update": None,
            "model_api_calls": 0,
        }

    def _get_embedding_dimension(self, model: str) -> int:
        """Get the embedding dimension for a given model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, 1536)

    def set_embedding_client(self, client: Any, model: Optional[str] = None) -> None:
        """Set the OpenAI client for embedding generation."""
        self._embedding_client = client
        if model:
            self._embedding_model = model
            self._embedding_dimension = self._get_embedding_dimension(model)

            # Recreate vector store with new dimension
            cache_path = self._vector_store.cache_dir if hasattr(self._vector_store, 'cache_dir') else None
            if isinstance(self._vector_store, HierarchicalVectorStore):
                self._vector_store = HierarchicalVectorStore(
                    dimension=self._embedding_dimension,
                    n_clusters=self._vector_store.n_clusters,
                    metric="cosine",
                    cache_dir=cache_path,
                )
            else:
                self._vector_store = VectorStore(
                    dimension=self._embedding_dimension,
                    metric="cosine",
                    cache_dir=cache_path,
                )

    def register_tool(self, tool: MCPTool) -> None:
        """Register a single tool."""
        with self._lock:
            self._tools[tool.name] = tool
            self._stats["total_tools"] = len(self._tools)

            # Generate embedding immediately if client is available
            if self._embedding_client:
                self._generate_embedding_for_tool(tool)

    def register_tools_batch(self, tools: List[MCPTool]) -> None:
        """Register multiple tools efficiently."""
        with self._lock:
            for tool in tools:
                self._tools[tool.name] = tool

            self._stats["total_tools"] = len(self._tools)

            # Generate embeddings in batch if client is available
            if self._embedding_client:
                self._generate_embeddings_batch(tools)

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow."""
        with self._lock:
            self._workflows[workflow.name] = workflow

    def _generate_embedding_for_tool(self, tool: MCPTool) -> Optional[np.ndarray]:
        """Generate embedding for a single tool."""
        if not self._embedding_client:
            return None

        try:
            # Create rich text representation of the tool
            text = self._tool_to_text(tool)

            # Generate embedding
            response = self._embedding_client.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            embedding = np.array(response.data[0].embedding)

            # Add to vector store
            self._vector_store.add(
                id=tool.name,
                embedding=embedding,
                metadata={
                    "description": tool.description,
                    "categories": tool.metadata.get("categories", []),
                    "input_schema": json.dumps(tool.input_schema),
                }
            )

            self._stats["embedded_tools"] += 1
            self._stats["model_api_calls"] += 1

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for tool {tool.name}: {e}")
            return None

    def _generate_embeddings_batch(self, tools: List[MCPTool]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple tools in batches."""
        if not self._embedding_client:
            return [None] * len(tools)

        embeddings = []
        start_time = time.perf_counter()

        # Process in batches
        for i in range(0, len(tools), self._embedding_batch_size):
            batch = tools[i:i + self._embedding_batch_size]

            try:
                # Create text representations for batch
                texts = [self._tool_to_text(tool) for tool in batch]

                # Generate embeddings in batch (OpenAI API supports batch embedding)
                response = self._embedding_client.embeddings.create(
                    model=self._embedding_model,
                    input=texts,
                )

                # Process results
                batch_items = []
                for j, tool in enumerate(batch):
                    embedding = np.array(response.data[j].embedding)
                    embeddings.append(embedding)

                    batch_items.append((
                        tool.name,
                        embedding,
                        {
                            "description": tool.description,
                            "categories": tool.metadata.get("categories", []),
                            "input_schema": json.dumps(tool.input_schema),
                        }
                    ))

                # Add batch to vector store
                if isinstance(self._vector_store, HierarchicalVectorStore):
                    self._vector_store.add_batch(batch_items)
                else:
                    self._vector_store.add_batch(batch_items)

                self._stats["embedded_tools"] += len(batch)
                self._stats["model_api_calls"] += 1

                logger.info(f"Generated embeddings for batch {i//self._embedding_batch_size + 1}")

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                embeddings.extend([None] * len(batch))

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["embedding_generation_time_ms"] = elapsed_ms
        self._stats["last_embedding_update"] = time.time()

        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_ms:.2f}ms")

        return embeddings

    def _tool_to_text(self, tool: MCPTool) -> str:
        """Convert a tool to a rich text representation for embedding."""
        # Include all relevant information for semantic search
        components = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
        ]

        # Add categories if available
        categories = tool.metadata.get("categories", [])
        if categories:
            components.append(f"Categories: {', '.join(categories)}")

        # Add input schema details
        if tool.input_schema.get("properties"):
            props = tool.input_schema["properties"]
            prop_descriptions = []
            for prop_name, prop_schema in props.items():
                desc = prop_schema.get("description", "")
                if desc:
                    prop_descriptions.append(f"{prop_name}: {desc}")
            if prop_descriptions:
                components.append(f"Parameters: {'; '.join(prop_descriptions)}")

        # Add intent patterns as additional context
        patterns = tool.metadata.get("intent_patterns", [])
        if patterns:
            # Convert regex patterns to readable text
            readable_patterns = []
            for pattern in patterns[:10]:  # Limit to avoid too long text
                # Simple conversion - can be enhanced
                readable = pattern.replace(r"\b", "").replace("\\s+", " ")
                readable_patterns.append(readable)
            components.append(f"Triggers: {', '.join(readable_patterns)}")

        return " | ".join(components)

    def ensure_embeddings(self) -> None:
        """Ensure all tools have embeddings (lazy batch generation)."""
        if not self._embedding_client:
            logger.warning("No embedding client configured")
            return

        with self._lock:
            logger.info(f"ensure_embeddings: Starting. Total tools registered: {len(self._tools)}")
            logger.info(f"ensure_embeddings: Vector store has {len(self._vector_store.entries)} entries")

            # Find tools without embeddings
            tools_to_embed = []
            for tool_name, tool in self._tools.items():
                if not self._vector_store.get(tool_name):
                    tools_to_embed.append(tool)
                    logger.debug(f"Tool '{tool_name}' needs embedding")

            if tools_to_embed:
                logger.info(f"Generating embeddings for {len(tools_to_embed)} tools: {[t.name for t in tools_to_embed]}")
                self._generate_embeddings_batch(tools_to_embed)
                logger.info(f"After generation: Vector store has {len(self._vector_store.entries)} entries")
            else:
                logger.info("All tools already have embeddings")

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_categories: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[MCPTool, float]]:
        """
        Perform semantic search for tools.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_categories: Optional list of categories to filter by
            similarity_threshold: Optional minimum similarity score

        Returns:
            List of (tool, similarity_score) tuples
        """
        if not self._embedding_client:
            return []

        try:
            # Generate query embedding
            response = self._embedding_client.embeddings.create(
                model=self._embedding_model,
                input=query,
            )
            query_embedding = np.array(response.data[0].embedding)
            self._stats["model_api_calls"] += 1

            # Prepare filter
            filter_metadata = None
            if filter_categories:
                filter_metadata = {"categories": filter_categories}

            # Search
            if isinstance(self._vector_store, HierarchicalVectorStore):
                results = self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    n_probe=5,  # Number of clusters to search
                    filter_metadata=filter_metadata,
                    threshold=similarity_threshold,
                )
            else:
                results = self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    threshold=similarity_threshold,
                )

            # Convert results to tools
            tool_results = []
            for tool_name, score, metadata in results:
                tool = self._tools.get(tool_name)
                if tool:
                    tool_results.append((tool, score))

            logger.debug(f"Semantic search found {len(results)} raw results, {len(tool_results)} with valid tools")
            if tool_results:
                logger.debug(f"Top result: {tool_results[0][0].name} with score {tool_results[0][1]:.3f}")

            return tool_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        pattern_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 10,
    ) -> List[Tuple[MCPTool, float]]:
        """
        Hybrid search combining pattern matching and semantic search.

        Args:
            query: Search query
            pattern_weight: Weight for pattern matching score
            semantic_weight: Weight for semantic similarity score
            top_k: Number of results to return

        Returns:
            List of (tool, combined_score) tuples
        """
        query_lower = query.lower()
        combined_scores: Dict[str, float] = {}

        # Pattern matching scores
        for tool_name, tool in self._tools.items():
            patterns = tool.metadata.get("intent_patterns", [])
            pattern_score = 0
            for pattern in patterns:
                import re
                if re.search(pattern, query_lower):
                    pattern_score += 1

            if pattern_score > 0:
                # Normalize pattern score (assuming max 5 patterns match)
                normalized_pattern = min(pattern_score / 5, 1.0)
                combined_scores[tool_name] = pattern_weight * normalized_pattern

        # Semantic search scores
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        for tool, semantic_score in semantic_results:
            if tool.name in combined_scores:
                combined_scores[tool.name] += semantic_weight * semantic_score
            else:
                combined_scores[tool.name] = semantic_weight * semantic_score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Return tools with scores
        results = []
        for tool_name, score in sorted_results:
            tool = self._tools.get(tool_name)
            if tool:
                results.append((tool, score))

        return results

    # Standard registry methods
    def list_tools(self) -> List[MCPTool]:
        """List all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        with self._lock:
            return self._tools.get(name)

    def get_workflows(self) -> List[WorkflowDefinition]:
        """Get all workflows."""
        with self._lock:
            return list(self._workflows.values())

    def execute_tool(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with caching."""
        tool = self.get_tool(name)
        if not tool:
            raise ToolExecutionError(f"Unknown tool: {name}")

        cache_key = f"{name}:{json.dumps(payload, sort_keys=True)}"
        cached = self._cache.get(cache_key)
        if cached:
            return {"cached": True, **cached}

        result = tool.handler(payload)
        self._cache.set(cache_key, result)
        return {"cached": False, **result}

    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        return self._cache.hit_rate

    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        if hasattr(self._vector_store, 'save_to_cache'):
            self._vector_store.save_to_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        vector_stats = {}
        if hasattr(self._vector_store, 'get_stats'):
            vector_stats = self._vector_store.get_stats()

        return {
            **self._stats,
            "cache_hit_rate": self.cache_hit_rate(),
            "workflow_count": len(self._workflows),
            "embedding_model": self._embedding_model,
            "embedding_dimension": self._embedding_dimension,
            "vector_store": vector_stats,
        }

    def metadata_snapshot(self) -> Dict[str, Any]:
        """Get metadata snapshot (backward compatibility)."""
        return {
            "tool_count": self._stats["total_tools"],
            "embedded_tools": self._stats["embedded_tools"],
            "workflow_count": len(self._workflows),
            "cache_hit_rate": round(self.cache_hit_rate(), 3),
            "embedding_model": self._embedding_model,
        }