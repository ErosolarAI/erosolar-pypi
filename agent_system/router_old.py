"""Three-tier routing system for MCP tools."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tool_registry import MCPTool, ToolRegistry, WorkflowDefinition


@dataclass
class RouteDecision:
    tier: str
    tool_sequence: List[str]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentRouter:
    """Fast regex-based intent matcher with enhanced question detection."""

    def __init__(self, registry: ToolRegistry, min_confidence: float = 0.35):
        self.registry = registry
        self.min_confidence = min_confidence
        self._search_category_tags = {"search", "web", "news"}
        # Compile regex patterns for better performance
        self._question_pattern = re.compile(r"\b(what|who|where|when|why|how|which|whose|whom)\b", re.IGNORECASE)
        self._info_seeking_pattern = re.compile(
            r"\b(tell|explain|describe|show|find|search|look|check|verify|investigate|"
            r"know|learn|understand|discover|research|explore|analyze|examine|study|"
            r"information|details?|facts?|data|evidence|proof|source|reference)\b",
            re.IGNORECASE
        )

    def route(self, query: str) -> Optional[RouteDecision]:
        query_lower = query.lower()

        is_programming_query = self._looks_like_programming_query(query)

        # Priority 1: Check tool-specific pattern matching FIRST
        # This ensures specialized tools (like weather) take precedence over general search
        candidates: List[tuple[float, MCPTool, List[str]]] = []
        for tool in self.registry.list_tools():
            patterns = tool.metadata.get("intent_patterns") or []
            weight = 0
            matched_patterns = []
            categories = tool.metadata.get("categories", [])

            # Skip broad search/web tools when the user clearly wants code help
            if is_programming_query and categories:
                if "search" in categories or "web" in categories:
                    continue

            for pattern in patterns:
                if re.search(pattern, query_lower):
                    weight += 1
                    matched_patterns.append(pattern)
            if weight:
                # Give small bonus weight to specialized tools (non-search)
                # But only if they have strong matches already
                if "weather" in categories and weight >= 2:
                    weight += 1  # Bonus only for strong weather matches
                elif "search" not in categories and weight >= 2:
                    weight += 0.5  # Small bonus for other specialized tools with strong matches
                candidates.append((float(weight), tool, matched_patterns))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            score, best_tool, matched_patterns = candidates[0]

            # Check if specialized tool has strong enough match
            categories = best_tool.metadata.get("categories", [])
            is_specialized = "search" not in categories and "web" not in categories

            # Higher confidence for specialized tools or many pattern matches
            if is_specialized and score >= 1:
                # Specialized tool with at least one pattern match
                confidence = min(0.95, 0.60 + 0.10 * score)
                reasoning = f"Specialized tool {best_tool.name} matched {int(score)} patterns"
                return RouteDecision(
                    tier="intent",
                    tool_sequence=[best_tool.name],
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={"score": score, "matched_patterns": len(matched_patterns), "specialized": True},
                )
            elif score >= 2:
                # Any tool with strong pattern matches
                confidence = min(0.95, 0.50 + 0.08 * score)
                reasoning = f"Matched {int(score)} patterns for {best_tool.name}"
                return RouteDecision(
                    tier="intent",
                    tool_sequence=[best_tool.name],
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={"score": score, "matched_patterns": len(matched_patterns)},
                )

        # Priority 2: Check if this is an information-seeking query
        if self._is_information_seeking_query(query):
            search_tool = self._pick_search_tool()
            if search_tool:
                # Check pattern matches for confidence scoring
                patterns = search_tool.metadata.get("intent_patterns") or []
                pattern_matches = sum(1 for pattern in patterns if re.search(pattern, query_lower))

                # Base confidence from question type
                base_confidence = 0.65 if self._question_pattern.search(query) else 0.55
                # Boost confidence based on pattern matches
                confidence = min(0.95, base_confidence + 0.05 * pattern_matches)

                reasoning = "Information-seeking query detected; routing to search"
                if pattern_matches > 0:
                    reasoning += f" (matched {pattern_matches} patterns)"

                return RouteDecision(
                    tier="intent",
                    tool_sequence=[search_tool.name],
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={"type": "information_seeking", "pattern_matches": pattern_matches},
                )

        # Priority 3: Fallback to current event detection (legacy)
        if self._looks_like_current_event(query_lower):
            search_tool = self._pick_search_tool()
            if search_tool:
                reasoning = "Event-style question detected; defaulting to search tool"
                return RouteDecision(
                    tier="intent",
                    tool_sequence=[search_tool.name],
                    confidence=0.52,
                    reasoning=reasoning,
                    metadata={"heuristic": "current_events"},
                )
        return None

    def requires_search(self, query: str) -> bool:
        """Check if a query requires search tool usage."""
        return self._is_information_seeking_query(query)

    def _pick_search_tool(self) -> Optional[MCPTool]:
        fallback = None
        for tool in self.registry.list_tools():
            categories = set(tool.metadata.get("categories") or [])
            if not categories:
                continue
            if categories & self._search_category_tags:
                if "search" in categories:
                    return tool
                if not fallback:
                    fallback = tool
        return fallback

    def _looks_like_current_event(self, query_lower: str) -> bool:
        event_terms = (
            "what happened",
            "happened",
            "happening",
            "latest",
            "update",
            "breaking",
            "current events",
            "news about",
            "recent",
        )
        if any(term in query_lower for term in event_terms):
            return True

        if "?" in query_lower or re.search(r"\b(what|who|where|when|why|how)\b", query_lower):
            year_matches = re.findall(r"\b20\d{2}\b", query_lower)
            if year_matches:
                current_year = time.gmtime().tm_year
                threshold = max(2015, current_year - 6)
                years = [int(year) for year in year_matches]
                if any(year >= threshold for year in years):
                    return True

        recency_terms = ("today", "yesterday", "last week", "this week", "earlier this")
        return any(term in query_lower for term in recency_terms)

    def _is_information_seeking_query(self, query: str) -> bool:
        """
        Enhanced detection for information-seeking queries that should trigger search.

        Returns True if the query:
        1. Contains question words (what, who, where, when, why, how)
        2. Contains information-seeking verbs (tell, explain, find, etc.)
        3. Appears to be asking for facts, events, or information
        4. References specific entities, dates, or events
        """
        query_lower = query.lower()

        # Programming/code-related queries should stay within the model instead of forcing search
        if self._looks_like_programming_query(query):
            return False

        # Check for question patterns
        if self._question_pattern.search(query):
            return True

        # Check for information-seeking patterns
        if self._info_seeking_pattern.search(query):
            return True

        # Check for queries ending with question mark
        if query.rstrip().endswith("?"):
            return True

        # Check for event-related queries
        event_indicators = [
            "happened", "happening", "occur", "took place", "takes place",
            "event", "incident", "situation", "development", "news"
        ]
        if any(indicator in query_lower for indicator in event_indicators):
            return True

        # Check for fact-seeking patterns
        fact_patterns = [
            r"\b(is|are|was|were)\s+\w+\s+(a|an|the)",  # "Is X a Y?" pattern
            r"\b(did|does|do)\s+\w+",  # "Did X do Y?" pattern
            r"\b(can|could|will|would|should)\s+\w+",  # Modal verb questions
            r"\blist\s+(of|all|the)",  # List requests
            r"\b(definition|meaning|significance)\s+of\b",  # Definition requests
            r"\b(about|regarding|concerning)\b",  # Topic indicators
            r"\bresults?\b",  # Results queries (e.g., "World Cup results")
            r"\b(winner|champion|victory|defeat)\b",  # Competition outcomes
            r"\b(score|standings|rankings?)\b",  # Sports/competition queries
        ]
        for pattern in fact_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for specific entity mentions with temporal indicators
        # (e.g., "Trump's 2025 trip to Asia")
        entity_temporal_pattern = r"\b\w+'s\s+\d{4}\b|\b(in|during|at|on)\s+\d{4}\b"
        if re.search(entity_temporal_pattern, query_lower):
            return True

        # Check for comparison queries
        comparison_terms = ["versus", "vs", "compared to", "difference between", "better than", "worse than"]
        if any(term in query_lower for term in comparison_terms):
            return True

        # If none of the above, check if it looks like a current event
        return self._looks_like_current_event(query_lower)

    def _looks_like_programming_query(self, query: str) -> bool:
        """Return True if the query clearly asks for code help rather than factual search."""
        query_lower = query.lower()

        # Quick exits for inline code or stack traces
        if "```" in query or "traceback (most recent call last)" in query_lower:
            return True

        language_terms = [
            "python", "java", "javascript", "js", "typescript", "ts", "c++", "c#", "go",
            "golang", "rust", "ruby", "kotlin", "swift", "php", "scala", "perl", "bash",
            "shell", "powershell", "sql", "html", "css", "react", "vue", "angular", "node",
        ]
        strict_programming_keywords = [
            "code", "function", "method", "class", "script", "module", "package", "library",
            "api", "sdk", "framework", "algorithm", "loop", "variable", "regex", "snippet",
            "stack trace", "stacktrace", "traceback", "exception", "error", "bug",
            "unit test", "pytest", "unittest", "compile",
        ]
        action_programming_keywords = [
            "implement", "write", "refactor", "build", "deploy", "import", "export",
            "run", "execute", "debug", "optimize", "test",
        ]
        install_phrases = [
            "pip install", "npm install", "yarn add", "poetry add",
            "conda install", "brew install",
        ]

        if any(term in query_lower for term in install_phrases):
            return True

        if "in python" in query_lower or "using python" in query_lower:
            return True

        def contains_term(term: str) -> bool:
            # Use word boundaries for alphanumeric terms to avoid partial matches (e.g., "api" in "capital")
            if term.replace(" ", "").isalnum():
                return re.search(rf"\b{re.escape(term)}\b", query_lower) is not None
            return term in query_lower

        has_language = any(contains_term(term) for term in language_terms)

        if any(contains_term(keyword) for keyword in strict_programming_keywords):
            return True

        if has_language and any(contains_term(keyword) for keyword in action_programming_keywords):
            return True

        if re.search(r"\.(py|js|ts|java|cs|rb|go|rs|php|sh)\b", query_lower):
            return True

        if re.search(r"\bdef\s+\w+\b", query_lower) or re.search(r"\bclass\s+\w+\b", query_lower):
            return True

        return False


class GraphRouter:
    """Workflow aware router that understands tool chains."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, query: str) -> Optional[RouteDecision]:
        workflows = self.registry.get_workflows()
        if not workflows:
            return None
        query_lower = query.lower()
        scored: List[tuple[float, List[str], WorkflowDefinition]] = []
        for workflow in workflows:
            triggers = [t.lower() for t in workflow.triggers]
            score = sum(1 for trig in triggers if trig in query_lower)
            if score:
                sequence = [step.tool_name for step in workflow.steps]
                scored.append((float(score), sequence, workflow))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, sequence, workflow = scored[0]
        confidence = min(0.9, 0.5 + 0.1 * best_score)
        reasoning = f"Workflow '{workflow.name}' matched {best_score} triggers"
        return RouteDecision(
            tier="graph",
            tool_sequence=sequence,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"workflow": workflow.name},
        )


class SemanticRouter:
    """Embedding-based fallback router."""

    def __init__(self, registry: ToolRegistry, embedding_client: Any):
        self.registry = registry
        self.embedding_client = embedding_client

    def route(self, query: str) -> Optional[RouteDecision]:
        if not self.embedding_client:
            return None
        self.registry.ensure_embeddings()
        if not self.registry.list_tools():
            return None
        query_embedding = (
            self.embedding_client.embeddings.create(
                model="text-embedding-3-large",
                input=query,
            ).data[0].embedding
        )
        matches = self.registry.semantic_search(query_embedding, top_k=2)
        if not matches:
            return None
        confidence = 0.55
        reasoning = "Semantic similarity fallback"
        return RouteDecision(
            tier="semantic",
            tool_sequence=[tool.name for tool in matches],
            confidence=confidence,
            reasoning=reasoning,
        )


class EnhancedSemanticRouter:
    """
    Enhanced embedding-based router for massive tool collections.

    Features:
    - Configurable embedding model (text-embedding-3-small/large)
    - Multi-tool selection with confidence scores
    - Category filtering
    - Hybrid search combining patterns and semantics
    - Adaptive top-k selection based on query complexity
    """

    def __init__(
        self,
        registry: Any,  # EnhancedToolRegistry
        embedding_client: Any,
        model: str = "text-embedding-3-small",
        min_confidence: float = 0.3,
        default_top_k: int = 5,
    ):
        self.registry = registry
        self.embedding_client = embedding_client
        self.model = model
        self.min_confidence = min_confidence
        self.default_top_k = default_top_k

        # Configure registry with embedding client
        if hasattr(registry, 'set_embedding_client'):
            registry.set_embedding_client(embedding_client, model)

    def route(self, query: str, top_k: Optional[int] = None) -> Optional[RouteDecision]:
        """
        Route using semantic search with adaptive tool selection.

        Args:
            query: User query
            top_k: Override default number of tools to consider

        Returns:
            RouteDecision with potentially multiple tools
        """
        if not self.embedding_client:
            return None

        # Determine how many tools to retrieve based on query complexity
        if top_k is None:
            top_k = self._estimate_tools_needed(query)

        try:
            # Use hybrid search if registry supports it
            if hasattr(self.registry, 'hybrid_search'):
                results = self.registry.hybrid_search(
                    query=query,
                    pattern_weight=0.3,
                    semantic_weight=0.7,
                    top_k=top_k
                )
            elif hasattr(self.registry, 'semantic_search'):
                results = self.registry.semantic_search(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=self.min_confidence
                )
            else:
                # Fallback to old method
                self.registry.ensure_embeddings()
                query_embedding = (
                    self.embedding_client.embeddings.create(
                        model=self.model,
                        input=query,
                    ).data[0].embedding
                )
                tools = self.registry.semantic_search(query_embedding, top_k=top_k)
                results = [(tool, 0.5) for tool in tools]

            if not results:
                return None

            # Filter by minimum confidence
            filtered_results = [
                (tool, score) for tool, score in results
                if score >= self.min_confidence
            ]

            if not filtered_results:
                return None

            # Build tool sequence and calculate overall confidence
            tool_sequence = []
            scores = []
            for tool, score in filtered_results:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                tool_sequence.append(tool_name)
                scores.append(score)

            # Calculate aggregate confidence
            if scores:
                # Use weighted average with decay for lower-ranked tools
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                aggregate_confidence = weighted_sum / total_weight
            else:
                aggregate_confidence = 0.5

            # Build reasoning
            reasoning = self._build_reasoning(filtered_results, query)

            return RouteDecision(
                tier="semantic",
                tool_sequence=tool_sequence,
                confidence=min(0.95, aggregate_confidence),
                reasoning=reasoning,
                metadata={
                    "model": self.model,
                    "top_k": top_k,
                    "num_tools": len(tool_sequence),
                    "scores": scores[:5],  # Include top 5 scores for debugging
                }
            )

        except Exception as e:
            # Log error and return None
            import logging
            logging.error(f"EnhancedSemanticRouter error: {e}")
            return None

    def route_with_categories(
        self,
        query: str,
        categories: List[str],
        top_k: Optional[int] = None
    ) -> Optional[RouteDecision]:
        """
        Route with category filtering.

        Args:
            query: User query
            categories: List of categories to filter by
            top_k: Number of tools to retrieve

        Returns:
            RouteDecision with tools from specified categories
        """
        if not self.embedding_client:
            return None

        if top_k is None:
            top_k = self.default_top_k

        try:
            # Use category-filtered search if available
            if hasattr(self.registry, 'semantic_search'):
                results = self.registry.semantic_search(
                    query=query,
                    top_k=top_k,
                    filter_categories=categories,
                    similarity_threshold=self.min_confidence
                )
            else:
                # Fallback: get all tools and filter manually
                return self.route(query, top_k)

            if not results:
                return None

            # Build decision
            tool_sequence = [
                tool.name if hasattr(tool, 'name') else str(tool)
                for tool, score in results
            ]

            confidence = results[0][1] if results else 0.5

            return RouteDecision(
                tier="semantic",
                tool_sequence=tool_sequence,
                confidence=min(0.95, confidence),
                reasoning=f"Semantic search in categories: {', '.join(categories)}",
                metadata={
                    "model": self.model,
                    "categories": categories,
                    "num_tools": len(tool_sequence),
                }
            )

        except Exception as e:
            import logging
            logging.error(f"Category routing error: {e}")
            return None

    def _estimate_tools_needed(self, query: str) -> int:
        """
        Estimate how many tools might be needed based on query complexity.

        Args:
            query: User query

        Returns:
            Estimated number of tools to retrieve
        """
        query_lower = query.lower()

        # Complex queries need more tools
        complexity_indicators = [
            ("and", 2),
            ("then", 2),
            ("also", 2),
            ("multiple", 3),
            ("several", 3),
            ("various", 3),
            ("all", 5),
            ("everything", 5),
            ("comprehensive", 5),
        ]

        base_k = self.default_top_k
        for indicator, boost in complexity_indicators:
            if indicator in query_lower:
                base_k = max(base_k, boost)

        # Long queries might need more tools
        word_count = len(query.split())
        if word_count > 20:
            base_k = max(base_k, 7)
        elif word_count > 10:
            base_k = max(base_k, 5)

        return min(base_k, 10)  # Cap at 10 tools

    def _build_reasoning(
        self,
        results: List[Tuple[Any, float]],
        query: str
    ) -> str:
        """Build human-readable reasoning for the decision."""
        if not results:
            return "No matching tools found"

        num_tools = len(results)
        top_score = results[0][1]

        if num_tools == 1:
            tool_name = results[0][0].name if hasattr(results[0][0], 'name') else str(results[0][0])
            return f"Semantic match: {tool_name} (score: {top_score:.2f})"
        else:
            return f"Found {num_tools} relevant tools via semantic search (top score: {top_score:.2f})"

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = {
            "model": self.model,
            "min_confidence": self.min_confidence,
            "default_top_k": self.default_top_k,
        }

        if hasattr(self.registry, 'get_stats'):
            stats["registry_stats"] = self.registry.get_stats()

        return stats


class ThreeTierRouter:
    """Coordinator that promotes fast routing while remaining extensible."""

    def __init__(
        self,
        intent_router: IntentRouter,
        graph_router: GraphRouter,
        semantic_router: SemanticRouter,
    ) -> None:
        self.intent_router = intent_router
        self.graph_router = graph_router
        self.semantic_router = semantic_router
        self.stats = {"intent": 0, "graph": 0, "semantic": 0, "fallback": 0}

    def route(self, query: str) -> RouteDecision:
        start = time.perf_counter()

        # Try Intent Router first (fastest, pattern-based)
        decision = self.intent_router.route(query)
        if decision:
            tier = decision.tier
        else:
            # Try Graph Router next (workflow-based)
            decision = self.graph_router.route(query)
            if decision:
                tier = decision.tier
            else:
                # Try Semantic Router as last resort (embedding-based)
                decision = self.semantic_router.route(query)
                if decision:
                    tier = "semantic"
                else:
                    tier = "fallback"

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Create fallback decision if nothing matched
        if not decision:
            # Check if it's an information-seeking query that should have been caught
            is_info_query = self.intent_router._is_information_seeking_query(query)
            if is_info_query:
                reasoning = "No tool matched (WARNING: information-seeking query not routed to search)"
            else:
                reasoning = "No tool matched; responding directly."

            decision = RouteDecision(
                tier="fallback",
                tool_sequence=[],
                confidence=0.0,
                reasoning=reasoning,
                metadata={"warning": is_info_query}
            )

        # Add routing metadata
        decision.metadata.setdefault("latency_ms", elapsed_ms)
        decision.metadata.setdefault("query_length", len(query))
        decision.metadata.setdefault("query_has_question_mark", "?" in query)

        # Update statistics
        self.stats[tier] = self.stats.get(tier, 0) + 1
        return decision

    def snapshot(self) -> Dict[str, Any]:
        total = sum(self.stats.values()) or 1
        distribution = {
            k: round(v / total, 3)
            for k, v in self.stats.items()
        }
        return distribution
