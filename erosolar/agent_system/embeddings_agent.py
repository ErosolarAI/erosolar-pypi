"""Embeddings-driven ReAct agent with model-controlled tool selection and iterative discovery."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .embeddings_router import (
    EmbeddingsRouter,
    ModelDrivenToolSelector,
    ToolDiscoveryResult,
)
from .enhanced_tool_registry import EnhancedToolRegistry

try:
    from langgraph.graph import START, END, StateGraph
except ImportError:  # pragma: no cover
    START = END = None
    StateGraph = None

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")


class EmbeddingsReActAgent:
    """
    ReAct agent powered by embeddings-based tool discovery.

    Key differences from traditional intent-based routing:
    1. Uses embeddings to discover relevant tools (no pattern matching)
    2. LLM decides which tools to actually use
    3. Supports iterative tool discovery if initial results insufficient
    4. More flexible and scalable to large tool collections
    """

    def __init__(
        self,
        registry: EnhancedToolRegistry,
        embedding_client: Any,
        llm_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        system_prompt: Optional[str] = None,
        initial_tool_count: int = 5,
        max_tool_count: int = 20,
        max_discovery_iterations: int = 2,
    ):
        """
        Initialize embeddings-driven agent.

        Args:
            registry: Enhanced tool registry with embedding support
            embedding_client: OpenAI client for embeddings
            llm_client: OpenAI client for LLM calls
            embedding_model: Model for embeddings
            system_prompt: Optional system prompt override
            initial_tool_count: Initial number of tools to discover
            max_tool_count: Maximum tools to discover across iterations
            max_discovery_iterations: Max tool discovery expansion iterations
        """
        self.registry = registry
        self.llm_client = llm_client

        # Initialize embeddings router
        self.router = EmbeddingsRouter(
            registry=registry,
            embedding_client=embedding_client,
            model=embedding_model,
            initial_top_k=initial_tool_count,
            max_top_k=max_tool_count,
        )

        # Initialize model-driven tool selector
        self.tool_selector = ModelDrivenToolSelector(
            llm_client=llm_client,
        )

        self.system_prompt = system_prompt or (
            "You are Erosolar, a ReAct-style assistant that reasons over tool outputs before answering. "
            "You have access to various tools that are discovered based on the user's query. "
            "Use tool outputs to provide accurate, well-reasoned responses."
        )

        self.max_discovery_iterations = max_discovery_iterations

        self._graph = self._build_graph() if StateGraph else None

        # Statistics
        self.stats = {
            "total_queries": 0,
            "tools_discovered": 0,
            "tools_selected": 0,
            "tools_executed": 0,
            "discovery_iterations": 0,
            "no_tools_queries": 0,
        }

    def _build_graph(self):
        """Build LangGraph state machine."""
        graph = StateGraph(dict)
        graph.add_node("discover", self._discover_node)
        graph.add_node("select", self._select_node)
        graph.add_node("execute", self._execute_node)

        graph.add_edge(START, "discover")
        graph.add_edge("discover", "select")
        graph.add_edge("select", "execute")
        graph.add_edge("execute", END)

        return graph.compile()

    def _discover_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Discover relevant tools using embeddings."""
        query = state.get("query")
        logger.info(f"Discovering tools for query: {query[:100]}...")

        discovery_result = self.router.discover_tools(query)

        self.stats["tools_discovered"] += len(discovery_result.tools)

        return {
            **state,
            "discovery_result": discovery_result,
            "iteration": 0,
        }

    def _select_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Let model select which tools to use."""
        query = state.get("query")
        discovery_result: ToolDiscoveryResult = state.get("discovery_result")
        iteration = state.get("iteration", 0)

        if not discovery_result.tools:
            logger.info("No tools discovered, proceeding without tools")
            self.stats["no_tools_queries"] += 1
            return {
                **state,
                "selected_tools": [],
                "needs_expansion": False,
            }

        logger.info(
            f"Selecting from {len(discovery_result.tools)} tools "
            f"(iteration {iteration})"
        )

        # Let model decide which tools to use
        decision = self.tool_selector.select_tools(
            query=query,
            available_tools=discovery_result.tools,
            max_selections=3,
        )

        self.stats["tools_selected"] += len(decision.selected_tools)

        # Check if we need to expand tool search
        needs_expansion = (
            decision.needs_more_tools and
            iteration < self.max_discovery_iterations and
            len(discovery_result.tools) < self.router.max_top_k
        )

        if needs_expansion:
            logger.info("Model requested more tools, expanding search...")
            self.stats["discovery_iterations"] += 1

            # Expand tool search
            expanded_result = self.router.expand_tool_search(
                previous_result=discovery_result,
                query=query,
                expansion_multiplier=1.5,
            )

            self.stats["tools_discovered"] += (
                len(expanded_result.tools) - len(discovery_result.tools)
            )

            # Re-select with expanded tools
            new_decision = self.tool_selector.select_tools(
                query=query,
                available_tools=expanded_result.tools,
                max_selections=3,
            )

            self.stats["tools_selected"] += len(new_decision.selected_tools)

            return {
                **state,
                "discovery_result": expanded_result,
                "selected_tools": new_decision.selected_tools,
                "selection_decision": new_decision,
                "needs_expansion": False,
                "iteration": iteration + 1,
            }

        return {
            **state,
            "selected_tools": decision.selected_tools,
            "selection_decision": decision,
            "needs_expansion": False,
        }

    def _execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node: Execute selected tools."""
        query = state.get("query")
        selected_tools = state.get("selected_tools", [])

        if not selected_tools:
            logger.info("No tools selected for execution")
            return {**state, "tool_outputs": []}

        logger.info(f"Executing {len(selected_tools)} tools: {selected_tools}")

        tool_outputs = []
        for tool_name in selected_tools:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                logger.warning(f"Tool not found: {tool_name}")
                continue

            try:
                # Prepare payload based on tool type
                payload = self._prepare_payload(tool_name, query, tool)

                # Execute tool
                result = self.registry.execute_tool(tool_name, payload)

                tool_outputs.append({
                    "tool": tool_name,
                    "payload": payload,
                    "result": result,
                    "success": True,
                })

                self.stats["tools_executed"] += 1

            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                tool_outputs.append({
                    "tool": tool_name,
                    "payload": {},
                    "error": str(e),
                    "success": False,
                })

        return {**state, "tool_outputs": tool_outputs}

    def _prepare_payload(
        self,
        tool_name: str,
        query: str,
        tool: Any
    ) -> Dict[str, Any]:
        """
        Prepare tool input payload based on tool schema and query.

        This is a simple heuristic-based approach. For more sophisticated
        payload generation, consider using an LLM to extract parameters.
        """
        # Get input schema
        schema = tool.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        payload = {}

        # Common parameter names and how to fill them
        if "query" in properties:
            payload["query"] = query
        elif "q" in properties:
            payload["q"] = query
        elif "text" in properties:
            payload["text"] = query
        elif "prompt" in properties:
            payload["prompt"] = query
        elif "question" in properties:
            payload["question"] = query

        # URL extraction for tools that need URLs
        if "url" in properties:
            url_match = URL_RE.search(query)
            if url_match:
                payload["url"] = url_match.group(0)

        # Location extraction for weather/location tools
        if "location" in properties:
            # Simple location extraction (can be enhanced)
            location = self._extract_location(query)
            if location:
                payload["location"] = location

        # Fill required fields that we haven't handled yet
        for field in required:
            if field not in payload:
                # Default value based on type
                field_schema = properties.get(field, {})
                field_type = field_schema.get("type", "string")

                if field_type == "string":
                    payload[field] = query  # Default to query
                elif field_type == "integer":
                    payload[field] = field_schema.get("default", 10)
                elif field_type == "number":
                    payload[field] = field_schema.get("default", 1.0)
                elif field_type == "boolean":
                    payload[field] = field_schema.get("default", False)
                elif field_type == "array":
                    payload[field] = []

        return payload

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query (simple heuristic)."""
        # Look for common location patterns
        patterns = [
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "in Paris"
            r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "at Boston"
            r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "for Seattle"
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    # Public API

    def plan(self, prompt: str) -> Dict[str, Any]:
        """
        Plan tool usage for a query (discovery + selection + execution).

        Args:
            prompt: User query

        Returns:
            State dict with discovery_result, selected_tools, tool_outputs
        """
        self.stats["total_queries"] += 1

        if self._graph:
            return self._graph.invoke({"query": prompt})

        # Manual execution without LangGraph
        state = {"query": prompt}
        state = self._discover_node(state)
        state = self._select_node(state)
        state = self._execute_node(state)
        return state

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Full execution: plan + LLM response generation.

        Args:
            prompt: User query

        Returns:
            Dict with tool_outputs and final_response
        """
        # Execute planning
        plan_state = self.plan(prompt)

        # Generate final response
        tool_outputs = plan_state.get("tool_outputs", [])
        selection_decision = plan_state.get("selection_decision")
        discovery_result = plan_state.get("discovery_result")

        final_response = self._render_final(
            prompt,
            tool_outputs,
            selection_decision,
            discovery_result,
        )

        return {
            "discovery_result": discovery_result,
            "selection_decision": selection_decision,
            "tool_outputs": tool_outputs,
            "final_response": final_response,
        }

    def stream_final(
        self,
        prompt: str,
        tool_outputs: List[Dict[str, Any]],
        selection_decision: Optional[Any] = None,
        discovery_result: Optional[ToolDiscoveryResult] = None,
    ):
        """
        Stream final LLM response.

        Args:
            prompt: User query
            tool_outputs: Executed tool results
            selection_decision: Model's tool selection decision
            discovery_result: Tool discovery result

        Yields:
            Streaming chunks from LLM
        """
        messages = self._build_messages(
            prompt,
            tool_outputs,
            selection_decision,
            discovery_result,
        )

        stream = self.llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk

    def _render_final(
        self,
        prompt: str,
        tool_outputs: List[Dict[str, Any]],
        selection_decision: Optional[Any] = None,
        discovery_result: Optional[ToolDiscoveryResult] = None,
    ) -> Dict[str, Any]:
        """Generate final LLM response."""
        messages = self._build_messages(
            prompt,
            tool_outputs,
            selection_decision,
            discovery_result,
        )

        completion = self.llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
        )

        message = completion.choices[0].message

        return {
            "content": message.content,
            "reasoning_content": getattr(message, "reasoning_content", None),
        }

    def _build_messages(
        self,
        prompt: str,
        tool_outputs: List[Dict[str, Any]],
        selection_decision: Optional[Any] = None,
        discovery_result: Optional[ToolDiscoveryResult] = None,
    ) -> List[Dict[str, str]]:
        """Build message history for LLM."""
        # System message
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        # Tool context (if tools were used)
        if tool_outputs:
            tool_log = self._format_tool_log(tool_outputs)
            messages.append({
                "role": "system",
                "name": "tool_logger",
                "content": f"Tool execution results:\n{tool_log}",
            })

        # Tool discovery context (optional, for transparency)
        if discovery_result and discovery_result.tools:
            discovery_summary = (
                f"Discovered {len(discovery_result.tools)} relevant tools "
                f"via embeddings search (tier {discovery_result.tier}). "
            )
            if selection_decision:
                discovery_summary += (
                    f"Selected {len(selection_decision.selected_tools)} "
                    f"for execution based on relevance."
                )
            messages.append({
                "role": "system",
                "name": "tool_discovery",
                "content": discovery_summary,
            })

        # User query
        messages.append({
            "role": "user",
            "content": prompt,
        })

        return messages

    def _format_tool_log(self, tool_outputs: List[Dict[str, Any]]) -> str:
        """Format tool outputs for LLM context."""
        if not tool_outputs:
            return "No tools executed."

        import json

        blocks = []
        for output in tool_outputs:
            tool_name = output.get("tool")
            payload = output.get("payload", {})
            result = output.get("result")
            error = output.get("error")
            success = output.get("success", True)

            block = {
                "tool": tool_name,
                "payload": payload,
                "success": success,
            }

            if success and result:
                block["result"] = result
            elif error:
                block["error"] = error

            blocks.append(json.dumps(block, ensure_ascii=False))

        return "\n\n".join(blocks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        router_stats = self.router.get_stats()
        selector_stats = self.tool_selector.get_stats()

        return {
            "agent_stats": self.stats,
            "router_stats": router_stats,
            "selector_stats": selector_stats,
        }
