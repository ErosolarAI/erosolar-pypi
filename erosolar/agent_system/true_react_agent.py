"""True ReAct agent with iterative Reason-Act-Observe loops."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

from .enhanced_tool_registry import EnhancedToolRegistry
from .embeddings_router import EmbeddingsRouter, ModelDrivenToolSelector

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """Represents a single step in the ReAct loop."""
    step_number: int
    thought: str  # Agent's reasoning about what to do
    action: Optional[str] = None  # Tool name to execute (None if deciding to finish)
    action_input: Optional[Dict[str, Any]] = None  # Tool parameters
    observation: Optional[str] = None  # Result from tool execution
    should_continue: bool = True  # Whether to continue the loop
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActResult:
    """Complete result from ReAct agent execution."""
    steps: List[ReActStep]
    final_answer: str
    reasoning_content: Optional[str] = None
    total_steps: int = 0
    tools_used: List[str] = field(default_factory=list)


class TrueReActAgent:
    """
    True ReAct agent that iteratively:
    1. Reasons about what to do next
    2. Acts by executing tools
    3. Observes the results
    4. Repeats until deciding no more actions are needed
    5. Generates final response
    """

    def __init__(
        self,
        registry: EnhancedToolRegistry,
        embedding_client: Any,
        llm_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        reasoning_model: str = "deepseek-chat",
        final_model: str = "deepseek-reasoner",
        max_iterations: int = 5,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize True ReAct agent.

        Args:
            registry: Enhanced tool registry
            embedding_client: OpenAI client for embeddings
            llm_client: OpenAI client for LLM calls
            embedding_model: Model for embeddings
            reasoning_model: Fast model for reasoning steps
            final_model: Model for final response (can be reasoning model)
            max_iterations: Maximum ReAct loop iterations
            system_prompt: Optional system prompt override
        """
        self.registry = registry
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.reasoning_model = reasoning_model
        self.final_model = final_model

        # Initialize embeddings router for tool discovery
        self.router = EmbeddingsRouter(
            registry=registry,
            embedding_client=embedding_client,
            model=embedding_model,
            initial_top_k=10,
            max_top_k=20,
        )

        # Initialize tool selector
        self.tool_selector = ModelDrivenToolSelector(
            llm_client=llm_client,
        )

        self.system_prompt = system_prompt or self._default_system_prompt()

        # Statistics
        self.stats = {
            "total_queries": 0,
            "avg_iterations": 0.0,
            "total_tools_used": 0,
            "finished_early": 0,
            "max_iterations_reached": 0,
        }

    def _default_system_prompt(self) -> str:
        return """You are Erosolar, a ReAct-style assistant.

ReAct means you follow this loop:
1. **Reason**: Think about what you need to do next to answer the user's query
2. **Act**: Decide which tool(s) to use (or decide to finish)
3. **Observe**: Examine the results from your actions
4. **Repeat**: Go back to step 1 until you have enough information

You will be given a query and available tools discovered via semantic search.

For each reasoning step, respond with a JSON object:
{
    "thought": "your reasoning about what to do next",
    "action": "tool_name",  // or "FINISH" when ready to answer
    "action_input": {"param": "value"},  // tool parameters (omit if FINISH)
    "confidence": 0.85  // confidence in this decision (0-1)
}

Guidelines:
- Think step by step about what information you need
- Use tools when you need external information or computation
- When you have enough information, set action to "FINISH"
- Be efficient - don't use unnecessary tools
- Each iteration builds on previous observations"""

    def run(self, prompt: str, allowed_tools: Optional[Iterable[str]] = None) -> ReActResult:
        """
        Execute full ReAct loop (non-streaming).

        Args:
            prompt: User query

        Returns:
            ReActResult with all steps and final answer
        """
        self.stats["total_queries"] += 1

        (
            available_tools,
            conversation_history,
            requested_tool_filter,
            allowed_set,
            allowed_display,
            missing_allowed,
        ) = self._prepare_context(prompt, allowed_tools)

        self._handle_tool_filter_logging(
            requested_tool_filter,
            allowed_set,
            allowed_display,
            missing_allowed,
        )

        steps = []
        # Conversation history already initialized with system/tool/user context

        # ReAct loop
        for iteration in range(self.max_iterations):
            step_number = iteration + 1
            logger.info(f"ReAct iteration {step_number}/{self.max_iterations}")

            # REASON: Get agent's next decision
            try:
                decision = self._reason(conversation_history)

                step = ReActStep(
                    step_number=step_number,
                    thought=decision.get("thought", ""),
                    action=decision.get("action"),
                    action_input=decision.get("action_input"),
                    metadata={"confidence": decision.get("confidence", 0.5)}
                )

                # Check if agent wants to finish
                if step.action and step.action.upper() == "FINISH":
                    step.should_continue = False
                    steps.append(step)
                    self.stats["finished_early"] += 1
                    logger.info(f"Agent decided to finish after {step_number} iterations")
                    break

                # ACT: Execute the tool
                if step.action:
                    observation = self._act(step.action, step.action_input or {})
                    step.observation = observation
                    self.stats["total_tools_used"] += 1

                    # Add to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "thought": step.thought,
                            "action": step.action,
                            "action_input": step.action_input
                        })
                    })
                    conversation_history.append({
                        "role": "system",
                        "name": "observation",
                        "content": f"Tool result: {observation}"
                    })
                else:
                    # No action specified, finish
                    step.should_continue = False
                    steps.append(step)
                    break

                steps.append(step)

            except Exception as e:
                logger.error(f"Error in ReAct iteration {step_number}: {e}")
                step = ReActStep(
                    step_number=step_number,
                    thought=f"Error occurred: {str(e)}",
                    should_continue=False,
                    metadata={"error": str(e)}
                )
                steps.append(step)
                break

        # Update iteration stats
        if len(steps) >= self.max_iterations:
            self.stats["max_iterations_reached"] += 1

        self.stats["avg_iterations"] = (
            self.stats["avg_iterations"] * 0.9 + len(steps) * 0.1
        )

        # Generate final answer
        final_response = self._generate_final_answer(prompt, steps, conversation_history)

        tools_used = [step.action for step in steps if step.action and step.action.upper() != "FINISH"]

        return ReActResult(
            steps=steps,
            final_answer=final_response.get("content", ""),
            reasoning_content=final_response.get("reasoning_content"),
            total_steps=len(steps),
            tools_used=tools_used
        )

    def stream(self, prompt: str, allowed_tools: Optional[Iterable[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Execute ReAct loop with streaming updates.

        Yields:
            Dict with type and data for each event:
            - {"type": "step", "data": {...}}  # Each ReAct step
            - {"type": "final", "data": "text"}  # Final response chunks
            - {"type": "reasoning", "data": "text"}  # Reasoning chunks
            - {"type": "done"}  # Completion marker
        """
        self.stats["total_queries"] += 1

        (
            available_tools,
            conversation_history,
            requested_tool_filter,
            allowed_set,
            allowed_display,
            missing_allowed,
        ) = self._prepare_context(prompt, allowed_tools)

        allowed_label = self._handle_tool_filter_logging(
            requested_tool_filter,
            allowed_set,
            allowed_display,
            missing_allowed,
        )

        # Yield discovery event
        yield {
            "type": "discovery",
            "data": {
                "tools_found": len(available_tools),
                "tool_names": [tool.name for tool, _ in available_tools[:5]]
            }
        }

        if allowed_set:
            notice = f"Prioritizing user-selected tools: {allowed_label or 'selected tools'}"
            if missing_allowed:
                notice += f" (ignored unknown tools: {', '.join(missing_allowed)})"
            yield {
                "type": "reasoning",
                "data": notice + " — other tools remain available.\n\n",
            }
        elif requested_tool_filter:
            warn_msg = f"No valid tools matched selection: {', '.join(requested_tool_filter)}"
            yield {
                "type": "reasoning",
                "data": warn_msg + " — using full toolset.\n\n",
            }

        steps = []

        # ReAct loop
        for iteration in range(self.max_iterations):
            step_number = iteration + 1
            logger.info(f"ReAct iteration {step_number}/{self.max_iterations}")

            # REASON: Get agent's next decision
            try:
                decision = self._reason(conversation_history)

                step = ReActStep(
                    step_number=step_number,
                    thought=decision.get("thought", ""),
                    action=decision.get("action"),
                    action_input=decision.get("action_input"),
                    metadata={"confidence": decision.get("confidence", 0.5)}
                )

                # Yield reasoning event
                yield {
                    "type": "step",
                    "data": {
                        "step_number": step_number,
                        "thought": step.thought,
                        "action": step.action,
                        "action_input": step.action_input,
                    }
                }

                # Check if agent wants to finish
                if step.action and step.action.upper() == "FINISH":
                    step.should_continue = False
                    steps.append(step)
                    self.stats["finished_early"] += 1
                    logger.info(f"Agent decided to finish after {step_number} iterations")
                    break

                # ACT: Execute the tool
                if step.action:
                    observation = self._act(step.action, step.action_input or {})
                    step.observation = observation
                    self.stats["total_tools_used"] += 1

                    # Yield observation event
                    yield {
                        "type": "observation",
                        "data": {
                            "step_number": step_number,
                            "tool": step.action,
                            "result": observation[:500]  # Truncate long results
                        }
                    }

                    # Add to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "thought": step.thought,
                            "action": step.action,
                            "action_input": step.action_input
                        })
                    })
                    conversation_history.append({
                        "role": "system",
                        "name": "observation",
                        "content": f"Tool result: {observation}"
                    })
                else:
                    # No action specified, finish
                    step.should_continue = False
                    steps.append(step)
                    break

                steps.append(step)

            except Exception as e:
                logger.error(f"Error in ReAct iteration {step_number}: {e}")
                yield {
                    "type": "error",
                    "data": {
                        "step_number": step_number,
                        "error": str(e)
                    }
                }
                break

        # Update iteration stats
        if len(steps) >= self.max_iterations:
            self.stats["max_iterations_reached"] += 1

        self.stats["avg_iterations"] = (
            self.stats["avg_iterations"] * 0.9 + len(steps) * 0.1
        )

        # Stream final answer
        yield from self._stream_final_answer(prompt, steps, conversation_history)

        yield {"type": "done"}

    def _reason(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Reasoning step: Let agent decide what to do next.

        Args:
            conversation_history: Current conversation context

        Returns:
            Decision dict with thought, action, action_input, confidence
        """
        # Add instruction for this reasoning step
        reasoning_prompt = conversation_history + [
            {
                "role": "system",
                "content": "What should you do next? Respond with JSON containing: thought, action, action_input, confidence"
            }
        ]

        response = self.llm_client.chat.completions.create(
            model=self.reasoning_model,
            messages=reasoning_prompt,
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        decision = json.loads(response.choices[0].message.content)
        return decision

    def _act(self, action: str, action_input: Dict[str, Any]) -> str:
        """
        Action step: Execute a tool.

        Args:
            action: Tool name
            action_input: Tool parameters

        Returns:
            Observation string
        """
        tool = self.registry.get_tool(action)
        if not tool:
            return f"Error: Tool '{action}' not found"

        try:
            result = self.registry.execute_tool(action, action_input)

            # Format result as string
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Tool execution failed for {action}: {e}")
            return f"Error executing {action}: {str(e)}"

    def _generate_final_answer(
        self,
        prompt: str,
        steps: List[ReActStep],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate final answer after ReAct loop completes.

        Args:
            prompt: Original user query
            steps: All ReAct steps taken
            conversation_history: Full conversation context

        Returns:
            Dict with content and reasoning_content
        """
        # Build final context
        steps_summary = self._format_steps_summary(steps)

        final_messages = [
            {
                "role": "system",
                "content": "You are Erosolar. Based on the ReAct steps above, provide a comprehensive answer to the user's query."
            },
            {
                "role": "system",
                "name": "react_summary",
                "content": f"ReAct execution summary:\n{steps_summary}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = self.llm_client.chat.completions.create(
            model=self.final_model,
            messages=final_messages,
        )

        message = completion.choices[0].message

        return {
            "content": message.content,
            "reasoning_content": getattr(message, "reasoning_content", None),
        }

    def _stream_final_answer(
        self,
        prompt: str,
        steps: List[ReActStep],
        conversation_history: List[Dict[str, str]]
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream final answer after ReAct loop completes.

        Yields:
            Dicts with type "final" or "reasoning" and data
        """
        # Build final context
        steps_summary = self._format_steps_summary(steps)

        final_messages = [
            {
                "role": "system",
                "content": "You are Erosolar. Based on the ReAct steps above, provide a comprehensive answer to the user's query."
            },
            {
                "role": "system",
                "name": "react_summary",
                "content": f"ReAct execution summary:\n{steps_summary}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        stream = self.llm_client.chat.completions.create(
            model=self.final_model,
            messages=final_messages,
            stream=True,
        )

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning_chunk = getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                yield {
                    "type": "reasoning",
                    "data": reasoning_chunk
                }

            content_chunk = getattr(delta, "content", None)
            if content_chunk:
                yield {
                    "type": "final",
                    "data": content_chunk
                }

    def _find_tool_by_name(self, name: str):
        """Look up a tool by name (case-insensitive)."""
        if not name:
            return None

        tool = self.registry.get_tool(name)
        if tool:
            return tool

        lowered = name.lower()
        for registry_tool in self.registry.list_tools():
            if registry_tool.name.lower() == lowered:
                return registry_tool

        return None

    def _resolve_allowed_tools(
        self,
        requested_tools: Optional[Iterable[str]],
        available_tools: Sequence[Tuple[Any, float]],
    ) -> Tuple[List[Tuple[Any, float]], Optional[Set[str]], List[str], List[str]]:
        """
        Normalize user-selected tools and merge them with discovered tools.

        The selected tools are treated as preferred options that should be tried
        first, but the agent can still fall back to other discovered tools.

        Returns:
            merged_tools: Preferred tools (in selection order) followed by discovered tools
            allowed_set: Lowercased set of preferred tool names (or None)
            allowed_display: Canonical tool names for display/logging
            missing: Requested names that could not be resolved
        """
        normalized = [
            str(name).strip()
            for name in (requested_tools or [])
            if name and str(name).strip()
        ]
        if not normalized:
            return list(available_tools), None, [], []

        available_list = list(available_tools)
        available_map = {
            tool.name.lower(): (tool, score)
            for tool, score in available_list
        }

        preferred: List[Tuple[Any, float]] = []
        allowed_set: Set[str] = set()
        allowed_display: List[str] = []
        missing: List[str] = []

        for raw_name in normalized:
            lowered = raw_name.lower()
            if lowered in allowed_set:
                continue

            if lowered in available_map:
                tool, score = available_map[lowered]
            else:
                registry_tool = self._find_tool_by_name(raw_name)
                if registry_tool:
                    tool, score = registry_tool, 1.0
                else:
                    missing.append(raw_name)
                    continue

            preferred.append((tool, score))
            allowed_set.add(tool.name.lower())
            allowed_display.append(tool.name)

        if not allowed_set:
            return available_list, None, [], missing

        seen = {tool.name for tool, _ in preferred}
        merged = preferred + [
            (tool, score)
            for tool, score in available_list
            if tool.name not in seen
        ]

        return merged, allowed_set, allowed_display, missing

    def _handle_tool_filter_logging(
        self,
        requested_tool_filter: List[str],
        allowed_set: Optional[Set[str]],
        allowed_display: List[str],
        missing_allowed: List[str],
    ) -> str:
        """Log tool preference status and return a human-readable label."""
        allowed_label = ", ".join(allowed_display) if allowed_set else ""
        if allowed_set:
            notice = f"Prioritizing user-selected tools: {allowed_label or 'selected tools'}"
            if missing_allowed:
                notice += f" (ignored unknown tools: {', '.join(missing_allowed)})"
            logger.info(notice)
        elif requested_tool_filter:
            warn_msg = f"No valid tools matched selection: {', '.join(requested_tool_filter)}"
            if missing_allowed:
                warn_msg += f" (unknown tools: {', '.join(missing_allowed)})"
            logger.warning(warn_msg)
        return allowed_label

    def _prepare_context(
        self,
        prompt: str,
        allowed_tools: Optional[Iterable[str]],
    ) -> Tuple[
        List[Tuple[Any, float]],
        List[Dict[str, Any]],
        List[str],
        Optional[Set[str]],
        List[str],
        List[str],
    ]:
        """Prepare tool discovery info and base conversation history."""
        discovery_result = self.router.discover_tools(prompt)
        available_tools = list(discovery_result.tools)

        requested_tool_filter = [
            str(name).strip()
            for name in (allowed_tools or [])
            if name and str(name).strip()
        ]

        if requested_tool_filter:
            logger.info(f"Applying user-selected tools: {requested_tool_filter}")

        (
            filtered_tools,
            allowed_set,
            allowed_display,
            missing_allowed,
        ) = self._resolve_allowed_tools(requested_tool_filter, available_tools)

        logger.info(f"Discovered {len(filtered_tools)} tools for query: {prompt[:100]}")

        conversation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if filtered_tools:
            tools_desc = self._format_available_tools(filtered_tools)
            conversation_history.append({
                "role": "system",
                "name": "tool_discovery",
                "content": f"Available tools discovered via semantic search:\n{tools_desc}"
            })
        if allowed_display:
            conversation_history.append({
                "role": "system",
                "name": "tool_preferences",
                "content": (
                    "User manually selected these tools: "
                    f"{', '.join(allowed_display)}. Prioritize them when relevant, "
                    "but other tools remain available if needed."
                )
            })

        conversation_history.append({
            "role": "user",
            "content": prompt
        })

        return (
            filtered_tools,
            conversation_history,
            requested_tool_filter,
            allowed_set,
            allowed_display,
            missing_allowed,
        )

    def _format_available_tools(self, tools: List[tuple]) -> str:
        """Format available tools for context."""
        if not tools:
            return "No tools available"

        lines = []
        for i, (tool, score) in enumerate(tools[:10], 1):  # Limit to top 10
            lines.append(
                f"{i}. {tool.name} (relevance: {score:.3f})\n"
                f"   Description: {tool.description}\n"
                f"   Parameters: {json.dumps(tool.input_schema.get('properties', {}), ensure_ascii=False)}"
            )

        return "\n\n".join(lines)

    def _format_steps_summary(self, steps: List[ReActStep]) -> str:
        """Format ReAct steps for final context."""
        if not steps:
            return "No steps executed"

        lines = [f"Total steps: {len(steps)}"]

        for step in steps:
            lines.append(f"\nStep {step.step_number}:")
            lines.append(f"  Thought: {step.thought}")

            if step.action and step.action.upper() != "FINISH":
                lines.append(f"  Action: {step.action}")
                if step.action_input:
                    lines.append(f"  Input: {json.dumps(step.action_input, ensure_ascii=False)}")
                if step.observation:
                    # Truncate long observations
                    obs = step.observation[:300] + "..." if len(step.observation) > 300 else step.observation
                    lines.append(f"  Observation: {obs}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "router_stats": self.router.get_stats(),
            "tool_selector_stats": self.tool_selector.get_stats(),
        }
