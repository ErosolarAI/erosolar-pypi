"""
Configuration example for using the enhanced routing system in deepflask.py

This shows how to integrate the scalable embedding-based routing system
designed for hundreds of thousands of MCP tools.
"""

import os
from openai import OpenAI

# Import the enhanced components
from agent_system.enhanced_tool_registry import EnhancedToolRegistry
from agent_system.router import (
    IntentRouter,
    GraphRouter,
    EnhancedSemanticRouter,
    ThreeTierRouter,
)
from agent_system.langgraph_agent import LangGraphReActAgent
from agent_system.tool_registry import WorkflowDefinition, WorkflowStep

# Import tool builders
from agent_system.tools.tavily import build_tavily_tools
from agent_system.tools.open_metro_weather import build_open_metro_weather_tool


def create_enhanced_routing_system(
    deepseek_client,
    use_hierarchical: bool = False,
    embedding_model: str = "text-embedding-3-small",
    enable_hybrid_router: bool = True,
):
    """
    Create an enhanced routing system optimized for massive tool collections.

    Args:
        deepseek_client: OpenAI client for DeepSeek API
        use_hierarchical: Use hierarchical vector store for >10k tools
        embedding_model: "text-embedding-3-small" (cheaper) or "text-embedding-3-large" (better)
        enable_hybrid_router: Enable the hybrid routing system

    Returns:
        Configured agent with enhanced routing
    """

    # 1. Check for OpenAI API key (required for embeddings)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Semantic routing will be disabled.")
        print("   Set it for full functionality: export OPENAI_API_KEY='your-key'")
        embedding_client = None
    else:
        embedding_client = OpenAI(api_key=openai_api_key)

    # 2. Create enhanced tool registry
    print("Initializing enhanced tool registry...")
    tool_registry = EnhancedToolRegistry(
        use_hierarchical=use_hierarchical,
        n_clusters=100 if use_hierarchical else 10,
        cache_dir="./cache/vectors",  # Persistent vector storage
        embedding_model=embedding_model,
        embedding_batch_size=100,  # Batch size for embedding generation
    )

    # Set embedding client if available
    if embedding_client:
        tool_registry.set_embedding_client(embedding_client, embedding_model)

    # 3. Register existing tools
    print("Registering tools...")

    # Register Tavily tools
    for tool in build_tavily_tools():
        tool_registry.register_tool(tool)

    # Register weather tool
    for tool in build_open_metro_weather_tool():
        tool_registry.register_tool(tool)

    # TODO: Add more tools here as they become available
    # Example for batch registration:
    # additional_tools = load_mcp_tools_from_directory("./mcp_tools")
    # tool_registry.register_tools_batch(additional_tools)

    # 4. Register workflows
    tool_registry.register_workflow(
        WorkflowDefinition(
            name="web_research",
            triggers=[
                "research", "report", "news", "summarize",
                "investigate", "analyze", "explore", "study",
                "fact", "information", "details", "evidence",
            ],
            description="Multi-step workflow for web research",
            steps=[
                WorkflowStep(tool_name="tavily_search"),
                WorkflowStep(
                    tool_name="tavily_extract",
                    optional=True,
                    depends_on="tavily_search"
                ),
            ],
        )
    )

    # 5. Generate embeddings for all tools (if embedding client available)
    if embedding_client:
        print(f"Generating embeddings using {embedding_model}...")
        tool_registry.ensure_embeddings()
        stats = tool_registry.get_stats()
        print(f"  ✓ Generated embeddings for {stats['embedded_tools']} tools")
        print(f"  ✓ API calls: {stats['model_api_calls']}")

    # 6. Create routers
    print("Initializing routers...")

    # Intent Router (fast pattern matching)
    intent_router = IntentRouter(tool_registry, min_confidence=0.35)

    # Graph Router (workflow aware)
    graph_router = GraphRouter(tool_registry)

    # Enhanced Semantic Router (embedding-based)
    if embedding_client:
        semantic_router = EnhancedSemanticRouter(
            registry=tool_registry,
            embedding_client=embedding_client,
            model=embedding_model,
            min_confidence=0.3,  # Lower threshold for semantic matches
            default_top_k=5,  # Number of tools to consider
        )
        print(f"  ✓ Semantic router enabled with {embedding_model}")
    else:
        # Fallback to basic semantic router or None
        semantic_router = None
        print("  ⚠ Semantic router disabled (no OpenAI API key)")

    # 7. Create three-tier router
    router = ThreeTierRouter(
        intent_router=intent_router,
        graph_router=graph_router,
        semantic_router=semantic_router,
    )

    # 8. Create agent
    agent = LangGraphReActAgent(
        router=router,
        registry=tool_registry,
        llm_client=deepseek_client,
        system_prompt=(
            "You are LifePilot, an AI assistant with access to various tools. "
            "Use the provided tools to help users with their requests. "
            "When multiple tools are available, select the most appropriate ones."
        ),
    )

    print(f"✓ Enhanced routing system initialized with {tool_registry._stats['total_tools']} tools")

    return agent, tool_registry, router


def create_flask_app_with_enhanced_routing():
    """
    Create Flask app with enhanced routing system.

    This is a drop-in replacement for the initialization in deepflask.py
    """
    from flask import Flask, Response, request
    from openai import OpenAI

    app = Flask(__name__)

    # Initialize DeepSeek client
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable")

    deepseek_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
    )

    # Create enhanced routing system
    agent, tool_registry, router = create_enhanced_routing_system(
        deepseek_client=deepseek_client,
        use_hierarchical=False,  # Set to True for >10k tools
        embedding_model="text-embedding-3-small",  # or "text-embedding-3-large"
        enable_hybrid_router=True,
    )

    # Configuration
    AB_TEST_ENABLED = os.environ.get("ENABLE_HYBRID_ROUTER", "true").lower() not in {"0", "false"}
    HYBRID_ROUTER_PERCENT = float(os.environ.get("HYBRID_ROUTER_PERCENT", "0.5"))  # Increase default

    @app.route("/stream")
    def stream():
        """Enhanced streaming endpoint with scalable routing."""
        prompt = request.args.get("prompt")
        if not prompt:
            return "Missing 'prompt' query parameter", 400

        # Choose router variant (can be overridden by query param)
        router_version = request.args.get("router_version", "").lower()
        if router_version not in {"legacy", "hybrid"}:
            # Use hybrid router more often by default
            import random
            router_version = "hybrid" if random.random() < HYBRID_ROUTER_PERCENT else "legacy"

        if router_version == "hybrid":
            # Use enhanced routing
            plan_state = agent.plan(prompt)
            decision = plan_state.get("decision")
            tool_outputs = plan_state.get("tool_outputs", [])

            # Stream response
            def generate():
                # Send reasoning
                reasoning = f"[Enhanced Router] {decision.reasoning if decision else 'No tools matched'}"
                yield f"event: reasoning\ndata: {reasoning}\n\n"

                # Stream final response
                if decision and tool_outputs:
                    for chunk in agent.stream_final(prompt, decision, tool_outputs):
                        if chunk.choices[0].delta.content:
                            yield f"data: {chunk.choices[0].delta.content}\n\n"
                else:
                    # Direct response without tools
                    stream = deepseek_client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield f"data: {chunk.choices[0].delta.content}\n\n"

            return Response(generate(), mimetype="text/event-stream")
        else:
            # Legacy routing (direct to LLM)
            def generate():
                stream = deepseek_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield f"data: {chunk.choices[0].delta.content}\n\n"

            return Response(generate(), mimetype="text/event-stream")

    @app.route("/router-metrics")
    def router_metrics():
        """Get enhanced router metrics."""
        import json

        metrics = {
            "router_distribution": router.snapshot(),
            "registry_stats": tool_registry.get_stats(),
            "cache_hit_rate": tool_registry.cache_hit_rate(),
        }

        # Add semantic router stats if available
        if hasattr(router.semantic_router, 'get_stats'):
            metrics["semantic_router"] = router.semantic_router.get_stats()

        return json.dumps(metrics, indent=2)

    @app.route("/tools")
    def list_tools():
        """List all registered tools."""
        import json

        tools = []
        for tool in tool_registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "categories": tool.metadata.get("categories", []),
            })

        return json.dumps({"total": len(tools), "tools": tools}, indent=2)

    @app.route("/search-tools")
    def search_tools():
        """Search for tools using semantic search."""
        import json

        query = request.args.get("q", "")
        top_k = int(request.args.get("top_k", "10"))
        categories = request.args.getlist("category")

        if not query:
            return json.dumps({"error": "Missing 'q' query parameter"}), 400

        # Use semantic search
        if categories:
            # Search within categories
            results = tool_registry.semantic_search(
                query=query,
                top_k=top_k,
                filter_categories=categories,
            )
        else:
            # General search
            results = tool_registry.hybrid_search(
                query=query,
                pattern_weight=0.3,
                semantic_weight=0.7,
                top_k=top_k,
            )

        tools = []
        for tool, score in results:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "categories": tool.metadata.get("categories", []),
                "score": round(score, 3),
            })

        return json.dumps({"query": query, "results": tools}, indent=2)

    return app


# Environment variable configuration guide
CONFIGURATION_GUIDE = """
ENHANCED ROUTING SYSTEM - CONFIGURATION GUIDE
============================================

Required Environment Variables:
-------------------------------
DEEPSEEK_API_KEY         : Your DeepSeek API key (required)
OPENAI_API_KEY           : Your OpenAI API key (required for embeddings)
TAVILY_API_KEY           : Your Tavily API key (for web search)

Optional Configuration:
-----------------------
EMBEDDING_MODEL          : "text-embedding-3-small" (default) or "text-embedding-3-large"
HYBRID_ROUTER_PERCENT    : Percentage of traffic to route through hybrid system (0.0-1.0, default: 0.5)
ENABLE_HYBRID_ROUTER     : Enable/disable hybrid routing ("true"/"false", default: "true")
USE_HIERARCHICAL_STORE   : Use hierarchical vector store for >10k tools ("true"/"false", default: "false")
VECTOR_CACHE_DIR         : Directory for persistent vector storage (default: "./cache/vectors")
EMBEDDING_BATCH_SIZE     : Batch size for embedding generation (default: 100)

Model Selection Guide:
----------------------
text-embedding-3-small:
  - Dimensions: 1536
  - Cost: $0.00002 per 1k tokens
  - Speed: Faster
  - Use for: Large-scale deployments, cost optimization
  - Recommended for: 10k+ tools

text-embedding-3-large:
  - Dimensions: 3072
  - Cost: $0.00013 per 1k tokens
  - Quality: Better semantic understanding
  - Use for: Higher accuracy requirements
  - Recommended for: <10k tools or critical applications

Performance Considerations:
---------------------------
- <1,000 tools:      Use standard VectorStore
- 1,000-10,000:      Use standard VectorStore with batch operations
- 10,000-100,000:    Use HierarchicalVectorStore
- >100,000:          Consider external vector database (Pinecone, Weaviate, etc.)

Example Setup:
--------------
export DEEPSEEK_API_KEY='your-deepseek-key'
export OPENAI_API_KEY='your-openai-key'
export TAVILY_API_KEY='your-tavily-key'
export EMBEDDING_MODEL='text-embedding-3-small'
export HYBRID_ROUTER_PERCENT='0.7'
export USE_HIERARCHICAL_STORE='false'

python deepflask_enhanced.py
"""

if __name__ == "__main__":
    print(CONFIGURATION_GUIDE)

    # Example: Create and run the enhanced Flask app
    print("\nStarting enhanced Flask app...")
    app = create_flask_app_with_enhanced_routing()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)