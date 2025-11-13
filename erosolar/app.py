import os
import json
import logging
import random
import sqlite3
import time
from typing import List
from flask import Flask, Response, request, render_template_string, stream_with_context
from openai import OpenAI

from .agent_system.langgraph_agent import LangGraphReActAgent
from .agent_system.embeddings_router import EmbeddingsRouter
from .agent_system.tool_registry import ToolRegistry, WorkflowDefinition, WorkflowStep
from .agent_system.tools.open_metro_weather import build_open_metro_weather_tool
from .agent_system.tools.tavily import build_tavily_tools
from .agent_system.true_react_agent import TrueReActAgent
from .agent_system.enhanced_tool_registry import EnhancedToolRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLIENT_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

app = Flask(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY environment variable is not set!")
    logger.error("Please set it with: export DEEPSEEK_API_KEY='your-api-key-here'")
    raise RuntimeError("DEEPSEEK_API_KEY is required. Please set the environment variable and restart the app.")

try:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    logger.info("DeepSeek client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize DeepSeek client: {e}")
    raise RuntimeError(f"Failed to initialize DeepSeek client: {e}")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if OPENAI_API_KEY:
    embedding_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI embeddings client initialized (for better tool matching)")
else:
    embedding_client = None
    logger.warning("OPENAI_API_KEY not set - will use DeepSeek for embeddings (may be slower)")
    logger.warning("To improve performance, set OPENAI_API_KEY: export OPENAI_API_KEY='your-key-here'")

tool_registry = ToolRegistry()

# Register Tavily tools if API key is available
if TAVILY_API_KEY:
    try:
        for tool in build_tavily_tools():
            tool_registry.register_tool(tool)
        logger.info("Tavily web search tools registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register Tavily tools: {e}")
else:
    logger.warning("TAVILY_API_KEY not set - web search tools will not be available")
    logger.warning("To enable web search, set TAVILY_API_KEY: export TAVILY_API_KEY='your-key-here'")

# Register weather tools
try:
    for tool in build_open_metro_weather_tool():
        tool_registry.register_tool(tool)
    logger.info("Weather tools registered successfully")
except Exception as e:
    logger.warning(f"Failed to register weather tools: {e}")

tool_registry.register_workflow(
    WorkflowDefinition(
        name="web_research",
        triggers=[
            "research", "report", "news", "summarize",
            "investigate", "analyze", "explore", "study",
            "fact", "information", "details", "evidence"
        ],
        description="Multi-step workflow that searches the web then optionally extracts a key page.",
        steps=[
            WorkflowStep(tool_name="tavily_search"),
            WorkflowStep(tool_name="tavily_extract", optional=True, depends_on="tavily_search"),
        ],
    )
)

if embedding_client:
    tool_registry.set_embedding_client(embedding_client, model="text-embedding-3-large")

# Use EmbeddingsRouter (replaces deprecated IntentRouter, GraphRouter, SemanticRouter, ThreeTierRouter)
router = EmbeddingsRouter(tool_registry, embedding_client) if embedding_client else None

agent = LangGraphReActAgent(router=router, registry=tool_registry, llm_client=client) if router else None

# Create enhanced tool registry for TrueReActAgent
enhanced_registry = EnhancedToolRegistry()
# Migrate tools from regular registry to enhanced registry
for tool in tool_registry.list_tools():
    enhanced_registry.register_tool(tool)
# Set embedding client for enhanced registry
if embedding_client:
    enhanced_registry.set_embedding_client(embedding_client, model="text-embedding-3-small")

# Create TrueReActAgent - always create it, use DeepSeek client as fallback
true_react_agent = TrueReActAgent(
    registry=enhanced_registry,
    embedding_client=embedding_client or client,  # Fallback to DeepSeek client if no OpenAI
    llm_client=client,
    max_iterations=5,
)

# IMPORTANT: Ensure embeddings are generated AFTER creating the agent
# (EmbeddingsRouter recreates the vector store when setting embedding client)
if embedding_client:
    enhanced_registry.ensure_embeddings()
    logger.info(f"TrueReActAgent initialized with {len(enhanced_registry.list_tools())} tools and OpenAI embeddings")
else:
    logger.warning(f"TrueReActAgent initialized with {len(enhanced_registry.list_tools())} tools using DeepSeek embeddings (OPENAI_API_KEY not set)")

AB_TEST_ENABLED = os.environ.get("ENABLE_HYBRID_ROUTER", "true").lower() not in {"0", "false"}
HYBRID_ROUTER_PERCENT = float(os.environ.get("HYBRID_ROUTER_PERCENT", "0.1"))

ROUTER_METRICS = {
    "legacy": {"requests": 0, "avg_latency_ms": 0.0, "tool_invocations": 0, "last_tier": "legacy"},
    "hybrid": {"requests": 0, "avg_latency_ms": 0.0, "tool_invocations": 0, "last_tier": "intent"},
    "react": {"requests": 0, "avg_latency_ms": 0.0, "tool_invocations": 0, "last_tier": "react"},
}

def get_user_data_dir():
    """Get OS-specific user data directory for Erosolar."""
    from pathlib import Path
    if os.name == 'nt':  # Windows
        base_dir = os.environ.get('APPDATA', os.path.expanduser('~'))
        data_dir = Path(base_dir) / 'Erosolar'
    elif os.name == 'posix':
        if 'darwin' in os.sys.platform.lower():  # macOS
            data_dir = Path.home() / 'Library' / 'Application Support' / 'Erosolar'
        else:  # Linux
            data_dir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'erosolar'
    else:
        # Fallback
        data_dir = Path.home() / '.erosolar'

    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


DATABASE_PATH = str(get_user_data_dir() / "chat_history.db")


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def clear_all_history():
    """Clear all chat history from the database."""
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM chat_history")
        conn.commit()
        conn.close()
        logger.info("All chat history cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        return False


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def fetch_chat_history(limit=25):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, prompt, response, reasoning, created_at FROM chat_history ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "reasoning": row["reasoning"] or "",
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_chat_entry(prompt: str, response: str, reasoning: str = "") -> int:
    conn = get_db_connection()
    cursor = conn.execute(
        "INSERT INTO chat_history (prompt, response, reasoning) VALUES (?, ?, ?)",
        (prompt, response, reasoning),
    )
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    return entry_id


init_db()


def choose_router_variant(req) -> str:
    """Choose which router variant to use based on request parameters or A/B testing."""
    override = (req.args.get("router_version") or "").lower()
    if override in {"legacy", "hybrid"}:
        return override
    # Note: removed intent_router.requires_search() - EmbeddingsRouter doesn't have this method
    if not AB_TEST_ENABLED:
        return "legacy"
    bucket = random.random()
    return "hybrid" if bucket < HYBRID_ROUTER_PERCENT else "legacy"


def record_metrics(variant: str, latency_ms: float, decision=None, tool_outputs=None):
    bucket = ROUTER_METRICS.setdefault(
        variant, {"requests": 0, "avg_latency_ms": 0.0, "tool_invocations": 0, "last_tier": variant}
    )
    bucket["requests"] += 1
    requests_count = bucket["requests"]
    bucket["avg_latency_ms"] = round(
        ((requests_count - 1) * bucket["avg_latency_ms"] + latency_ms) / requests_count, 2
    )
    bucket["tool_invocations"] += len(tool_outputs or [])
    if decision:
        bucket["last_tier"] = decision.tier
        bucket["last_tools"] = decision.tool_sequence
        bucket["confidence"] = decision.confidence
    # EmbeddingsRouter uses get_stats() instead of snapshot()
    bucket["router_stats"] = router.get_stats() if router and hasattr(router, "get_stats") else {}
    bucket["cache_hit_rate"] = tool_registry.cache_hit_rate()


def format_reasoning(decision, tool_outputs) -> str:
    def friendly_tool_name(name: str) -> str:
        if not name:
            return "Unknown Tool"
        return name.replace("_", " ").title()

    if not decision:
        return "Router: responding directly — no tools required."

    router_label = f"{(decision.tier or 'router').title()} Router"
    confidence = f"{decision.confidence * 100:.0f}%" if decision.confidence is not None else "unknown"
    planned_tools = ", ".join(friendly_tool_name(t) for t in (decision.tool_sequence or [])) or "None"
    lines = [f"{router_label} • Confidence {confidence} • Tools: {planned_tools}"]

    if not tool_outputs:
        lines.append("No tools executed.")
        return "\n".join(lines)

    for obs in tool_outputs:
        tool_label = friendly_tool_name(obs.get("tool"))
        result = obs.get("result")
        cached = result.get("cached") if isinstance(result, dict) else None
        if cached is True:
            lines.append(f"Using cached {tool_label}")
        elif cached is False:
            lines.append(f"Performing {tool_label}")
        else:
            lines.append(f"Running {tool_label}")
    return "\n".join(lines)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Erosolar - Agentic Suite</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #343541;
      --surface: #000000;
      --surface-alt: #2a2b32;
      --border: #4d4d4f;
      --border-strong: #565869;
      --accent: #19c37d;
      --accent-strong: #19c37d;
      --text: #ececf1;
      --text-subtle: #c5c5d2;
    }
    * { box-sizing: border-box; }
    body {
      font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    /* Top Navigation Bar */
    .top-nav {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 1px 0 rgba(255,255,255,0.05);
    }
    .nav-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.85rem 1.5rem;
      max-width: 100%;
    }
    .nav-brand {
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text);
      letter-spacing: -0.02em;
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .account-menu {
      position: relative;
    }
    .account-button {
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 50%;
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 150ms ease;
      padding: 0;
    }
    .account-button:hover {
      background: var(--border-strong);
      border-color: var(--accent);
    }
    .account-icon {
      font-size: 1.25rem;
      display: block;
    }
    .account-dropdown {
      position: absolute;
      top: calc(100% + 0.5rem);
      right: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.85rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      min-width: 200px;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-8px);
      transition: opacity 200ms ease, transform 200ms ease, visibility 200ms;
    }
    .account-dropdown.open {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }
    .dropdown-item {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.85rem 1rem;
      background: transparent;
      border: none;
      color: var(--text);
      font-size: 0.95rem;
      text-align: left;
      cursor: pointer;
      transition: background 150ms ease;
      border-radius: 0.75rem;
    }
    .dropdown-item:hover {
      background: var(--surface-alt);
    }
    .dropdown-icon {
      font-size: 1.1rem;
    }
    /* Modal Styles */
    .modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .modal.open {
      display: flex;
    }
    .modal-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.7);
      backdrop-filter: blur(4px);
    }
    .modal-content {
      position: relative;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      box-shadow: 0 20px 60px rgba(0,0,0,0.5);
      max-width: 500px;
      width: 90%;
      max-height: 85vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: modalSlideIn 250ms ease;
    }
    @keyframes modalSlideIn {
      from {
        opacity: 0;
        transform: scale(0.95) translateY(20px);
      }
      to {
        opacity: 1;
        transform: scale(1) translateY(0);
      }
    }
    .modal-header {
      padding: 1.5rem 1.5rem 1rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .modal-header h2 {
      margin: 0;
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--text);
    }
    .modal-close {
      background: transparent;
      border: none;
      color: var(--text-subtle);
      font-size: 2rem;
      line-height: 1;
      cursor: pointer;
      padding: 0;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 0.5rem;
      transition: all 150ms ease;
    }
    .modal-close:hover {
      background: var(--surface-alt);
      color: var(--text);
    }
    .modal-body {
      padding: 1.5rem;
      overflow-y: auto;
      flex: 1;
    }
    .setting-group {
      margin-bottom: 1.5rem;
    }
    .setting-group:last-child {
      margin-bottom: 0;
    }
    .setting-group label {
      display: block;
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text);
      margin-bottom: 0.5rem;
    }
    .setting-group input {
      width: 100%;
      padding: 0.75rem;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      color: var(--text);
      font-size: 0.95rem;
      transition: border-color 150ms ease;
    }
    .setting-group input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(25,195,125,0.1);
    }
    .setting-hint {
      margin: 0.5rem 0 0;
      font-size: 0.85rem;
      color: var(--text-subtle);
    }
    .modal-footer {
      padding: 1rem 1.5rem 1.5rem;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 0.75rem;
      justify-content: flex-end;
    }
    .app-layout {
      display: flex;
      min-height: calc(100vh - 57px);
    }
    .sidebar {
      width: 218px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }
    .sidebar-header {
      padding: 0.85rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .sidebar-header button {
      width: 100%;
    }
    .sidebar-header p {
      margin: 0;
      color: var(--text-subtle);
      font-size: 0.8rem;
    }
    .history-list {
      flex: 1;
      overflow-y: auto;
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .history-item {
      padding: 0.6rem 0.7rem;
      border-radius: 0.85rem;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
      font-size: 0.8rem;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .history-item:hover {
      background: var(--surface);
      border-color: var(--primary);
      transform: translateX(2px);
    }
    .history-item:active {
      transform: translateX(0px);
      opacity: 0.8;
    }
    .history-empty {
      color: var(--text-subtle);
      text-align: center;
      margin-top: 4rem;
      font-size: 0.8rem;
    }
    .chat-pane {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 1rem;
      gap: 0.75rem;
      max-height: calc(100vh - 57px);
      overflow: hidden;
    }
    .chat-window {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      overflow-y: auto;
      min-height: 0;
      box-shadow: none;
    }
    .message-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .message {
      display: flex;
      gap: 0.5rem;
      align-items: flex-start;
    }
    .message.user {
      flex-direction: row-reverse;
      text-align: right;
    }
    .message .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: var(--surface-alt);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--accent);
      flex-shrink: 0;
    }
    .message.user .avatar { color: var(--accent-strong); }
    .message-content {
      position: relative;
      background: var(--surface-alt);
      border: 1px solid rgba(255,255,255,0.04);
      border-radius: 1rem;
      padding: 0.7rem 0.85rem 2rem;
      width: 100%;
      font-size: 0.9rem;
    }
    .message.user .message-content {
      background: var(--surface-alt);
      border-color: var(--border);
    }
    .message.streaming .message-content {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(25,195,125,0.3);
    }
    .message-body {
      white-space: pre-wrap;
      line-height: 1.45;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .message-text {
      white-space: pre-wrap;
    }
    .copy-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-subtle);
      border-radius: 999px;
      font-size: 0.75rem;
      padding: 0.25rem 0.85rem;
      position: absolute;
      bottom: 0.55rem;
      right: 0.6rem;
    }
    .copy-btn:hover {
      color: var(--text);
      border-color: var(--accent);
    }
    .thought-inline {
      background: var(--surface-alt);
      border: 1px dashed var(--border-strong);
      border-radius: 0.75rem;
      padding: 0.5rem 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0;
      position: relative;
    }
    .thought-inline.expanded {
      padding: 0.65rem 0.85rem 2rem;
      gap: 0.4rem;
    }
    .thought-toggle {
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 0.65rem;
      padding: 0.25rem 0.6rem;
      font-size: 0.65rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--text-subtle);
      align-self: flex-start;
    }
    .thought-toggle[aria-expanded="true"] {
      color: var(--accent);
      border-color: var(--accent);
    }
    .thought-inline .content {
      white-space: pre-wrap;
      font-size: 0.8rem;
      color: var(--text);
      line-height: 1.35;
      max-height: 200px;
      overflow-y: auto;
      display: none;
    }
    .thought-inline .content.visible {
      display: block;
    }
    .thought-inline .thought-copy-btn {
      display: none;
      right: 0.75rem;
    }
    .thought-inline.expanded .thought-copy-btn {
      display: inline-flex;
    }
    .code-box {
      position: relative;
      background: #0c111b;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.6rem 0.85rem 2.2rem;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }
    .code-header {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-subtle);
      margin-bottom: 0.4rem;
    }
    .code-box pre {
      margin: 0;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.78rem;
      color: var(--text);
      background: #111728;
      border-radius: 0.6rem;
      padding: 0.55rem 0.75rem;
      overflow: auto;
      white-space: pre;
      line-height: 1.5;
    }
    .code-box code {
      display: block;
      min-width: 100%;
    }
    .hljs {
      color: #d4d4d4;
      background: transparent;
    }
    .hljs-comment,
    .hljs-quote {
      color: #6a9955;
      font-style: italic;
    }
    .hljs-keyword,
    .hljs-selector-tag,
    .hljs-literal,
    .hljs-built_in,
    .hljs-type {
      color: #c586c0;
    }
    .hljs-number,
    .hljs-symbol,
    .hljs-bullet,
    .hljs-link {
      color: #b5cea8;
    }
    .hljs-string,
    .hljs-meta .hljs-string,
    .hljs-template-tag,
    .hljs-template-variable {
      color: #ce9178;
    }
    .hljs-title,
    .hljs-section,
    .hljs-selector-id,
    .hljs-selector-class {
      color: #4ec9b0;
    }
    .hljs-attr,
    .hljs-attribute,
    .hljs-property,
    .hljs-variable,
    .hljs-params {
      color: #9cdcfe;
    }
    .hljs-meta,
    .hljs-name {
      color: #dcdcaa;
    }
    .hljs-emphasis {
      font-style: italic;
    }
    .hljs-strong {
      font-weight: 700;
    }
    .code-copy-btn {
      position: absolute;
      bottom: 0.55rem;
      right: 0.7rem;
    }
    /* Markdown formatting styles */
    .md-h1 {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text);
      margin: 1rem 0 0.75rem;
      line-height: 1.3;
    }
    .md-h2 {
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--text);
      margin: 0.9rem 0 0.65rem;
      line-height: 1.3;
    }
    .md-h3 {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text);
      margin: 0.8rem 0 0.55rem;
      line-height: 1.3;
    }
    .md-table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.75rem 0;
      font-size: 0.85rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      overflow: hidden;
    }
    .md-table th {
      background: var(--surface-alt);
      color: var(--text);
      font-weight: 600;
      padding: 0.6rem 0.75rem;
      text-align: left;
      border-bottom: 2px solid var(--border);
    }
    .md-table td {
      padding: 0.55rem 0.75rem;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      color: var(--text);
    }
    .md-table tr:last-child td {
      border-bottom: none;
    }
    .md-table tr:hover {
      background: rgba(255,255,255,0.02);
    }
    .md-list {
      margin: 0.6rem 0;
      padding-left: 1.5rem;
      color: var(--text);
    }
    .md-list li {
      margin: 0.3rem 0;
      line-height: 1.5;
    }
    .md-inline-code {
      background: var(--surface);
      color: var(--accent);
      padding: 0.15rem 0.4rem;
      border-radius: 0.3rem;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.85em;
      border: 1px solid var(--border);
    }
    .md-blockquote {
      border-left: 3px solid var(--accent);
      padding: 0.5rem 0.75rem;
      margin: 0.6rem 0;
      background: var(--surface-alt);
      border-radius: 0.5rem;
      color: var(--text-subtle);
      font-style: italic;
    }
    .md-hr {
      border: none;
      border-top: 1px solid var(--border);
      margin: 1rem 0;
    }
    .md-link {
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid transparent;
      transition: border-color 150ms ease;
    }
    .md-link:hover {
      border-bottom-color: var(--accent);
    }
    .json-block {
      background: #0c111b;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.75rem;
      margin: 0.6rem 0;
      overflow-x: auto;
    }
    .json-block code {
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.8rem;
      color: #9cdcfe;
      line-height: 1.5;
    }
    .message-text strong {
      font-weight: 600;
      color: var(--text);
    }
    .message-text em {
      font-style: italic;
      color: var(--text);
    }
    .input-area {
      position: relative;
      padding: 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }
    .input-wrapper {
      position: relative;
    }
    .input-area textarea {
      width: 100%;
      border-radius: 0.75rem;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--text);
      padding: 0.75rem 0.75rem 3.5rem 0.75rem;
      font-size: 0.9rem;
      min-height: 120px;
      resize: vertical;
    }
    .input-area textarea::placeholder {
      color: var(--text-subtle);
    }
    .input-controls {
      position: absolute;
      bottom: 0.5rem;
      right: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      align-items: flex-end;
    }
    .tools-selector {
      position: relative;
    }
    .tools-toggle {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-subtle);
      border-radius: 0.5rem;
      font-size: 0.75rem;
      padding: 0.35rem 0.7rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 150ms ease;
    }
    .tools-toggle:hover {
      color: var(--text);
      border-color: var(--accent);
      background: var(--surface-alt);
    }
    .tools-toggle.active {
      color: var(--accent);
      border-color: var(--accent);
    }
    .tools-dropdown {
      position: absolute;
      bottom: calc(100% + 0.5rem);
      right: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      min-width: 280px;
      max-height: 400px;
      overflow-y: auto;
      display: none;
      flex-direction: column;
      z-index: 10;
    }
    .tools-dropdown.open {
      display: flex;
    }
    .tools-header {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 0.9rem;
      color: var(--text);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .tools-clear {
      background: transparent;
      border: none;
      color: var(--text-subtle);
      font-size: 0.75rem;
      padding: 0.2rem 0.5rem;
      cursor: pointer;
      border-radius: 0.4rem;
    }
    .tools-clear:hover {
      color: var(--accent);
      background: var(--surface-alt);
    }
    .tool-item {
      padding: 0.6rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      cursor: pointer;
      transition: background 150ms ease;
    }
    .tool-item:hover {
      background: var(--surface-alt);
    }
    .tool-item:last-child {
      border-bottom: none;
    }
    .tool-checkbox {
      width: 16px;
      height: 16px;
      border: 1px solid var(--border);
      border-radius: 0.3rem;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .tool-checkbox.checked {
      background: var(--accent);
      border-color: var(--accent);
    }
    .tool-checkbox.checked::after {
      content: '✓';
      color: #000;
      font-size: 0.7rem;
      font-weight: 700;
    }
    .tool-info {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .tool-name {
      font-size: 0.85rem;
      color: var(--text);
      font-weight: 500;
    }
    .tool-desc {
      font-size: 0.7rem;
      color: var(--text-subtle);
      line-height: 1.3;
    }
    .tool-meta {
      font-size: 0.65rem;
      color: var(--text-subtle);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 0.55rem 1.25rem;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 150ms ease, opacity 150ms ease;
    }
    button.primary {
      background: var(--accent);
      color: #000;
    }
    button.secondary {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
    }
    button:hover:not(:disabled) { transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .input-area.disabled textarea { opacity: 0.6; }
    @media (max-width: 960px) {
      .app-layout { flex-direction: column; }
      .sidebar { width: 100%; border-right: none; border-bottom: 1px solid var(--border); }
      .chat-pane { padding: 1rem; }
    }
  </style>
</head>
<body>
  <!-- Top Navigation Bar -->
  <nav class="top-nav">
    <div class="nav-content">
      <div class="nav-brand">Erosolar - Agentic Suite</div>
      <div class="nav-actions">
        <div class="account-menu">
          <button class="account-button" id="account-button" type="button" aria-haspopup="true" aria-expanded="false">
            <span class="account-icon">⚙</span>
          </button>
          <div class="account-dropdown" id="account-dropdown">
            <button class="dropdown-item" id="settings-button" type="button">
              <span class="dropdown-icon">🔑</span>
              API Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  </nav>

  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <button class="primary" type="button" id="new-chat">New Chat</button>
      </div>
      <div id="history-list" class="history-list"></div>
    </aside>
    <main class="chat-pane">
      <section id="chat-history" class="chat-window"></section>
      <form id="chat-form" class="input-area">
        <div class="input-wrapper">
          <textarea id="prompt" placeholder="Ask me anything..."></textarea>
          <div class="input-controls">
            <div class="tools-selector">
              <button type="button" class="tools-toggle" id="tools-toggle" aria-haspopup="true" aria-expanded="false">
                <span>🛠</span>
                <span id="tools-label">Tools</span>
              </button>
              <div class="tools-dropdown" id="tools-dropdown">
                <div class="tools-header">
                  <span id="tools-count">Available Tools</span>
                  <button type="button" class="tools-clear" id="tools-clear">Clear All</button>
                </div>
                <div id="tools-list"></div>
              </div>
            </div>
            <button type="submit" class="primary" id="send-btn">Send</button>
          </div>
        </div>
      </form>
    </main>
  </div>

  <!-- Settings Modal -->
  <div class="modal" id="settings-modal">
    <div class="modal-overlay" id="modal-overlay"></div>
    <div class="modal-content">
      <div class="modal-header">
        <h2>API Settings</h2>
        <button class="modal-close" id="modal-close" type="button" aria-label="Close">&times;</button>
      </div>
      <div class="modal-body">
        <div class="setting-group">
          <label for="deepseek-key">DeepSeek API Key</label>
          <input type="password" id="deepseek-key" placeholder="Enter your DeepSeek API key" />
          <p class="setting-hint">Used for AI model inference</p>
        </div>
        <div class="setting-group">
          <label for="tavily-key">Tavily API Key</label>
          <input type="password" id="tavily-key" placeholder="Enter your Tavily API key" />
          <p class="setting-hint">Used for web search functionality</p>
        </div>
        <div class="setting-group">
          <label for="openai-key">OpenAI API Key</label>
          <input type="password" id="openai-key" placeholder="Enter your OpenAI API key" />
          <p class="setting-hint">Used for embeddings (optional)</p>
        </div>
        <div class="setting-group" style="border-top: 1px solid #e2e8f0; padding-top: 20px; margin-top: 20px;">
          <label>Data Management</label>
          <button class="secondary" id="clear-history-btn" type="button" style="width: 100%; margin-top: 10px;">
            Clear All Chat History
          </button>
          <p class="setting-hint">Permanently delete all chat history from the database</p>
        </div>
      </div>
      <div class="modal-footer">
        <button class="secondary" id="cancel-settings" type="button">Cancel</button>
        <button class="primary" id="save-settings" type="button">Save Settings</button>
      </div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script>
    console.log('[Init] Script loaded, waiting for DOM...');

    document.addEventListener('DOMContentLoaded', function() {
      console.log('[Init] DOM ready, initializing application...');

      try {
        const form = document.getElementById('chat-form');
        const promptInput = document.getElementById('prompt');
        const chatHistory = document.getElementById('chat-history');
        const newChatBtn = document.getElementById('new-chat');
        const historyList = document.getElementById('history-list');
        const sendButton = document.getElementById('send-btn');
        const appParams = new URLSearchParams(window.location.search);
        const agentOverride = (appParams.get('agent') || '').toLowerCase();

        // Debug element selection
        console.log('[Init] Form element:', form);
        console.log('[Init] Prompt input:', promptInput);
        console.log('[Init] Send button:', sendButton);
        console.log('[Init] Chat history:', chatHistory);

        if (!form) console.error('[Init] ERROR: Form element not found!');
        if (!promptInput) console.error('[Init] ERROR: Prompt input not found!');
        if (!sendButton) console.error('[Init] ERROR: Send button not found!');
        if (!chatHistory) console.error('[Init] ERROR: Chat history not found!');

        let es = null;
        let isStreaming = false;
        let activeGroup = null;
        let reasoningBuffer = '';
        let finalBuffer = '';
        let currentPrompt = '';
        let hasFinalStarted = false;
        let currentSelectedToolList = [];
        let streamOpened = false;
        let streamOpenTimeout = null;
        let streamCompleted = false;
        let fallbackActive = false;
        let availableTools = [];
        let selectedTools = new Set();

    function reportClientLog(level, message, details = {}) {
      try {
        const payload = {
          level: (level || 'info').toLowerCase(),
          message,
          details,
          url: window.location.href,
          timestamp: new Date().toISOString(),
        };
        const body = JSON.stringify(payload);
        if (navigator.sendBeacon) {
          const blob = new Blob([body], { type: 'application/json' });
          navigator.sendBeacon('/client-log', blob);
        } else {
          fetch('/client-log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
            keepalive: true,
          }).catch((err) => console.warn('[ClientLog] Fallback failed', err));
        }
      } catch (logError) {
        console.warn('[ClientLog] Unable to send log', logError);
      }
    }

    window.addEventListener('error', (event) => {
      reportClientLog('error', 'Unhandled error event', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error ? event.error.stack || event.error.message : null,
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      reportClientLog('error', 'Unhandled promise rejection', {
        reason: event.reason ? (event.reason.stack || event.reason.message || String(event.reason)) : null,
      });
    });

    if (window.hljs && hljs.configure) {
      hljs.configure({ ignoreUnescapedHTML: true });
    }

    const LANGUAGE_ALIASES = {
      'c++': 'cpp',
      'cpp': 'cpp',
      'c#': 'csharp',
      'f#': 'fsharp',
      'js': 'javascript',
      'ts': 'typescript',
      'py': 'python',
      'shell': 'bash',
      'sh': 'bash',
      'powershell': 'powershell',
      'plaintext': 'plaintext',
    };

    function normalizeLanguageName(raw) {
      if (!raw) return 'plaintext';
      const cleaned = raw.trim().toLowerCase();
      return LANGUAGE_ALIASES[cleaned] || cleaned.replace(/[^a-z0-9]+/g, '-');
    }

    function highlightCodeElement(codeEl, rawLanguage) {
      const language = normalizeLanguageName(rawLanguage);
      codeEl.classList.add('hljs');
      if (language) {
        codeEl.dataset.language = language;
        codeEl.classList.add('language-' + language);
      }
      if (window.hljs && hljs.highlightElement) {
        hljs.highlightElement(codeEl);
      }
    }

    function scrollChatToBottom() {
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function createCopyButton(getText, label = 'Copy') {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-btn';
      const idleLabel = label || 'Copy';
      btn.textContent = idleLabel;
      btn.addEventListener('click', async () => {
        const value = typeof getText === 'function' ? getText() : getText;
        if (!value) return;
        try {
          await navigator.clipboard.writeText(value);
          btn.textContent = 'Copied!';
        } catch (error) {
          console.error('Copy failed', error);
          btn.textContent = 'Error';
        }
        setTimeout(() => {
          btn.textContent = idleLabel;
        }, 1500);
      });
      return btn;
    }

    function renderAssistantSegments(container, text) {
      container.innerHTML = '';
      if (!text) return;
      const regex = /```([\\w+-]+)?\\n([\\s\\S]*?)```/g;
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(text)) !== null) {
        const preceding = text.slice(lastIndex, match.index);
        appendText(preceding);

        const language = (match[1] || '').trim();
        const codeContent = match[2] || '';
        container.appendChild(createCodeBox(codeContent, language));

        lastIndex = match.index + match[0].length;
      }

      // Check for incomplete code block at the end (for streaming)
      const remaining = text.slice(lastIndex);
      const incompleteMatch = remaining.match(/```([\\w+-]+)?\\n([\\s\\S]*)$/);

      if (incompleteMatch) {
        // Found an incomplete code block
        const beforeCode = remaining.slice(0, incompleteMatch.index);
        appendText(beforeCode);

        const language = (incompleteMatch[1] || '').trim();
        const codeContent = incompleteMatch[2] || '';
        container.appendChild(createCodeBox(codeContent, language));
      } else {
        // No incomplete code block, just append remaining text
        appendText(remaining);
      }

      function appendText(segment) {
        if (!segment) return;
        const block = document.createElement('div');
        block.className = 'message-text';
        block.innerHTML = parseMarkdown(segment);
        container.appendChild(block);
      }
    }

    function parseInlineMarkdown(text) {
      // Helper function to parse inline formatting only
      let html = text;

      // Parse inline code (must be before bold/italic)
      html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');

      // Parse bold (** or __)
      html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');

      // Parse italic (* or _)
      html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
      html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

      // Parse links [text](url)
      html = html.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');

      return html;
    }

    function parseMarkdown(text) {
      if (!text) return '';

      // Try to detect if the entire text is JSON
      const trimmed = text.trim();
      if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
          (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
        try {
          const parsed = JSON.parse(trimmed);
          const jsonStr = JSON.stringify(parsed, null, 2);
          const pre = document.createElement('pre');
          pre.className = 'json-block';
          const code = document.createElement('code');
          code.className = 'language-json hljs';
          code.textContent = jsonStr;
          if (window.hljs && hljs.highlightElement) {
            hljs.highlightElement(code);
          }
          pre.appendChild(code);
          return pre.outerHTML;
        } catch (e) {
          // Not valid JSON, continue with markdown parsing
        }
      }

      let html = escapeHtml(text);

      // Parse tables (must be done before other formatting)
      html = html.replace(/(^|\\n)\\|(.+)\\|\\n\\|([:\\-\\| ]+)\\|\\n((?:\\|.+\\|\\n?)+)/g, (match, prefix, header, separator, rows) => {
        const headerCells = header.split('|').map(cell => cell.trim()).filter(cell => cell);
        const rowLines = rows.trim().split('\\n');

        let table = '<table class="md-table"><thead><tr>';
        headerCells.forEach(cell => {
          table += `<th>${parseInlineMarkdown(cell)}</th>`;
        });
        table += '</tr></thead><tbody>';

        rowLines.forEach(row => {
          const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell);
          table += '<tr>';
          cells.forEach(cell => {
            table += `<td>${parseInlineMarkdown(cell)}</td>`;
          });
          table += '</tr>';
        });

        table += '</tbody></table>';
        return prefix + table + '\\n';
      });

      // Parse headers (with inline formatting)
      html = html.replace(/^### (.+)$/gm, (match, content) => {
        return `<h3 class="md-h3">${parseInlineMarkdown(content)}</h3>`;
      });
      html = html.replace(/^## (.+)$/gm, (match, content) => {
        return `<h2 class="md-h2">${parseInlineMarkdown(content)}</h2>`;
      });
      html = html.replace(/^# (.+)$/gm, (match, content) => {
        return `<h1 class="md-h1">${parseInlineMarkdown(content)}</h1>`;
      });

      // Parse horizontal rules
      html = html.replace(/\\n---\\n/g, '\\n<hr class="md-hr">\\n');

      // Parse blockquotes (with inline formatting)
      html = html.replace(/^&gt; (.+)$/gm, (match, content) => {
        return `<blockquote class="md-blockquote">${parseInlineMarkdown(content)}</blockquote>`;
      });

      // Parse unordered lists (at start or after newline)
      html = html.replace(/(^|\\n)((?:[-*+] .+(?:\\n|$))+)/g, (match, prefix, list) => {
        const items = list.trim().split('\\n').map(line => {
          const content = line.replace(/^[-*+] /, '');
          return `<li>${parseInlineMarkdown(content)}</li>`;
        }).join('');
        return `${prefix}<ul class="md-list">${items}</ul>\\n`;
      });

      // Parse ordered lists (at start or after newline)
      html = html.replace(/(^|\\n)((?:\\d+\\. .+(?:\\n|$))+)/g, (match, prefix, list) => {
        const items = list.trim().split('\\n').map(line => {
          const content = line.replace(/^\\d+\\. /, '');
          return `<li>${parseInlineMarkdown(content)}</li>`;
        }).join('');
        return `${prefix}<ol class="md-list">${items}</ol>\\n`;
      });

      // Apply inline formatting to remaining text (after block elements are processed)
      // Split by HTML tags to avoid processing inside them
      const parts = html.split(/(<[^>]+>)/g);
      html = parts.map((part, i) => {
        // Only process text parts (odd indices are tags)
        if (i % 2 === 0 && !part.match(/^</) && part.trim()) {
          return parseInlineMarkdown(part);
        }
        return part;
      }).join('');

      // Preserve line breaks
      html = html.replace(/\\n/g, '<br>');

      return html;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function createCodeBox(codeContent, language) {
      const box = document.createElement('div');
      box.className = 'code-box';

      const header = document.createElement('div');
      header.className = 'code-header';
      header.textContent = (language || 'Code').trim() || 'Code';
      box.appendChild(header);

      const pre = document.createElement('pre');
      const codeEl = document.createElement('code');
      codeEl.textContent = codeContent;
      highlightCodeElement(codeEl, language);
      pre.appendChild(codeEl);
      box.appendChild(pre);

      const copyBtn = createCopyButton(() => codeContent);
      copyBtn.classList.add('code-copy-btn');
      box.appendChild(copyBtn);

      return box;
    }

    function buildMessage(role, initialText = '') {
      let currentText = initialText;

      const wrapper = document.createElement('div');
      wrapper.className = 'message ' + role;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = role === 'user' ? 'You' : 'AI';
      const bubble = document.createElement('div');
      bubble.className = 'message-content';
      const body = document.createElement('div');
      body.className = 'message-body';
      const copyBtn = createCopyButton(() => currentText);
      bubble.appendChild(body);
      bubble.appendChild(copyBtn);
      wrapper.appendChild(avatar);
      wrapper.appendChild(bubble);

      function setText(newText) {
        currentText = newText;
        if (role === 'assistant') {
          renderAssistantSegments(body, currentText);
        } else {
          body.textContent = currentText;
        }
      }

      setText(initialText);

      return { wrapper, bubble, body, setText };
    }

    function createMessageGroup(prompt) {
      const group = document.createElement('div');
      group.className = 'message-group';

      const userMsg = buildMessage('user', prompt);
      const assistantMsg = buildMessage('assistant', '');
      assistantMsg.wrapper.classList.add('streaming');

      const thought = document.createElement('div');
      thought.className = 'thought-inline';
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'thought-toggle';
      toggle.textContent = 'Thought & Action Log';
      toggle.setAttribute('aria-expanded', 'false');
      const content = document.createElement('div');
      content.className = 'content';
      thought.appendChild(toggle);
      thought.appendChild(content);
      const thoughtCopyBtn = createCopyButton(() => content.textContent || '', 'Copy log');
      thoughtCopyBtn.classList.add('thought-copy-btn');
      thoughtCopyBtn.disabled = true;
      thought.appendChild(thoughtCopyBtn);

      group.appendChild(userMsg.wrapper);
      group.appendChild(thought);
      group.appendChild(assistantMsg.wrapper);

      chatHistory.appendChild(group);
      scrollChatToBottom();

      const state = {
        reasoning: content,
        assistantMessage: assistantMsg,
        assistantWrapper: assistantMsg.wrapper,
        toggle,
        thought,
        thoughtCopyBtn,
        userThoughtPreference: null,
      };

      toggle.addEventListener('click', () => {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        const next = !expanded;
        state.userThoughtPreference = next;
        applyThoughtVisibility(state, next);
      });

      return state;
    }

    function applyThoughtVisibility(group, expanded) {
      if (!group || !group.reasoning || !group.toggle) return;
      group.toggle.setAttribute('aria-expanded', String(expanded));
      group.reasoning.classList.toggle('visible', expanded);
      if (group.thought) {
        group.thought.classList.toggle('expanded', expanded);
      }
      if (group.thoughtCopyBtn) {
        const hasContent = Boolean(group.reasoning.textContent && group.reasoning.textContent.trim());
        group.thoughtCopyBtn.disabled = !hasContent;
      }
      if (!expanded) {
        group.reasoning.scrollTop = 0;
      }
    }

    function autoThoughtVisibility(group, expanded) {
      if (!group || group.userThoughtPreference !== null) return;
      applyThoughtVisibility(group, expanded);
    }

    function resetStreamState() {
      if (es) {
        es.close();
        es = null;
      }
      if (streamOpenTimeout) {
        clearTimeout(streamOpenTimeout);
        streamOpenTimeout = null;
      }
      isStreaming = false;
      streamOpened = false;
      fallbackActive = false;
      form.classList.remove('disabled');
      promptInput.disabled = false;
      sendButton.disabled = false;
      promptInput.focus();
      if (activeGroup && activeGroup.assistantWrapper) {
        activeGroup.assistantWrapper.classList.remove('streaming');
      }
    }

    function buildStreamURL(prompt) {
      const params = new URLSearchParams();
      params.set('prompt', prompt);
      if (currentSelectedToolList && currentSelectedToolList.length > 0) {
        params.set('tools', currentSelectedToolList.join(','));
      }
      if (agentOverride) {
        params.set('agent', agentOverride);
      }
      return `/stream?${params.toString()}`;
    }

    function triggerSyncFallback(reason, errorDetails) {
      if (fallbackActive) {
        return;
      }
      fallbackActive = true;
      if (streamOpenTimeout) {
        clearTimeout(streamOpenTimeout);
        streamOpenTimeout = null;
      }
      if (es) {
        try {
          es.close();
        } catch (closeError) {
          console.warn('[Stream] Failed to close EventSource before fallback', closeError);
        }
        es = null;
      }
      reportClientLog('warning', 'Triggering sync fallback', {
        reason,
        error: errorDetails ? (errorDetails.message || String(errorDetails)) : null,
        promptLength: currentPrompt.length,
      });
      startSyncFallback(currentPrompt, reason);
    }

    async function startSyncFallback(prompt, reason) {
      if (!prompt) {
        reportClientLog('error', 'Sync fallback aborted - empty prompt', { reason });
        resetStreamState();
        return;
      }

      try {
        reportClientLog('info', 'Starting sync fallback', {
          reason,
          agent: agentOverride || 'react',
          tools: currentSelectedToolList,
        });
        const payload = {
          prompt,
          tools: currentSelectedToolList,
        };
        if (agentOverride) {
          payload.agent = agentOverride;
        }
        const response = await fetch('/chat_sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`chat_sync failed with status ${response.status}`);
        }
        const data = await response.json();
        reasoningBuffer = data.reasoning || '';
        finalBuffer = data.final || '';
        streamCompleted = true;
        if (activeGroup && activeGroup.reasoning) {
          activeGroup.reasoning.textContent = reasoningBuffer;
          if (reasoningBuffer.trim()) {
            autoThoughtVisibility(activeGroup, true);
            if (activeGroup.thoughtCopyBtn) {
              activeGroup.thoughtCopyBtn.disabled = false;
            }
          }
        }
        if (activeGroup && activeGroup.assistantMessage) {
          activeGroup.assistantMessage.setText(finalBuffer || '[No response]');
        }
        hasFinalStarted = true;
        persistHistory();
        reportClientLog('info', 'Sync fallback completed', {
          finalLength: finalBuffer.length,
          reasoningLength: reasoningBuffer.length,
        });
      } catch (syncError) {
        streamCompleted = true;
        const message = `[ERROR] Sync fallback failed: ${syncError.message}`;
        if (activeGroup && activeGroup.assistantMessage) {
          activeGroup.assistantMessage.setText(message);
        }
        reportClientLog('error', 'Sync fallback failed', {
          reason,
          error: syncError.message,
        });
      } finally {
        fallbackActive = false;
        resetStreamState();
      }
    }

    function startStream(prompt) {
      console.log('[Stream] startStream called with prompt:', prompt);
      reportClientLog('info', 'Opening EventSource stream', {
        promptLength: prompt.length,
        agent: agentOverride || 'react',
        tools: currentSelectedToolList,
      });
      if (es) {
        console.log('[Stream] Closing existing EventSource');
        es.close();
      }

      const url = buildStreamURL(prompt);
      console.log('[Stream] Connecting to:', url);
      try {
        es = new EventSource(url);
      } catch (error) {
        console.error('[Stream] Failed to create EventSource', error);
        reportClientLog('error', 'EventSource constructor failed', { message: error.message });
        triggerSyncFallback('eventsource-constructor-error', error);
        return;
      }
      isStreaming = true;
      hasFinalStarted = false;
      streamOpened = false;
      streamCompleted = false;

      if (streamOpenTimeout) {
        clearTimeout(streamOpenTimeout);
      }
      streamOpenTimeout = setTimeout(() => {
        if (!streamOpened && !fallbackActive) {
          console.warn('[Stream] EventSource open timeout, switching to fallback');
          triggerSyncFallback('eventsource-open-timeout');
        }
      }, 4000);

      es.addEventListener('open', () => {
        streamOpened = true;
        if (streamOpenTimeout) {
          clearTimeout(streamOpenTimeout);
          streamOpenTimeout = null;
        }
        console.log('[Stream] Connection opened');
        reportClientLog('info', 'EventSource connection opened');
      });

      es.addEventListener('reasoning', (event) => {
        const text = JSON.parse(event.data);
        reasoningBuffer += text;
        if (activeGroup && activeGroup.reasoning) {
          autoThoughtVisibility(activeGroup, true);
          activeGroup.reasoning.textContent = reasoningBuffer;
          activeGroup.reasoning.scrollTop = activeGroup.reasoning.scrollHeight;
          if (activeGroup.thoughtCopyBtn) {
            activeGroup.thoughtCopyBtn.disabled = !reasoningBuffer.trim();
          }
        }
        scrollChatToBottom();
      });

      es.addEventListener('final', (event) => {
        const text = JSON.parse(event.data);
        if (!text) return;
        finalBuffer += text;
        if (!hasFinalStarted) {
          autoThoughtVisibility(activeGroup, false);
          hasFinalStarted = true;
        }
        if (activeGroup && activeGroup.assistantMessage) {
          activeGroup.assistantMessage.setText(finalBuffer);
          scrollChatToBottom();
        }
      });

      es.addEventListener('done', () => {
        console.log('[Stream] Done event received');
        streamCompleted = true;
        reportClientLog('info', 'Stream done event', {
          finalLength: finalBuffer.length,
          reasoningLength: reasoningBuffer.length,
        });
        resetStreamState();
        persistHistory();
      });

      es.onerror = (error) => {
        console.error('[Stream] Error:', error);
        reportClientLog('error', 'EventSource stream error', {
          readyState: es && typeof es.readyState !== 'undefined' ? es.readyState : 'unknown',
          message: error && error.message ? error.message : null,
        });
        if (streamCompleted || fallbackActive) {
          resetStreamState();
          return;
        }
        triggerSyncFallback('eventsource-error', error);
      };
    }

    async function loadHistory() {
      try {
        const response = await fetch('/history');
        if (!response.ok) throw new Error('Failed to load history');
        const data = await response.json();
        renderHistory(data.history || []);
      } catch (error) {
        console.error(error);
        reportClientLog('error', 'Failed to load history', {
          message: error.message,
        });
      }
    }

    function truncateText(text, maxLength = 48) {
      if (!text) return '';
      return text.length > maxLength ? text.slice(0, maxLength - 1) + '…' : text;
    }

    function loadConversationFromHistory(item) {
      // Clear current chat
      chatHistory.innerHTML = '';

      // Reset stream state
      if (es) es.close();
      resetStreamState();

      // Create message group for historical conversation
      const group = document.createElement('div');
      group.className = 'message-group';

      // Add user message
      const userMsg = buildMessage('user', item.prompt);
      group.appendChild(userMsg.wrapper);

      // Add reasoning if available
      if (item.reasoning && item.reasoning.trim()) {
        const thought = document.createElement('div');
        thought.className = 'thought-inline';
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'thought-toggle';
        toggle.textContent = 'Thought & Action Log';
        toggle.setAttribute('aria-expanded', 'false');
        const content = document.createElement('div');
        content.className = 'content';
        content.textContent = item.reasoning;
        thought.appendChild(toggle);
        thought.appendChild(content);
        const thoughtCopyBtn = createCopyButton(() => content.textContent || '', 'Copy log');
        thoughtCopyBtn.classList.add('thought-copy-btn');
        thought.appendChild(thoughtCopyBtn);

        toggle.addEventListener('click', () => {
          const expanded = toggle.getAttribute('aria-expanded') === 'true';
          toggle.setAttribute('aria-expanded', String(!expanded));
          content.classList.toggle('visible', !expanded);
          thought.classList.toggle('expanded', !expanded);
        });

        group.appendChild(thought);
      }

      // Add assistant response
      const assistantMsg = buildMessage('assistant', item.response);
      group.appendChild(assistantMsg.wrapper);

      chatHistory.appendChild(group);
      scrollChatToBottom();

      // Clear the prompt input
      promptInput.value = '';
      promptInput.focus();
    }

    function renderHistory(items) {
      if (!items.length) {
        historyList.innerHTML = '<div class="history-empty">No stored chats yet.</div>';
        return;
      }
      historyList.innerHTML = '';
      items.forEach(item => {
        const block = document.createElement('div');
        block.className = 'history-item';
        block.textContent = truncateText(item.prompt, 48);
        block.title = item.prompt; // Show full text on hover
        block.addEventListener('click', () => {
          loadConversationFromHistory(item);
        });
        historyList.appendChild(block);
      });
    }

    async function persistHistory() {
      if (!currentPrompt.trim() || !finalBuffer.trim()) return;
      try {
        await fetch('/history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: currentPrompt,
            response: finalBuffer,
            reasoning: reasoningBuffer,
          }),
        });
        loadHistory();
      } catch (error) {
        console.error('Failed to save history', error);
        reportClientLog('error', 'Failed to persist chat history', {
          message: error.message,
        });
      }
    }

    function handlePromptSubmit(event) {
      console.log('[Form] Submit trigger received');
      console.log('[Form] Event type:', event ? event.type : 'no event');
      console.log('[Form] isStreaming:', isStreaming);
      console.log('[Form] promptInput value:', promptInput.value);

      if (event) {
        event.preventDefault();
        if (typeof event.stopPropagation === 'function') {
          event.stopPropagation();
        }
      }

      if (isStreaming) {
        console.log('[Form] Already streaming, ignoring submit');
        return;
      }

      const prompt = promptInput.value.trim();
      console.log('[Form] Trimmed prompt:', prompt);

      if (!prompt) {
        console.log('[Form] Empty prompt, ignoring');
        reportClientLog('warning', 'Empty prompt submission ignored');
        return;
      }

      console.log('[Form] Processing prompt:', prompt);
      console.log('[Form] selectedTools:', selectedTools);
      currentPrompt = prompt;
      reasoningBuffer = '';
      finalBuffer = '';
      hasFinalStarted = false;
      streamCompleted = false;
      fallbackActive = false;

      // Safely convert selectedTools to array
      try {
        currentSelectedToolList = selectedTools ? Array.from(selectedTools) : [];
        console.log('[Form] Selected tools array:', currentSelectedToolList);
      } catch (error) {
        console.error('[Form] Error converting selectedTools:', error);
        currentSelectedToolList = [];
      }

      reportClientLog('info', 'handlePromptSubmit', {
        promptLength: prompt.length,
        tools: currentSelectedToolList,
        agent: agentOverride || 'react',
      });

      promptInput.value = '';
      promptInput.disabled = true;
      sendButton.disabled = true;
      form.classList.add('disabled');

      try {
        activeGroup = createMessageGroup(prompt);
        console.log('[Form] Message group created:', activeGroup);
      } catch (error) {
        console.error('[Form] Error creating message group:', error);
        resetStreamState();
        return;
      }

      try {
        console.log('[Form] Starting stream...');
        startStream(prompt);
      } catch (error) {
        console.error('[Form] Error starting stream:', error);
        resetStreamState();
        return;
      }
    }

    form.addEventListener('submit', handlePromptSubmit);

    // Add explicit click handler for send button to ensure it works
    sendButton.addEventListener('click', (event) => {
      console.log('[Send Button] Click detected');
      event.preventDefault(); // Prevent default form submission
      handlePromptSubmit(event);
    });

    newChatBtn.addEventListener('click', () => {
      if (es) es.close();
      chatHistory.innerHTML = '';
      reasoningBuffer = '';
      finalBuffer = '';
      currentPrompt = '';
      activeGroup = null;
      hasFinalStarted = false;
      resetStreamState();
      promptInput.value = '';
      promptInput.focus();
    });

    promptInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!isStreaming && promptInput.value.trim()) {
          // Use dispatchEvent for better compatibility than requestSubmit
          handlePromptSubmit(event);
        }
      }
    });

    loadHistory();

    // Tools Selection
    const toolsToggle = document.getElementById('tools-toggle');
    const toolsDropdown = document.getElementById('tools-dropdown');
    const toolsList = document.getElementById('tools-list');
    const toolsLabel = document.getElementById('tools-label');
    const toolsCountLabel = document.getElementById('tools-count');
    const toolsClear = document.getElementById('tools-clear');

    // Check if all required tools elements exist
    const toolsUIAvailable = toolsToggle && toolsDropdown && toolsList &&
                             toolsLabel && toolsCountLabel && toolsClear;

    if (!toolsUIAvailable) {
      console.warn('[Tools] Some tools UI elements are missing. Tools selection will be disabled.');
      console.warn('[Tools] Missing elements:', {
        toolsToggle: !!toolsToggle,
        toolsDropdown: !!toolsDropdown,
        toolsList: !!toolsList,
        toolsLabel: !!toolsLabel,
        toolsCountLabel: !!toolsCountLabel,
        toolsClear: !!toolsClear
      });
    }

    function normalizeToolsList(rawTools) {
      if (!Array.isArray(rawTools)) return [];
      const seen = new Set();
      const cleaned = [];
      rawTools.forEach((tool) => {
        if (!tool || !tool.name) return;
        const key = tool.name.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        const metadata = tool.metadata || {};
        let category = tool.category || metadata.category || null;
        const metaCategories = metadata.categories;
        if (!category && Array.isArray(metaCategories) && metaCategories.length) {
          category = metaCategories[0];
        }
        cleaned.push({
          name: tool.name,
          description: tool.description || 'No description available',
          category: category || 'general',
          source: tool.source || 'registry',
        });
      });
      cleaned.sort((a, b) => a.name.localeCompare(b.name));
      return cleaned;
    }

    // Fetch available tools from backend
    async function loadTools() {
      try {
        console.log('[Tools] Fetching tools from /tools endpoint...');
        const response = await fetch('/tools');
        if (!response.ok) {
          console.error('[Tools] Response not OK:', response.status, response.statusText);
          throw new Error(`Failed to load tools: ${response.status}`);
        }
        const data = await response.json();
        console.log('[Tools] Received data:', data);
        console.log('[Tools] Raw tools count:', data.tools ? data.tools.length : 0);
        availableTools = normalizeToolsList(data.tools);
        console.log('[Tools] Normalized tools count:', availableTools.length);
        console.log('[Tools] Available tools:', availableTools);
        renderTools();
      } catch (error) {
        console.error('[Tools] Failed to load tools:', error);
        reportClientLog('error', 'Failed to load tools list', {
          message: error.message,
        });
        availableTools = [];
        renderTools();
      } finally {
        updateToolsLabel();
      }
    }

    function renderTools() {
      if (!toolsUIAvailable) return;

      const totalLabel = availableTools.length
        ? `Available Tools (${availableTools.length})`
        : 'Available Tools';
      if (toolsCountLabel) {
        toolsCountLabel.textContent = totalLabel;
      }

      if (!availableTools.length) {
        toolsList.innerHTML = '<div style="padding:1rem;text-align:center;color:var(--text-subtle);font-size:0.8rem;">No tools available</div>';
        selectedTools.clear();
        return;
      }

      const validNames = new Set(availableTools.map((tool) => tool.name));
      selectedTools.forEach((name) => {
        if (!validNames.has(name)) {
          selectedTools.delete(name);
        }
      });

      toolsList.innerHTML = '';
      availableTools.forEach((tool) => {
        const item = document.createElement('div');
        item.className = 'tool-item';
        item.dataset.toolName = tool.name;

        const checkbox = document.createElement('div');
        checkbox.className = 'tool-checkbox';
        if (selectedTools.has(tool.name)) {
          checkbox.classList.add('checked');
        }

        const info = document.createElement('div');
        info.className = 'tool-info';

        const name = document.createElement('div');
        name.className = 'tool-name';
        name.textContent = tool.name;

        const desc = document.createElement('div');
        desc.className = 'tool-desc';
        desc.textContent = tool.description || 'No description available';

        const metaParts = [];
        if (tool.category) metaParts.push(tool.category);
        if (tool.source && tool.source !== 'registry') metaParts.push(tool.source);

        info.appendChild(name);
        if (metaParts.length) {
          const meta = document.createElement('div');
          meta.className = 'tool-meta';
          meta.textContent = metaParts.join(' • ');
          info.appendChild(meta);
        }
        info.appendChild(desc);
        item.appendChild(checkbox);
        item.appendChild(info);

        item.addEventListener('click', () => {
          toggleTool(tool.name);
          checkbox.classList.toggle('checked');
          updateToolsLabel();
        });

        toolsList.appendChild(item);
      });
    }

    function toggleTool(toolName) {
      if (selectedTools.has(toolName)) {
        selectedTools.delete(toolName);
      } else {
        selectedTools.add(toolName);
      }
    }

    function updateToolsLabel() {
      if (!toolsUIAvailable) return;

      const count = selectedTools.size;
      const total = availableTools.length;
      if (!total) {
        toolsLabel.textContent = count ? `Tools (${count})` : 'Tools';
      } else if (count === 0) {
        toolsLabel.textContent = `Tools (0/${total})`;
      } else {
        toolsLabel.textContent = `Tools (${count}/${total})`;
      }

      if (count === 0) {
        toolsToggle.classList.remove('active');
      } else {
        toolsToggle.classList.add('active');
      }
    }

    // Only setup tools UI if all elements are available
    if (toolsUIAvailable) {
      // Toggle tools dropdown
      toolsToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = toolsDropdown.classList.toggle('open');
        toolsToggle.setAttribute('aria-expanded', String(isOpen));
      });

      // Clear all selected tools
      toolsClear.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedTools.clear();
        renderTools();
        updateToolsLabel();
      });

      // Close dropdown when clicking outside
      document.addEventListener('click', (e) => {
        if (!toolsToggle.contains(e.target) && !toolsDropdown.contains(e.target)) {
          if (toolsDropdown.classList.contains('open')) {
            toolsDropdown.classList.remove('open');
            toolsToggle.setAttribute('aria-expanded', 'false');
          }
        }
      });

      // Load tools on page load
      console.log('[Init] Starting tools loading...');
      loadTools().then(() => {
        console.log('[Init] Tools loading completed');
      }).catch(error => {
        console.error('[Init] Tools loading failed:', error);
      });
    } else {
      console.log('[Init] Tools UI not available, skipping tools initialization');
    }
    // Account Menu & Settings Modal
    const accountButton = document.getElementById('account-button');
    const accountDropdown = document.getElementById('account-dropdown');
    const settingsButton = document.getElementById('settings-button');
    const settingsModal = document.getElementById('settings-modal');
    const modalOverlay = document.getElementById('modal-overlay');
    const modalClose = document.getElementById('modal-close');
    const cancelSettings = document.getElementById('cancel-settings');
    const saveSettings = document.getElementById('save-settings');

    const deepseekKeyInput = document.getElementById('deepseek-key');
    const tavilyKeyInput = document.getElementById('tavily-key');
    const openaiKeyInput = document.getElementById('openai-key');

    // Toggle account dropdown
    accountButton.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = accountDropdown.classList.contains('open');
      accountDropdown.classList.toggle('open');
      accountButton.setAttribute('aria-expanded', String(!isOpen));
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!accountButton.contains(e.target) && !accountDropdown.contains(e.target)) {
        accountDropdown.classList.remove('open');
        accountButton.setAttribute('aria-expanded', 'false');
      }
    });

    // Open settings modal
    function openSettingsModal() {
      settingsModal.classList.add('open');
      loadAPIKeys();
      accountDropdown.classList.remove('open');
      accountButton.setAttribute('aria-expanded', 'false');
    }

    // Close settings modal
    function closeSettingsModal() {
      settingsModal.classList.remove('open');
    }

    settingsButton.addEventListener('click', openSettingsModal);
    modalClose.addEventListener('click', closeSettingsModal);
    cancelSettings.addEventListener('click', closeSettingsModal);
    modalOverlay.addEventListener('click', closeSettingsModal);

    // Prevent modal content clicks from closing modal
    document.querySelector('.modal-content').addEventListener('click', (e) => {
      e.stopPropagation();
    });

    // Load API keys from localStorage
    function loadAPIKeys() {
      deepseekKeyInput.value = localStorage.getItem('DEEPSEEK_API_KEY') || '';
      tavilyKeyInput.value = localStorage.getItem('TAVILY_API_KEY') || '';
      openaiKeyInput.value = localStorage.getItem('OPENAI_API_KEY') || '';
    }

    // Save API keys to localStorage
    function saveAPIKeys() {
      const deepseekKey = deepseekKeyInput.value.trim();
      const tavilyKey = tavilyKeyInput.value.trim();
      const openaiKey = openaiKeyInput.value.trim();

      if (deepseekKey) {
        localStorage.setItem('DEEPSEEK_API_KEY', deepseekKey);
      } else {
        localStorage.removeItem('DEEPSEEK_API_KEY');
      }

      if (tavilyKey) {
        localStorage.setItem('TAVILY_API_KEY', tavilyKey);
      } else {
        localStorage.removeItem('TAVILY_API_KEY');
      }

      if (openaiKey) {
        localStorage.setItem('OPENAI_API_KEY', openaiKey);
      } else {
        localStorage.removeItem('OPENAI_API_KEY');
      }

      closeSettingsModal();
    }

    saveSettings.addEventListener('click', saveAPIKeys);

    // Clear history button handler
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    clearHistoryBtn.addEventListener('click', async () => {
      if (confirm('Are you sure you want to clear all chat history? This action cannot be undone.')) {
        try {
          const response = await fetch('/clear_history', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          });
          const data = await response.json();
          if (response.ok) {
            alert('All chat history has been cleared successfully');
            // Clear the chat display
            chatHistory.innerHTML = '';
          } else {
            alert('Failed to clear history: ' + (data.error || 'Unknown error'));
          }
        } catch (error) {
          console.error('Error clearing history:', error);
          alert('Failed to clear history: ' + error.message);
        }
      }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && settingsModal.classList.contains('open')) {
        closeSettingsModal();
      }
    });

      console.log('[Init] Script completed successfully');
      reportClientLog('info', 'Chat UI initialized');
    } catch (error) {
      console.error('[Init] CRITICAL ERROR in main script:', error);
      console.error('[Init] Stack trace:', error.stack);
      reportClientLog('critical', 'Chat UI initialization failed', {
        error: error && error.stack ? error.stack : error ? error.message : 'unknown',
      });
    }
  }); // End of DOMContentLoaded
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "GET":
        return {"history": fetch_chat_history()}

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    response_text = (data.get("response") or "").strip()
    reasoning_text = (data.get("reasoning") or "").strip()

    if not prompt or not response_text:
        return {"error": "Both prompt and response are required."}, 400

    entry_id = save_chat_entry(prompt, response_text, reasoning_text)
    return {"id": entry_id}, 201


@app.route("/clear_history", methods=["POST"])
def clear_history_route():
    """Clear all chat history."""
    success = clear_all_history()
    if success:
        return {"message": "All history cleared successfully"}, 200
    else:
        return {"error": "Failed to clear history"}, 500


@app.route("/tools", methods=["GET"])
def get_tools():
    """Return list of available tools for user selection."""
    logger.info("[Tools API] Received request for tools list")
    tools_map = {}

    def classify_category(tool) -> str:
        metadata = getattr(tool, "metadata", {}) or {}
        categories = metadata.get("categories")
        if isinstance(categories, (list, tuple)) and categories:
            return categories[0]
        return metadata.get("category") or metadata.get("type") or getattr(tool, "category", None) or "general"

    registries = []
    if enhanced_registry:
        registries.append(("enhanced", enhanced_registry))
        logger.info("[Tools API] Added enhanced_registry to registries list")
    if tool_registry:
        registries.append(("registry", tool_registry))
        logger.info("[Tools API] Added tool_registry to registries list")

    logger.info(f"[Tools API] Processing {len(registries)} registries")

    for source_label, registry_obj in registries:
        if not registry_obj:
            logger.warning(f"[Tools API] Registry '{source_label}' is None, skipping")
            continue
        tools_from_registry = list(registry_obj.list_tools())
        logger.info(f"[Tools API] Processing {len(tools_from_registry)} tools from '{source_label}' registry")
        for tool in tools_from_registry:
            if tool.name in tools_map:
                logger.debug(f"[Tools API] Skipping duplicate tool: {tool.name}")
                continue
            tools_map[tool.name] = {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "category": classify_category(tool),
                "source": source_label,
                "metadata": getattr(tool, "metadata", {}) or {},
            }

    tools_list = sorted(tools_map.values(), key=lambda item: item["name"].lower())
    logger.info(f"[Tools API] Returning {len(tools_list)} tools to frontend")
    return {"tools": tools_list}


@app.route("/client-log", methods=["POST"])
def client_log():
    """Receive client-side diagnostic logs and mirror them to the server console."""
    payload = request.get_data(cache=False, as_text=True) or ""
    data = request.get_json(silent=True)
    if data is None:
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {}

    level = str(data.get("level", "info")).lower()
    log_level = CLIENT_LOG_LEVELS.get(level, logging.INFO)
    message = data.get("message") or "Client log message"
    context = data.get("details") or data.get("context") or {}
    if not isinstance(context, dict):
        context = {"context": context}

    metadata = {
        "url": data.get("url"),
        "timestamp": data.get("timestamp"),
        **context,
    }
    logger.log(log_level, "[ClientLog] %s | %s", message, metadata)
    return {"status": "ok"}


@app.route("/router-metrics", methods=["GET"])
def router_metrics():
    return {
        "ab_test_enabled": AB_TEST_ENABLED,
        "hybrid_rollout": HYBRID_ROUTER_PERCENT,
        "variants": ROUTER_METRICS,
        "router_stats": router.get_stats() if router and hasattr(router, "get_stats") else {},
        "tool_registry": tool_registry.metadata_snapshot(),
    }


@app.route("/stream")
def stream():
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return "Missing 'prompt' query parameter", 400

    # Get selected tools from request
    tools_param = request.args.get("tools", "").strip()
    selected_tools = [t.strip() for t in tools_param.split(",") if t.strip()] if tools_param else None

    # Check if user wants to force a specific agent
    force_agent = request.args.get("agent", "").lower()

    prompt_preview = prompt[:200].replace("\n", " ")
    logger.info(
        "[Stream] Prompt received (len=%s, agent=%s, tools=%s): %s",
        len(prompt),
        force_agent or "auto",
        ", ".join(selected_tools) if selected_tools else "all",
        prompt_preview,
    )

    if force_agent == "legacy":
        generator = legacy_stream(prompt, "legacy", selected_tools)
    elif force_agent == "hybrid":
        variant = choose_router_variant(request)
        generator = hybrid_stream(prompt, variant, selected_tools)
    else:
        # Default to TrueReActAgent (true ReAct loop)
        generator = true_react_stream(prompt, "react", selected_tools)

    return Response(stream_with_context(generator), mimetype="text/event-stream")


def legacy_stream(prompt: str, variant: str, selected_tools=None):
    start = time.perf_counter()
    logger.info("[Legacy] Starting stream (prompt_len=%s)", len(prompt))

    # Note: legacy stream doesn't use tools, but we accept the parameter for consistency
    if selected_tools:
        logger.info("[Legacy] User selected tools (ignored): %s", selected_tools)

    stream = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    try:
        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning_chunk = getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                yield "event: reasoning\n"
                yield "data: " + json.dumps(reasoning_chunk) + "\n\n"

            final_chunk = getattr(delta, "content", None)
            if final_chunk:
                yield "event: final\n"
                yield "data: " + json.dumps(final_chunk) + "\n\n"

        yield "event: done\n"
        yield "data: \"[DONE]\"\n\n"
    except Exception as exc:
        logger.exception("[Legacy] Stream failed")
        err_msg = f"Server error: {str(exc)}"
        yield "event: final\n"
        yield "data: " + json.dumps("\\n[ERROR] " + err_msg) + "\n\n"
        yield "event: done\n"
        yield "data: \"[DONE]\"\n\n"
    finally:
        latency = (time.perf_counter() - start) * 1000
        record_metrics(variant, latency)


def hybrid_stream(prompt: str, variant: str, selected_tools=None):
    start = time.perf_counter()
    logger.info("[Hybrid] Starting stream (prompt_len=%s)", len(prompt))

    # If user selected specific tools, we should filter the agent's tool access
    # For now, we'll just log it and let the agent use all tools
    # TODO: Implement tool filtering in LangGraphReActAgent
    if selected_tools:
        logger.info("[Hybrid] User selected tools (not enforced): %s", selected_tools)

    plan_state = agent.plan(prompt)
    decision = plan_state.get("decision")
    tool_outputs = plan_state.get("tool_outputs", [])
    reasoning_text = format_reasoning(decision, tool_outputs)
    yield "event: reasoning\n"
    yield "data: " + json.dumps(reasoning_text) + "\n\n"

    try:
        stream = agent.stream_final(prompt, decision, tool_outputs)
        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning_chunk = getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                yield "event: reasoning\n"
                yield "data: " + json.dumps(reasoning_chunk) + "\n\n"

            final_chunk = getattr(delta, "content", None)
            if final_chunk:
                yield "event: final\n"
                yield "data: " + json.dumps(final_chunk) + "\n\n"

        yield "event: done\n"
        yield "data: \"[DONE]\"\n\n"
    except Exception as exc:
        logger.exception("[Hybrid] Stream failed")
        err_msg = f"Hybrid router error: {str(exc)}"
        yield "event: final\n"
        yield "data: " + json.dumps("\\n[ERROR] " + err_msg) + "\n\n"
        yield "event: done\n"
        yield "data: \"[DONE]\"\n\n"
    finally:
        latency = (time.perf_counter() - start) * 1000
        record_metrics(variant, latency, decision, tool_outputs)


def true_react_stream(prompt: str, variant: str, selected_tools=None):
    """Stream using TrueReActAgent with iterative ReAct loop."""
    if not true_react_agent:
        # Fallback to legacy if TrueReActAgent not available
        yield from legacy_stream(prompt, variant, selected_tools)
        return

    start = time.perf_counter()
    done_sent = False  # Track if we've sent the done event
    logger.info("[TrueReAct] Starting stream (prompt_len=%s)", len(prompt))

    # If user selected specific tools, filter the available tools
    if selected_tools:
        logger.info("[TrueReAct] User selected tools: %s", selected_tools)

    try:
        # Stream ReAct execution with optional tool filtering
        stream_kwargs = {"allowed_tools": selected_tools} if selected_tools else {}
        for event in true_react_agent.stream(prompt, **stream_kwargs):
            event_type = event.get("type")
            event_data = event.get("data")

            if event_type == "discovery":
                # Tool discovery event
                tools_found = event_data.get("tools_found", 0)
                tool_names = event_data.get("tool_names", [])
                discovery_msg = f"Discovered {tools_found} relevant tools"
                if tool_names:
                    discovery_msg += f": {', '.join(tool_names)}"
                discovery_msg += "\n\n"

                yield "event: reasoning\n"
                yield "data: " + json.dumps(discovery_msg) + "\n\n"

            elif event_type == "step":
                # ReAct step (Thought + Action)
                step_num = event_data.get("step_number", 0)
                thought = event_data.get("thought", "")
                action = event_data.get("action", "")

                step_msg = f"Step {step_num}: {thought}"
                if action and action.upper() != "FINISH":
                    step_msg += f"\nAction: {action}"
                step_msg += "\n\n"

                yield "event: reasoning\n"
                yield "data: " + json.dumps(step_msg) + "\n\n"

            elif event_type == "observation":
                # Tool execution result
                tool = event_data.get("tool", "")
                result = event_data.get("result", "")

                obs_msg = f"Executed {tool}"
                if result:
                    # Show truncated result
                    obs_msg += f"\nResult: {result[:200]}{'...' if len(result) > 200 else ''}"
                obs_msg += "\n\n"

                yield "event: reasoning\n"
                yield "data: " + json.dumps(obs_msg) + "\n\n"

            elif event_type == "reasoning":
                # Final response reasoning
                yield "event: reasoning\n"
                yield "data: " + json.dumps(event_data) + "\n\n"

            elif event_type == "final":
                # Final response content
                yield "event: final\n"
                yield "data: " + json.dumps(event_data) + "\n\n"

            elif event_type == "error":
                # Error occurred
                error_msg = f"Error in step {event_data.get('step_number', '?')}: {event_data.get('error', 'Unknown error')}"
                yield "event: final\n"
                yield "data: " + json.dumps("\\n[ERROR] " + error_msg) + "\n\n"

            elif event_type == "done":
                # Completion marker
                yield "event: done\n"
                yield "data: \"[DONE]\"\n\n"
                done_sent = True

        # CRITICAL FIX: Always send done event if not already sent
        if not done_sent:
            yield "event: done\n"
            yield "data: \"[DONE]\"\n\n"

    except Exception as exc:
        err_msg = f"TrueReAct agent error: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        yield "event: final\n"
        yield "data: " + json.dumps("\\n[ERROR] " + err_msg) + "\n\n"
        # Always send done event on error
        if not done_sent:
            yield "event: done\n"
            yield "data: \"[DONE]\"\n\n"
    finally:
        latency = (time.perf_counter() - start) * 1000
        # Record metrics (simplified - no decision/tool_outputs needed)
        ROUTER_METRICS[variant]["requests"] += 1
        prev_avg = ROUTER_METRICS[variant]["avg_latency_ms"]
        ROUTER_METRICS[variant]["avg_latency_ms"] = prev_avg * 0.95 + latency * 0.05


def normalize_tool_selection(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        return []
    cleaned = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def format_true_react_reasoning(result) -> str:
    if not result or not getattr(result, "steps", None):
        return "TrueReAct agent executed without intermediate reasoning."

    lines = []
    for step in result.steps:
        if not step:
            continue
        header = f"Step {step.step_number}: {step.thought or ''}".strip()
        if header:
            lines.append(header)
        if step.action and step.action.upper() != "FINISH":
            lines.append(f"Action: {step.action}")
        if step.observation:
            obs = step.observation
            if len(obs) > 400:
                obs = obs[:400] + "..."
            lines.append(f"Observation: {obs}")
        lines.append("")

    reasoning = (result.reasoning_content or "").strip()
    if reasoning:
        lines.append("Final reasoning:")
        lines.append(reasoning)

    formatted = "\n".join(line for line in lines if line.strip())
    return formatted or "TrueReAct agent finished without detailed reasoning."


def legacy_chat_sync(prompt: str, variant: str):
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    message = completion.choices[0].message
    final_text = message.content or ""
    reasoning_text = getattr(message, "reasoning_content", None) or "Legacy router responded directly."
    latency = (time.perf_counter() - start) * 1000
    record_metrics(variant, latency)
    return final_text, reasoning_text


def hybrid_chat_sync(prompt: str, variant: str, selected_tools=None):
    start = time.perf_counter()
    if selected_tools:
        logger.info("[HybridSync] User selected tools (not enforced): %s", selected_tools)
    run_result = agent.run(prompt) if agent else {}
    decision = run_result.get("decision")
    tool_outputs = run_result.get("tool_outputs", [])
    final_response = run_result.get("final_response") or {}
    final_text = final_response.get("content") or ""
    reasoning_text = format_reasoning(decision, tool_outputs)
    reasoning_chunk = final_response.get("reasoning_content")
    if reasoning_chunk:
        reasoning_text = f"{reasoning_text}\n\n{reasoning_chunk}".strip()
    latency = (time.perf_counter() - start) * 1000
    record_metrics(variant, latency, decision, tool_outputs)
    return final_text, reasoning_text


def true_react_chat_sync(prompt: str, variant: str, selected_tools=None):
    if not true_react_agent:
        return legacy_chat_sync(prompt, "legacy")

    start = time.perf_counter()
    result = true_react_agent.run(prompt, allowed_tools=selected_tools)
    final_text = getattr(result, "final_answer", "") or ""
    reasoning_text = format_true_react_reasoning(result)
    latency = (time.perf_counter() - start) * 1000
    ROUTER_METRICS[variant]["requests"] += 1
    prev_avg = ROUTER_METRICS[variant]["avg_latency_ms"]
    ROUTER_METRICS[variant]["avg_latency_ms"] = prev_avg * 0.95 + latency * 0.05
    ROUTER_METRICS[variant]["last_tier"] = "react"
    return final_text, reasoning_text


@app.route("/chat_sync", methods=["POST"])
def chat_sync():
    """Non-streaming chat endpoint used as a fallback when SSE fails."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return {"error": "Prompt is required."}, 400

    selected_tools = normalize_tool_selection(data.get("tools"))
    force_agent = (data.get("agent") or "").lower()

    logger.info(
        "[chat_sync] Prompt received (agent=%s, tools=%s, len=%s)",
        force_agent or "auto",
        ", ".join(selected_tools) if selected_tools else "all",
        len(prompt),
    )

    try:
        if force_agent == "legacy":
            final_text, reasoning_text = legacy_chat_sync(prompt, "legacy")
        elif force_agent == "hybrid":
            variant = choose_router_variant(request)
            final_text, reasoning_text = hybrid_chat_sync(prompt, variant, selected_tools)
        else:
            final_text, reasoning_text = true_react_chat_sync(prompt, "react", selected_tools)
    except Exception as exc:
        logger.exception("[chat_sync] Failed to process prompt")
        return {"error": str(exc)}, 500

    return {
        "final": final_text,
        "reasoning": reasoning_text,
    }


# Entry point is handled by __main__.py when running as a package
