"""Tavily MCP tools."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from ..tool_registry import MCPTool

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    body = {"api_key": TAVILY_API_KEY, **payload}
    response = requests.post(url, json=body, timeout=20)
    response.raise_for_status()
    return response.json()


def tavily_search_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
    query = inputs.get("query")
    if not query:
        raise ValueError("'query' is required")
    params = {
        "query": query,
        "search_depth": inputs.get("search_depth", "advanced"),
        "max_results": min(int(inputs.get("max_results", 5)), 10),
    }
    results = _post(TAVILY_SEARCH_URL, params)
    return {"results": results}


def tavily_extract_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
    urls = inputs.get("urls")
    if not urls:
        raise ValueError("'urls' is required")
    params = {
        "urls": urls,
        "include_images": inputs.get("include_images", False),
        "extract_depth": inputs.get("extract_depth", "basic"),
        "format": inputs.get("format", "markdown"),
    }
    # Add optional parameters if provided
    if "include_favicon" in inputs:
        params["include_favicon"] = inputs["include_favicon"]
    if "timeout" in inputs:
        params["timeout"] = inputs["timeout"]

    results = _post(TAVILY_EXTRACT_URL, params)
    return {"extract": results}


def build_tavily_tools() -> List[MCPTool]:
    return [
        MCPTool(
            name="tavily_search",
            description="Perform web search using Tavily with synthesis optimized for reasoning agents.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "default": "advanced",
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            output_schema={"type": "object", "properties": {"results": {"type": "object"}}},
            handler=tavily_search_handler,
            metadata={
                "categories": ["search", "web"],
                "latency_budget_ms": 800,
            },
        ),
        MCPTool(
            name="tavily_extract",
            description="Extract structured content from one or more URLs using Tavily's extraction API.",
            input_schema={
                "type": "object",
                "properties": {
                    "urls": {
                        "oneOf": [
                            {"type": "string", "format": "uri"},
                            {"type": "array", "items": {"type": "string", "format": "uri"}, "maxItems": 20}
                        ],
                        "description": "A single URL or array of URLs (max 20) to extract content from"
                    },
                    "include_images": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include extracted image URLs"
                    },
                    "include_favicon": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include favicon URL"
                    },
                    "extract_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "default": "basic",
                        "description": "Extraction depth (basic: 1 credit/5 extractions, advanced: 2 credits/5 extractions with tables)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "text"],
                        "default": "markdown",
                        "description": "Output format"
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 60.0,
                        "description": "Timeout in seconds"
                    }
                },
                "required": ["urls"],
            },
            output_schema={"type": "object", "properties": {"extract": {"type": "object"}}},
            handler=tavily_extract_handler,
            metadata={
                "categories": ["web", "extraction"],
            },
        ),
    ]
