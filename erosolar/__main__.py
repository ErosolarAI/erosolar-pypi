"""
Main entry point for the Erosolar CLI command.
Launches the Flask server and opens a browser window.
"""

import os
import sys
import time
import webbrowser
import logging
from threading import Timer

logger = logging.getLogger(__name__)


def open_browser(port=5051):
    """Open web browser after a short delay to ensure server is running."""
    url = f"http://localhost:{port}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)


def main():
    """Main entry point for the erosolar command."""
    from .app import app

    port = int(os.environ.get("PORT", "5051"))

    # Print startup message
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    EROSOLAR AI Chat                      ║
╠══════════════════════════════════════════════════════════╣
║  Server starting on http://localhost:{port}            ║
║  Opening browser automatically...                        ║
║                                                          ║
║  Press Ctrl+C to stop the server                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Open browser after 1.5 seconds
    Timer(1.5, open_browser, args=(port,)).start()

    try:
        # Run Flask app
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,  # Set to False for production
            use_reloader=False  # Disable reloader to prevent double browser opening
        )
    except KeyboardInterrupt:
        print("\n\nShutting down Erosolar AI Chat. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"\nError starting server: {e}")
        print("Please ensure all required environment variables are set:")
        print("  - DEEPSEEK_API_KEY (required)")
        print("  - OPENAI_API_KEY (optional, for embeddings)")
        print("  - TAVILY_API_KEY (optional, for web search)")
        sys.exit(1)


if __name__ == "__main__":
    main()
