# Erosolar AI Chat

AI-powered chat interface with DeepSeek, featuring advanced tool integrations, embeddings-based routing, and a modern web UI.

## Features

- 🤖 **AI-Powered Chat**: Leverages DeepSeek for intelligent conversations
- 🔧 **Tool Integration**: Built-in support for web search (Tavily) and weather tools
- 🧠 **Smart Embeddings**: Optional OpenAI embeddings for better tool matching
- 💾 **Persistent History**: Chat history stored safely in OS-specific user directories
- 🌐 **Modern Web UI**: Clean, responsive interface with syntax highlighting
- ⚙️ **Easy Configuration**: Simple API key management through settings
- 🗑️ **Privacy Controls**: Clear all chat history with one click

## Installation

Install Erosolar from PyPI:

```bash
pip install erosolar
```

## Quick Start

### 1. Set Required Environment Variables

```bash
# Required: DeepSeek API Key
export DEEPSEEK_API_KEY='your-deepseek-api-key-here'

# Optional: For better embeddings performance
export OPENAI_API_KEY='your-openai-api-key-here'

# Optional: For web search capabilities
export TAVILY_API_KEY='your-tavily-api-key-here'
```

### 2. Launch Erosolar

Simply run:

```bash
erosolar
```

This will:
- Start the Flask server on `http://localhost:5051`
- Automatically open your default web browser
- Display the chat interface

### 3. Start Chatting!

The interface will open automatically. You can:
- Ask questions and get AI-powered responses
- Use tools for web search and weather information
- View your chat history
- Configure API keys through the settings modal
- Clear chat history when needed

## Data Storage

Erosolar stores your chat history in OS-specific directories:

- **Windows**: `%APPDATA%\Erosolar\chat_history.db`
- **macOS**: `~/Library/Application Support/Erosolar/chat_history.db`
- **Linux**: `~/.local/share/erosolar/chat_history.db`

You can clear all history at any time through the Settings menu.

## Configuration

### API Keys

You can set API keys in two ways:

1. **Environment Variables** (recommended for security):
   ```bash
   export DEEPSEEK_API_KEY='your-key'
   export OPENAI_API_KEY='your-key'
   export TAVILY_API_KEY='your-key'
   ```

2. **Settings UI**: Click the settings icon in the web interface to configure keys (stored in browser localStorage)

### Custom Port

By default, Erosolar runs on port 5051. To use a different port:

```bash
PORT=8080 erosolar
```

## Development

### Installing from Source

```bash
git clone https://github.com/ErosolarAI/erosolar-pypi.git
cd erosolar-pypi
pip install -e .
```

### Optional Dependencies

For advanced features with LangGraph:

```bash
pip install erosolar[langgraph]
```

For development tools:

```bash
pip install erosolar[dev]
```

## Contributing

We welcome contributions! Please visit our repository to get involved:

**🔗 [Contribute on GitHub](https://github.com/ErosolarAI/erosolar-pypi)**

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Architecture

Erosolar uses a sophisticated multi-tier architecture:

- **Embeddings Router**: Intelligent tool selection based on query embeddings
- **ReAct Agent**: Reasoning and action framework for complex queries
- **Tool Registry**: Extensible system for adding new capabilities
- **Vector Store**: Efficient similarity search for tool matching

## Requirements

- Python 3.8 or higher
- DeepSeek API key (required)
- OpenAI API key (optional, for embeddings)
- Tavily API key (optional, for web search)

## Troubleshooting

### Server Won't Start

Ensure `DEEPSEEK_API_KEY` is set:
```bash
echo $DEEPSEEK_API_KEY  # Should output your key
```

### Port Already in Use

Change the port:
```bash
PORT=8080 erosolar
```

### Browser Doesn't Open

Manually navigate to `http://localhost:5051` in your browser.

## License

MIT License - see LICENSE file for details

## Links

- **PyPI**: https://pypi.org/project/erosolar/
- **GitHub**: https://github.com/ErosolarAI/erosolar-pypi
- **Issues**: https://github.com/ErosolarAI/erosolar-pypi/issues

## Support

For questions, issues, or feature requests, please:
- Open an issue on [GitHub](https://github.com/ErosolarAI/erosolar-pypi/issues)
- Check existing documentation
- Review closed issues for solutions

---

**Made with ❤️ by the ErosolarAI team**
