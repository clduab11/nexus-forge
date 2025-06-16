# MCP Configuration Setup

This document explains how to configure the Model Context Protocol (MCP) servers for Nexus Forge.

## Initial Setup

1. Copy the template configuration:
   ```bash
   cp .roo/mcp.json.template .roo/mcp.json
   ```

2. Edit `.roo/mcp.json` and replace the placeholder values with your actual API keys:
   - `YOUR_PERPLEXITY_API_KEY_HERE` - Get from https://perplexity.ai
   - `YOUR_TAVILY_API_KEY_HERE` - Get from https://tavily.com  
   - `YOUR_FIRECRAWL_API_KEY_HERE` - Get from https://firecrawl.dev
   - `YOUR_GITHUB_PERSONAL_ACCESS_TOKEN_HERE` - Generate from GitHub Settings > Developer settings > Personal access tokens
   - `YOUR_SUPABASE_ACCESS_TOKEN_HERE` - Get from your Supabase dashboard

## MCP Servers Included

- **Ask Perplexity**: Web search and research capabilities
- **Redis**: Caching and data storage
- **Git Tools**: Git operations and version control
- **Tavily**: Advanced web crawling and extraction
- **Puppeteer**: Browser automation and testing
- **Firecrawl**: Intelligent web scraping
- **Sequential Thinking**: Advanced reasoning capabilities
- **Filesystem**: File system operations
- **GitHub**: GitHub repository management
- **Mem0**: Knowledge graph and memory management
- **Supabase**: Database and backend services

## Security Notes

- Never commit the actual `mcp.json` file with API keys
- The `.roo/` directory is included in `.gitignore` to prevent accidental commits
- Use environment variables or secure secret management in production
- Regularly rotate API keys for security