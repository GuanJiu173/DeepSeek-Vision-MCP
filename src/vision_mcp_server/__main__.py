"""Vision MCP Server 入口

用法:
    python -m vision_mcp_server           # 启动 MCP Server (stdio 传输)
    python -m vision_mcp_server --http    # 启动 HTTP 传输模式（调试用）
"""

import sys


def main():
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"

    from vision_mcp_server.server import mcp

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
