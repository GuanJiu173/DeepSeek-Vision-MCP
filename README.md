# Vision MCP Server

为 MCP 客户端（Claude Code 等）提供图片理解能力，通过阿里云百炼/OpenAI/OpenRouter 等视觉模型分析图片内容，返回面向软件开发的描述。

## 快速开始

```bash
pip install -e .
python -m vision_mcp_server
```

## 环境变量

### 必选

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 百炼 API Key（默认 Provider） |

### Provider 切换

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VISION_PROVIDER` | `bailian` | Provider 名称：`bailian` / `openai` / `openrouter` |
| `VISION_BASE_URL` | 按 Provider | 覆盖 API 端点地址 |
| `VISION_MODEL` | 按 Provider | 覆盖模型名称 |
| `VISION_API_KEY` | 按 Provider | 覆盖 API Key |
| `VISION_MAX_TOKENS` | `600` (quick) / `1500` (detailed) | 最大输出 token 数 |

### 各 Provider 默认值

| Provider | 模型 | 地址 |
|----------|------|------|
| `bailian` | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `openrouter` | `openai/gpt-4o` | `https://openrouter.ai/api/v1` |

## Tool: `image_understand`

```
image_understand(image_path: str, prompt: str | None = None, mode: str = "quick") -> dict
```

### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image_path` | string | 必填 | 本地图片路径（PNG/JPG/GIF/WebP）或 HTTP URL |
| `prompt` | string | `None` | 自定义提问，不传则自动选择提示词 |
| `mode` | string | `"quick"` | `"quick"` 精简快速（5-10s）/ `"detailed"` 七维度详细分析 |

### 返回

```json
{
  "description": "图片内容描述（Markdown 格式）",
  "model": "qwen-vl-max",
  "status": "success"
}
```

### 两种模式

| 模式 | 耗时 | 输出 | 适用场景 |
|------|------|------|----------|
| `quick` | 5-10s | 3-4 要点 | 日常识图、快速了解 |
| `detailed` | 15-30s | 七维度分析 | UI 还原、设计评审、图表提取 |

### detailed 模式的七个分析维度

1. UI 布局 — 整体结构、区块位置比例
2. 组件结构 — 按钮/表单/表格的层次嵌套
3. 页面层级 — 信息层级关系
4. 配色风格 — 主色调、设计风格、明暗模式
5. OCR 文字 — 所有可见文字及位置
6. 图表信息 — 图表类型、数据维度、关键数值
7. 前端实现特征 — CSS 框架、响应式、动画、图标库

## Claude Code 配置

项目根目录创建 `.mcp.json`：

```json
{
  "mcpServers": {
    "vision": {
      "command": "python",
      "args": ["-m", "vision_mcp_server"],
      "cwd": "E:/MCP",
      "env": {
        "DASHSCOPE_API_KEY": "sk-xxx"
      }
    }
  }
}
```

安装后 `/mcp` → Reconnect 生效。

## 项目结构

```
src/vision_mcp_server/
├── __init__.py
├── __main__.py       # 入口
├── server.py         # FastMCP + image_understand tool
├── vision.py         # 多 Provider 视觉客户端
└── image_utils.py    # 图片路径检测 + Base64 编码
```
