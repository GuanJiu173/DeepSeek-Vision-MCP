# Vision MCP Server

## 为什么需要它

许多 AI 模型（包括部分 Claude 版本）**不能直接"看"图片**。当你给它一张 UI 截图或设计稿时，它无法理解画面内容，也就无法帮你分析界面、提取文字、写代码复刻。

Vision MCP Server 填补了这个缺口——它充当 AI 的"眼睛"，让你在对话中直接丢图片给模型分析，同时提供缓存、模型回退、UI 对比等实用能力。

## 它能做什么

| 场景 | 用法 |
|------|------|
| 给我描述这张截图的内容 | `image_understand` — 5 秒快速识图 |
| 这张设计稿用了什么配色和字体 | `image_understand` + `mode="detailed"` — 七维度分析 |
| 实现截图和设计稿对比，找出差异 | `image_compare` — 自动发现 UI 偏差 |
| 额度用完了自动换模型接着跑 | `VISION_MODELS` 模型回退 |
| 同一张图昨天分析过，今天免费用 | 磁盘缓存，7 天 TTL |

### 典型工作流

```
"生成这个页面的 React 代码"
    ↓ Claude 写完
    ↓ Playwright 截图
    ↓ image_compare(设计稿, 截图)
    ↓ 返回差异 → Claude 修复
```

第一次分析 5-15 秒，第二次相同图 0.01 秒（命中原缓存），**不花二次钱**。

## 快速开始

```bash
pip install -e .
python -m vision_mcp_server
```

Claude Code 在项目根目录配 `.mcp.json` 即可用，不需要额外配置（前提是已设置 API Key）。

## 工具

### `image_understand` — 单图分析

```python
image_understand(image_path, prompt=None, mode="quick", force_refresh=False)
```

| 参数 | 说明 |
|------|------|
| `image_path` | 本地图片路径或 HTTP URL |
| `prompt` | 自定义提问，不传则自动选择提示词 |
| `mode` | `"quick"` 精简（5-10s）\| `"detailed"` 七维度（15-30s） |
| `force_refresh` | 跳过缓存 |

### `image_compare` — 双图对比

```python
image_compare(expected_image, actual_image, mode="ui", force_refresh=False)
```

返回结构化 JSON，包含差异类型、严重度、区域和预期/实际对比。

## 配置

### 环境变量

```bash
# 必选
DASHSCOPE_API_KEY=sk-xxx       # 百炼 API Key（默认 Provider）

# Provider 切换
VISION_PROVIDER=bailian         # bailian | openai | openrouter
VISION_API_KEY=sk-xxx           # 覆盖 API Key
VISION_BASE_URL=https://...     # 覆盖 API 端点

# 模型回退
VISION_MODELS=qwen-vl-max,qwen-vl-plus,qwen-3.7  # 按顺序尝试，遇到 quota/429 自动切换

# 缓存
VISION_CACHE_ENABLED=false      # 禁用缓存
VISION_CACHE_TTL=604800         # TTL（秒），默认 7 天
VISION_CACHE_DIR=/path/to/cache # 自定义缓存目录
```

### Provider 默认值

| Provider | 模型 | API 地址 |
|----------|------|----------|
| `bailian` | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `openrouter` | `openai/gpt-4o` | `https://openrouter.ai/api/v1` |

## 扩展

启动 HTTP 模式调试：

```bash
python -m vision_mcp_server --http
```
