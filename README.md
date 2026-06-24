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
| `VISION_MODELS` | — | 模型回退列表，英文逗号分隔（见下文） |

### 模型回退 Router

设置 `VISION_MODELS` 环境变量，指定一组模型按顺序尝试：

```bash
VISION_MODELS=qwen-vl-max,qwen-vl-plus,qwen-3.7 python -m vision_mcp_server
```

当当前模型返回 **HTTP 429**（限流）、**quota exceeded**（配额超限）或 **insufficient balance**（余额不足）时，自动切换到下一个模型。所有模型都失败才返回错误。

未设置 `VISION_MODELS` 时行为不变（由 `VISION_MODEL` 或 Provider 默认决定）。

### 缓存

支持磁盘缓存，默认 7 天 TTL。环境变量控制：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VISION_CACHE_ENABLED` | `true` | `false` 时禁用缓存 |
| `VISION_CACHE_TTL` | `604800` | 缓存过期时间（秒） |
| `VISION_CACHE_DIR` | 按平台 | 缓存存储目录，Windows: `%LOCALAPPDATA%/vision-mcp-server/cache` |

### 各 Provider 默认值

| Provider | 模型 | 地址 |
|----------|------|------|
| `bailian` | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `openrouter` | `openai/gpt-4o` | `https://openrouter.ai/api/v1` |

## Tool: `image_understand`

```
image_understand(image_path: str, prompt: str | None = None, mode: str = "quick", force_refresh: bool = False) -> dict
```

### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image_path` | string | 必填 | 本地图片路径（PNG/JPG/GIF/WebP）或 HTTP URL |
| `prompt` | string | `None` | 自定义提问，不传则自动选择提示词 |
| `mode` | string | `"quick"` | `"quick"` 精简快速（5-10s）/ `"detailed"` 七维度详细分析 |
| `force_refresh` | bool | `false` | 跳过缓存，重新调用模型分析 |

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

## Tool: `image_compare`

对比两张图片（设计稿 vs 实现截图），返回结构化差异列表。

```
image_compare(expected_image: str, actual_image: str, mode: str = "ui", force_refresh: bool = False) -> dict
```

### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `expected_image` | string | 必填 | 预期图片路径（设计稿）或 URL |
| `actual_image` | string | 必填 | 实际图片路径（实现截图）或 URL |
| `mode` | string | `"ui"` | 对比模式，默认 UI 对比 |
| `force_refresh` | bool | `false` | 跳过缓存重新对比 |

### 返回

```json
{
  "summary": "发现 4 处差异",
  "differences": [
    {
      "type": "layout",
      "severity": "high",
      "area": "header",
      "expected": "导航栏高度约 64px",
      "actual": "导航栏高度约 48px"
    }
  ],
  "model": "qwen-vl-max",
  "cached": false
}
```

### 差异类型

| 类型 | 说明 |
|------|------|
| `layout` | 布局偏差：位置、间距、对齐 |
| `color` | 颜色差异：主色调、背景色 |
| `spacing` | 间距问题：padding、margin |
| `typography` | 字体差异：字号、字重、行高 |
| `missing` | 缺少元素 |
| `extra` | 多余元素 |

### 推荐流程

```
Claude 生成页面 → Playwright 截图 → Vision MCP 对比 → 输出差异 → Claude 修复
```

## 验证记录

`image_compare` 实测结果（设计稿 vs 含有意差异的实现截图）：

| 引入的差异 | 是否检出 |
|-----------|----------|
| 按钮颜色 #2563eb → #dc2626 | 检出 |
| 职位文字"高级"缺失 | 检出 |
| 头像 64px → 48px | 漏检 |
| 卡片圆角 12px → 4px | 漏检 |
| 背景色 #f0f2f5 → #fafafa | 漏检 |
| 姓名字号 20px → 16px | 漏检 |

**结论**：视觉模型在颜色偏差和文字内容差异上表现良好，但不擅长精确数值比较（像素级大小、圆角半径）。适合做"有没有差异 + 大概差在哪"的定性检查，像素级精确定位需叠加工具补充。

**已验证的完整链路**：
```
设计稿 HTML → Playwright 截图 → image_compare → 结构化 JSON 差异 → 人工确认
``

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
