---
title: 개발 및 SDK 가이드
parent: MCP (Model Context Protocol)
nav_order: 3
---

# 개발 및 SDK 가이드

MCP 서버를 구축하기 위해 다양한 언어별 SDK가 제공됩니다. 가장 널리 쓰이는 TypeScript와 Python 예시를 다룹니다.

## 1. TypeScript SDK (Node.js)

### 설치
```bash
npm install @modelcontextprotocol/sdk
```

### 서버 구현 예시 (Stdio 방식)
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server({
  name: "my-mcp-server",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
  },
});

// 1. 도구 목록 등록
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "calculate_sum",
    description: "두 숫자의 합을 계산합니다.",
    inputSchema: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
    },
  }],
}));

// 2. 도구 실행 핸들러
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "calculate_sum") {
    const { a, b } = request.params.arguments as { a: number, b: number };
    return {
      content: [{ type: "text", text: `결과: ${a + b}` }],
    };
  }
  throw new Error("Tool not found");
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

## 2. Python SDK

### 설치
```bash
pip install mcp
```

### 서버 구현 예시
```python
from mcp.server.fastmcp import FastMCP

# FastMCP를 사용하면 매우 간단하게 서버를 구축할 수 있습니다.
mcp = FastMCP("My API Server")

@mcp.tool()
def get_weather(location: str) -> str:
    """해당 위치의 날씨 정보를 가져옵니다."""
    # 실제 API 호출 로직...
    return f"{location}의 날씨는 맑음입니다."

if __name__ == "__main__":
    mcp.run()
```

## 3. 디버깅 도구: MCP Inspector
서버가 제대로 작동하는지 시각적으로 테스트할 수 있는 독립 도구입니다.
```bash
npx @modelcontextprotocol/inspector <server-start-command>
# 예: npx @modelcontextprotocol/inspector node build/index.js
```

## 4. 2025 최신 팁
- **Typed SDK**: 최신 SDK들은 타입 추론이 매우 강력해져서, JSON 스키마와 실제 함수의 타입 정의가 일치하도록 보장합니다.
- **Async/Await**: 모든 핸들러는 비동기 방식으로 동작하며, 원격 API 호출 중에도 호스트가 응답을 대기할 수 있도록 설계해야 합니다.
