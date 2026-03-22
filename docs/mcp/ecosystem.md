---
title: 에코시스템 및 통합
parent: MCP (Model Context Protocol)
nav_order: 4
---

# 에코시스템 및 통합

MCP는 공개 직후부터 AI 개발 도구와 서비스들 사이에서 급격히 채택되었습니다. 현재 어떤 환경에서 MCP를 사용할 수 있는지 정리합니다.

## 1. 주요 호스트(Hosts) 및 통합 현황

### Claude Desktop
- Anthropic의 공식 데스크탑 앱으로, 가장 먼저 MCP를 지원했습니다.
- 사용자의 설정 파일(`claude_desktop_config.json`)에 MCP 서버를 등록하여 즉시 도구로 활용할 수 있습니다.

### Cursor & VS Code
- **Cursor**: AI 특화 IDE인 Cursor는 MCP 서버를 통해 프로젝트 외부의 데이터나 특수 도구들을 에이전트에게 제공합니다.
- **VS Code**: GitHub Copilot이나 각종 AI 확장 프로그램들이 MCP 클라이언트를 내장하여 에디터의 능력을 확장하고 있습니다.

### Claude Code (CLI)
- Anthropic이 2025년 공개한 에이전트 기반 CLI 도구입니다.
- 로컬 개발 환경과 MCP를 통해 긴밀하게 통합되어, 터미널 상에서 즉시 복잡한 작업을 수행합니다.

## 2. MCP 서버 레지스트리 (Servers & Marketplaces)

개발자가 직접 만들지 않아도 즉시 사용할 수 있는 수만 개의 MCP 서버를 다음 플랫폼에서 찾을 수 있습니다.

- **Smithery ([smithery.ai](https://smithery.ai))**: "AI 도구의 npm"이라 불리는 가장 큰 커뮤니티 플랫폼입니다. 7,300개 이상의 도구를 제공하며 전용 CLI를 통한 쉬운 설치를 지원합니다.
- **Glama ([glama.ai/mcp/servers](https://glama.ai/mcp/servers))**: 보안과 품질을 강조하는 마켓플레이스입니다. 모든 서버에 대해 수동 리뷰와 취약점 스캔을 진행하여 안전한 도구 목록을 제공합니다.
- **Official MCP Registry ([registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io))**: 2025년 9월 출시된 공식 레지스트리입니다. 호스트 애플리케이션들이 도구를 자동으로 검색하고 설치할 수 있는 표준 API를 제공합니다.
- **MCP Directory ([mcpdirectory.org](https://mcpdirectory.org))**: 여러 레지스트리의 정보를 통합하여 보여주는 메타 디렉토리입니다.
- **Awesome MCP Servers (GitHub)**: 커뮤니티에서 큐레이션한 오픈소스 MCP 서버들의 목록입니다. (`punkpeye/awesome-mcp-servers`)

## 3. MCP Registry (서버 탐색)
공식 또는 커뮤니티에서 운영하는 서버 리스트를 탐색할 수 있는 곳들입니다.
- **MCP 공식 서버 목록**: [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
- **Smithery**: MCP 서버를 쉽게 찾고 설치할 수 있는 커뮤니티 대시보드.

## 4. MCP 호환 모델
현재 MCP 프로토콜 자체는 모델 독립적이지만, 이를 가장 잘 활용하는 모델은 다음과 같습니다.
- **Claude 3.5 Sonnet / Haiku**: MCP의 복잡한 도구 사용 로직을 매우 정확하게 수행합니다.
- **GPT-4o / GPT-4.5**: OpenAI Agents SDK 등을 통해 MCP 서버와 연동 가능합니다.
- **Llama-3 (8B/70B)**: 오픈소스 에이전트 프레임워크를 통해 MCP와 결합하여 사용됩니다.
