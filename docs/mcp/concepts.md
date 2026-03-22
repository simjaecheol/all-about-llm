---
title: 핵심 개념 및 아키텍처
parent: MCP (Model Context Protocol)
nav_order: 1
---

# 핵심 개념 및 아키텍처

MCP는 **Host - Client - Server**라는 세 가지 주요 구성 요소를 기반으로 설계되었습니다. 이 구조는 모델과 데이터 소스 간의 결합도를 낮추어 유연한 통합을 가능하게 합니다.

## 1. 주요 구성 요소 (Architecture Roles)

- **MCP Host**: AI 애플리케이션 그 자체입니다. (예: Claude Desktop, Cursor, VS Code, Gemini CLI) 호스트는 에이전트 기능을 수행하기 위해 MCP 클라이언트를 포함합니다.
- **MCP Client**: 호스트 내부에 위치하며, 여러 MCP 서버와 통신하고 모델에게 제공할 수 있는 기능을 협상(Negotiation)합니다.
- **MCP Server**: 실제 데이터 소스나 도구(GitHub, PostgreSQL, Slack 등)와 연결되어 기능을 노출하는 가벼운 서비스입니다.

## 2. 통신 프로토콜 (The Protocol)

MCP는 **JSON-RPC 2.0** 프로토콜을 사용하여 메시지를 주고받습니다.
- **기본 통신 모델**: 양방향 비동기 통신. 클라이언트가 서버에 요청을 보내거나, 서버가 클라이언트에 알림을 보낼 수 있습니다.
- **Capability Negotiation**: 연결 시작 시, 클라이언트와 서버는 서로 어떤 기능을 지원하는지 확인하는 단계를 거칩니다.

## 3. 전송 레이어 (Transport Layers)

MCP는 크게 두 가지 방식으로 데이터를 전송합니다.

### Stdio Transport (로컬)
- **방식**: 호스트가 MCP 서버를 자식 프로세스(Child Process)로 실행하고 표준 입출력(stdin/stdout)을 통해 통신합니다.
- **장점**: 별도의 서버 구동 없이 즉시 실행 가능하며, 로컬 파일 시스템이나 데이터베이스에 접근하기에 가장 안전합니다.
- **주요 활용**: VS Code 확장 프로그램, 로컬 CLI 도구.

### HTTP + SSE (Server-Sent Events) (원격)
- **방식**: 클라이언트는 서버에 HTTP POST를 통해 메시지를 보내고, 서버는 SSE를 통해 클라이언트에 실시간으로 이벤트를 전송합니다.
- **장점**: 원격 클라우드 환경에 배포된 서버와 통신할 수 있으며, 여러 호스트가 공유할 수 있습니다.
- **주요 활용**: 클라우드 기반 AI 서비스, 공유 데이터 허브.

## 4. 보안 모델
- **Local Sandbox**: Stdio 서버는 호스트의 권한 아래에서 실행되므로, 호스트 수준의 샌드박싱이 가능합니다.
- **Authentication**: SSE 환경에서는 표준적인 HTTP 인증(Bearer Tokens 등)을 통해 접근을 제어합니다.
