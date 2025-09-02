---
title: MCP (Model Context Protocol)
parent: Tool Call
nav_order: 6
---

# MCP (Model Context Protocol)

## MCP란?

**Model Context Protocol (MCP)**는 Anthropic이 2024년 11월에 오픈소스로 공개한 AI 시스템과 데이터 소스 간의 표준화된 연결을 제공하는 프로토콜입니다.

## MCP의 핵심 특징

### 표준화된 프로토콜

MCP는 USB-C와 같은 범용 연결 표준을 제공합니다:

- **서버-클라이언트 아키텍처**: 명확한 역할 분담과 확장 가능한 구조
- **다양한 SDK 지원**: TypeScript, Python 등 다양한 프로그래밍 언어 지원
- **사전 구축된 서버**: Google Drive, Slack, GitHub, Postgres 등 주요 서비스와의 연동

### 보안 및 확장성

- **안전한 양방향 연결**: 암호화된 통신과 인증 메커니즘
- **로컬 및 원격 서버 지원**: 온프레미스와 클라우드 환경 모두 지원
- **기업 환경 최적화**: 보안 정책과 규정 준수 요구사항 충족

## MCP 아키텍처

![MCP 아키텍처](../assets/images/mcp_protocol.png)

MCP는 다음과 같은 구성 요소로 이루어집니다:

1. **MCP Host**: Claude, IDEs, Tools 등 클라이언트 애플리케이션
2. **MCP Servers**: 다양한 데이터 소스와 연결하는 서버들
3. **MCP Protocol**: 표준화된 통신 프로토콜
4. **Resources**: 로컬 및 원격 데이터 소스

## MCP 활용 사례

### 기업 도입 현황

실제 기업들의 MCP 도입 사례:

- **Block, Apollo**: MCP를 시스템에 통합하여 AI 에이전트의 정보 검색 능력 확장
- **Zed, Replit, Codeium**: 개발 환경에서 MCP를 활용한 코드 분석 및 생성
- **Sourcegraph**: 코드 검색 및 분석에 MCP 적용하여 개발자 경험 향상

### 개발자 도구 통합

- **IDE 확장**: VS Code, IntelliJ 등에서 MCP 서버 연동
- **CLI 도구**: 명령줄에서 MCP 서버와 상호작용
- **웹 애플리케이션**: 브라우저 기반 AI 도구와 MCP 서버 연결

## MCP 서버 구현 예시

### Slack MCP 서버

**주요 기능**:
- 채널 및 스레드 지원 (#이름, @검색)
- 스마트 히스토리 (날짜별, 개수별 페이지네이션)
- DM 및 그룹 DM 지원
- 메시지 검색 및 안전한 메시지 게시
- Stdio/SSE 전송 및 프록시 지원

**사용 시나리오**:
- AI 에이전트가 Slack 채널에서 정보 검색
- 자동화된 메시지 게시 및 알림
- 채널 활동 분석 및 요약

### GitHub MCP 서버

**주요 기능**:
- 저장소 정보 조회 및 검색
- 이슈 및 PR 관리
- 코드 리뷰 및 코멘트 생성
- 워크플로우 자동화

### PostgreSQL MCP 서버

**주요 기능**:
- 데이터베이스 스키마 조회
- SQL 쿼리 실행 및 결과 반환
- 데이터 분석 및 시각화
- 보안 정책 적용

## MCP 개발 가이드

### 서버 개발

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

class MyMCPServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'my-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers() {
    // 도구 핸들러 설정
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: [
          {
            name: 'example_tool',
            description: 'An example tool',
            inputSchema: {
              type: 'object',
              properties: {
                input: { type: 'string' }
              }
            }
          }
        ]
      };
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}
```

### 클라이언트 개발

```typescript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

class MyMCPClient {
  private client: Client;

  constructor() {
    this.client = new Client({
      name: 'my-mcp-client',
      version: '1.0.0',
    });
  }

  async connectToServer(serverPath: string) {
    const transport = new StdioClientTransport(serverPath);
    await this.client.connect(transport);
  }

  async listTools() {
    const response = await this.client.request('tools/list', {});
    return response.tools;
  }

  async callTool(name: string, arguments_: any) {
    const response = await this.client.request('tools/call', {
      name,
      arguments: arguments_
    });
    return response.content;
  }
}
```

## MCP의 장점과 한계

### 장점

- **표준화**: 다양한 AI 시스템과 도구 간의 호환성 보장
- **확장성**: 새로운 데이터 소스와 도구를 쉽게 추가 가능
- **보안**: 기업 환경에서 요구하는 보안 수준 충족
- **개방성**: 오픈소스 프로토콜로 커뮤니티 기반 발전

### 한계 및 과제

- **성능**: 네트워크 지연 및 처리 오버헤드
- **복잡성**: 대규모 시스템에서의 관리 복잡성
- **학습 곡선**: 새로운 프로토콜에 대한 개발자 학습 필요
- **에코시스템**: 아직 성숙하지 않은 도구 및 라이브러리

## 미래 전망

MCP는 AI 시스템과 데이터 소스 간의 연결을 표준화하는 중요한 역할을 할 것으로 예상됩니다:

- **산업 표준화**: 더 많은 기업과 개발자 커뮤니티의 채택
- **도구 생태계 확장**: 다양한 도메인별 MCP 서버 개발
- **성능 최적화**: 대용량 데이터 처리 및 실시간 응답 개선
- **보안 강화**: 더욱 정교한 인증 및 권한 관리 시스템

MCP의 성공적인 채택과 발전은 AI 시스템이 실제 세계의 데이터와 더욱 효과적으로 상호작용할 수 있는 기반을 마련할 것입니다.
