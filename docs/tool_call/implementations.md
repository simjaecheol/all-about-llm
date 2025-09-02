---
title: 주요 구현 사례
parent: Tool Call
nav_order: 3
---

# 주요 구현 사례

## 1. MCP (Model Context Protocol) - Anthropic

2024년 11월 Anthropic이 오픈소스로 공개한 **MCP**는 AI 시스템과 데이터 소스 간의 표준화된 연결을 제공하는 프로토콜입니다.

### MCP의 핵심 특징

**표준화된 프로토콜**: USB-C와 같은 범용 연결 표준 제공
- 서버-클라이언트 아키텍처로 구성
- TypeScript, Python 등 다양한 SDK 지원
- Google Drive, Slack, GitHub, Postgres 등 사전 구축된 서버 제공

**보안 및 확장성**:
- 안전한 양방향 연결 구축
- 로컬 및 원격 서버 모두 지원
- 기업 환경에서의 안전한 데이터 접근

![MCP 아키텍처](../assets/images/mcp_protocol.png)

### MCP 활용 사례

실제 기업들의 MCP 도입 현황:
- **Block, Apollo**: MCP를 시스템에 통합하여 AI 에이전트 정보 검색 확장
- **Zed, Replit, Codeium**: 개발 환경에서 MCP 활용
- **Sourcegraph**: 코드 검색 및 분석에 MCP 적용

### MCP 서버 구현 예시

**Slack MCP 서버**의 주요 기능:
- 채널 및 스레드 지원 (#이름, @검색)
- 스마트 히스토리 (날짜별, 개수별 페이지네이션)
- DM 및 그룹 DM 지원
- 메시지 검색 및 안전한 메시지 게시
- Stdio/SSE 전송 및 프록시 지원

## 2. Claude Code - Anthropic

**Claude Code**는 Anthropic이 개발한 명령줄 기반의 에이전틱 코딩 도구입니다.

### 주요 특징 및 활용 사례

**Anthropic 내부 팀 사용 현황**:
- **코드베이스 탐색**: 새로운 개발자의 빠른 온보딩 지원
- **테스트 자동화**: 포괄적인 단위 테스트 작성 
- **코드 리뷰**: Pull Request 자동 코멘트 생성
- **디버깅**: 복잡한 버그 추적 및 수정

**비개발자 활용 사례**:
- **법무팀**: 전화 트리 시스템 구축
- **마케팅팀**: 수백 개의 광고 변형 자동 생성
- **데이터 사이언티스트**: JavaScript 지식 없이 복잡한 시각화 생성

### Claude Code 모범 사례

**환경 최적화**:
- `CLAUDE.md` 파일을 통한 프로젝트별 문서화
- 공통 bash 명령어, 코드 스타일 가이드라인 정리
- 개발 환경 설정 및 예상치 못한 동작 기록

**효율적인 워크플로우**:
- `/clear` 명령어를 통한 컨텍스트 정리
- VS Code 확장 프로그램 활용
- 여러 인스턴스 병렬 실행

## 3. 기타 주요 구현 사례

### OpenAI Function Calling

OpenAI는 GPT-4와 GPT-3.5 Turbo에서 Function Calling 기능을 제공하며, 이를 통해 개발자들이 LLM에 외부 함수를 연결할 수 있게 했습니다.

### LangChain Tool Calling

LangChain은 다양한 도구와 LLM을 연결하는 프레임워크를 제공하며, Tool Calling을 위한 표준화된 인터페이스를 지원합니다.

## 구현 사례의 의의

이러한 구현 사례들은 Tool Call 기술이 이론적 연구를 넘어서 실제로 활용 가능한 수준에 도달했음을 보여줍니다. 특히 MCP의 표준화와 Claude Code의 실용적 접근은 개발자뿐만 아니라 일반 사용자도 AI 도구를 활용할 수 있는 길을 열어주고 있습니다.
