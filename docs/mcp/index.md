---
title: MCP (Model Context Protocol)
nav_order: 17
has_children: true
---

# MCP (Model Context Protocol)

## 개요

**Model Context Protocol (MCP)**는 AI 시스템과 데이터 소스 간의 연결을 표준화하는 개방형 표준입니다. 2024년 11월 Anthropic에 의해 공개되었으며, 현재는 리눅스 재단의 **Agentic AI Foundation**에서 관리되는 업계 표준으로 자리 잡았습니다.

흔히 **"AI를 위한 USB-C"**라고 불리는 MCP는 서로 다른 AI 모델(N)과 서로 다른 데이터 소스/도구(M) 간의 복잡한 통합 문제를 해결합니다.

### MCP의 핵심 가치

- **범용성**: 한 번 구축된 MCP 서버는 Claude, Cursor, VS Code, Gemini 등 다양한 호스트에서 즉시 사용 가능합니다.
- **보안성**: 로컬 프로세스(Stdio) 또는 보안 HTTP(SSE)를 통해 데이터 흐름을 안전하게 제어합니다.
- **확장성**: JSON-RPC 2.0 기반의 가벼운 프로토콜로 누구나 새로운 기능을 쉽게 추가할 수 있습니다.

---

## 학습 내용

이 섹션에서는 MCP의 기본 개념부터 실제 서버 구축, 그리고 최신 생태계 통합까지 체계적으로 학습합니다.

### 1. [핵심 개념 및 아키텍처](./concepts)
**학습 목표**: MCP의 작동 원리와 Host-Client-Server 구조 이해
- Host, Client, Server의 역할 분담
- Transport Layer: Stdio vs SSE

### 2. [주요 기능 프리미티브](./capabilities)
**학습 목표**: 에이전트가 활용하는 3대 핵심 기능 이해
- **Tools**: 실행 가능한 함수 (Actions)
- **Resources**: 읽기 전용 데이터 (Knowledge)
- **Prompts**: 지능적인 템플릿 (Workflows)

### 3. [개발 및 SDK 가이드](./sdk_guide)
**학습 목표**: 주요 SDK를 활용한 실제 MCP 서버 구축 방법 습득
- TypeScript 및 Python SDK 활용법
- 서버 및 클라이언트 구현 예제

### 4. [에코시스템 및 통합](./ecosystem)
**학습 목표**: Claude Code, Cursor 등 주요 도구에서의 MCP 활용 현황 파악
- IDE 통합 (VS Code, Cursor)
- CLI 에이전트 (Claude Code)
- 사전에 구축된 서버(Registry) 목록

### 5. [배포 및 보안 운영](./deployment)
**학습 목표**: 실제 프로덕션 환경에서의 배포 및 보안 고려 사항 학습
- 로컬 vs 원격 배포 전략
- 권한 관리 및 엔터프라이즈 가버넌스

---

## Tool Call vs MCP vs Skills

| 특징 | Tool Call | MCP | Agent Skills |
| :--- | :--- | :--- | :--- |
| **역할** | 실행 메커니즘 (행위) | 연결 표준 (규격) | 전문 지식 (노하우) |
| **비유** | 도구를 잡는 **손동작** | 범용 **연결 단자** (USB-C) | 도구 사용 **매뉴얼** |
| **상세 가이드** | [Tool Call 섹션](../tool_call/index) | **현재 섹션 (MCP)** | [Skills 섹션](../skills/index) |
