---
title: 주요 기능 프리미티브
parent: MCP (Model Context Protocol)
nav_order: 2
---

# 주요 기능 프리미티브 (Capabilities)

MCP 서버는 모델에게 세 가지 핵심 능력을 제공합니다: **Tools, Resources, Prompts**입니다.

## 1. Tools (실행 가능한 도구)
모델이 외부 세계에서 실제로 **동작(Action)**을 수행하기 위해 호출하는 실행 가능한 함수입니다.

- **특징**: 모델이 인자(Arguments)를 채워 호출하고, 서버가 실행 후 결과를 반환합니다.
- **예시**: 
    - `search_github(query: string)`: 저장소 검색
    - `run_sql_query(sql: string)`: DB 쿼리 실행
    - `send_slack_message(channel: string, message: string)`: 알림 전송

## 2. Resources (구조화된 데이터)
모델이 참고할 수 있는 **지식(Data)**을 읽기 전용으로 제공하는 방식입니다.

- **특징**: 파일, 데이터베이스 레코드, API 문서 등을 모델이 읽을 수 있는 형태로 노출합니다.
- **Resource URIs**: `mcp://github/repo/file.md`와 같은 고유 주소를 통해 접근합니다.
- **종류**: 
    - **Text Resources**: 일반 마크다운, 코드 파일 등
    - **Binary Resources**: 이미지, PDF 등 (Base64 인코딩 등으로 전달)
- **구독(Subscriptions)**: 리소스의 내용이 바뀌면 서버가 클라이언트에게 알림을 보내 모델의 컨텍스트를 업데이트할 수 있습니다.

## 3. Prompts (지능적인 템플릿)
특정 작업을 수행하기 위해 검증된 **프롬프트 템플릿(Workflows)**을 제공합니다.

- **특징**: 단순한 도구 사용법뿐만 아니라, 모델이 해당 작업을 수행할 때 가져야 할 '페르소나'나 '단계별 가이드'를 포함합니다.
- **예시**:
    - `analyze-log-file`: 대량의 로그 파일을 분석하기 위한 지침이 담긴 템플릿
    - `code-review-assistant`: 특정 코딩 표준에 따른 리뷰 가이드라인

---

### 기능 요약 비교

| 프리미티브 | 목적 | 상호작용 방식 | 주요 사용 사례 |
| :--- | :--- | :--- | :--- |
| **Tools** | 행위(Doing) | 호출 -> 결과 | 데이터 수정, API 연동 |
| **Resources** | 정보(Knowing) | 읽기 (URI) | 문서 참조, 로그 조회 |
| **Prompts** | 가이드(Thinking) | 템플릿 적용 | 복잡한 워크플로우 초기화 |
