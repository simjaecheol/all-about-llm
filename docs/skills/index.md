---
title: Agent Skills
nav_order: 16
has_children: true
---

# Agent Skills (SKILL.md)

## 개요

**에이전트 스킬(Agent Skills)**은 AI 에이전트에게 모듈화되고 재사용 가능한 **절차적 지식(Procedural Knowledge)**과 전문 지식을 제공하기 위한 2025-2026년의 가장 혁신적인 개방형 표준입니다. 

기존의 **MCP(Model Context Protocol)**가 에이전트에게 "무엇을 할 수 있는가(도구)"를 제공하는 데 집중했다면, **Agent Skills**는 "그 일을 어떻게 수행해야 하는가(방법론)"에 대한 전문적인 가이드를 제공합니다. 최근 업계에서는 도구의 연결(Connectivity)을 넘어, 도구를 활용하는 **지능(Intelligence)**과 **워크플로우(Workflow)**를 모듈화하는 스킬의 중요성이 MCP보다 더 강조되고 있습니다.

### Agent Skills의 핵심 가치

- **모듈화**: 특정 작업(예: 보안 감사, 배포, 코드 리뷰)에 대한 워크플로우를 독립된 폴더로 캡슐화.
- **도구 활용의 극대화**: MCP 서버로 연결된 수많은 도구들을 어떤 순서로, 어떤 제약 조건 하에 사용해야 하는지 명시.
- **재사용성**: 한 프로젝트에서 정의한 스킬을 다른 프로젝트나 다른 AI 에이전트(Claude Code, Gemini CLI, Cursor 등)에서 즉시 사용 가능.
- **컨텍스트 최적화**: 모든 정보를 한 번에 읽지 않고, 필요할 때만 스킬을 활성화하여 컨텍스트 윈도우 효율 극대화.
- **버전 관리**: 스킬 자체가 코드로 존재하여 Git을 통한 버전 관리 및 협업 가능.

---

## 학습 내용

이 섹션에서는 Agent Skills의 표준 사양부터 실제 구축 방법까지 체계적으로 학습합니다.

### 1. [스킬의 개념과 원리](./concepts)
**학습 목표**: 도구(Tools)와 스킬(Skills)의 차이 및 점진적 공개 모델 이해
- "How-to" 중심의 절차적 지식 정의
- Discovery -> Activation -> Deep Dive 로딩 메커니즘

### 2. [비교 분석: Tool Call vs Skills vs MCP](./comparison)
**학습 목표**: 혼동하기 쉬운 세 가지 핵심 개념의 차이점과 상호 관계 명확화
- 각각의 정의 및 역할 비유
- 기술적 구현 층위(Layer) 분석
- 시너지 효과 및 결합 사례

### 2. [SKILL.md 사양 가이드](./specification)
**학습 목표**: 표준 `SKILL.md` 파일의 구조와 작성 방법 습득
- YAML Frontmatter: 메타데이터 및 트리거 설정
- Markdown 본문: 지시사항, 제약 조건, 예시 작성법

### 3. [스킷 구조 및 관리](./directory_structure)
**학습 목표**: 표준 스킬 디렉토리 레이아웃 구성 방법 이해
- `references/`, `scripts/`, `assets/` 등 하위 폴더 활용
- 실행 가능한 스크립트 결합 방법

### 4. [에이전트 스킬과 도구 활용](./tool_integration)
**학습 목표**: 에이전트 스킬 내에서 도구(Tool Call/MCP)를 지능적으로 제어하는 방법 습득
- 스킬(Logic)과 도구(Action)의 상호작용 설계
- 도구 실행 결과의 해석 및 워크플로우 반영 지침
- MCP(Model Context Protocol)와의 시너지 효과

### 5. [Skill-Creator 활용법](./skill_creator)
**학습 목표**: Gemini CLI 등에서 제공하는 스킬 생성 도구 사용법 익히기
- 새로운 스킬 템플릿 생성 및 배포
- 기존 워크플로우의 스킬화 과정

### 6. [커뮤니티 및 생태계](./community_resources)
**학습 목표**: 전 세계 개발자들이 공유하는 `SKILL.md` 검색 및 설치 방법 습득
- 공식 스킬 저장소 및 레지스트리
- `npx skills add`를 활용한 간편 설치
- 자동 검색(Automatic Discovery) 메커니즘

---

## 비교 분석: Skills vs Others

| 특징 | Agent Skills (`SKILL.md`) | MCP (Protocol) | README (`CLAUDE.md`) |
| :--- | :--- | :--- | :--- |
| **목적** | 전문적인 워크플로우/지식 전수 | 데이터 소스 및 API 연결 | 프로젝트 전반의 규칙 정의 |
| **단위** | 독립된 모듈 (폴더 단위) | 클라이언트-서버 프로토콜 | 단일 마크다운 파일 |
| **로딩 방식** | 필요 시 트리거 (On-demand) | 지속적 연결 (Persistent) | 세션 시작 시 로드 |
| **범위** | 여러 프로젝트 간 재사용 가능 | 외부 시스템 연동 표준 | 해당 프로젝트 전용 |

---

## 학습 효과

- **효율적인 컨텍스트 관리**: 대규모 프롬프트를 스킬로 분리하여 모델의 추론 성능 향상.
- **AI 행동의 표준화**: 팀 내 AI 에이전트의 작업 방식(Code Style, Security Check 등)을 통일.
- **유지보수 용이성**: 복잡한 지시사항을 중앙에서 관리하고 버전업 가능.
