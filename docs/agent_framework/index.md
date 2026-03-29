---
layout: default
---
title: Agent Framework
has_children: true
nav_order: 73
---
# Agent Framework

에이전트 개발을 가속화하고 복잡한 워크플로우를 관리하기 위한 주요 프레임워크와 플랫폼들을 다루는 디렉터리입니다. 

## 학습 내용

에이전트 구현의 다양한 접근 방식을 제공하는 주요 프레임워크 및 오픈소스 프로젝트들의 특징과 사용법을 학습합니다.

### 1. [LangGraph](./langgraph)
**핵심 키워드**: 상태 기반(Stateful), 순환(Cycles), 정교한 제어
- 복잡한 로직의 그래프 모델링
- Human-in-the-loop과 Persistence 구현

### 2. [CrewAI](./crewai)
**핵심 키워드**: 역할 기반(Role-based), 프로세스 중심, 비즈니스 자동화
- Role, Goal, Backstory를 활용한 에이전트 팀 구성
- Sequential, Hierarchical 프로세스 설계

### 3. [PydanticAI](./pydanticai)
**핵심 키워드**: 타입 안정성(Type-safe), 엔지니어링 완성도, FastAPI 유사성
- 강력한 데이터 검증 및 정적 타입 체크
- 의존성 주입(Dependency Injection)과 모델 독립적 설계

### 4. [AutoGen (AG2)](./autogen)
**핵심 키워드**: 대화형(Conversational), 오케스트레이션, 강력한 코드 실행
- 에이전트 간의 자유로운 대화 시나리오 구성
- 다중 에이전트 간의 자동 협업 및 실행

### 5. [Smolagents](./smolagents)
**핵심 키워드**: 코드 액션(Action-as-Code), 가볍고 빠름, 토큰 효율
- JSON 대신 파이썬 코드로 도구 호출 수행
- Hugging Face 생태계와의 긴밀한 통합

### 6. [OpenHands](./openhands) (구 OpenDevin)
**핵심 키워드**: 소프트웨어 엔지니어링 특화, 자율형 에이전트
- 소프트웨어 엔지니어링에 특화된 자율형 에이전트 프레임워크

### 7. [Dify](./dify)
**핵심 키워드**: 로우코드(Low-code), 시각적 설계, 워크플로우
- 로우코드 기반으로 복잡한 에이전트 워크플로우를 시각적으로 설계할 수 있는 플랫폼

### 8. [OpenFang](./open_fang)
**핵심 키워드**: 모듈형, 에이전트 전용 운영체제
- 단순한 프레임워크를 넘어 '에이전트 전용 운영체제'를 표방하는 모듈형 프로젝트

### 9. [ZeroClaw](./zero_claw)
**핵심 키워드**: 고성능, 빠른 속도
- Rust 기반의 고성능 에이전트 런타임 프로젝트

### 10. [Deep Agents](./deep_agent)
**핵심 키워드**: 심층 추론(Deep Reasoning), 장기 메모리, 계획 수립
- LangChain의 고급 에이전트 아키텍처로 복잡한 다단계 태스크를 자율적으로 수행
- LangGraph 기반의 내구성 있는 실행과 전문화된 하위 에이전트 위임 지원

---

## 기업용 프레임워크 (Enterprise Frameworks)

- **OpenAI Agents SDK**: OpenAI에서 공식 배포한 에이전트 구축 프레임워크입니다.
- **Google ADK**: Gemini 생태계와 밀접하게 연동되는 Google의 에이전트 개발 키트입니다.
