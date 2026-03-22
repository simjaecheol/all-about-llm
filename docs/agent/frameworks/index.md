---
title: 주요 프레임워크 및 생태계
parent: Agent
nav_order: 4
has_children: true
---

# 주요 프레임워크 및 생태계

에이전트 개발을 가속화하고 복잡한 워크플로우를 관리하기 위한 주요 프레임워크들을 세부적으로 살펴봅니다. 

## 학습 내용

에이전트 구현의 다양한 접근 방식을 제공하는 주요 프레임워크 5종의 특징과 사용법을 다룹니다.

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

---

## 프레임워크 선택 가이드

어떤 프레임워크를 선택할지는 프로젝트의 목적과 개발 스타일에 따라 다릅니다.

- **복잡한 비즈니스 로직과 세밀한 상태 제어**가 필요하다면? -> **LangGraph**
- **빠르게 팀 협업 워크플로우를 구축**하고 싶다면? -> **CrewAI**
- **견고한 백엔드 엔지니어링과 타입 체크**가 중요하다면? -> **PydanticAI**
- **에이전트들이 서로 대화하며 해결하는 시뮬레이션**이 목적이라면? -> **AutoGen**
- **토큰 비용을 아끼고 소형 모델(SLM)을 활용**하고 싶다면? -> **Smolagents**
