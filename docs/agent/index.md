---
title: Agent
nav_order: 15
has_children: true
---

# Agent

## 개요

**에이전트(Agent)**는 AI가 단순히 질문에 답하는 수준을 넘어, 주어진 목표를 달성하기 위해 스스로 계획을 세우고, 도구를 사용하며, 환경과 상호작용하는 자율적인 시스템을 의미합니다. 2024년 이후 AI 기술의 중심은 단순한 '채팅'에서 '실행' 중심의 **에이전트 워크플로우(Agentic Workflow)**로 이동하고 있습니다.

### Agent의 특징

- **자율성(Autonomy)**: 인간의 개입을 최소화하고 스스로 판단하여 작업 수행
- **도구 사용(Tool Use)**: 외부 API, 웹 검색, 데이터베이스 등을 활용하여 능력 확장
- **추론 및 계획(Reasoning & Planning)**: 복잡한 문제를 하위 작업으로 분해하고 순차적으로 해결
- **상태 유지(Statefulness)**: 작업의 진행 상황과 과거의 경험을 기억하여 일관성 유지

---

## 학습 내용

이 섹션에서는 에이전트의 핵심 구성 요소부터 최신 프레임워크, 그리고 실제 프로덕션 환경에서의 고려 사항까지 체계적으로 학습합니다.

### 1. [Agent의 개념과 워크플로우](./concepts)
**학습 목표**: 에이전트의 정의와 기존 LLM 활용 방식과의 차이점 이해
- 에이전트의 정의 및 발전 과정
- Agentic Workflow: Andrew Ng이 제시한 반복적 워크플로우 개념
- 자율성의 단계와 에이전트의 유형

### 2. [에이전트 아키텍처 (Core Pillars)](./architecture)
**학습 목표**: 에이전트를 구성하는 4가지 핵심 요소(Planning, Memory, Tools, Action) 이해
- **Planning**: CoT, ReAct 등 추론 및 계획 기법
- **Memory**: 단기 기억(State)과 장기 기억(RAG, Episodic)
- **Tools**: Function Calling과 MCP(Model Context Protocol)
- **Action**: 실행 엔진 및 환경과의 상호작용

### 3. [에이전트 설계 패턴](./design_patterns)
**학습 목표**: 효율적인 에이전트 시스템 구축을 위한 주요 설계 패턴 습득
- Reflection: 스스로 결과를 검토하고 개선하는 패턴
- Self-Correction: 오류 발생 시 스스로 수정하는 메커니즘
- Planning & Execution: 계획자와 실행자의 분리 구조

### 4. [주요 프레임워크 및 생태계](./frameworks)
**학습 목표**: 최신 에이전트 개발 프레임워크의 특징과 선택 기준 파악
- **LangGraph**: 상태 중심의 복잡한 그래프 설계
- **CrewAI**: 역할 기반의 멀티 에이전트 협업
- **PydanticAI**: 타입 안정성을 강조한 개발자 친화적 프레임워크
- **AutoGen (AG2)**: 대화형 오케스트레이션

### 5. [멀티 에이전트 시스템 (MAS)](./multi_agent_systems)
**학습 목표**: 여러 에이전트가 협업하여 복잡한 문제를 해결하는 구조 이해
- Hierarchical vs. Joint Collaboration 모델
- Agent Hand-off: 에이전트 간 작업 이관 및 상태 공유
- 협업 프로토콜 및 통신 방식

### 6. [평가 및 관측성](./evaluation_and_observability)
**학습 목표**: 에이전트의 성능을 측정하고 사고 과정을 추적하는 방법 이해
- 에이전트 전용 벤치마크 (WebArena, Tau-Bench)
- LLM-as-a-Judge를 활용한 실행 결과 평가
- Tracing 도구 (Langfuse, Arize Phoenix) 활용법

### 7. [실무 고려 사항 및 보안](./production_considerations)
**학습 목표**: 실제 서비스 도입 시 직면하는 기술적, 윤리적 과제 해결
- Human-in-the-Loop (HITL) 설계
- 보안: 프롬프트 인젝션 방어 및 도구 실행 권한 관리
- 비용 및 성능 최적화: SLM(Small Language Model) 활용 전략

---

## 학습 효과

- **기술적 심화**: 단순 프롬프팅을 넘어 자율적인 AI 시스템의 내부 구조 이해
- **시스템 설계 능력**: 복잡한 비즈니스 로직을 에이전트 워크플로우로 변환하는 능력 함양
- **최신 트렌드 파악**: MCP, LangGraph 등 빠르게 변화하는 에이전트 생태계 적응
- **신뢰성 확보**: 에이전트의 오류를 제어하고 안전하게 배포하는 실무 지식 습득
