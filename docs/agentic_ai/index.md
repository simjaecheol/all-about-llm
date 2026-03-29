---
layout: default
title: Agentic AI
nav_order: 70
has_children: true
---

# Agentic AI

## 개요

**Agentic AI**는 단순한 개별 에이전트(Agent)를 넘어, 인공지능이 자율적인 사고 루프와 도구 활용을 통해 목표를 달성하는 **시스템적 패러다임**을 의미합니다. 2025년 AI 기술의 핵심은 모델의 크기를 키우는 'Scale-up'에서, 시스템의 자율성과 반복적 추론을 강화하는 **'Agentic Reasoning'**으로 이동하고 있습니다.

### 핵심 가치

- **자율적 루프(Autonomous Loops)**: 한 번의 프롬프트 응답이 아닌, 추론-실행-반성을 반복하는 워크플로우.
- **성능 극대화**: 작은 모델도 에이전틱 워크플로우를 통해 거대 모델 이상의 성능 도출 가능.
- **목표 지향성(Goal-Directed)**: 구체적인 명령이 없어도 최종 목표를 위해 스스로 경로를 계획하고 수정.
- **실제 세계 상호작용**: 외부 환경의 피드백을 실시간으로 수용하여 동작 최적화.

---

## 학습 내용

이 섹션에서는 Agentic AI의 설계 패턴부터 최신 추론 패러다임까지 심도 있게 학습합니다.

### 1. [Agentic AI의 개념과 차별점](./concepts)
**학습 목표**: 개별 에이전트와 에이전틱 시스템의 구조적 차이 이해
- Agent vs Agentic AI: 개체(Who)에서 시스템(How)으로
- System 1 (직관) vs System 2 (숙고) 추론 모델

### 2. [에이전틱 설계 패턴 (4대 패턴)](./patterns)
**학습 목표**: Andrew Ng이 제시한 4가지 핵심 설계 패턴 습득
- Reflection (반성/비판)
- Tool Use (지능적 도구 활용)
- Planning (계획 수립 및 수정)
- Multi-agent Collaboration (다중 에이전트 협업)

### 3. [에이전틱 워크플로우 설계](./workflow)
**학습 목표**: 성공적인 에이전틱 시스템 구축을 위한 워크플로우 설계 기법
- Iterative Refinement 루프 구성 방법
- 환경 피드백(Observation)의 통합 및 처리
- 오류 발생 시의 자가 수정(Self-correction) 메커니즘

### 4. [2025년의 진화: Degrees of Agency](./evolution_2025)
**학습 목표**: 자율성의 수준(Agency)을 측정하고 시스템을 고도화하는 기준 파악
- 자율성의 단계별 분류
- Thinking Time 스케일링 (Test-time Compute)
- 2025년 최신 에이전틱 패러다임: System 2 Reasoning의 대중화

---

## Agentic AI vs AI Agent vs Skills

| 구분 | AI Agent | Agentic AI | Agent Skills |
| :--- | :--- | :--- | :--- |
| **관점** | 작업을 수행하는 **개체** | 시스템의 **작동 방식** | 에이전트의 **전문 지식** |
| **핵심 질문** | "누가 일을 하는가?" | "시스템이 어떻게 사고하는가?" | "그 일을 어떻게 수행하는가?" |
| **구성 요소** | Persona, Tools, [Memory](../agent/memory.md) | Iterative Loops, Reasoning | `SKILL.md`, Instructions |

---

## 학습 효과

- **시스템 사고 능력**: 단일 프롬프트를 넘어 복잡한 자율 시스템을 설계하는 안목 확보.
- **비용 효율적 개발**: 작은 모델을 활용하여 고성능 에이전틱 서비스를 구축하는 전략 습득.
- **미래 기술 적응**: 2025년 이후의 AI 표준인 '숙고형 추론'과 '자율 워크플로우' 기술 선점.
