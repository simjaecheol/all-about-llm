---
layout: default
title: 에이전트 하네스의 구성 요소
parent: Agent Harness
nav_order: 1
---

# 에이전트 하네스의 구성 요소

에이전트 하네스는 단순히 모델을 실행하는 것 이상의 **구조화된 환경**을 제공해야 합니다. 효과적인 하네스를 구성하는 4가지 핵심 요소를 살펴봅니다.

---

## 1. System Prompt & Persona (시스템 프롬프트 및 페르소나)

에이전트의 정체성과 행동 강령을 정의하는 부분입니다. 하네스는 모델이 일관된 방식으로 도구를 사용하고 추론하도록 가이드를 제공합니다.

- **Role Definition**: "너는 유능한 소프트웨어 엔지니어 에이전트이다"와 같이 역할을 명시합니다.
- **Behavioral Guardrails**: 답변 형식(예: JSON만 사용), 도구 사용 시 주의사항, 출력 언어 등을 제약합니다.
- **Prompt Versioning**: 프롬프트의 미세한 변화가 성능에 큰 영향을 미치므로, 하네스 내에서 프롬프트를 버전 관리하고 테스트해야 합니다.

## 2. Tool Definitions & Schema (도구 명세)

에이전트가 사용할 수 있는 '손'에 해당하는 부분입니다. 하네스는 모델이 도구를 호출할 수 있도록 표준화된 명세를 제공합니다.

- **Function Calling Schema**: 도구의 이름, 설명, 필요한 매개변수(Parameter) 타입 등을 JSON Schema 등으로 정의합니다.
- **Tool Selection Logic**: 수많은 도구 중 현재 작업에 필요한 도구만 선택해서 프롬프트에 주입하는 기법이 포함될 수 있습니다.
- **Mock vs. Real Tools**: 테스트 목적에 따라 실제 API를 호출하거나, 미리 정의된 결과만 반환하는 Mock 도구를 제공합니다.

## 3. Environment Sandbox (환경 샌드박스)

에이전트가 실제 행동을 취하고 그 결과를 관찰하는 격리된 공간입니다.

- **File System Sandbox**: 에이전트가 파일을 생성, 수정, 삭제할 수 있는 가상 파일 시스템을 제공합니다.
- **Terminal Execution**: 명령어를 실행하고 그 표준 출력(stdout) 및 에러(stderr)를 다시 에이전트에게 전달합니다.
- **State Reset**: 각 테스트 케이스가 끝날 때마다 환경을 초기 상태로 되돌려(Reset) 테스트의 독립성을 보장합니다.

## 4. Loop Control (루프 관리 및 오케스트레이션)

에이전트의 사고(Reasoning)와 실행(Action) 과정을 관리하는 두뇌의 실행 루프입니다.

- **Reasoning Loop**: 에이전트가 "Thought -> Action -> Observation" 단계를 반복하도록 제어합니다.
- **Stopping Criteria**: 최대 사고 횟수(Max Turns), 특정 단어(Stop tokens) 발견, 또는 목표 달성 시 루프를 종료합니다.
- **Tracing & Logging**: 모든 사고 과정과 도구 호출 이력을 기록하여 나중에 에이전트의 실패 원인을 분석할 수 있게 합니다.

---

## 요약: 하네스의 구조적 흐름

1.  **Input**: 사용자의 질문 또는 작업 목표가 입력됨.
2.  **Orchestration**: 하네스가 프롬프트와 도구 명세를 결합하여 LLM에 전달.
3.  **Action**: LLM이 도구 호출 명령을 내리면, 하네스가 샌드박스 내에서 이를 실행.
4.  **Observation**: 실행 결과가 다시 하네스를 통해 LLM의 다음 입력으로 들어감.
5.  **Final Answer**: 루프가 종료되면 최종 결과물을 사용자에게 반환.
