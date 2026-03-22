---
title: 추론 플랫폼(Inference Platform)
nav_order: 9
has_children: true
---

# 추론 플랫폼(Inference Platform)

2025년 LLM 생태계는 단순한 모델 실행기인 **Inference Engine**을 넘어, 모델이 실행되는 환경과 도구, 상태를 통합 관리하는 **Inference Platform**으로 진화하고 있습니다.

## Inference Engine vs Inference Platform

| 구분 | Inference Engine (기존) | Inference Platform (현재/미래) |
| :--- | :--- | :--- |
| **핵심 목표** | 추론 속도 및 메모리 최적화 | 실행 환경 관리 및 도구 오케스트레이션 |
| **주요 기능** | PagedAttention, Speculative Decoding | Sandboxing, State Persistence, Context Caching |
| **관리 단위** | 모델 가중치 및 KV 캐시 | 컨테이너, 파일 시스템, 세션 상태 |
| **대표 예시** | vLLM, TensorRT-LLM, SGLang | OpenAI Responses API, Claude Agent Skills |

## 주요 핵심 구성 요소

### 1. 실행 환경 관리 (Execution Environment)
모델이 생성한 코드나 쉘 명령어를 안전하게 실행할 수 있는 격리된 **샌드박스(Sandbox)**를 제공합니다.
- **Container Info**: 에이전트가 현재 실행 환경의 리소스(CPU, 메모리, 네트워크)를 인식.
- **Security**: 호스트 시스템으로부터의 완벽한 격리.

### 2. 상태 유지 (State & Persistence)
여러 번의 API 호출 사이에서 환경의 변화를 유지합니다.
- **Persistent Containers**: 설치된 라이브러리와 생성된 파일을 다음 턴에서도 사용.
- **Session Management**: 사용자 세션별 독립된 런타임 제공.

### 3. 도구 오케스트레이션 (Tool Orchestration)
외부 데이터 소스 및 서비스와의 연결을 표준화합니다.
- **MCP (Model Context Protocol)**: 다양한 도구와 데이터를 연결하는 개방형 표준.
- **Skill-driven Execution**: 전문적인 워크플로우를 기반으로 도구를 지능적으로 제어.

## 학습 내용

### 1. [주요 API 제공사별 비교](./providers_comparison)
- OpenAI, Anthropic, Google의 추론 플랫폼 전략 및 기능 차이 분석.
- `container_info`, `hosted runtime`, `managed code execution` 등 핵심 개념 비교.

### 2. [상태 유지와 컨테이너 관리](./container_management)
- 에이전트의 연속성을 보장하기 위한 컨테이너 재사용 및 세션 유지 기술.

### 3. [보안 및 샌드박스 전략](./security_sandboxing)
- 모델 생성 코드 실행 시 발생할 수 있는 보안 위협과 이를 방지하기 위한 아키텍처.
