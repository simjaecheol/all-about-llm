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
| 대표 예시 | vLLM, TensorRT-LLM, SGLang | OpenAI Responses API, Claude Agent Skills |

## 추론 플랫폼이 필요한 이유 (Why Inference Platform?)

2024년까지의 LLM 활용이 단순히 질문에 답을 하는 '챗봇' 중심이었다면, 2025년부터는 스스로 도구를 사용하고 코드를 실행하며 목표를 달성하는 **에이전틱 AI(Agentic AI)**가 주류가 되었습니다. 이러한 패러다임의 변화는 기존의 단순 추론 엔진(Inference Engine)만으로는 해결할 수 없는 새로운 기술적 요구사항을 발생시켰습니다.

### 1. Model-as-a-Service에서 Agent-as-a-Service로의 진화
과거에는 모델의 가중치를 서버에 올리고 텍스트를 생성하는 것(Inference)이 핵심이었습니다. 하지만 에이전트는 추론 과정에서 파이썬 코드를 작성하고, 파일을 수정하며, 터미널 명령을 실행합니다. 이를 위해서는 단순한 API 엔드포인트를 넘어, **모델이 활동할 수 있는 '운동장(Runtime Environment)'**이 필요해졌습니다.

### 2. 보안과 격리 (Security & Isolation)
모델이 생성한 신뢰할 수 없는 코드(Untrusted Code)를 서버에서 직접 실행하는 것은 매우 위험합니다. 추론 플랫폼은 각 에이전트 요청마다 독립된 **샌드박스(Sandbox)**를 즉시 생성하여, 호스트 시스템을 보호하면서도 모델에게는 자유로운 실행 권한을 부여합니다.

### 3. 상태 유지와 연속성 (State & Continuity)
에이전틱 워크플로우는 한 번의 호출로 끝나지 않고 여러 단계(Multi-turn)에 걸쳐 진행됩니다. 이전 단계에서 에이전트가 설치한 라이브러리나 생성한 파일이 다음 단계에서도 유지되어야 합니다. 추론 플랫폼은 **세션(Session) 기반의 상태 관리**를 통해 에이전트가 작업의 맥락을 잃지 않게 돕습니다.

### 4. 비용 및 리소스 최적화 (Context Caching)
에이전트는 사고 과정(Reasoning Loop)에서 동일한 대규모 코드베이스나 문서를 반복적으로 참조합니다. 매번 전체 컨텍스트를 모델에 전송하는 것은 비용과 지연시간 측면에서 비효율적입니다. 추론 플랫폼은 **컨텍스트 캐싱(Context Caching)** 기술을 통해 중복된 데이터 처리를 최소화하고 추론 속도를 극대화합니다.

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
