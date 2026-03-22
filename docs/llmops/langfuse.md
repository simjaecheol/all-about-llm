---
layout: default
title: Langfuse
parent: LLMOps
nav_order: 1
---

# Langfuse

[Langfuse](https://langfuse.com/)는 LLM 애플리케이션과 에이전트를 모니터링 및 최적화하기 위해 설계된 강력한 **오픈소스 기반 옵저버빌리티 플랫폼**입니다.

## 주요 특징 (Features)

### 1. 강력하고 세밀한 Tracing 기능
사용자 단위의 전체 세션(Session)을 "Trace"로 관리하며, 그 내부의 상호작용 과정들을 "Span"과 "Event"로 기록합니다.
* LLM이 호출된 기록, RAG를 위한 임베딩 단계, 에이전트의 의사결정 기록 및 도구(Tool) 실행 내역 등을 모두 세세하게 추적합니다.
* 다중 에이전트 시스템에서 복잡한 상호작용(Multi-agent interactions)의 흐름과 인과관계를 쉽게 파악할 수 있도록 돕습니다.

### 2. 실시간 Metrics 및 Analytics
Langfuse 대시보드를 통해 LLM을 다룰 때 가장 중요한 수치 지표들을 실시간으로 제공합니다.
* **비용(Cost)**, **토큰 사용량(Token Usage)**, **지연 시간(Latency)**의 실시간 추이를 모델, 유저, 세션 별로 분할하여 추적이 가능합니다.
* 어디서 비용 낭비가 발생하는지, 혹은 병목 현상(Bottleneck)이 일어나는 특정 추적 구간을 골라내어 성능 스케일링 전략에 도움을 줍니다.

### 3. 평가 및 피드백 (Evaluation & Feedback)
사용자 환경 및 평가 파이프라인과 직접적인 통합이 가능합니다.
* 사용자로부터 들어오는 긍정/부정(Thumbs up/down) 피드백이나 기타 텍스트 피드백을 직접 Trace에 기록하여 최적화 메트릭에 반영할 수 있습니다.
* LLM-as-a-Judge나 수동 채점 방식을 통해 에이전트 응답의 구조적 평가가 가능합니다.

### 4. 프롬프트 관리 시스템 (Prompt Management)
Langfuse의 내장 Prompt Management 기능을 사용해 프롬프트를 플랫폼 상에서 직접 배포, 관리합니다.
* 이를 통해 애플리케이션의 코드를 수정하거나 배포 과정을 거치지 않고도 프롬프트 교체가 가능하며, 손쉬운 A/B 테스트 환경을 제공합니다.

### 5. 프레임워크 종속성 최소화 적용 (Framework Agnostic)
다양한 언어 생태계와 LLM 프레임워크를 광범위하게 지원합니다.
* `LangChain`, `LlamaIndex`, `LangGraph`는 물론, OpenTelemetry 표준을 채택한 Pydantic AI, smolagents 등 다양하게 연결이 가능합니다.
* 오픈소스 버전이기 때문에 기업 환경에서 보안(데이터 주권) 문제로 외부 서비스 연결이 힘든 경우 **Self-hosted 서버 배포**가 가능합니다.
