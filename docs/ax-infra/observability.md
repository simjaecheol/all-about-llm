---
layout: default
title: 관측 가능성 (Observability & LLMOps)
parent: AI Transformation 인프라 (AX Infra)
nav_order: 3
---

# 관측 가능성 (Observability & LLMOps)

## 개요

일반적인 소프트웨어 개발과 달리 에이전트/LLM 개발은 필연적으로 **확률론적 블랙박스(Probabilistic Blackbox)**의 특성을 지닙니다. PoC에서 작동하던 프롬프트나 에이전트 로직이 프로덕션 환경에서 원인 불명의 오류나 환각(Hallucination)을 일으킬 때, 내부를 모니터링할 도구가 없으면 개선이 불가능합니다. 

이러한 문제를 해결하기 위해 도입되는 인프라가 바로 **관측 가능성(Observability) 도구**입니다.

> **연관 문서 파도타기**
> - [LLMOps 생태계 이해](../llmops/index.md)
> - [Langfuse 상세 활용 가이드](../llmops/langfuse.md)

---

## 1. 무엇을 관측해야 하는가?

에이전트 인프라의 관측 가능성 도구는 기존 APM(Application Performance Monitoring - 예: Datadog, New Relic)을 넘어 아래의 지표를 필수로 추적합니다.

- **비용 (Cost / Tokens)**: 매 API 호출에 발생한 입력/출력 토큰 수를 기반으로 사용자/세션 단위 비용 산출.
- **지연 시간 (Latency)**: 전체 에이전트 응답 시간 및 내부 RAG 처리, 외부 Tool API 호출 등 지표별 지연 병목 파악.
- **추적 (Tracing)**: 사용자의 단일 입력(Trace) 안에 얽혀있는 여러 LLM 호출, 검색, 도구 사용을 트리(Tree) 계층 구조(Spans)로 시각화.
- **평가 (Evaluation / QA)**: 사용자의 긍/부정 피드백 기록, 혹은 자체적인 LLM-as-a-Judge 지표(답변 정합성, 관련성 측정) 대시보드.

---

## 2. 대표적인 관측 스택

에이전트 옵저버빌리티 생태계를 이끄는 주요 플랫폼은 다음과 같습니다.

### [Langfuse (오픈소스 옵저버빌리티)](https://langfuse.com/)
- 개발자가 자가 호스팅(Self-hosting)이 가능하며, 직관적인 UI와 강력한 확장성 제공.
- 복잡한 LangChain, LangGraph, LlamaIndex 에이전트 워크플로우를 완벽하게 시각화. ([상세 문서](../llmops/langfuse.md) 참고)

### [LangSmith](https://www.langchain.com/langsmith)
- LangChain에서 자체 구축한 엔터프라이즈급 관측 플랫폼.
- LangChain 프레임워크와의 결합도가 압도적이며 프롬프트 버전관리, 프롬프트 배포, 데이버셋 기반 테스트 자동화에 최적화됨.

### [Arize Phoenix](https://phoenix.arize.com/)
- LLM에 특화된 오픈소스 평가 및 트레이싱 도구.
- RAG의 검색 정확도 평가, UMAP를 활용한 임베딩 시각화, 모델 드리프트(Drift) 분석 등에 강력한 강점을 보유.

### [Prometheus & Grafana (전통적 인프라 메트릭)](../llmops/prometheus-grafana.md)
- 모델 서빙 관점(vLLM 구동의 GPU 사용량, API 가용성 지표 등)인 하드웨어 레벨의 모니터링은 여전히 전통적인 스택의 결합이 필수적임.

---

## 3. 프롬프트 버전 관리 (Prompt Versioning)

옵저버빌리티 도구(특히 Langfuse, LangSmith)는 단순한 모니터링을 넘어 **프롬프트 관리 및 버전 컨트롤** 영역까지 제공합니다. 소스 코드(Git)와 프롬프트를 분리하여 관리하는 것이 최신 인프라 트렌드입니다.

### Langfuse 기반 프롬프트 관리의 이점
- **런타임 분리(Decoupling)**: 개발자가 코드를 수정 후 재배포하지 않아도, 기획자나 AI 엔지니어가 Langfuse 대시보드에서 프롬프트를 바로 수정하고 배포(Publish)할 수 있습니다. 에이전트 애플리케이션은 런타임에 가장 최신 버전의 프롬프트를 동적으로 당겨와(Fetch) 실행합니다.
- **버전 추적 및 관리 (A/B 테스트 등)**: 프롬프트 v1과 v2가 실제 환경에서 토큰량, 응답 품질 측면에서 어떻게 다르게 동작했는지 트레이싱과 연동하여 통합 분석이 가능합니다.
- **점진적 롤백 (Rollback)**: 새 프롬프트에 문제가 발생할 경우, 소스 코드 롤백 없이 즉시 이전 버전으로 스위칭할 수 있어 높은 운영 안정성을 보장합니다.
