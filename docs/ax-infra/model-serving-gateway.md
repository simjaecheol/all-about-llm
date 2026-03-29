---
layout: default
title: 모델 서빙 및 API 게이트웨이
parent: AI Transformation 인프라 (AX Infra)
nav_order: 1
---

# 모델 서빙 및 API 게이트웨이

## 개요

에이전트 시스템이 엔터프라이즈 레벨로 확장됨에 따라 가장 먼저 직면하는 문제는 **모델에 대한 접근 통제, 라우팅, 그리고 자체 호스팅을 통한 성능 최적화**입니다. 단일 공급자(예: OpenAI)의 API만 사용하는 구조를 넘어, 여러 외부 및 내부 모델을 효율적으로 섞어 쓰는(Multi-Model) 전략이 필수적입니다.

---

## 1. AI API 게이트웨이 (AI API Gateway)

LLM 게이트웨이는 애플리케이션 프론트/백엔드와 실제 생성형 AI 모델 사이에 위치하여 트래픽을 중재합니다.

### 왜 필요한가?
- **Unified API**: OpenAI, Anthropic, Google Gemini 등 규격이 다른 API들을 하나의 공통된 인터페이스(주로 OpenAI 호환 포맷)로 통일.
- **Failover / Fallback**: 메인 공급자의 API가 장애를 겪거나 Rate Limit에 걸리면 자동으로 대체 모델(Fallback)로 라우팅.
- **로드 밸런싱 (Load Balancing)**: 다수의 API Key를 보유했을 때 호출을 분산시켜 병목현상 방지.
- **비용 통제 및 캐싱 (Caching)**: 반복적인 질의에 대해 시맨틱 캐싱(Semantic Caching)하여 LLM 비용 절감 및 응답 속도 향상.

### 대표 스택 (LiteLLM)
- **[LiteLLM](https://litellm.ai/)**: 100개 이상의 LLM 공급자에 대한 표준화된 인터페이스를 제공하는 오픈소스 플랫폼. 엔터프라이즈 환경에서 프록시 서버로 매우 많이 사용됨.

---

## 2. 오픈소스 모델 서빙 (OpenSource Model Serving)

클라우드 기반의 상용 API 대신, 보안성 강화 및 특정 도메인(온프레미스) 운용을 위해 오픈소스 모델(Llama 3, Mistral, Qwen 등)을 직접 호스팅하여 서빙합니다.

### 오픈소스 모델 서빙의 핵심 요소
- **처리량(Throughput)** 극대화: vRAM 제약 하에서 얼마나 많은 동시 요청을 소화할 수 있는가.
- **지연 시간(Latency)** 최소화: TTFT (Time To First Token), 즉 첫 토큰 생성까지의 시간 단축.

### 대표 스택
1. **[vLLM](https://vllm.ai/)**:
   - PagedAttention 이라는 메모리 관리 기술을 통해 KV 캐시(KV Cache) 메모리 낭비를 줄이고 처리량을 극대화한 서빙 엔진.
   - 대규모 운영 환경에서 표준으로 자리잡음.
2. **[Ollama](https://ollama.ai/)**:
   - 개발자들 사이에서 로컬 환경에 가장 쉽게 모델을 올리고 테스트할 수 있는 플랫폼.
   - 복잡한 설정 없이 `ollama run llama3`와 같은 명령어로 바로 서빙 가능.
3. **[TGI (Text Generation Inference)](https://huggingface.co/docs/text-generation-inference/index)**:
   - HuggingFace에서 만든 고성능 텍스트 생성 추론 엔진. 지속적 배칭(Continuous Batching)과 텐서 병렬 처리 지원.
