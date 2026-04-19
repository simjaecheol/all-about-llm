---
layout: default
title: LLM API 및 서빙 제공자 종합 가이드
parent: 모델 서빙 및 API 게이트웨이
nav_order: 2
---

# LLM API 및 서빙 제공자 (Providers) 종합 가이드

## 개요
에이전트나 AI 애플리케이션을 구축할 때 가장 먼저 결정해야 하는 것은 **"어떤 LLM을 어떻게 호출할 것인가"**입니다. 모델의 성능, 비용, 응답 속도, 보안(데이터 거버넌스) 요구 사항에 따라 각기 다른 성격의 제공자(Provider)를 선택해야 합니다.

본 문서에서는 주요 LLM 서빙 환경을 목적에 따라 5가지 주요 카테고리로 상세히 분류합니다.

---

## 1. 퍼블릭 클라우드 기반 관리형 서비스 (CSP Managed Services)
엔터프라이즈 환경에서 보안, 사내망(VPC) 연동 기능, 그리고 중앙화된 빌링 및 거버넌스 관리가 필수적일 때 가장 강력한 옵션입니다.

### 🔹 Azure OpenAI Service & Azure AI Studio
*   **특징**: 마이크로소프트의 글로벌 클라우드 인프라를 바탕으로 제공되는 서비스. 
*   **장점**: 철저한 데이터 프라이버시(학습에 사용되지 않음), Entra ID를 통한 엔터프라이즈 보안, 안정적인 SLA가 큰 특징입니다. 최근 AI Studio를 통해 Llama 3, Mistral 등 오픈소스 모델 서빙(MaaS) 기능도 대폭 강화되었습니다.
*   **적합한 케이스**: 대기업이나 금융/공공 기관과 같이 철저한 보안 통제와 SLA 보장이 필요한 B2B 프로젝트.

### 🔹 AWS Bedrock (Amazon)
*   **특징**: 서드파티(Anthropic 등) 및 오픈소스 모델에 단일 인터페이스로 접근할 수 있게 해주는 완전 관리형 파운데이션 모델 서비스.
*   **장점**: Anthropic Claude 모델 생태계를 가장 안정적으로 지원합니다. 더불어 Amazon Titan, Meta Llama, Mistral 등 다양한 모델을 동일한 API 구조로 쉽게 교체하며 사용할 수 있으며 AWS 인프라(IAM 등)에 완벽하게 녹아듭니다.
*   **적합한 케이스**: 기존에 대규모 AWS 인프라를 운영 중이거나 Claude 모델을 주력으로 사용하려는 기업.

### 🔹 Google Cloud Vertex AI
*   **특징**: 구글의 자체 모델(Gemini)과 엔터프라이즈 레벨의 MLOps 플랫폼이 결합된 서비스.
*   **장점**: Vertex AI Vector Search 등 생태계 내 다른 AI 빌딩 블록과 연동이 쉽고, RAG 구현을 위한 다양한 엔터프라이즈 솔루션을 자체 제공합니다.

---

## 2. 독점 파운데이션 모델 제공자 (Proprietary Foundation Models)
가장 강력한 지능과 혁신적인 기능을 지닌 모델을 자체적으로 훈련하여 API 형태로 직접 제공하는 주체들입니다.

### 🔹 OpenAI API
*   **특징**: GPT-4o, o1-preview 등 업계 벤치마크의 표준 모델 패밀리.
*   **장점**: 놀라운 범용 추론 성능과 뛰어난 한국어 이해력. Tool Calling (Function Calling), Structured Outputs, Assistant API 등 에이전트를 구축하는 데 필요한 도구들을 API 레벨에서 완벽하게 지원합니다.

### 🔹 Anthropic API (Console)
*   **특징**: Claude 3.5 Sonnet, Claude 3 Opus 모델 제공. 
*   **장점**: 최근 코딩(Coding) 지원과 복잡한 텍스트 분석에 있어 개발자들 사이에서 최고의 평가를 받고 있습니다. 환각(Hallucination)이 적고 최대 200k 토큰이라는 거대한 컨텍스트 윈도우 스티어링에 매우 능숙합니다.

### 🔹 Google Gemini API (AI Studio)
*   **특징**: Google의 멀티모달 네이티브 모델 (Gemini 1.5 Pro / Flash).
*   **장점**: 최대 2M(200만) 토큰이라는 압도적인 컨텍스트를 제공하여 수백 페이지의 PDF나 1시간 이상의 동영상/음성을 파인튜닝이나 RAG 없이 롱컨텍스트 추론으로 해결할 수 있습니다. 적극적인 무료 티어 제공도 특징입니다.

---

## 3. 고성능 오픈소스 인퍼런스 특화 (High-Performance Open-Source Specialists)
Llama, Mistral, Qwen과 같은 오픈소스 및 오픈웨이트(Open-weights) 모델을 누구나 API 형태로 호출할 수 있게 해주는 서버리스 클라우드 제공자입니다. **초저지연(Latency)과 저렴한 비용**이 무기입니다.

### 🔹 Together AI
*   **특징**: 업계에서 가장 광범위한 오픈소스 모델 카탈로그를 자랑하는 안정적인 인퍼런스 허브.
*   **장점**: 빠른 처리량과 합리적인 비용. 자체 데이터를 업로드하여 손쉽게 LoRA 파인튜닝을 진행하고 그 결과를 즉시 서빙할 수 있는 플랫폼 통합성.

### 🔹 Groq
*   **특징**: GPU가 아닌 자체 하드웨어 생태계(LPU - Language Processing Unit)를 활용한 제공자.
*   **장점**: 타의 추종을 불허하는 수준의 압도적인 초저지연(Ultra-low Latency). 음성 대화형 에이전트, 실시간 게임 NPC 등 처리 속도가 사용자 경험을 좌우하는 애플리케이션에 필수적입니다.

### 🔹 기타 주요 제공자
*   **Fireworks AI**: VRAM 최적화를 통한 매우 빠른 속도 및 로드 밸런싱. 최상급 가성비 제공.
*   **DeepInfra**: 시장에서 가장 저렴한 토큰당 단가를 제공하여 토큰 비용 절감이 최우선인 배치 작업에 적합.
*   **Replicate**: 모델 추론뿐 아니라 이미지 생성(Stable Diffusion), 음성, 영상 등 다양한 멀티모달 오픈소스 모델들을 간편하게 컨테이너화하여 일관된 API로 배포 가능.

---

## 4. API 애그리게이터 및 라우터 (Aggregators & Routers)
수십 개의 서로 다른 모델 제공사 API를 하나로 묶어(마켓플레이스 통합) 단일 엔드포인트로 제공하는 중계 및 라우팅 서비스입니다.

### 🔹 OpenRouter
*   **특징**: OpenAI, Anthropic부터 Together AI, Groq 등 서드파티 제공자들의 모델까지 수백 가지 모델을 단일(OpenAI 호환) API 포맷으로 통일시켜 제공.
*   **장점**: 
    1. 단 한 번의 결제로 거의 모든 모델 사용 가능 (카드 등록 최소화).
    2. 여러 제공자가 동일 모델(예: Llama 3)을 서빙할 때 실시간 가격과 지연 시간을 비교해 가장 좋은 경로로 라우팅.
    3. 특정 공급자 장애 시 대비가 가능한 **Vender Lock-in(종속) 회피**의 핵심.

*(참고: 클라이언트 사이드의 프록시 라이브러리인 LiteLLM과 달리, OpenRouter는 실제 서버 측 요금을 종합/청구하는 마켓플레이스입니다.)*

---

## 5. 자체 호스팅 및 서빙 (Self-Serving / On-Premise)
데이터 보안(Air-gapped Network)이나 장기적인 클라우드 구독 비용 절감(TCO), 그리고 특수한 파인튜닝 모델을 극도로 제어해야 할 때 직접 인스턴스(GPU 서버)를 띄워 서빙하는 방식입니다. 

자세한 구동 기술은 **[LLM 인퍼런스 병목 및 최적화 방안](./inference-optimization.md)**을 참고하세요.

### 🔹 vLLM
*   **특징**: PagedAttention을 도입하여 메모리 파편화를 극적으로 줄인 현존하는 가장 대중적인 고성능 서빙 엔진.
*   **장점**: 처리량(Throughput)이 매우 높아 프로덕션(Production) 환경에서 표준처럼 사용됩니다. OpenAI 스펙의 API 서버를 손쉽게 열 수 있습니다.

### 🔹 Ollama
*   **특징**: Docker처럼 사용자 친화적인 CLI를 제공해 모델 가중치(GGUF 등)를 다운로드하고 실행하는 과정을 극도로 단축한 도구.
*   **장점**: 로컬 개발자 PC 수준에서 가장 쉽게 구동 및 테스트할 수 있어 로컬 에이전트 개발 및 사이드 프로젝트 생태계의 표준이 되었습니다.

### 🔹 Hugging Face TGI (Text Generation Inference)
*   **특징**: Hugging Face 생태계에서 지원하는 C++ 기반 고성능 추론 엔진.
*   **장점**: 텐서 병렬화 모델이나 대규모 분산 처리에 있어 탁월한 안정성을 보여줍니다.

### 🔹 NVIDIA Triton Inference Server (with TensorRT-LLM)
*   **특징**: NVIDIA가 만든 엔터프라이즈용 딥러닝 추론 게이트웨이.
*   **장점**: TensorRT-LLM과 결합 시 NVIDIA GPU 하드웨어 자원을 가장 극한으로 짜내어 최고의 효율(속도/처리량)을 달성할 수 있으며, 여러 모델 체인을 통합 관리할 수 있으나 진입 장벽이 가장 높습니다.
