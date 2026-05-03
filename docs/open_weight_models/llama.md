---
layout: default
title: Llama (Meta)
parent: 오픈 웨이트 모델 (Open-Weight Models)
nav_order: 1
---

# Llama (Meta)

## 개요
Meta(구 Facebook)에서 개발한 Llama(Large Language Model Meta AI) 시리즈는 오픈 웨이트 모델 생태계의 가장 강력한 표준입니다. Llama 1의 발표 이후 오픈 소스 커뮤니티의 폭발적인 성장을 이끌었으며, 현재는 상업적 이용이 가능한 Llama 3와 Llama 4 시리즈로 진화했습니다.

## 주요 시리즈 및 특징

### 1. Llama 1 & 2
*   **Llama 1**: 연구 목적으로만 공개되었으나, 로컬 LLM 구동의 시작점이 됨.
*   **Llama 2**: 상업적 이용 허용. 7B, 13B, 70B 모델 제공. RLHF(Reinforcement Learning from Human Feedback)를 본격적으로 도입하여 안전성과 유용성을 개선.

### 2. Llama 3 (3.1 & 3.2)
*   **Llama 3.1**: 최대 405B 파라미터 모델을 포함하며, GPT-4o 급의 성능을 오픈 웨이트로 제공. 128K 컨텍스트 윈도우 지원.
*   **Llama 3.2**: 비전(Vision) 기능을 갖춘 멀티모달 모델(11B, 90B)과 모바일에 최적화된 경량 모델(1B, 3B) 추가.

### 3. Llama 4 (2025-2026 로드맵)
*   **에이전트 중심 설계**: 복잡한 추론과 도구 사용 능력 대폭 강화.
*   **네이티브 멀티모달**: 텍스트, 이미지, 음성을 하나의 모델에서 통합 처리.
*   **추론 성능 최적화**: MoE(Mixture of Experts) 도입 가능성 제기 및 성능 향상.

## 기술적 강점
*   **GQA (Grouped-Query Attention)**: 추론 시 메모리 효율성 극대화.
*   **RoPE (Rotary Positional Embeddings)**: 긴 문맥 처리를 위한 효율적인 위치 인코딩.
*   **생태계 호환성**: 대부분의 LLM 라이브러리(llama.cpp, vLLM, Ollama 등)에서 기본적으로 지원.

## 주요 파생 모델: Hermes 시리즈 (Nous Research)

Llama 아키텍처의 대표적인 에이전트 특화 파생 모델로, **Nous Research**가 Llama 3.1을 기반으로 개발한 Hermes 시리즈가 있습니다. 이 모델들은 자율형 에이전트([Hermes Agent](../open_source_project/hermes-agent.md))의 백엔드 추론 엔진으로 설계되어, Llama 생태계의 확장 가능성을 입증하는 핵심 사례입니다.

| 비교 항목 | Hermes 3 (2024.08) | Hermes 4 (2025.08) |
| :--- | :--- | :--- |
| **파라미터** | 3B, 8B, 70B, 405B | 14B, 70B, 405B |
| **훈련 방식** | DPO + LoRA(r=32, α=16) + NEFTune | 60B 토큰 추론 궤적 데이터셋 |
| **추론 방식** | 단일 패스 지시 이행 | `<think>` 태그 기반 하이브리드 추론 |
| **핵심 강점** | 안정적 JSON 출력, 함수 호출 정확성 SOTA | 무검열 중립 정렬, LiveCodeBench 54.6% |

*   **Hermes 3**: 중립적 정렬(Neutrally-aligned) 기반 범용 모델. 에이전트 프레임워크에 필수적인 함수 호출 및 구조화된 출력 안정성에서 오픈 웨이트 모델 중 최상위 수준을 달성했습니다.
*   **Hermes 4**: 사용자가 `<think>` 태그로 모델의 사고 깊이를 동적으로 제어하는 하이브리드 추론 모델. 에이전트 파이프라인에서 안전 필터로 인한 실행 거부 문제를 해결하기 위해 **검열 없는 사용자 지시 이행**을 극대화했습니다.

## 평가 및 벤치마크
Llama 시리즈는 MMLU, GPQA, HumanEval 등 주요 지표에서 오픈 소스 모델 중 항상 최상위권을 유지하며, 특히 지시 이행(Instruction Following) 능력에서 높은 신뢰도를 보입니다.

