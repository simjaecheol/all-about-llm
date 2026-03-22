---
layout: default
title: 추론 가속
parent: LLM 최적화
nav_order: 2
---

# 추론 가속 (Inference Acceleration)

LLM의 추론 성능을 향상시키는 기법들은 크게 메모리 관리(VRAM), 계산 효율화(Attention), 디코딩 전략으로 나뉩니다. 2025년 현재 가장 널리 쓰이는 가속 기술들을 소개합니다.

---

## 1. PagedAttention (vLLM)
- **핵심 원리:** 운영체제의 가상 메모리 관리(Paging) 개념을 KV 캐시에 도입한 기술입니다.
- **특징:**
  - **메모리 효율:** 기존 방식은 KV 캐시를 위해 미리 큰 메모리를 할당하여 낭비(Fragment)가 심했으나, PagedAttention은 블록 단위로 쪼개어 동적 할당합니다.
  - **성능:** 메모리 낭비를 4% 이하로 줄여, 동일한 GPU에서 더 많은 동시 요청(Throughput)을 처리할 수 있게 합니다.

## 2. Flash Attention (1, 2, 3)
- **핵심 원리:** GPU의 메모리 계층 구조(SRAM vs HBM)를 고려하여 IO 연산을 최적화한 Attention 알고리즘입니다.
- **특징:**
  - **Flash Attention 1 & 2:** 커널 융합(Kernel Fusion)을 통해 중간 계산 데이터를 HBM에 쓰지 않고 SRAM에서 처리하여 속도를 높였습니다.
  - **Flash Attention 3 (2025):** 최신 NVIDIA H100 GPU의 하드웨어 특성(Warp-specialized kernels)을 활용하여 FP8 및 비동기 연산을 극대화, 이전 버전 대비 2배 이상의 속도 향상을 제공합니다.

## 3. 투기적 디코딩 (Speculative Decoding)
- **핵심 원리:** 작고 빠른 모델(Draft Model)이 미리 여러 개의 토큰을 예측하고, 거대 모델(Target Model)이 이를 한 번에 검증하는 방식입니다.
- **특징:**
  - **속도 향상:** 거대 모델이 매 토큰마다 직접 계산하는 대신 '병렬 검증'만 수행하므로, 지연 시간(Latency)을 획기적으로 단축할 수 있습니다.
  - **성능 유지:** 검증 결과가 틀리면 거대 모델이 즉시 수정하므로, 최종 결과의 품질은 거대 모델 단독 사용 시와 100% 동일합니다.

## 4. KV 캐시 최적화
- **핵심 원리:** 대화가 길어질수록 기하급수적으로 커지는 KV 캐시 용량을 줄이는 기술입니다.
- **주요 기법:**
  - **Prefix Caching:** 공통된 시스템 프롬프트나 이전 대화 맥락을 재사용하여 중복 계산을 방지합니다.
  - **Quantized KV Cache:** KV 캐시 데이터 자체를 4비트 또는 8비트로 양자화하여 VRAM 사용량을 절반으로 줄입니다.

---

## 추론 가속 기술 비교 (2025)

| 기술 | 주요 효과 | 추천 상황 |
| :--- | :--- | :--- |
| **PagedAttention** | 처리량(Throughput) 증대 | 대규모 동시 접속 서비스 (vLLM) |
| **Flash Attention 3** | 계산 시간 단축 | H100/B200 기반 고성능 인프라 |
| **Speculative Decoding** | 지연 시간(Latency) 단축 | 실시간 대화형 AI 서비스 |
| **Prefix Caching** | 초기 응답 시간 단축 | RAG, 장문 대화 반복 서비스 |
