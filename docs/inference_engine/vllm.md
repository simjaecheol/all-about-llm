---
title: vLLM
parent: LLM 서빙
nav_order: 1
---

# vLLM (UC Berkeley)

## 아키텍처 및 핵심 기술

### PagedAttention 메모리 관리
- KV 캐시의 메모리 효율적 관리를 위한 핵심 혁신
- 메모리 단편화를 방지하고 동적 배치 처리 최적화
- GPU 메모리 사용량을 크게 줄이면서 처리량 향상

### 3-Process 비동기 아키텍처
- AsyncLLM: 비동기 래퍼로 요청 관리
- EngineCore: 핵심 추론 엔진, 모델 실행 담당
- 토큰화, 모델 추론, 디토큰화를 비동기로 처리하여 GPU 활용도 극대화

## 지원하는 디코딩 전략

### 고급 디코딩 알고리즘
- Parallel sampling, beam search
- Continuous batching으로 동적 요청 처리
- Speculative decoding 지원
- Chunked prefill 최적화

### 양자화 기법
- GPTQ, AWQ, AutoRound 지원
- INT4, INT8, FP8 양자화
- FlashAttention 및 FlashInfer 통합

## 성능 특성

### 처리량 우선
- 높은 처리량과 낮은 지연시간 동시 달성
- 표준 Hugging Face 파이프라인 대비 2-4배 성능 향상
- GPU 활용률 최적화로 비용 효율성 제공

### 하드웨어 지원
- NVIDIA GPU, AMD CPU/GPU, Intel CPU/GPU
- TPU, AWS Neuron, PowerPC CPU 지원
- 다중 GPU 텐서/파이프라인 병렬 처리

## 장단점

### 장점
- OpenAI API 호환성으로 쉬운 마이그레이션
- 동적 배치 처리와 캐싱으로 실시간 추론 최적화
- 수평 확장성 및 클라우드 네이티브 배포 지원
- 광범위한 모델 지원 (Llama, Mistral, Mixtral 등)

### 단점
- 특정 하드웨어 최적화 부족 (TensorRT-LLM 대비)
- 대용량 입력 처리 시 지연시간 증가 가능
- 초기 설정 시 GPU 인프라 요구사항

## 사용 시나리오
- 클라우드 기반 LLM 서빙
- 높은 처리량이 필요한 프로덕션 환경
- 다양한 하드웨어 환경에서의 배포
- 연구 및 실험을 위한 범용 플랫폼
