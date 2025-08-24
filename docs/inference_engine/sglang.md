---
title: SGLang
parent: LLM 서빙
nav_order: 2
---

# SGLang

## 아키텍처 및 핵심 기술

### 도메인별 언어
- Python에 임베디드된 DSL (Domain-Specific Language)
- 저지연 추론과 동적 작업 분산에 최적화
- 멀티 레벨 최적화 지원

### 고급 최적화 기법
- FP8 blockwise quantization tuning
- CUDA Graph 실행으로 커널 오버헤드 감소
- Torch Compile을 통한 커널 융합
- NextN Speculative Decoding (EAGLE-2)

## DeepSeek R1 최적화 결과
- 단일 노드에서 2배 처리량 향상
- 긴 컨텍스트와 높은 동시성 처리 최적화
- 수십 명의 사용자 동시 처리 가능

## 지원하는 디코딩 전략

### 고급 디코딩 기법
- DP Attention for Multi-Head Latent Attention
- FlashInfer MLA 최적화
- Overlap Scheduler로 CPU-GPU 오버랩
- FP8 정확도 개선 (blockwise/tilewise scaling)

## 성능 특성

### 벤치마크 결과
- Qwen-1.5B: 210.48 tokens/s (vLLM 98.27 대비)
- Hermes-3: 118.34 tokens/s (vLLM 60.69 대비)
- 낮은 지연시간과 높은 GPU 활용률

## 장단점

### 장점
- 극도로 높은 토큰 생성 속도
- 메모리 효율적 (vLLM 대비 1/6 메모리 사용)
- 긴 컨텍스트 처리에 최적화
- 연구 친화적 프레임워크

### 단점
- 상대적으로 새로운 프레임워크
- 제한적인 모델 지원
- 복잡한 최적화 설정

## 사용 시나리오
- 극한 성능이 필요한 연구 환경
- 긴 컨텍스트 처리 요구사항
- 최신 추론 기법 실험
- 저지연 추론이 중요한 애플리케이션
