---
title: LMDeploy
parent: LLM 서빙
nav_order: 6
---

# LMDeploy (MMRazor/MMDeploy)

## 아키텍처 및 핵심 기술

### 이중 엔진 구조
- TurboMind: FasterTransformer 기반 고성능 엔진
- PyTorch 백엔드: 호환성 및 실험용

### 고급 최적화 기법
- Persistent batching (continuous batching)
- Blocked KV cache
- Dynamic split&fuse
- 고성능 CUDA 커널

## 지원하는 디코딩 전략

### 양자화 최적화
- Weight-only 및 KV 양자화
- 4-bit 추론 성능 FP16 대비 2.4배 향상
- INT8 KV Cache 지원

### 대화형 추론 모드
- 멀티라운드 대화 시 KV 캐시 활용
- 히스토리 캐싱으로 반복 처리 방지

## 성능 특성

### 벤치마크 결과
- vLLM 대비 최대 1.8배 높은 요청 처리량
- A100에서 Llama 3 70B Q4: 700 tokens/s (100 동시 사용자)
- 지연시간 40-60ms

## 장단점

### 장점
- 뛰어난 디코딩 속도
- 간단한 설정 (원커맨드 배포)
- 효율적인 멀티라운드 채팅
- 양자화 품질 검증

### 단점
- TurboMind는 NVIDIA GPU 전용
- 슬라이딩 윈도우 어텐션 미지원 (Mistral 등)
- PyTorch 엔진은 상대적으로 느림

## 사용 시나리오
- 빠른 배포가 필요한 환경
- 높은 처리량 요구사항
- 멀티라운드 채팅 애플리케이션
- NVIDIA GPU 환경에서의 최적화된 성능
