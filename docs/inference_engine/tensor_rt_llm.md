---
title: TensorRT-LLM
parent: LLM 서빙
nav_order: 3
---

# TensorRT-LLM (NVIDIA)

## 아키텍처 및 핵심 기술

### 컴파일 기반 최적화
- Model Definition API를 통한 모델 정의
- TensorRT 엔진으로 컴파일하여 NVIDIA GPU 최적화
- C++ 런타임 권장 (Python 런타임도 지원)

### 고급 최적화 기법
- In-flight batching으로 동적 배치 처리
- Paged KV cache, Quantized KV cache
- 커스텀 FP8 양자화
- 커스텀 CUDA 커널 최적화

## 지원하는 디코딩 전략

### Speculative Decoding
- EAGLE 기반 추측 디코딩으로 최대 3배 처리량 향상
- Tree attention과 multi-round speculative decoding
- CUDA Graph 최적화로 커널 실행 오버헤드 제거

### 샘플링 전략
- Beam search, top-K, top-P 샘플링
- Temperature scaling 등 다양한 logits warper 지원
- 고성능 샘플링 기능

## 성능 특성

### 극한 성능 최적화
- NVIDIA 하드웨어에서 최고 성능 달성
- Llama 3.3 70B에서 3배 처리량 향상
- 지연시간 최적화 (A100에서 50ms 미만)
- 예측 가능한 실시간 성능 보장

### 메모리 효율성
- 고급 양자화로 메모리 사용량 대폭 감소
- 제한된 GPU에서도 대형 모델 배포 가능

## 장단점

### 장점
- NVIDIA GPU에서 최고의 성능
- 금융, 실시간 AI 애플리케이션에 적합
- 고급 양자화 기법 지원
- 예측 가능한 성능

### 단점
- 복잡한 설정 및 튜닝 필요
- NVIDIA 하드웨어 종속성 (벤더 락인)
- 동적 배치 크기 변화에 취약
- 개발 및 배포 시간 오버헤드

## 사용 시나리오
- NVIDIA GPU 환경에서의 최고 성능 요구
- 금융, 실시간 AI 애플리케이션
- 예측 가능한 성능이 중요한 프로덕션 환경
- NVIDIA 생태계 기반 인프라
