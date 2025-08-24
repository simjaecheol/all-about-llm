---
title: HuggingFace TGI
parent: LLM 서빙
nav_order: 5
---

# HuggingFace Text Generation Inference (TGI)

## 아키텍처 및 핵심 기술

### Rust + Python 하이브리드
- 고성능 Rust 백엔드와 Python API
- OpenTelemetry 분산 추적, Prometheus 메트릭
- 프로덕션 그레이드 서빙 시스템

### 고급 최적화
- Flash Attention, Paged Attention
- Continuous batching 및 SSE 스트리밍
- 다양한 양자화 (bitsandbytes, GPTQ, AWQ, Marlin)

## TGI 3.0 성능 향상
- 긴 프롬프트에서 vLLM 대비 13배 속도 향상
- 3배 더 많은 토큰 처리 (L4 24GB에서 30K 토큰 vs vLLM 10K)
- 메모리 사용량 최적화

## 지원하는 디코딩 전략

### 풍부한 디코딩 기능
- Speculative decoding으로 2배 지연시간 감소
- Guidance/JSON 구조화 출력
- 워터마킹 지원
- 고급 logits 처리

## 성능 특성

### 균형잡힌 성능
- 다양한 하드웨어 지원 (NVIDIA, AMD ROCm, Intel, TPU)
- 안정적인 프로덕션 성능
- OpenAI API 완전 호환

## 장단점

### 장점
- Hugging Face 생태계 완전 통합
- 프로덕션 준비 완료
- 광범위한 하드웨어 지원
- 풍부한 기능 세트

### 단점
- 특정 하드웨어 최적화 부족
- 설정 복잡성
- 일부 시나리오에서 성능 제한

## 사용 시나리오
- Hugging Face 생태계 기반 프로젝트
- 프로덕션 환경에서의 안정적인 LLM 서빙
- 다양한 하드웨어 환경에서의 배포
- OpenAI API 호환성이 필요한 환경
