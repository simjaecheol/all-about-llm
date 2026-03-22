---
title: vLLM
parent: LLM 서빙
nav_order: 1
---

# vLLM (UC Berkeley)

## 아키텍처 및 핵심 기술

### vLLM V1 Alpha 아키텍처 (2025)
- **비동기 스케줄링**: 스케줄링 오버헤드를 최소화하기 위한 비동기 엔진 구조로의 대대적 전환
- **성능 최적화**: 텍스트와 멀티모달 데이터를 동일한 효율로 처리하기 위한 전역 메모리 관리자 도입

### PagedAttention 및 자동 접두사 캐싱 (Prefix Caching)
- **PagedAttention**: KV 캐시의 메모리 단편화를 방지하는 핵심 기술
- **Hash-based Prefix Caching**: V1에서 도입된 해시 기반 매칭을 통해 멀티턴 대화 및 공통 프롬프트의 KV 캐시 재사용성 극대화 (이미지/오디오 임베딩 캐싱 포함)

### Chunked Prefill 및 스케줄링
- **Chunked Prefill**: 긴 프롬프트를 작은 청크(Chunk)로 나누어 처리하여 첫 토큰 생성 시간(TTFT)과 토큰 간 지연 시간(ITL)의 균형을 최적화
- **Continuous Batching**: 동적 요청 처리를 통해 GPU 활용도 극대화

## 지원하는 디코딩 전략 및 멀티모달

### 멀티모달(Multi-modal) 지원 강화
- **Vision & Audio**: LLaVA, Qwen2-VL, Qwen2-Audio 등 최신 모델 지원
- **Encoder Caching**: 비전/오디오 인코더의 출력값을 캐싱하여 반복적인 미디어 처리 비용 절감

### 고급 디코딩 및 양자화
- **Speculative Decoding**: EAGLE-3 등 최신 기법 지원으로 처리량 향상
- **FlashInfer 통합**: CUDA 백엔드에서 FlashInfer를 기본 사용하여 Blackwell 등 최신 하드웨어 성능 대응
- **양자화**: GPTQ, AWQ, FP8, INT4/INT8 지원 및 AutoRound 통합

## 성능 및 하드웨어 지원

### 하드웨어 확장성
- NVIDIA GPU (Blackwell 지원), AMD ROCm (MI300X), Intel (Gaudi/CPU), TPU, AWS Inferentia/Neuron 등 광범위한 하드웨어 지원
- 다중 GPU 텐서/파이프라인 병렬 처리 최적화

## 장단점

### 장점
- **업계 표준**: 가장 넓은 모델 지원 범위(100+ 아키텍처)와 성숙한 생태계
- **멀티모달 최적화**: 텍스트와 미디어가 혼합된 복잡한 워크플로우에 강력함
- **OpenAI API 호환성**: 쉬운 마이그레이션과 클라우드 네이티브 배포 지원

### 단점
- **오버헤드**: 순수 C++ 엔진(LMDeploy 등) 대비 스케줄링 오버헤드가 존재할 수 있음
- **복잡도**: V1 전환기에 따른 설정 옵션의 변화 및 학습 곡선

## 사용 시나리오
- 클라우드 기반 범용 LLM/VLM 서빙
- 다양한 모델 아키텍처와 하드웨어를 혼용하는 환경
- 멀티턴 대화가 빈번한 에이전트 서비스

