---
title: HuggingFace TGI
parent: LLM 서빙
nav_order: 5
---

# HuggingFace Text Generation Inference (TGI)

## 2025년 주요 업데이트 및 상태 변경

### 유지보수 모드 전환 (2025년 12월)
- **전략적 변화**: Hugging Face는 TGI를 **유지보수 모드(Maintenance Mode)**로 전환했습니다.
- **배경**: vLLM, SGLang 등 오픈소스 추론 엔진이 충분히 성숙함에 따라, 신규 프로젝트에는 이 엔진들을 사용할 것을 권장하고 있습니다.
- **추후 계획**: 새로운 기능 추가보다는 안정성 확보 및 보안 업데이트에 집중할 예정입니다.

## 아키텍처 및 핵심 기술

### Rust + Python 하이브리드
- 고성능 Rust 백엔드와 유연한 Python API의 결합.
- OpenTelemetry 분산 추적 및 Prometheus 메트릭을 통한 프로덕션급 모니터링 지원.

### TGI v3.0 성능 혁신
- **긴 프롬프트(Long Prompt)**: 200k 이상의 매우 긴 프롬프트 처리 시 vLLM 대비 최대 13배 빠른 속도 기록.
- **메모리 효율**: L4 24GB GPU 기준, vLLM 대비 약 3배 더 많은 토큰(30k vs 10k)을 수용 가능.
- **Zero Config**: 하드웨어와 모델을 자동으로 분석하여 별도의 플래그 설정 없이도 최적의 성능을 끌어내는 자동화 기능.

## 지원하는 디코딩 전략 및 특징

### 풍부한 디코딩 옵션
- **Speculative Decoding**: 지연 시간을 절반 수준으로 줄이는 투기적 디코딩 기술 지원.
- **구조화된 출력**: Guidance를 이용한 JSON 등의 형식 보증 출력.
- **다양한 양자화**: FP8, bitsandbytes, GPTQ, AWQ, Marlin 등 폭넓은 정밀도 지원.

## 성능 특성

### 하드웨어 범용성
- NVIDIA GPU뿐만 아니라 AMD ROCm, Intel Gaudi, Google TPU 지원.
- OpenAI API와의 완전한 호환성 유지.

## 장단점

### 장점
- **생태계 통합**: Hugging Face 생태계 및 자격 증명(API Token)과의 원활한 통합.
- **안정성**: 수많은 엔터프라이즈 환경에서 검증된 견고한 설계.
- **편의성**: Zero Config 기능으로 누구나 쉽게 고성능 서빙 가능.

### 단점
- **성장 정체**: 유지보수 모드 전환으로 인한 미래 기술(Blackwell FP4 등) 도입 지연 가능성.
- **최고 처리량**: 일부 최신 벤치마크에서는 SGLang이나 vLLM V1에 비해 뒤처질 수 있음.

## 사용 시나리오
- 기존 Hugging Face 인프라를 활용하는 프로젝트의 안정적인 유지보수
- 매우 긴 컨텍스트를 처리해야 하는 특수 워크로드
- 다양한 하드웨어(TPU, Intel 등)에서의 안정적인 배포
- 복잡한 설정 없이 빠르게 고품질 서빙 환경을 구축해야 할 때
