---
title: LLM 서빙
nav_order: 8
---

# LLM 서빙

LLM 추론을 위한 다양한 inference engine들을 조사하여 각 엔진의 아키텍처, 특징, 최적화 기법, 장단점을 정리했습니다.

## 주요 LLM Inference Engine

### 1. [vLLM (UC Berkeley)](vllm.md)
- **PagedAttention** 메모리 관리로 메모리 효율성 극대화
- **3-Process 비동기 아키텍처**로 GPU 활용도 최적화
- 높은 처리량과 낮은 지연시간 동시 달성
- OpenAI API 호환성으로 쉬운 마이그레이션

### 2. [SGLang](sglang.md)
- **도메인별 언어(DSL)** 기반 최적화
- 극도로 높은 토큰 생성 속도 (vLLM 대비 2배 이상)
- 메모리 효율적 (vLLM 대비 1/6 메모리 사용)
- 연구 친화적 프레임워크

### 3. [TensorRT-LLM (NVIDIA)](tensor_rt_llm.md)
- **컴파일 기반 최적화**로 NVIDIA GPU 최고 성능
- **EAGLE 기반 Speculative Decoding**으로 3배 처리량 향상
- 극한 성능 최적화와 예측 가능한 실시간 성능
- 금융, 실시간 AI 애플리케이션에 최적

### 4. [Ollama](ollama.md)
- **로컬 배포 중심**의 사용자 친화적 프레임워크
- 원클릭 모델 다운로드 및 실행
- 개인정보 보호와 편의성 우선
- 개인 및 소규모 팀에 적합

### 5. [HuggingFace TGI](huggingface_tgi.md)
- **Rust + Python 하이브리드** 아키텍처
- 긴 프롬프트에서 vLLM 대비 13배 속도 향상
- Hugging Face 생태계 완전 통합
- 프로덕션 준비 완료

### 6. [LMDeploy](lmdeploy.md)
- **이중 엔진 구조** (TurboMind + PyTorch)
- vLLM 대비 최대 1.8배 높은 요청 처리량
- 간단한 설정 (원커맨드 배포)
- 효율적인 멀티라운드 채팅

### 7. [LightLLM](lightllm.md)
- **경량 설계**와 높은 확장성
- 3-process 비동기 협업으로 효율적인 처리
- 빠른 배포와 간단한 관리
- 확장 가능한 아키텍처

### 8. [FlexFlow Serve](flexflow_serve.md)
- **멀티 노드 분산 추론** 최적화
- SpecInfer를 통한 추측 추론
- 단일/멀티 노드에서 1.3-2.4배 성능 향상
- 대규모 LLM 서빙 인프라

### 9. [MLC LLM](mlc_llm.md)
- **Apache TVM 기반** 컴파일 최적화
- 크로스 플랫폼 지원 (브라우저, 모바일, CPU, GPU)
- WebLLM을 통한 브라우저 내 추론
- 엣지 디바이스 최적화

## 성능 비교 및 선택 가이드

### 처리량 중심 비교 (Llama 3 70B, A100 80GB)

| Engine | Tokens/s (100 users) | 특징 |
|--------|---------------------|------|
| LMDeploy | 700 | TurboMind 엔진 |
| TensorRT-LLM | ~700 | NVIDIA 최적화 |
| SGLang | 118 (Hermes-3) | 저지연 특화 |
| vLLM | ~400-500 | 균형잡힌 성능 |
| TGI | ~300-400 | 안정성 중심 |

### 사용 시나리오별 추천

**고성능 프로덕션**
- **TensorRT-LLM**: NVIDIA GPU + 최고 성능 필요시
- **LMDeploy**: 빠른 배포 + 높은 처리량

**연구 및 실험**
- **SGLang**: 최신 기법 실험
- **vLLM**: 범용적 연구 플랫폼

**로컬 배포**
- **Ollama**: 개인/소규모 팀
- **MLC LLM**: 모바일/엣지 디바이스

**엔터프라이즈**
- **TGI**: Hugging Face 생태계
- **vLLM**: 클라우드 확장성

**특수 요구사항**
- **FlexFlow Serve**: 멀티 노드 분산 추론
- **LightLLM**: 경량 고속 서빙

## 선택 시 고려사항

각 inference engine은 서로 다른 디코딩 전략과 최적화 기법을 적용하여 특정 사용 사례에 최적화되어 있습니다. 선택 시 다음 요소들을 종합적으로 고려해야 합니다:

- **하드웨어 환경**: GPU 종류, 메모리 용량, CPU 성능
- **성능 요구사항**: 처리량, 지연시간, 동시 사용자 수
- **개발 복잡성**: 설정 난이도, 학습 곡선, 유지보수
- **확장성**: 수평/수직 확장, 클라우드 배포, 분산 처리
- **생태계**: 모델 지원, 커뮤니티, 문서화 품질
- **라이선스**: 상업적 사용, 오픈소스 정책

자세한 내용은 각 엔진의 상세 페이지를 참조하시기 바랍니다.
