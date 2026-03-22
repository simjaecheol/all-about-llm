---
title: LLM 서빙
nav_order: 8
---

# LLM 서빙

LLM 추론을 위한 다양한 inference engine들을 조사하여 각 엔진의 아키텍처, 특징, 최적화 기법, 장단점을 정리했습니다.

## 주요 LLM Inference Engine

### 1. [vLLM (UC Berkeley)](vllm.md)
- **V1 Alpha 아키텍처**: 비동기 스케줄링과 성능 오버헤드 최소화로의 대대적 전환
- **Chunked Prefill**: 긴 프롬프트를 쪼개어 처리하여 TTFT와 ITL(지연 시간)의 균형 최적화
- **Automatic Prefix Caching**: 해시 기반 KV 캐시 재사용으로 멀티턴 대화 속도 극대화
- **멀티모달 지원**: Vision(LLaVA, Qwen2-VL) 및 Audio 모델의 인코더 캐싱 지원

### 2. [SGLang](sglang.md)
- **RadixAttention**: 기수 트리(Radix Tree) 구조를 이용한 계층적 KV 캐시 관리로 vLLM 대비 높은 적중률
- **압도적 처리량**: Llama 3.1 기준 vLLM 대비 최대 29%~116% 높은 토큰 처리량 달성
- **구조적 생성 최적화**: JSON, XML 등 구조화된 데이터 생성 시 Compressed FSM으로 속도 향상
- **DeepSeek 최적화**: FlashMLA 및 DeepGemm 커널 통합으로 DeepSeek V3/R1 최적 추론 지원

### 3. [TensorRT-LLM (NVIDIA)](tensor_rt_llm.md)
- **Blackwell (GB200) 지원**: 2세대 Transformer Engine을 활용한 **FP4(4-bit)** 정밀도 추론 지원
- **EAGLE-3 투기적 디코딩**: 하드웨어 성능을 넘어서는 최대 4배의 추가 속도 향상
- **Disaggregated Serving**: Prefill과 Decode 단계를 서로 다른 GPU 노드에서 처리하여 효율성 극대화
- **PyTorch 워크플로우**: v1.0부터 PyTorch 기반 설계가 기본 채택되어 개발 편의성 개선

### 4. [Ollama](ollama.md)
- **병렬 도구 호출 (Parallel Tool Calling)**: 한 번의 응답에서 여러 함수를 동시에 호출하고 처리
- **MCP (Model Context Protocol) 지원**: Anthropic의 개방형 표준을 통한 외부 도구 및 데이터 연결
- **동시성 관리**: 다중 GPU 환경에서 메모리 부족(OOM)을 방지하는 지능형 모델 스케줄링
- **API 호환성**: OpenAI 뿐만 아니라 Anthropic Messages API와의 완벽한 호환성 제공

### 5. [HuggingFace TGI](huggingface_tgi.md)
- **유지보수 모드 전환**: 2025년 12월부로 신규 기능 추가 중단 (vLLM, SGLang 사용 권장)
- **TGI v3.0**: 긴 프롬프트(200k+) 처리 시 이전 버전 대비 13배 속도 향상 달성
- **Zero Config**: 하드웨어별 최적 설정을 자동으로 적용하는 편의성 제공

### 6. [LMDeploy](lmdeploy.md)
- **TurboMind 엔진**: C++/CUDA 기반의 고도화된 엔진으로 vLLM 대비 최대 1.8배 높은 처리량
- **MXFP4 지원**: Microscaling Formats 지원으로 H800 등 최신 하드웨어 성능 극대화
- **양자화 특화**: W4A16, KV8/KV4 등 다양한 양자화 기법에서 업계 최고 수준의 속도 제공

### 7. [LightLLM](lightllm.md)
- **LiteLLM 통합**: LLM 게이트웨이인 LiteLLM과의 결합으로 관리 편의성 및 생태계 확장
- **Token Attention**: 효율적인 KV 캐시 관리를 통해 경량 서빙 환경에서의 처리량 개선

### 8. [FlexFlow Serve](flexflow_serve.md)
- **트리 기반 투기적 추론**: 작은 모델(SSM)을 이용한 트리 형태의 다중 토큰 예측 및 병렬 검증
- **CPU 오퍼로딩**: 대형 모델을 단일 GPU에서 구동하기 위한 효율적인 메모리 관리 기법 도입

### 9. [MLC LLM](mlc_llm.md)
- **MLCEngine**: iOS/Android 네이티브 앱 통합을 위한 Swift/Kotlin 바인딩 제공
- **WebLLM**: WebGPU 가속을 통한 브라우저 내 서버리스 LLM 추론 (Llama 3, Qwen 2.5 지원)
- **JSON 모드**: 브라우저 환경에서도 구조화된 데이터 생성 및 함수 호출 지원

### 10. [KTransformers](ktransformers.md)
- **하이브리드 추론**: 일반 PC(RTX 4090 + RAM)에서 DeepSeek-V3/R1(671B)과 같은 초거대 모델 구동 가능
- **DeepSeek 전용 최적화**: FlashMLA, DeepGemm 등 최신 전용 커널을 내장하여 추론 효율 극대화

### 11. [ExLlamaV2](exllamav2.md)
- **NVIDIA 극한 속도**: 순수 C++/CUDA 기반으로 Llama 아키텍처에서 타 엔진 대비 압도적인 디코딩 속도 달성
- **EXL2 양자화**: 정밀한 비트 레이트 조절로 가진 VRAM에 최적화된 모델 서빙 가능

### 12. [Aphrodite Engine](aphrodite.md)
- **vLLM 대안 서빙**: vLLM 아키텍처를 계승하면서도 프로덕션 환경에서의 안정성과 버그 수정에 집중
- **멀티 테넌트 최적화**: 대규모 사용자 대상 API 서비스에서 균일한 품질의 추론 결과 제공

## 성능 비교 및 선택 가이드

### 처리량 중심 비교 (Llama 3 70B, A100 80GB)

| Engine | Tokens/s (100 users) | 특징 |
|--------|---------------------|------|
| LMDeploy | 700 | TurboMind 엔진 |
| TensorRT-LLM | ~700 | NVIDIA 최적화 |
| ExLlamaV2 | 800+ (Single user) | NVIDIA 전용 극한의 속도 |
| SGLang | 118 (Hermes-3) | 저지연 특화 |
| vLLM | ~400-500 | 균형잡힌 성능 |
| TGI | ~300-400 | 안정성 중심 |

### 사용 시나리오별 추천

**고성능 프로덕션**
- **TensorRT-LLM**: NVIDIA GPU + 최고 성능 필요시
- **LMDeploy**: 빠른 배포 + 높은 처리량
- **Aphrodite Engine**: 안정성 중심의 서비스 운영

**연구 및 실험**
- **SGLang**: 최신 기법 실험
- **vLLM**: 범용적 연구 플랫폼

**로컬 배포**
- **Ollama**: 개인/소규모 팀 (최고의 편의성)
- **ExLlamaV2**: NVIDIA GPU 환경에서의 극한의 텍스트 생성 속도
- **KTransformers**: 일반 PC에서 DeepSeek-V3/R1 등 초거대 모델 구동
- **MLC LLM**: 모바일/엣지 디바이스

**엔터프라이즈**
- **TGI**: Hugging Face 생태계 유지보수
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
