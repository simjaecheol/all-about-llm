---
title: LLaMA Factory
parent: LLM 학습 프레임워크
nav_order: 4
---

# LLaMA Factory

## 개요
LlamaFactory는 베이항대학교(Beihang University) Yaowei Zheng 연구팀이 개발한 **통합형 LLM 미세조정 전용 플랫폼**으로, 100개 이상의 대형 언어모델을 코드 작성 없이 미세조정할 수 있도록 설계되었다. 2024년 ACL 데모 논문으로 발표되어 학술적 검증을 받았으며, 현재 GitHub에서 55,900개의 스타를 기록하며 Amazon, NVIDIA, Aliyun 등 주요 기업에서 활용되고 있다.

## 핵심 철학과 접근 방식

### 통합성(Unification)과 접근성(Accessibility)
LlamaFactory의 가장 큰 특징은 **"Unified Efficient Fine-Tuning"**이라는 철학으로, 다양한 모델과 미세조정 기법을 하나의 프레임워크로 통합한 점이다. 특히 **LlamaBoard**라는 웹 기반 GUI를 통해 비전문가도 복잡한 설정 없이 미세조정을 수행할 수 있도록 했다.

### Day-0/Day-1 모델 지원 정책
최신 모델에 대한 **당일 또는 익일 지원**을 목표로 하여, Llama 4, Gemma 3, Qwen3, GLM-4.1V, InternLM 3 등의 최신 모델을 즉시 사용할 수 있다. 이는 연구자와 실무진이 최신 모델을 빠르게 활용할 수 있게 하는 중요한 차별화 요소이다.

## 지원 모델 생태계 (100+ 모델)

### 주요 모델 패밀리 지원
- **LLaMA 계열**: LLaMA/LLaMA 2/LLaMA 3-3.3/LLaMA 4 (1B~402B)
- **Qwen 계열**: Qwen1-2.5/Qwen3/Qwen2-Audio/Qwen2.5-Omni/Qwen2-VL (0.5B~235B)
- **DeepSeek 계열**: DeepSeek/DeepSeek 2.5/3/DeepSeek R1 Distill (1.5B~671B)
- **멀티모달**: LLaVA-1.5/NeXT, PaliGemma/PaliGemma2, InternVL 2.5-3, MiniCPM-V 등
- **기타**: Gemma/Gemma 2-3, GLM-4, Mistral/Mixtral, ChatGLM, Phi, Yi 등

### 템플릿 시스템
각 모델별로 최적화된 **대화 템플릿**을 제공하여 모델 고유의 프롬프트 형식을 자동 처리하며, 사용자 정의 템플릿 추가도 가능하다.

## 미세조정 방법론의 포괄성

### 학습 단계별 접근법
- **사전학습**: 연속 사전학습(Continuous Pre-training) 지원
- **지도학습**: (멀티모달) 지도 미세조정(SFT), 보상 모델 학습
- **강화학습**: PPO, DPO, KTO, ORPO, SimPO 등 최신 선호도 학습 기법

### 파라미터 효율성 기법
- **Full Fine-tuning**: 16비트 전체 파라미터 학습
- **Freeze Fine-tuning**: 일부 레이어 고정 학습
- **LoRA 계열**: LoRA, QLoRA(2/3/4/5/6/8비트), DoRA, LongLoRA, LoRA+, LoftQ, PiSSA
- **메모리 최적화**: GaLore, BAdam, APOLLO, Adam-mini 등 고급 최적화 알고리즘

## 성능 최적화와 확장성

### 하드웨어 요구사항 최적화
|방법|비트|7B|70B|메모리 절약 효과|
|---|---|---|---|---|
|Full (bf16/fp16)|32|120GB|1200GB|기준|
|Full (pure_bf16)|16|60GB|600GB|50% 절약|
|LoRA/GaLore|16|16GB|160GB|87% 절약|
|QLoRA 8비트|8|10GB|80GB|92% 절약|
|QLoRA 4비트|4|6GB|48GB|95% 절약|
|QLoRA 2비트|2|4GB|24GB|98% 절약|

### 가속화 기술 통합
- **FlashAttention-2**: 메모리 효율적 어텐션 연산
- **Unsloth**: LoRA 학습에서 170% 속도 향상, 50% 메모리 절약
- **Liger Kernel**: 특정 모델(Qwen2-VL 등)에 대한 최적화
- **vLLM/SGLang**: 추론 시 270% 속도 향상

## LlamaBoard: 웹 기반 통합 인터페이스

### No-Code 미세조정 환경
LlamaBoard는 Gradio 기반의 직관적인 웹 인터페이스로, 다음 기능을 제공한다:
- **모델 선택**: 100+ 모델에서 원클릭 선택
- **데이터셋 관리**: 업로드, 미리보기, 전처리 자동화
- **하이퍼파라미터 설정**: GUI 기반 파라미터 조정
- **학습 모니터링**: 실시간 손실 그래프, TensorBoard/W&B/SwanLab 연동
- **체크포인트 관리**: 저장, 재개, 선택 기능
- **추론 테스트**: 학습된 모델 즉시 테스트

### 분산 학습 지원
단일 노드 분산 학습을 웹 UI에서 직접 설정할 수 있어, 복잡한 분산 설정을 GUI로 간단하게 처리한다.

## 데이터셋 생태계와 전처리

### 내장 데이터셋 (100+ 개)
- **사전학습**: Wiki Demo, RefinedWeb, RedPajama V2, SkyPile 등
- **지도 미세조정**: Stanford Alpaca, UltraChat, ShareGPT, Magpie 시리즈 등
- **선호도 학습**: DPO mixed, UltraFeedback, RLHF-V 등
- **멀티모달**: LLaVA mixed, Pokemon-gpt4o-captions 등

### 유연한 데이터 처리
- **포맷 지원**: HuggingFace, ModelScope, Modelers Hub, S3/GCS 클라우드 저장소
- **스트리밍**: 대용량 데이터셋의 배치 단위 스트리밍 처리
- **전처리 자동화**: 토큰화, 패킹, contamination-free 처리

## 실전 워크플로우와 사용성

### 3단계 CLI 워크플로우
```bash
# 1. 미세조정 실행
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# 2. 추론 테스트
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml

# 3. LoRA 병합
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### 웹 UI 실행
```bash
llamafactory-cli webui
```
단일 명령으로 브라우저 기반 미세조정 환경이 실행된다.

## 최신 기능과 2025년 업데이트

### 2025년 주요 업데이트
- **새로운 모델**: GPT-OSS, Gemma 3, Qwen3, GLM-4.1V-9B-Thinking, InternVL3
- **Muon Optimizer**: 최신 최적화 알고리즘 지원
- **SGLang 백엔드**: 추론 가속화를 위한 새로운 추론 엔진
- **EasyR1**: 효율적인 GRPO 학습을 위한 강화학습 프레임워크 통합
- **APOLLO Optimizer**: 고급 최적화 기법 추가

### 멀티모달 확장
Qwen2-Audio, MiniCPM-o-2.6을 통한 **오디오 이해 태스크** 지원으로 텍스트, 이미지, 비디오, 오디오를 아우르는 멀티모달 미세조정 생태계를 구축했다.

## 배포와 서비스 통합

### OpenAI 스타일 API
```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm
```
vLLM 백엔드를 통한 호환 가능한 API 서버 제공으로 기존 ChatGPT 기반 애플리케이션에 직접 통합 가능하다.

### 클라우드 및 컨테이너 지원
- **Docker**: CUDA, ROCm(AMD), NPU(Ascend) 전용 이미지 제공
- **클라우드 플랫폼**: Colab, PAI-DSW, Alaya NeW 등 주요 클라우드 환경 지원

## 경쟁 플랫폼 대비 차별화

### LitGPT 대비 장점
- **전문성 vs 범용성**: LitGPT가 사전학습-배포 전체를 다루는 반면, LlamaFactory는 **미세조정에 특화**
- **GUI vs CLI**: 웹 기반 LlamaBoard로 비전문가 접근성이 뛰어남
- **모델 지원**: 100+ 모델 지원으로 선택의 폭이 더 넓음

### Axolotl 대비 장점
- **통합성**: 다양한 학습 방법론을 하나의 인터페이스로 통합
- **사용성**: No-code 웹 인터페이스로 진입 장벽이 낮음
- **데이터셋**: 100+ 내장 데이터셋으로 즉시 실험 가능

## 커뮤니티와 생태계

### 산업계 도입 사례
Amazon, NVIDIA, Aliyun 등 글로벌 기업에서 활용 중이며, **98개 프로젝트**에서 LlamaFactory를 기반으로 한 연구와 애플리케이션을 개발했다. 대표적으로:
- **StarWhisper**: 천문학 특화 LLM
- **DISC-LawLLM**: 중국 법률 도메인 전문 모델
- **CareGPT**: 의료 도메인 시리즈
- **Chinese-LLaVA-Med**: 중국 의료 멀티모달 모델

### 학술적 영향력
55개 이상의 논문에서 LlamaFactory를 활용한 연구 결과를 발표했으며, ACL 2024에서 시연 논문으로 채택되어 학술적 인정을 받았다.

## 제약사항과 고려사항

### 하드웨어 의존성
QLoRA 등 최적화 기법을 사용해도 대형 모델(70B+)의 경우 상당한 GPU 메모리가 필요하며, 분산 학습 환경 구축이 복잡할 수 있다.

### 학습 방법론 깊이
폭넓은 지원을 제공하지만, Axolotl처럼 특정 기법(예: 최신 분산 최적화)에 대한 깊이 있는 전문성은 상대적으로 부족할 수 있다.

## 결론
LlamaFactory는 **"미세조정의 민주화"**를 목표로 한 가장 완성도 높은 통합 플랫폼으로, 학술 연구부터 산업 응용까지 광범위한 사용자층을 아우른다. LitGPT의 전체 파이프라인 접근법, Axolotl의 미세조정 전문성과 달리, **접근성과 통합성에 중점**을 두어 비전문가도 최신 LLM 미세조정을 쉽게 수행할 수 있게 한다.

특히 100+ 모델 지원, Day-0/1 최신 모델 업데이트, No-code 웹 인터페이스, 그리고 강력한 커뮤니티 생태계를 통해 LLM 미세조정 분야의 **표준 플랫폼**으로 자리잡고 있다.

**추천 대상**:
- LLM 미세조정 입문자 및 비전문가
- 빠른 프로토타이핑과 실험이 필요한 연구팀
- 다양한 모델과 기법을 체계적으로 비교 평가해야 하는 프로젝트
- 웹 기반 협업 환경을 선호하는 조직

**공식 리소스**:
- GitHub: https://github.com/hiyouga/LLaMA-Factory
- 문서: https://llamafactory.readthedocs.io
- 논문: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (ACL 2024)
- 데모: Colab 노트북, 웹 UI 체험 환경 제공
