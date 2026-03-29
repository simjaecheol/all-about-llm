---
title: LLM 학습 프레임워크
has_children: true
nav_order: 21
---

# LLM 학습 프레임워크

대규모 언어 모델(LLM)의 학습과 미세조정은 복잡하고 많은 자원을 필요로 하는 작업입니다. LLM 학습 프레임워크는 이러한 과정을 더 체계적이고 효율적으로 관리할 수 있도록 돕는 도구 모음입니다. 데이터 전처리부터 분산 학습, 모델 최적화, 배포에 이르기까지 LLM 개발의 전체 파이프라인을 지원하며, 연구자와 개발자가 핵심 로직에 집중할 수 있도록 돕습니다.

이 섹션에서는 현재 LLM 생태계를 구성하는 주요 학습 플랫폼들을 소개합니다. 모든 프레임워크의 기반이 되는 **Transformers**부터, 각기 다른 철학과 목적을 가진 전문 플랫폼들까지 다양하게 살펴봅니다. 각 도구는 해결하려는 문제와 지향하는 사용자 경험이 다르므로, 자신의 목표에 가장 적합한 것을 선택하는 것이 중요합니다.

### 생태계의 중심: 기반 라이브러리

- **[Hugging Face Transformers](./transformers.md)**: **LLM 개발의 표준 플랫폼이자 생태계의 중심**입니다. 100만 개 이상의 모델 허브를 기반으로, `pipeline`을 통한 간편한 추론부터 `Trainer` API를 활용한 전문적인 미세조정까지 모든 개발 단계의 기반을 제공합니다. 대부분의 다른 프레임워크가 Transformers와 호환되거나 그 위에 구축됩니다.

### 미세조정 전문 플랫폼: 각기 다른 접근법

미세조정은 가장 일반적인 LLM 학습 작업이며, 각 플랫폼은 서로 다른 강점을 내세웁니다.

- **[Unsloth](./unsloth.md)**: **성능과 효율의 극한을 추구**하는 프레임워크입니다. 커스텀 GPU 커널을 통해 정확도 손실 없이 2배 이상 빠른 학습과 70% 이상의 메모리 절약을 달성합니다. 최근 **48GB VRAM만으로 70B 파라미터 모델을 튜닝**할 수 있게 되었으며, **GGUF 1.58-bit 동적 양자화**, **Vision(멀티모달) 파인튜닝**, 노코드로 학습과 데이터 합성을 진행할 수 있는 **Unsloth Studio** 및 장문 컨텍스트 튜닝 등 공격적인 업데이트를 제공합니다.

- **[LLaMA Factory](./LLaMA_Factory.md)**: **웹 UI를 통한 미세조정의 민주화**를 목표로 합니다. LlamaBoard 등 직관적인 인터페이스를 통해 코드 작성 없이 손쉬운 미세조정을 지원합니다. 최근 **Qwen3, Llama 4, InternVL3 등 최신 다중 모달 모델**에 대한 지원을 빠르게 통합했으며, 도구 사용 능력을 학습시키는 **Agent Tuning**, **Muon optimizer**, 효율적인 **OFT 및 FP8 훈련**, **KTO 알고리즘**을 추가하여 초보자와 전문가 모두를 만족시킵니다.

- **[Axolotl](./axolotl.md)**: **YAML 기반의 정교하고 재현 가능한 실험**에 특화되어 있습니다. 단일 설정 파일로 복잡한 워크플로우를 제어하여 연구 환경에서 강점을 보입니다. 최근 VRAM 최적화를 극대화하는 **MoE Expert Quantization**, 직접적인 MoE 가중치 학습을 가능케하는 **ScatterMoE LoRA**, LLaVA나 Pixtral 등의 **멀티모달 프레임워크 튜닝**, 초대형 모델 스케일링을 위한 **ND Parallelism 및 Sequence Parallelism** 등 엔터프라이즈급 훈련 기법들을 통합했습니다.

### 강화학습(RLHF) 및 모델 정렬 플랫폼

인간의 선호도를 모델에 반영하는 정렬(Alignment) 작업은 LLM의 품질을 결정하는 핵심 단계입니다. 

- **[TRL](./trl.md)**: **Hugging Face의 공식 RLHF 라이브러리**입니다. 최근 RLHF의 복잡한 보상 모델 훈련을 우회하고 최적화 효율을 높인 **DPO(Direct Preference Optimization)와 GRPO, KTO가 주류**로 잡음에 따라, 간편한 `DPOTrainer` 사용 및 멀티모달 정렬(Vision Language Model Alignment)에 대한 지원을 획기적으로 향상시켰습니다.

- **[Verl](./verl.md)**: **ByteDance가 개발한 대규모 분산 RLHF 프레임워크**입니다. 데이터 의존성과 연산을 분리한 **HybridFlow** 엔진을 바탕으로 PPO와 GRPO 훈련을 효율화합니다. 최신 연구인 DAPO 및 VAPO 구현으로 AIME 등 수학/추론 벤치마크에서 SOTA를 달성했으며, Megatron-LM 백엔드 연동을 통해 DeepSeek나 Qwen3과 같은 초대규모 MoE 모델 파이프라인 관리에 투입되고 있습니다.

### 대규모 분산 학습 및 연구 플랫폼

사전학습부터 시작하여 수천억 파라미터 모델을 다루는 대규모 연구 및 개발을 위한 플랫폼입니다.

- **[Megatron-LM](./megatron_lm.md)**: **초대규모 모델의 분산 학습을 위한 연구 프레임워크**입니다. NVIDIA 기술의 집약체로 핵심 모듈인 **Megatron Core**를 완전히 오픈소스화했습니다. Mamba 혼합 구조, **멀티모달 트레이닝 파이프라인**, 훈련 속도 향상을 위한 **Dynamic Context Parallelism**, DeepSeek-V3 등 대형 **MoE 모델링 병렬 최적화**, Hugging Face와의 원활한 양방향 체크포인트 변환 도구(Megatron Bridge) 등을 새롭게 제공합니다.

- **[LitGPT](./litgpt.md)**: **사전학습부터 배포까지, 투명하고 해커블(hackable)한 툴킷**입니다. '스크립트 중심' 설계를 통해 사용자가 깊이 있는 커스터마이징을 할 수 있습니다. 405B에 달하는 초대형 Llama 3 시리즈 및 Qwen2.5에 대한 완전한 지원뿐만 아니라 단일 명령어로 FSDP를 활용한 다중 GPU 분산 학습 및 엔터프라이즈 환경 즉시 배포(Lifecycle management)를 폭넓게 지원합니다.

