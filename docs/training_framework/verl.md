---
title: Verl
parent: LLM 학습 프레임워크
nav_order: 6
---

# Verl(Volcano Engine Reinforcement Learning) 학습 플랫폼 완전 분석

## 개요  
Verl은 ByteDance Seed 팀이 개발한 **RLHF(Reinforcement Learning from Human Feedback) 전용 프레임워크**로, 대형 언어모델(LLM)의 사후 학습(post-training)을 효율적으로 지원한다. "HybridFlow" 프로그래밍 모델을 기반으로 다양한 RL 알고리즘(PPO, GRPO, DAPO 등)을 손쉽게 구현하며, PyTorch FSDP, Megatron-LM, vLLM, SGLang 등 주요 인프라와 **무결점 통합**을 제공한다.

## 핵심 철학과 구조  

- **HybridFlow 프로그래밍 모델**  
  단일-컨트롤러와 다중-컨트롤러 방식을 결합해, 복잡한 RL 데이터플로우를 몇 줄의 코드로 정의하고 실행할 수 있다.

- **모듈화 & 확장성**  
  보상 함수(reward function), 정책(policy), 데이터 전처리, 롤아웃(rollout) 엔진을 **모듈화**하여, 새로운 알고리즘·백엔드·하드웨어 지원을 쉽게 추가 가능하다.

- **유연한 디바이스 매핑**  
  1–n개의 GPU 및 수백 대의 클러스터까지 **자원 할당**을 최적화하는 디바이스 매핑 API를 제공한다.

## 지원 알고리즘 및 모델  

| 알고리즘         | 설명                                                         |
|------------------|--------------------------------------------------------------|
| PPO (Proximal Policy Optimization)            | 표준 온-폴리시 RL 기법                                    |
| GRPO (Group Relative Policy Optimization)     | DeepSeekMath 논문 기반, 규칙 함수(reward function) 지원   |
| DAPO (Direct Advantage Policy Optimization)   | DAPO 논문 기반 최신 선호도 학습 기법                      |
| ReMax, ReLOFI 등                             | 모델 기반 보상, 비지도 강화학습 등 다양한 기법 포함      |

- **PPOTrainer**, **GRPOTrainer**, **DAPOTrainer** 클래스로 각 기법별 간편 호출 지원  
- Hugging Face 모델 허브의 GPT 계열, LLaMA 계열을 포함한 **모든 사전훈련 LLM**과 호환  

## 주요 기능  

1. **간편 설치 & 통합**  
   ```bash
   pip install verl
   # 또는 소스 설치
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install -e .[vllm]
   ```
   설치 시 🤗Transformers, torch, accelerate 등 필수 의존성을 자동 설치한다.

2. **Rollout 및 Generation 엔진 연결**  
   - **vLLM**, **TGI**, **DeepSpeed Ulysses** 등의 롤아웃 백엔드를 **플러그인 방식**으로 교체 가능  
   - **3D-HybridEngine**을 통해 롤아웃에서 학습 단계로 전환 시 메모리 중복을 제거하여 통신 오버헤드 최소화

3. **데이터 파이프라인 지원**  
   - 다양한 포맷(JSONL, Parquet) 및 외부 스토리지(S3, GCS)에서 직접 로딩  
   - **병렬 전처리**, **시퀀스 패킹**, **FlashAttention-기반 샘플링**으로 대규모 샘플링 가속

4. **확장 가능한 컨트롤러**  
   - **SingleController**: 동기적(on-policy) 학습  
   - **MultiController**: 비동기적(off-policy), 멀티턴 RL, 툴 호출 등 복잡한 대화 시나리오 처리  
   - **AgentLoop**: 도구 호출(tool calling) 및 멀티턴 챗봇 시나리오 학습 지원

5. **하드웨어 최적화**  
   - PyTorch **FSDP**, **Megatron-LM** 분산 학습 백엔드 완전 통합  
   - **FlashAttention**, **LoRA/QLoRA**, **Adam-mini**, **APOLLO** 등 최신 최적화 기법 지원  
   - 소비자급 GPU(RTX 3090/4090, L4)부터 수백 GPU 클러스터까지 스케일링 가능

## 기본 워크플로우 예시 (GRPO)  
```python
from transformers import AutoTokenizer
from verl import VerlTrainer, VerlConfig, create_reference_model

# 1. 모델 및 토크나이저 로드
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model_ref = create_reference_model(model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Verl 설정
config = VerlConfig(
    algorithm="GRPO",
    batch_size=1,
    rollout_backend="vllm",
    max_length=128,
)

# 3. VerlTrainer 초기화
trainer = VerlTrainer(
    config=config,
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
)

# 4. 학습 스텝 실행 (query, response, reward)
query = ["Hello, how are you?"]
response = trainer.generate_rollout(query)
reward = [0.5]  # 사용자 정의 보상
trainer.step(query, response, reward)
```
이처럼 **10줄 이내 코드**로 GRPO, PPO, DAPO 등의 RLHF 학습 루프를 모두 구현할 수 있다.

## 성능 및 벤치마크  

- **Throughput**: Megatron-LM + vLLM 연동 시 초대형 모델(>70B) 학습 처리량 SOTA 달성  
- **메모리 효율**: 3D-HybridEngine으로 롤아웃↔학습 전환 시 메모리 중복 90% 제거  
- **확장성**: 1→1000+ GPU 클러스터 자동 확장, zero downtime 스케일링 지원

## 실제 적용 사례  

- **DeepSeek-R1**: 수리 문제 해결 성능 개선을 위한 GRPO 기반 RLHF, verl로 대규모 분산 학습 수행
- **MathBench**: DAPO 알고리즘으로 GSM8K 등 수학 벤치마크에서 SOTA 수준 성능 달성
- **멀티모달 챗봇**: 이미지 캡션 VLM의 허위 생성 감소를 위해 GRPO로 비전-언어 모델 정렬

## 커뮤니티 및 지원  

- GitHub: volcengine/verl (stars 8,000+, 1,300+ 이슈 및 PR 활발)
- Read the Docs: https://verl.readthedocs.io
- Slack/Discord: ByteDance Seed 팀 주최 채널에서 질문 및 업데이트  

## 결론  
Verl은 **"LLM RLHF 학습의 민주화"**를 목표로, 복잡한 RLHF 워크플로우를 **모듈화·추상화**하여 누구나 쉽게 구현할 수 있도록 돕는 플랫폼이다. Hugging Face 생태계와 완벽 통합되며, 최신 RL 알고리즘과 초고속 커널 최적화를 결합해 대규모 분산 환경에서도 **안정성과 성능**을 모두 확보한다. 연구자·엔지니어가 RLHF 실험을 빠르게 시작하고, 운영 환경으로 곧바로 이전(deploy)할 수 있는 표준 툴킷이다.

**핵심 키워드**: RLHF, HybridFlow, GRPO, DAPO, PPO, 3D-HybridEngine, 분산학습, ByteDance, 모듈화
