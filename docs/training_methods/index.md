---
title: LLM 학습 방법
nav_order: 3
---

# LLM 학습 방법

대규모 언어 모델(LLM)의 학습 패러다임은 단순히 데이터 양을 늘리는 단계에서 벗어나, **추론 시간 확장(Inference-time Scaling)**과 **검증 가능한 보상 기반의 정렬(Reasoning RL)** 중심으로 완전히 전환되었습니다. 2026년 현재, LLM 학습은 사전 학습 이후의 **Post-training Stack** 고도화와 추론 시점의 '생각하는 시간'을 확보하기 위한 전략에 집중하고 있습니다.

## 학습 방법론 개요

현대적인 LLM 학습 파이프라인은 다음과 같은 핵심 단계로 구성됩니다:

1. **Pre-Training**: 대규모 코퍼스를 통한 기본 지식 습득 (MLA 및 Sparse MoE 최적화가 핵심)
2. **Post-Training (Alignment & Preference)**: 지시사항 이행 및 선호도 정렬 (DPO, ORPO, **KTO**, **Online DPO**)
3. **Reasoning Training (RLVR)**: 추론 능력 강화를 위한 **검증 가능한 보상 기반 학습** (PRM, GRPO)
4. **Synthetic Data Pipeline**: 모델 간 피드백(RLAIF) 및 자가 개선 데이터 생성 (Self-Rewarding)
5. **PEFT**: 자원 효율적인 파라미터 미세 조정 (LoRA, DoRA, **KTO-PEFT**)
6. **Inference-time Compute Scaling**: 추론 시점에 모델이 스스로 사고 경로를 탐색하고 수정하게 하는 전략 (OpenAI o1, DeepSeek-R1 style)

---

## 학습 방법론 상세

### [Pre-Training](./pre_training)
**Foundation 모델 구축 및 아키텍처 효율화**
- **MLA (Multi-Head Latent Attention)**: KV 캐시 용량을 획기적으로 줄여 문맥 처리 효율성을 극대화 (DeepSeek-V3 표준).
- **Sparse MoE (Mixture of Experts)**: 토큰별로 최적의 전문가만 활성화하여 조 단위 파라미터 모델을 효율적으로 학습.
- **FP8 Mixed-Precision Training**: 정밀도를 최적화하여 학습 속도와 비용을 혁신적으로 절감.

### [Post-Training & Alignment](./alignment)
**인간 및 AI 선호도에 맞는 모델 정렬**
- **KTO (Kahneman-Tversky Optimization)**: 비교 쌍 없이 단일 라벨(좋음/나쁨)로 학습 가능한 경제적 정렬 기법.
- **Online DPO**: 모델이 실시간으로 생성한 답변을 평가하여 데이터 분포 괴리(Distribution Shift)를 방지.
- **ORPO & SimPO**: 참조 모델 없이 메모리를 절약하며 길이 편향을 억제하는 정렬 기법.

### [Reasoning & RLVR](./reasoning_training)
**논리적 사고 과정(Chain of Thought)과 결과의 정확성 확보**
- **RLVR (Reinforcement Learning with Verifiable Rewards)**: 수학적 정답이나 코드 실행 결과처럼 객관적으로 검증 가능한 보상을 통해 추론 능력을 극대화.
- **GRPO (Group Relative Policy Optimization)**: 비싼 비판 모델(Critic) 없이 그룹 내 상대적 보상으로 효율적인 강화학습 수행.
- **System 2 Reasoning**: 추론 시점에 즉각적인 응답 대신 '사고의 시간'을 투입하여 복잡한 문제를 해결.

### [PEFT (Parameter Efficient Fine-Tuning)](./PEFT)
**제한된 자원에서의 고성능 튜닝**
- **LoRA / QLoRA**: 저사양 환경에서의 표준 튜닝 기법.
- **DoRA**: 가중치의 크기와 방향을 분리하여 학습 안정성을 높인 LoRA의 진화형.

### [Synthetic Data & Self-Rewarding](./synthetic_data)
**데이터 벽(Data Wall) 돌파 및 지식 전수**
- **Self-Rewarding Models**: 모델이 스스로 답변의 품질을 평가하고 이를 다시 학습 데이터로 사용하는 자가 개선 루프.
- **Distillation (CoT Distillation)**: 대형 모델의 사고 과정(Chain of Thought)을 소형 모델에게 전수하여 '작지만 논리적인' 모델 구축.

---

## 2025-2026 학습 파이프라인 트렌드 비교

| 항목 | 기존 방식 (2023-2024) | 최신 방식 (2025-2026) |
| :--- | :--- | :--- |
| **정렬(Alignment)** | RLHF (PPO) 중심 | **KTO, Online DPO, RLAIF** |
| **보상 체계** | 인간 선호도 중심 | **검증 가능한 보상 (RLVR, Verifiable)** |
| **데이터 원천** | 인터넷 수집 데이터 | **자가 생성 및 AI 피드백 데이터 (Self-Rewarding)** |
| **성능 향상** | 파라미터 스케일링 | **추론 시간 스케일링 (Inference Scaling)** |
| **아키텍처** | Dense / 기본 MoE | **MLA + Sparse MoE (Shared Experts)** |

---

## 학습 파이프라인 예시 (Modern Stack)

```python
# 1. Pre-training (Sparse MoE 아키텍처 활용)
base_model = train_moe_model(huge_corpus)

# 2. ORPO 적용 (SFT와 Alignment를 한 번에 해결)
aligned_model = train_orpo(base_model, preference_dataset)

# 3. Process Supervision (PRM을 통한 추론 능력 강화)
reasoning_model = train_with_prm(aligned_model, step_by_step_reasoning_data)

# 4. Test-time Scaling (추론 시 Verifier와 함께 사용)
final_output = generate_with_search(reasoning_model, verifier_model, question)
```

이러한 최신 방법론들은 모델의 크기보다 **'데이터의 품질'**과 **'사고의 깊이'**를 우선시하며, 실제 비즈니스 환경에서 더 적은 비용으로 더 높은 신뢰성을 가진 모델을 구축하는 데 기여합니다.
