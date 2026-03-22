---
title: Reasoning & Process Supervision
parent: LLM 학습 방법
nav_order: 3
---

# Reasoning & Process Supervision

2025-2026년 AI의 핵심은 단순한 지식 인출을 넘어, 논리적으로 사고하고 스스로 검증하는 **추론(Reasoning)** 능력입니다. 이를 위해 결과뿐만 아니라 **'사고의 과정'**을 학습시키고, 추론 시점에 더 많은 연산량을 투입하는 **추론 시간 확장(Inference-time Scaling)**이 필수 기술로 자리 잡았습니다.

## 1. 왜 과정 중심 학습인가?
기존의 학습 방식은 최종 결과(Outcome)만 맞으면 보상을 주었습니다. 하지만 이는 과정이 틀려도 우연히 정답을 맞힌 경우(Hallucination)를 걸러내지 못합니다. **Process Supervision**은 추론의 모든 중간 단계(Step)를 검증하여 모델이 '올바른 논리'를 갖추게 합니다.

## 2. 핵심 기술

### RLVR (Reinforcement Learning with Verifiable Rewards)
- **개념**: 수학 문제의 정답, 코드 실행 결과, 컴파일 성공 여부 등 **객관적으로 검증 가능한 지표**를 보상으로 사용하여 강화학습을 수행합니다.
- **장점**: 인간의 주관적인 평가 없이도 모델이 스스로 정답에 도달하는 최적의 사고 경로를 찾을 수 있게 하며, 추론의 정확도를 비약적으로 높입니다.

### PRM (Process Reward Model)
- **개념**: 추론 체인(Chain of Thought)의 각 단계별로 타당성을 평가하는 보상 모델입니다.
- **효과**: 중간 단계의 오류를 즉시 감지하여 모델이 엉뚱한 방향으로 사고를 전개하지 않도록 가이드합니다.

### GRPO (Group Relative Policy Optimization)
- **개념**: 하나의 질문에 대해 여러 사고 경로를 생성하고, 그룹 내에서 상대적인 정확도와 효율성을 비교하여 학습합니다. 비싼 Critic 모델 없이 대규모 추론 모델을 학습시키는 데 최적화되어 있습니다.

## 3. 추론 시간 확장 (Inference-time Scaling)
학습 단계뿐만 아니라 실제 서비스 시점(Test-time)에서도 모델이 '더 많이 생각'하게 하여 성능을 높이는 전략입니다.

- **Chain of Thought (CoT) 강화학습**: 모델이 "생각해 보자..."로 시작하여 내부적으로 복잡한 사고 과정을 거친 뒤 최종 답을 내놓도록 RL을 통해 유도합니다.
- **Search & Optimization (MCTS)**: 여러 사고 경로를 시뮬레이션해보고 가장 유망한 경로를 선택하거나, 중간에 막히면 스스로 돌아와 다른 길을 찾는 능력을 학습합니다.
- **Self-Correction (자가 수정)**: 자신의 첫 번째 답변을 스스로 검토하고 논리적 허점을 찾아 수정하는 과정을 반복합니다.

## 4. 데이터 구성 (Reasoning Data)
- **Step-by-step Solutions**: 정답만 있는 것이 아니라 상세한 풀이 과정이 포함된 데이터.
- **Synthetic CoT**: 고성능 모델이 생성한 논리적 사고 과정을 소형 모델 학습에 활용하여 추론 능력을 증류(Distillation).
- **Self-Generated Trajectories**: 모델이 스스로 문제를 풀며 생성한 다양한 경로 중, 정답에 도달한 경로(Positive)와 실패한 경로(Negative)를 모두 학습에 활용.

---

## 미래 전망: System 3 Reasoning
현대적인 모델은 즉각적인 응답(System 1)과 숙고형 사고(System 2)를 넘어, **도구 활용과 장기적인 계획 수립**이 결합된 **'System 3' 추론**으로 진화하고 있습니다. 2026년 이후의 모든 고성능 모델은 이러한 '생각하는 힘'의 스케일링을 통해 지능의 한계를 돌파하고 있습니다.
