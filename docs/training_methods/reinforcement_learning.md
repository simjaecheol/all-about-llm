---
title: 강화학습
parent: LLM 학습 방법
nav_order: 4
---

# 강화학습 (Reinforcement Learning)

LLM 학습에서 가장 핵심적인 기술 중 하나인 **강화학습**에 대해 알아보겠습니다.

## 강화학습이란?

강화학습은 **"좋은 행동에는 보상을, 나쁜 행동에는 벌을 주면서 AI를 가르치는 방법"**입니다. 마치 강아지를 훈련시키는 것과 유사한 개념입니다.

### 강화학습이 필요한 이유

기존의 지도학습만으로는 AI가 **인간이 원하는 방식**으로 답변하기 어려웠습니다. 예를 들어:
- 정확하지만 무례한 답변
- 도움이 되지 않는 답변  
- 위험하거나 부적절한 답변

이런 문제들을 해결하기 위해 **RLHF (Reinforcement Learning from Human Feedback)**가 등장했습니다.

## 강화학습 개념

### 기본 구성 요소

강화학습의 핵심 구성 요소들은 다음과 같습니다:

- **에이전트 (Agent)**: 학습하는 주체로, 환경과 상호작용하며 정책을 개선
- **환경 (Environment)**: 에이전트가 상호작용하는 외부 세계
- **상태 (State)**: 환경의 현재 상황을 나타내는 정보
- **행동 (Action)**: 에이전트가 취할 수 있는 선택
- **보상 (Reward)**: 행동의 결과로 받는 피드백
- **정책 (Policy)**: 상태에서 행동을 선택하는 규칙
- **가치함수 (Value Function)**: 상태나 상태-행동 쌍의 장기적 가치를 추정

### 강화학습의 목표

강화학습의 목표는 **누적 보상(return)을 최대화하는 정책을 찾는 것**입니다:

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

여기서 γ는 할인율(discount factor)로, 미래 보상의 중요도를 조절합니다.

### 탐험 vs 활용 (Exploration vs Exploitation)

강화학습에서 핵심적인 딜레마입니다:

- **탐험 (Exploration)**: 새로운 행동을 시도하여 더 나은 정책을 발견
- **활용 (Exploitation)**: 현재까지 학습한 최선의 행동을 선택

### 주요 알고리즘 유형

#### 1. 가치 기반 방법 (Value-based Methods)
- **Q-Learning**: 상태-행동 가치함수를 학습
- **SARSA**: 온-폴리시 TD 학습
- **Deep Q-Network (DQN)**: 딥러닝을 활용한 Q-Learning

#### 2. 정책 기반 방법 (Policy-based Methods)
- **REINFORCE**: 정책 그래디언트 방법
- **Actor-Critic**: 정책과 가치함수를 동시에 학습
- **PPO**: 안정적인 정책 최적화

#### 3. 모델 기반 방법 (Model-based Methods)
- **Dyna-Q**: 환경 모델을 학습하여 시뮬레이션 활용
- **Monte Carlo Tree Search (MCTS)**: 트리 탐색 기반 계획

### Episode와 Trajectory

**Episode (에피소드)**는 에이전트가 초기 상태에서 종료 상태까지 환경과 상호작용하는 완전한 시퀀스를 의미합니다. **Trajectory (궤적)**는 상태와 행동의 연속된 경로를 나타냅니다.

### LLM에서의 재정의

LLM 환경에서는 이러한 개념들이 다음과 같이 적용됩니다:

**Episode**: 하나의 프롬프트에서 시작하여 완전한 응답 생성까지의 전체 과정
**Trajectory**: 토큰 단위의 생성 시퀀스 `(프롬프트, 토큰₁, 토큰₂, ..., 토큰ₙ)`

LLM에서 Episode과 Trajectory는 전통적인 강화학습과는 다른 특성을 가지며, 다음과 같은 핵심 특징을 보입니다:

1. **토큰 기반 시퀀셜 처리**: 각 토큰이 행동이 되고, 누적 시퀀스가 상태가 됨
2. **지연된 보상**: 주로 시퀀스 완료 후에 보상이 주어짐  
3. **가변 길이**: Episode마다 다른 길이를 가짐
4. **대규모 병렬 처리**: 효율성을 위한 배치 처리 필수

### MDP (Markov Decision Process) 구조

LLM의 RLHF에서 MDP는 다음과 같이 구성됩니다:

- **상태 (State)**: 현재까지 생성된 토큰 시퀀스 `s = (x, y₁, y₂, ..., yₜ)`
- **행동 (Action)**: 다음에 생성할 토큰 `a = yₜ₊₁`
- **환경 (Environment)**: 언어 모델 자체와 보상 함수
- **보상 (Reward)**: 주로 시퀀스 종료 시점에 제공되는 스칼라 값

```python
# LLM에서의 Episode 구조
episode = {
    "prompt": "사용자 질문",
    "states": [(prompt,), (prompt, token1), (prompt, token1, token2), ...],
    "actions": [token1, token2, token3, ..., ],
    "rewards": [0, 0, 0, ..., final_reward],
    "terminated": True  # <EOS> 토큰 또는 최대 길이 도달 시
}
```

### Episode 종료 조건과 보상 할당

#### 종료 조건 (Termination Conditions)
LLM에서 episode는 다음 조건에서 종료됩니다:

1. **자연스러운 종료**: `<EOS>` 토큰 생성
2. **길이 제한**: 최대 토큰 수 도달
3. **특별한 마커**: 백트래킹 마커 등 특수 토큰

#### 보상 할당 방식

**Sequence-Level Reward (기본 방식)**:
```python
# 전통적인 RLHF 보상
rewards = [0, 0, 0, ..., 0, final_reward]
# 마지막 토큰에만 보상 할당
```

**Token-Level Dense Reward (밀집 보상)**:
```python
# TLCR 등 토큰 수준 보상
rewards = [r₁, r₂, r₃, ..., rₜ]
# 각 토큰에 개별 보상 할당
```

## RLHF의 3단계 과정

### 1단계: 지도 미세조정 (SFT)
```
강화학습의 경우, 직접적인 답변을 가르쳐주지 않기 때문에 좋은 답변이 어떤 것인지 모르는 상태에서 강화학습을 적용할 경우, 모델이 학습하는 데까지 걸리는 시간이 오래 걸리게 됩니다.
이런 시간을 단축 시키기 위해 모델이 기본적인 선호 답변을 생성하도록 학습을 진행합니다.
이 때 데이터는 너무 많은 데이터를 사용하지 않습니다.
- 좋은 예시 데이터로 기본 훈련 → "이렇게 답하면 좋습니다!"
```

### 2단계: 보상 모델 만들기
```
사람들이 선호하는 답변을 1, 선호하지 않는 답변을 0으로 학습
선호하는 답변일 수록 1에 가까운 값이 만들어지는 보상 모델을 만듭니다.
- 인간이 답변들을 평가 → "이 답변이 더 좋습니다!" 
```

### 3단계: 강화학습으로 최적화
```
SFT에서 예제로 학습된 모델과 보상 모델을 가지고 강화학습 매커니즘을 적용하여 학습을 진행합니다.
- 보상 모델을 따라 좋은 답변만 하도록 훈련 → "보상을 최대화합니다!"
```

## 주요 기법들

### PPO (Proximal Policy Optimization)
**가장 널리 사용되는 강화학습 알고리즘**입니다.

**특징:**
- 안정적 (큰 변화 방지)
- 효율적 (여러 번 업데이트 가능)
- 간단함 (구현이 쉬움)

#### PPO에서의 Episode 처리
```python
# 실제 TRL 라이브러리 예시
for batch in dataloader:
    query_tensors = batch['input_ids']  # 프롬프트
    
    # Episode 생성: 프롬프트에서 완전한 응답까지
    response_tensors = ppo_trainer.generate(
        query_tensors, 
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 각 response_tensor가 하나의 episode trajectory
    trajectories = list(zip(query_tensors, response_tensors))
```

### DPO (Direct Preference Optimization)
**2023년에 제안된 혁신적인 기법**입니다.

**장점:**
- 보상 모델 없이 직접 선호도 학습
- 훈련이 더 안정적
- 계산 자원 절약

### Constitutional AI
**Anthropic에서 개발한 독특한 방법**입니다.

**핵심 아이디어:**
- AI에게 "헌법(규칙)"을 제공
- AI가 스스로 답변을 평가
- 문제가 있으면 스스로 수정

## 최신 기법들의 Episode 처리

### Macro Actions (MA-RLHF)
**MA-RLHF**에서는 토큰 단위가 아닌 더 큰 단위의 행동을 사용합니다:

```python
# 기존: 토큰 수준 행동
actions = [token1, token2, token3, token4, token5]

# MA-RLHF: 구문 수준 행동
macro_actions = [phrase1, phrase2, phrase3]
# phrase1 = [token1, token2]
# phrase2 = [token3, token4] 
# phrase3 = [token5]
```

**장점:**
- 신용 할당 문제 완화 (temporal distance 단축)
- 30% 성능 향상 달성

### Dynamic Sampling (DAPO)
**DAPO** 기법에서는 episode의 품질에 따라 동적으로 샘플링합니다:

```python
# 효과적인 gradient가 있는 episode만 선별
effective_episodes = []
for episode in generated_episodes:
    accuracy = evaluate_episode(episode)
    if 0 < accuracy < 1:  # 완벽하지도, 완전 실패도 아닌 경우
        effective_episodes.append(episode)
```

## 실제 구현

### TRL 라이브러리 사용하기

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer

# 1. 모델 준비
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 2. PPO 설정
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1e-5,
    mini_batch_size=4
)

# 3. 훈련 시작
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

for batch in dataloader:
    # 답변 생성
    response_tensors = ppo_trainer.generate(query_tensors)
    
    # 보상 계산
    rewards = reward_model(query_tensors, response_tensors)
    
    # PPO 업데이트
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### 감정 분석으로 보상 주기

```python
# 감정 분석으로 좋은 답변인지 판단
sentiment_pipe = pipeline("sentiment-analysis")

def compute_reward(texts):
    rewards = []
    for text in texts:
        sentiment = sentiment_pipe(text)[0]
        if sentiment['label'] == 'POSITIVE':
            rewards.append(torch.tensor(sentiment['score']))  # 좋은 감정 = 높은 보상
        else:
            rewards.append(torch.tensor(-sentiment['score'])) # 나쁜 감정 = 낮은 보상
    return rewards
```

## Episode/Trajectory 처리의 도전과제

### Credit Assignment Problem
토큰 수준에서 보상이 지연되어 발생하는 문제:

```python
# 문제 상황: 긴 시퀀스에서 어느 토큰이 최종 보상에 기여했는지 불분명
sequence = "질문에 대한 매우 긴 답변이 여러 문장에 걸쳐 생성되고 마지막에 정답이 나옴"
reward = 1.0  # 전체 시퀀스에 대한 보상

# 해결책 1: GAE (Generalized Advantage Estimation)
advantages = compute_gae(values, rewards, gamma=0.99, lambda=0.95)

# 해결책 2: Dense Token-Level Rewards
token_rewards = reward_model.get_token_rewards(sequence)
```

### Episode Length Variation
다양한 길이의 episode 처리:

```python
# 길이 가변성 문제
short_episode = [prompt, token1, token2, EOS]  # 길이: 4
long_episode = [prompt, token1, ..., token50, EOS]  # 길이: 52

# 해결책: Padding과 Masking
def pad_trajectories(trajectories):
    max_len = max(len(traj) for traj in trajectories)
    padded_trajs = []
    attention_masks = []
    
    for traj in trajectories:
        padded = traj + [PAD_TOKEN] * (max_len - len(traj))
        mask = [1] * len(traj) + [0] * (max_len - len(traj))
        padded_trajs.append(padded)
        attention_masks.append(mask)
    
    return padded_trajs, attention_masks
```

## 최신 트렌드

### 2024-2025 주요 동향

#### 1. 추론 능력이 뛰어난 모델들
- **OpenAI o1/o3**: 생각하는 과정을 내재화한 모델
- **DeepSeek-R1**: 오픈소스 추론 모델의 새로운 기준
- **Qwen with QwQ**: 32B로 강력한 추론 성능

#### 2. 더 효율적인 알고리즘들
- **GRPO**: PPO의 개선된 버전
- **DPO, IPO**: 직접 최적화 기법들
- **Verifiable Rewards**: 수학, 코딩 등에서 자동 평가

#### 3. 멀티모달 RLHF
- **Vision-Language Models**: 이미지와 텍스트 동시 처리
- **Diffusion RLHF**: 이미지 생성 모델에도 RLHF 적용

## 한국어 RLHF 연구

### KULLM-RLHF 프로젝트
성균관대학교에서 진행한 한국어 친화형 LLM 개발 프로젝트입니다.

**특징:**
- 한국어 데이터 증강
- 다단계 보상 모델 개선
- 효율적인 훈련 시스템

**성과:**
- KCC 2024 우수발표 논문상 수상
- 자연스럽고 윤리적인 한국어 대화

## 미래 전망

### 1. 더 스마트한 피드백 수집
- **AI-to-AI 피드백**: 인간 피드백 비용 절약
- **자동화된 평가**: 다양한 분야에서 객관적 평가

### 2. 개인 맞춤형 모델
- **개인별 선호도 학습**: 개인화된 AI 어시스턴트
- **적은 데이터로 학습**: 효율적인 개인화

### 3. 안전성 강화
- **더 정교한 헌법**: AI가 더 안전하게 행동
- **적대적 공격 방어**: 악의적 공격에 강인한 모델

## 결론

강화학습은 LLM이 단순한 텍스트 생성기를 넘어서 **진정으로 유용한 AI 어시스턴트**가 되도록 도와주는 핵심 기술입니다.

ChatGPT부터 시작해서 지금은 더욱 정교하고 효율적인 방법들이 계속 개발되고 있습니다. 앞으로는 개인마다 맞춤형 AI 어시스턴트를 가질 수 있는 시대가 올 것으로 예상됩니다.

---

**참고 자료:**
- [InstructGPT 논문](https://arxiv.org/abs/2203.02155)
- [DPO 논문](https://arxiv.org/abs/2305.18290)
- [Constitutional AI 논문](https://arxiv.org/abs/2212.08073)
- [TRL 라이브러리](https://github.com/huggingface/trl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
