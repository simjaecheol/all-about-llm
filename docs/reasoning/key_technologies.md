---
title: 핵심 추론 기법
nav_order: 2
parent: Reasoning
---

# 핵심 추론 기법

Reasoning 모델의 성능을 극대화하기 위해 다양한 추론 기법들이 사용됩니다. 이 기법들은 모델이 더 논리적이고 체계적으로 문제에 접근하도록 돕습니다.

## 1. Chain-of-Thought (CoT) 변형 기법

#### Zero-shot CoT
**"Let's think step by step"** 문구만으로 추론 능력을 향상시키는 가장 간단한 방법입니다.

다른 효과적인 문구들:
- "Let's work this out in a step-by-step way to be sure we have the right answer."
- "First, let's think about this logically."

#### Few-shot CoT
수작업으로 작성된 추론 예시를 제공하여 모델을 안내하는 방식으로, **최대 28.2%의 정확도 향상**을 달성합니다.

#### Auto-CoT (Automatic Chain-of-Thought)
**2단계 자동화 과정**을 통해 수작업의 한계를 극복합니다:
1. **Question Clustering**: 데이터셋 내 질문들을 다양한 클러스터로 분류
2. **Demonstration Sampling**: 각 클러스터에서 대표 질문 선택 후 Zero-shot-CoT로 추론 체인 생성

#### Thread of Thought (ThoT)
긴 대화나 다중 턴 상황에서 **일관된 사고 흐름을 유지**하는 기법입니다:
- 프롬프트: "Walk me through this context in manageable parts, step by step, summarizing and analyzing as we go."

#### Contrastive CoT
**올바른 추론과 잘못된 추론을 함께 제시**하여 모델이 잘못된 논리를 학습하도록 하는 대조학습 방식입니다.

#### Faithful CoT
자연어와 **기호적 추론(Python 코드 등)을 결합**하여 추론의 신뢰성을 높이는 2단계 프로세스입니다:
1. 자연어 쿼리를 기호적 추론 체인으로 변환
2. 결정론적 솔버로 최종 답 도출

## 2. Self-Consistency 기법

**다양한 추론 경로를 샘플링**한 후 가장 일관된 답변을 선택하는 방법입니다. 주요 성능 향상:
- GSM8K: +17.9%
- SVAMP: +11.0% 
- AQuA: +12.2%
- StrategyQA: +6.4%

**작동 원리**:
1. 탐욕적 디코딩 대신 다수의 추론 경로 샘플링
2. 각 경로별 최종 답변 수집
3. 가장 빈번히 나타나는 답변 선택

## 3. Tree of Thoughts (ToT)

**트리 구조로 다양한 추론 경로를 탐색**하는 고급 기법입니다:

**핵심 구성요소**:
- **Thought Generation**: 각 단계에서 다양한 사고 후보 생성
- **Thought Evaluation**: 각 사고의 품질과 실행 가능성 평가
- **Search Algorithm**: BFS/DFS를 통한 최적 경로 탐색
- **Backtracking**: 막다른 길에서 되돌아가기

**적용 사례**: Game of 24와 같은 수학적 추론 문제에서 3단계로 분해하여 각 단계마다 5개 최적 후보를 유지합니다.

## 4. Progressive-Hint Prompting (PHP)

**이전 응답을 힌트로 활용**하여 점진적으로 정답에 접근하는 방법입니다:

**작동 과정**:
1. 초기 문제 제시 및 답변 생성
2. 이전 답변을 힌트로 포함하여 재질문
3. 올바른 답에 도달할 때까지 반복

**성능 향상**:
- GSM8K: 4.2% 향상 (text-davinci-003)
- SVAMP: 89.1% → 91.9% (GPT-4)
- MATH: 50.3% → 53.9% (GPT-4)

**Self-consistency와 결합 시** 샘플 경로를 46.17% 감소시키면서도 높은 성능을 유지합니다.

## 5. 다중 에이전트 기반 추론

### Diverse Multi-Agent Debate (DMAD)
**서로 다른 추론 전략을 사용하는 에이전트들의 토론**을 통해 고정된 사고 패턴을 극복합니다:
- 각 에이전트에게 서로 다른 추론 접근법 할당
- 다양한 관점에서 문제 해결책 논의
- 집단 지성을 통한 최적해 도출

### CortexDebate
**인간 뇌의 피질 영역 연결 구조에서 영감**을 받은 희소 토론 그래프를 구성합니다:
- **McKinsey-based Debate Matter (MDM)**: 사회학의 McKinsey Trust Formula를 통합하여 신뢰할 수 있는 평가 시스템을 구축

## 6. 전략별 추론 기법

### Strategy-Conditioned Prompting
**인간의 인지 과학 연구**에 기반하여 문제 유형별로 최적의 추론 전략을 선택하는 방법입니다:
- **Supposition Following**: 가정을 세우고 그 결과를 추적
- **Chain Construction**: 논리적 관계를 식별하고 순차적 논증 구성
- **Compound Strategy**: 여러 논리적 관계를 통합하여 중간 결론 도출

### Reasoning Strategy Adaptation
모델이 문제 특성에 따라 **동적으로 추론 전략을 선택**하도록 훈련하는 방식입니다.

## 7. Think 길이 조절 방법

### Length Controlled Policy Optimization (LCPO)
**강화학습을 통한 추론 길이 제어**를 구현합니다.
- **LCPO-Exact**: 정확히 지정된 토큰 수만큼 추론 생성
- **LCPO-Max**: 최대 지정된 토큰 수 이하로 추론 제한
- **프롬프트 기반 제어**: `"Generate your reasoning using exactly 200 tokens"` 형태로 길이 명시
