---
title: System 2 사고와 LLM 추론
parent: Tool Call
nav_order: 10
---

# System 2 사고와 LLM 추론

## 1. 개요 (Dual Process Theory)

심리학자 다니엘 카너먼(Daniel Kahneman)은 저서 *Thinking, Fast and Slow*에서 인간의 인지 체계를 두 가지로 분류했습니다:

- **System 1 (Fast)**: 직관적이고 빠르며 자동적인 사고. (예: 2+2 계산, 모국어 이해)
- **System 2 (Slow)**: 논리적이고 느리며 노력이 필요한 사고. (예: 복잡한 수학 문제 풀이, 체스 전략 수립)

## 2. LLM에서의 적용

전통적인 LLM(GPT-3.5, GPT-4 초기 모델)은 주로 System 1 방식으로 작동했습니다. 즉, 다음 토큰을 확률적으로 빠르게 예측하는 데 집중했습니다. 2025~2026년의 현대적 모델(OpenAI o-series, DeepSeek-R1 등)은 **추론 시간 스케일링(Inference-time Scaling)**을 통해 System 2 사고를 구현합니다.

### 핵심 메커니즘
- **Chain of Thought (CoT)**: 모델이 최종 답변을 내놓기 전 내부적인 '사고의 사슬'을 생성합니다.
- **Inference-time Compute**: 답변 생성 시 더 많은 계산 자원을 할당하여 여러 경로를 탐색(Search)하고 스스로 검증합니다.
- **Self-Correction**: 자신의 계획이 틀렸음을 인지하면 중간에 경로를 수정합니다.

## 3. Tool Call에서의 System 2 역할

현대적인 **에이전틱 런타임**에서 System 2 사고는 Tool Call의 성공률을 극적으로 높이는 핵심 요소입니다.

1. **계획 수립 (Planning)**: 복잡한 사용자 요청을 받았을 때 바로 도구를 호출하지 않고, 어떤 도구를 어떤 순서로 사용할지 '미리 생각'합니다.
2. **환각 방지 (Anti-Hallucination)**: 존재하지 않는 도구를 사용하려 하거나 잘못된 인자를 넣으려는 시도를 내부 추론 과정에서 스스로 걸러냅니다.
3. **결과 검증 (Verification)**: 도구 실행 결과가 예상과 다를 경우, 왜 그런 결과가 나왔는지 분석하고 대안을 찾습니다.

## 4. 참고 문헌 및 심화 학습
- [Daniel Kahneman - Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- [OpenAI: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs](https://github.com/deepseek-ai/DeepSeek-R1)
