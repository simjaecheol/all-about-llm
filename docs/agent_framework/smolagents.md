---
title: Smolagents
parent: Agent Framework
nav_order: 5
---

# Smolagents

Hugging Face에서 발표한 Smolagents는 매우 가볍고(Minimalist), 빠르며, '코드 액션(Action-as-Code)'이라는 독특한 접근 방식을 가진 라이브러리입니다.

## 1. 핵심 철학
- **Action-as-Code**: 기존 에이전트들이 복잡한 JSON 형식으로 도구 호출을 지시했다면, Smolagents는 모델이 직접 파이썬 코드를 작성하게 하여 도구를 호출합니다.
- **Efficiency**: JSON 스키마를 프롬프트에 구구절절 설명할 필요가 없어 토큰을 적게 소모하며, 소형 모델(SLM)에서도 뛰어난 도구 활용 능력을 보여줍니다.
- **Simplicity**: 단 수십 줄의 코드로 에이전트를 완성할 수 있을 만큼 구조가 단순합니다.

## 2. 주요 기능
- **CodeInterpreterAgent**: 파이썬 인터프리터를 내장하여 코드를 직접 실행하고 결과를 관찰합니다.
- **Toolbox**: LLM이 사용할 도구들을 간단한 파이썬 함수로 정의하여 등록합니다.
- **Hugging Face Hub Integration**: 허브에 올라와 있는 수많은 모델과 툴들을 즉시 활용할 수 있습니다.

## 3. 장점
- **성능**: 복잡한 다단계 도구 사용 시 JSON 방식보다 코드 방식이 훨씬 성공률이 높다는 연구 결과를 바탕으로 설계되었습니다.
- **가독성**: 에이전트가 한 행동이 실제 코드로 남기 때문에 개발자가 무엇이 잘못되었는지 추적하기 매우 쉽습니다.
- **SLM 친화적**: Llama-3-8B, Phi-4 등 작은 모델들도 파이썬 문법을 잘 따르기 때문에 저비용 에이전트 구축에 유리합니다.
