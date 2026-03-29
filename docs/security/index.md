---
title: 보안 (Security)
has_children: true
nav_order: 120
---

# LLM Security & Alignment

대규모 언어 모델(LLM)이 생성하는 결과물의 안전성, 신뢰성, 윤리성을 확보하기 위한 보안 및 정렬(Alignment) 기법들을 다룹니다.

## 주요 개념

### 1. 보안 위협 (Security Threats)
- **Prompt Injection**: 시스템 프롬프트를 탈취하거나 악의적인 지시를 내리는 공격
- **Jailbreak**: 모델의 자체적인 안전 장치를 우회하여 금지된 정보를 생성하게 하는 기법
- **Data Poisoning**: 학습 데이터에 악의적인 데이터를 주입하여 모델의 행동을 왜곡하는 공격
- **Data Leakage**: 모델이 학습 데이터에 포함된 민감한 정보(개인정보, 기업 기밀 등)를 유출하는 현상

### 2. 정렬 (Alignment)
모델의 출력을 인류의 가치와 윤리에 부합하도록 조정하는 과정입니다.
- **RLHF (Reinforcement Learning from Human Feedback)**: 사람의 피드백을 수집하여 최적화
- **Constitutional AI (CAI)**: 모델이 사전에 정의된 윤리 헌장(Constitution)을 기반으로 스스로 출력을 검증 및 수정
- **DPO (Direct Preference Optimization)**: 보상 모델 없이 직접 선호도 데이터를 기반으로 최적화

### 3. [동적 에이전트 보안 위협 (Dynamic Agent Risks)](./dynamic-agent-risks.md)
단순 LLM을 넘어 MCP, Tools, Skills 및 로컬 환경을 조작하는 에이전트가 가지는 치명적인 위협과 아키텍처 방어 방안을 다룹니다.
- **Confused Deputy & Indirect Prompt Injection**
- **도구 실행 권한 분리 및 Human-in-the-Loop (HITL)**

### 4. 방어 기법 (Defense Mechanisms)
- **Guardrails**: 모델의 입출력을 중간에 가로채어 검열, 필터링, 정제(예: NeMo Guardrails, Llama Guard)
- **System Prompt Hardening**: 안전 관련 명령어를 포함하여 룰을 강화한 시스템 프롬프트 작성
- **Input/Output Filtering**: 입력 텍스트와 출력 텍스트에 대한 의도 분석 및 비속어/비정상 패턴 차단 시스템
- **Red Teaming**: 인위적인 공격을 통해 모델의 취약점을 선제적으로 도출하고 개선

## 구현 및 활용

- 최신 Jailbreak 방어 대책
- NeMo Guardrails 설정 및 연동 가이드
- Llama Guard 등 안전성 평가 전용 모델의 활용법

> 본 섹션에 추가 문서를 작성하여 세부 내용을 보강할 수 있습니다.
