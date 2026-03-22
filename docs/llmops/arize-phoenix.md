---
layout: default
title: Arize Phoenix
parent: LLMOps
nav_order: 3
---

# Arize Phoenix

[Arize Phoenix](https://phoenix.arize.com/)는 LLM 애플리케이션의 개발, 평가, 문제 해결(Troubleshooting)을 돕기 위해 구축된 **오픈소스 AI 옵저버빌리티 플랫폼**입니다.

## 주요 특징 (Features)

### 1. 트래킹 툴 및 OpenTelemetry 통합 (Tracing)
OpenTelemetry 표준을 토대로 런타임 성능을 세밀하게 측정할 수 있는 광범위한 트레이싱 기능을 제공합니다.
* LLM 호출, 검색 과정(Retrieval), 도구(Tool) 실행 등 각각의 단계를 Span과 Trace로 파악합니다.
* 속도 분석(Latency), 토큰 소비 비용 산정 등을 통해 복잡한 RAG 파이프라인이나 다중 에이전트 시스템에서 병목점이 생기는 곳을 파악할 수 있도록 돕습니다.

### 2. 평가 (Evaluation)
앱의 성능 벤치마킹을 위한 다양한 검증 수단을 제공합니다.
* LLM-assisted Evaluation을 통해 LLM을 심판(Judge) 모델로 삼아 성능을 평가하는 템플릿들을 제공합니다. (예: Hallucination 감지 점수 등)
* 커스텀 평가 지표 생성을 지원하고, 사람이 남기는 긍정/부정 피드백과 사용자 행동 지표들을 수집하여 문제 항목 클러스터링을 수월하게 진행합니다.

### 3. 디버깅 및 에이전트 그래프 (Debugging & Agent Graph)
특히 에이전트 위주의 복잡한 시스템의 디버깅 시간을 현저히 단축시킵니다.
* **Agent Graph**: 여러 실행 경로 및 재귀적(Self-looping) 에이전트 동작 구조를 개별 Span 대신 노드 기반의 그래프(Node-based graph)로 시각화합니다.
* 온갖 예외 상황(예방하기 힘든 토큰 Limit 초과 등)이 발생했을 때 이를 추적하여 모델 로그에 즉각 남겨 보완점을 제시합니다.

### 4. 프롬프트 최적화 (Prompt Management & Playground)
내장된 Prompt Playground를 통해 모델 간 비교, 최적화 작업을 빠르게 돕습니다.
* 에러나 환각을 유발한 나쁜 프롬프트를 따로 분석하여 버전 관리를 거쳐 최적의 템플릿(Templates)으로 재 가공합니다.
* 이전의 흔적(Trace)을 다시 재현(Replay)하며 여러 가지 시스템 프롬프트(Temperature 등의 파라미터 포함)를 조정해볼 수 있습니다.

### 5. 인프라 확장성
* Local 환경 (노트북 직접 실행), Jupyter 커널, Docker 컨테이너 배포 등 오픈소스 기반에서 가능한 매우 유연한 설치 환경을 보유하고 있습니다.
* LangChain, LlamaIndex, OpenAI Agents SDK, LangGraph 등을 비롯하여 폭넓은 써드파티 호환 시너지를 제공합니다.
