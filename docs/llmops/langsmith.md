---
layout: default
title: LangSmith
parent: LLMOps
nav_order: 2
---

# LangSmith

[LangSmith](https://smith.langchain.com/)는 LangChain의 제작 개발팀이 제공하는 전 방위적 LLM 옵저버빌리티(Observability) 및 디버깅 플랫폼입니다. 에이전트 개발부터 운영, 테스트 자동화까지 넓은 커버리지를 제공합니다.

## 주요 특징 (Features)

### 1. End-To-End Tracing
애플리케이션의 실행 단계별 기록을 UI 상에서 가장 직관적인 형태로 나타냅니다.
* 프롬프트 입력부터 API 호출 결과 통신 내용, 개별 Tool의 결과, 에이전트의 워크플로우를 모두 노드 기반의 트리 다이어그램으로 한눈에 볼 수 있도록 시각화하여 디버깅을 단순화합니다.
* LangChain, LangGraph의 네이티브(Native) 생태계답게 두 프레임워크를 함께 사용했을 때 별다른 설정 없이 완벽한 추적이 가능합니다.

### 2. 자동화된 애플리케이션 평가 (Automated Evaluation)
에이전트가 얼마나 잘 실행되고 있는지 점수를 측정하고 배포 안정성을 평가합니다.
* **LLM-as-a-Judge**: 보다 큰 모델이나 평가 전용 모델을 사용하여 온라인에서 품질을 스코어링하는 시스템.
* 사용자가 직접 커스텀 Evaluator를 정의하여 데이터셋과 실제 런타임 결과물의 성능(Accuracy 등) 차이를 분석할 수 있게 해줍니다.

### 3. 데이터셋 및 테스트 (Datasets and Testing)
Production 환경에서 발생한 유의미한 Trace 이력들을 클릭 몇 번으로 즉시 테스트 전용 데이터셋으로 전환할 수 있습니다.
* 모델을 교체(GPT -> Claude) 하거나, 프롬프트를 대규모 수정했을 때 과거 이력을 바탕으로 한 테스트 데이터셋을 통해 성능을 재 측정하는 CI/CD 통합 환경을 구축하기 용이합니다.

### 4. 프롬프트 허브 (Prompt Hub & Optimization)
* 다수의 팀원들이 프롬프트 버전을 공유하고 관리 및 테스트할 수 있는 협업 툴킷을 제공합니다.
* 특정 Trace에 문제가 발생했을 때, 해당 Trace에서 사용된 프롬프트를 샌드박스 환경(Playground)에서 바로 수정해 보며 어떻게 달라지는지 테스트할 수 있습니다.

### 5. 인프라 유연성
* 관리형 Cloud SaaS, BYOC(Bring-Your-Own-Cloud), 사내 서버를 위한 Self-Hosted 배포 등 여러 보안 요구 규정 충족을 위한 세분화된 플랜을 제공합니다.
