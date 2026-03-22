---
title: PydanticAI
parent: 주요 프레임워크 및 생태계
grand_parent: Agent
nav_order: 3
---

# PydanticAI

PydanticAI는 Pydantic 팀에서 개발한 에이전트 라이브러리로, 현대적인 파이썬 백엔드 엔지니어링의 정수를 에이전트 개발에 도입했습니다. '타입 안정성(Type-safe)'과 '테스트 가능성'을 극대화한 것이 특징입니다.

## 1. 핵심 철학
- **Type-safe by default**: 입출력 데이터가 항상 정의된 타입을 따르도록 강제하여 런타임 에러를 사전에 방어합니다.
- **Developer Experience (DX)**: FastAPI나 SQLModel과 유사한 개발 경험을 제공하여 백엔드 개발자들이 빠르게 적응할 수 있습니다.
- **Model Agnostic**: 특정 LLM 벤더에 종속되지 않으며, 모델을 쉽게 교체하고 테스트할 수 있는 구조를 제공합니다.

## 2. 주요 기능
- **Schema Validation**: Pydantic 모델을 사용하여 에이전트의 입력과 출력을 정의합니다. 모델이 생성한 JSON 결과가 스키마에 맞는지 자동으로 검증하고, 틀릴 경우 재시도를 요청합니다.
- **Dependency Injection (DI)**: 데이터베이스 연결, 설정값 등 외부 의존성을 에이전트에게 안전하게 주입할 수 있어 단위 테스트가 매우 용이합니다.
- **Structured Outputs**: 복잡한 정보 추출이나 특정 형식이 필요한 작업에서 강력한 성능을 발휘합니다.
- **Integration with IDEs**: 타입 힌트를 통해 IDE에서 자동 완성 및 타입 체크를 완벽하게 지원받을 수 있습니다.

## 3. 왜 PydanticAI인가?
기존 프레임워크들이 '추론 루프' 자체의 유연성에 집중했다면, PydanticAI는 **"에이전트를 어떻게 기존 서비스 시스템의 일부분으로 안정적으로 통합할 것인가?"**에 대한 답을 제시합니다. 프로덕션 환경에서 오류 없는 데이터 처리가 중요한 시스템에 가장 적합합니다.
