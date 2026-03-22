---
title: API 제공사별 비교 분석
parent: 추론 플랫폼(Inference Platform)
nav_order: 1
---

# API 제공사별 추론 플랫폼 비교 (2025)

2025년 현재 OpenAI, Anthropic, Google 등 주요 AI 기업들은 단순한 텍스트 생성을 넘어, 모델이 실행되는 환경(Container)과 도구를 통합적으로 관리하는 **추론 플랫폼(Inference Platform)** 전략을 취하고 있습니다.

## 핵심 기능 비교표

| 기능 | Anthropic (Claude) | OpenAI (Responses API) | Google Gemini |
| :--- | :--- | :--- | :--- |
| **대표 명칭** | **Agent Skills** | **Responses API (Hosted)** | **Managed Code Execution** |
| **컨테이너 식별** | `container_id` (Explicit) | `session_id` / `run_id` | Internal / Task-based |
| **환경 정보 제공** | `container_info` 필드 지원 | API를 통한 리소스 조회 | 시스템 프롬프트 주입 방식 |
| **상태 유지 방식** | 수동 (Container ID 재사용) | 자동 (세션 내 상태 관리) | 세션 기반 (Vertex AI 연결) |
| **주요 실행 도구** | Shell, Skills (Excel, PPT) | Shell, Code Interpreter | Python Interpreter |
| **보안 모델** | Ephemeral Micro-VM | Managed Hosted Runtime | Google Cloud Sandbox |
| **표준 프로토콜** | MCP (Native) | MCP Bridge / Function Call | MCP (Integrated) |

---

## 1. Anthropic (Claude): "Agent Skills & Persistent Environment"

Anthropic은 에이전트가 특정 작업을 위해 필요한 '숙련도'와 '환경'을 결합한 **Agent Skills** 모델을 제시합니다.

- **`container_info` 도입**: API 응답에 모델이 현재 사용 중인 컨테이너의 고유 ID와 정보를 포함합니다.
- **수동 상태 관리**: 개발자가 응답받은 `container_id`를 다음 요청에 명시적으로 전달함으로써, 이전에 설치한 라이브러리나 작업 중인 파일 시스템을 그대로 유지할 수 있습니다.
- **다양한 스킬 런타임**: 단순 코드 실행뿐만 아니라 오피스 문서(Excel, PowerPoint)를 직접 수정할 수 있는 전용 스킬 런타임을 컨테이너 형태로 제공합니다.

## 2. OpenAI: "Managed Hosted Runtimes"

OpenAI는 기존 Assistants API를 계승하고 확장한 **Responses API**를 통해 '에이전트 인프라'를 플랫폼화하고 있습니다.

- **자동화된 오케스트레이션**: 컨테이너의 생성, 유지, 소멸 주기를 OpenAI가 직접 관리합니다. 개발자는 복잡한 ID 관리를 하지 않아도 세션 내에서 모델이 환경 정보를 유지합니다.
- **컴팩션(Compaction)**: 긴 대화 중에도 컨테이너의 상태(파일, 환경 변수)를 효율적으로 압축하여 유지함으로써 컨텍스트 윈도우 폭발을 방지합니다.
- **Hosted Computer Environment**: 모델이 실제 컴퓨터를 사용하듯이 쉘 명령어를 실행하고 브라우징을 수행할 수 있는 완벽한 호스팅 환경을 제공합니다.

## 3. Google Gemini: "Integrated Vertex AI Ecosystem"

Google은 자사의 클라우드 인프라인 **Vertex AI**와 밀접하게 연동된 추론 플랫폼을 제공합니다.

- **Managed Code Execution**: Gemini 1.5/2.0 Pro 모델은 작업 수행 중 코드가 필요하다고 판단하면 즉시 Google 관리형 샌드박스에서 Python 코드를 생성하고 실행합니다.
- **에코시스템 중심**: 개별 컨테이너 관리보다는 Google Cloud의 다양한 서비스(BigQuery, Google Drive 등)와 모델을 연결하는 데 집중하며, 이를 위해 **MCP**를 적극적으로 활용합니다.
- **상황 인식 프롬프트**: 모델이 실행되는 환경 정보를 별도의 API 필드가 아닌, 시스템 프롬프트 레벨에서 동적으로 주입하여 모델의 가독성과 제어력을 높입니다.

---

## 결론: 개발자 선택 가이드

- **세밀한 환경 제어**가 필요하다면: **Anthropic Claude**의 `container_id` 기반 접근 방식이 유리합니다.
- **관리 오버헤드 없이** 에이전트 기능을 구현하고 싶다면: **OpenAI Responses API**의 자동화된 런타임이 적합합니다.
- **기업용 데이터 및 클라우드 연동**이 중요하다면: **Google Gemini**의 Managed 서비스와 MCP 통합 모델을 권장합니다.
