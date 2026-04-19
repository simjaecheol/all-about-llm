---
layout: default
title: 메모리 및 데이터 인프라
parent: AI Transformation 인프라 (AX Infra)
nav_order: 2
---

# 메모리 및 데이터 인프라 (Memory & Data Infra)

## 개요

에이전트는 근본적으로 '기억(Memory)'이 없는 상태(Stateless)인 LLM을 기반으로 작동합니다. 따라서 에이전트가 단기/장기 대화 맥락을 유지하고 기업 내부 문서를 기반으로 답변하며, 기존 시스템과 상호작용하기 위해서는, **인지적 인프라(Cognitive Infrastructure)** 역할을 하는 다양한 데이터베이스 및 상태 저장 기술이 결합되어야 합니다. 단순히 Vector DB에 그치지 않고, 에이전트가 직접 파일 시스템에 상태를 기록하거나 기존 기업형 관계형 DB(RDBMS)와 맞물려 동작하는 아키텍처로 진화하고 있습니다.

> **연관 문서 파도타기**
> - [Agent Memory 상세 (단기/장기 기억 이론)](../agent/memory.md)
> - [RAG (검색 증강 생성) 개요](../RAG/index.md)

---

## 1. 장기 기억 인프라: 벡터 데이터베이스 (Vector DB)

벡터 데이터베이스는 비정형 데이터(텍스트, 이미지, 문서)를 수학적 임베딩(Vector)으로 변환 후 저장하여, **의미적 유사도(Semantic Similarity)**에 기반해 연관 정보를 초고속으로 검색(Retrieval)하는 핵심 인프라입니다. RAG의 근간을 이룹니다.

### 주요 역할
- 사내 지식 베이스 검색(Knowledge Retrieval)
- 사용자의 취향이나 선호도 등 장기 기억(Long-term Memory) 적재 및 검색 연동

### 대표 스택
- **[Milvus](https://milvus.io/) / [Zilliz](https://zilliz.com/)**: 대규모 엔터프라이즈 환경에서 10억 개 단위의 방대한 벡터 데이터 관리에 탁월한 분산형 벡터 DB.
- **[Qdrant](https://qdrant.tech/)**: Rust로 작성되어 빠르고 가벼우며, 페이로드(메타데이터) 필터링 처리에 강력한 오픈소스 제공.
- **[ChromaDB](https://www.trychroma.com/)**: AI 네이티브 개발자 친화적 오픈소스로, 프로토타이핑과 소규모 프로젝트에 매우 인기.
- **[Pinecone](https://www.pinecone.io/)**: 서버리스 형태의 완전 관리형 SaaS 모델로 설정 부담이 없음.

---

## 2. 상태(State) 및 단기 기억 저장소: Session DB

단기 기억(대화 이력 저장 등)이나 에이전트의 현재 작업 추적(State)은 고속 I/O가 중요하므로, 전통적이고 안정적인 데이터 구조를 필요로 합니다.

### 왜 필요한가?
- 에이전트 다이얼로그나 턴(Turn) 마다 이전 채팅 트레이스를 즉각 불러와 주입시켜야 함.
- LangGraph 등 워크플로우 프레임워크 상에서, 특정 노드(Node) 상태를 영속적(Durable)으로 저장(`checkpointing`)하기 위해.

### 대표 스택
- **Redis (키-값 스토어)**:
  - 메모리(In-Memory) 기반으로 극도의 낮은 지연율(Low Latency) 제공.
  - 대화 히스토리 및 세션 관리를 캐싱하는 데 업계 표준.
- **PostgreSQL / SQLite**:
  - LangChain 백엔드나 에이전트 워크플로우 추적기의 영구 보관용 상태 저장소. (최근에는 `pgvector` 확장을 통해 VectorDB 역할까지 겸비하는 하이브리드로 사용되기도 함)

---

## 3. 파일 시스템 및 워크스페이스 기반 메모리 (Memory Bank)

최근에는 코딩 에이전트(예: Cline, AutoGPT, OpenDevin 등) 및 자율형 에이전트가 등장하면서, 외부의 전용 데이터베이스에 의존하지 않고 에이전트가 파일 시스템(Workspace) 자체를 자신의 영구 작업 메모리 및 지식 베이스로 활용하는 아키텍처가 부상하고 있습니다.

### 작동 방식 및 특징
- **에이전트 스킬 활용**: 에이전트는 `ls` (디렉토리 조회), `grep` (텍스트 검색), `cat` (파일 읽기), `write` (파일 쓰기) 등의 도구를 활용하여, 사람과 동일한 방식으로 정보를 보존합니다.
- **Memory Bank 구성**: 프로젝트 초기 설정, 현재 진행 상태, 시스템 아키텍처 등의 문맥을 특정 디렉토리(`#/memory/` 또는 `.memory-bank/`) 하위의 `Markdown` 파일로 기록합니다.
- **투명성과 제어**: 파일 시스템 기반이므로 개발자(사람)가 언제든 에디터로 열어 에이전트가 "무엇을 기억하고 무슨 규칙을 따르는지" 명확하게 확인 및 직접 편집(Human-in-the-loop)할 수 있습니다. 시스템 오류 시 디버깅이 매우 쉽습니다.
- **인프라 관점 (Sandbox Volume)**: 이 메모리가 증발하지 않으려면 샌드박스 환경(Docker, E2B)에 마운트된 영구적인 스토리지 볼륨이 필수적입니다. 

---

## 4. 기존 시스템 연동 기반 메모리 (NL2SQL / RDBMS)

기업 환경(Enterprise)에서 신뢰성 있는 에이전트는 고립되어 작동할 수 없습니다. 전사적 자원 관리(ERP), 고객 관계 관리(CRM) 등 수십 년간 축적된 RDBMS야말로 기업의 가장 완벽하고 거대한 **'접지 가능한 메모리(Grounding Memory)'**입니다. 기존 시스템의 데이터를 LLM이 동적으로 인출하여 에이전트의 문맥으로 통합하는 환경 구성이 필요합니다.

### 아키텍처 및 구현 기법
- **Text-to-SQL (NL2SQL) 에이전트**: 데이터 베이스 스키마를 에이전트의 컨텍스트에 주입하여, 사용자의 자연어 질문을 SQL 쿼리로 실시간 변환하여 실행합니다. 이를 통해 방대한 테이블에서 필요한 팩트만 즉각 발췌해 단기 기억으로 활용합니다.
- **Semantic Layer (의미론적 계층)**: 복잡한 원천 DB 테이블을 에이전트가 직접 접근하게 하는 대신, 비즈니스 용어로 정리된 '의미론적 뷰(Semantic View)' 계층 또는 메트릭 레이어를 제공함으로써 쿼리 오류나 혼란을 통제합니다.
- **보안 샌드박싱과 권한 제어**: NL2SQL 방식에서는 에이전트 모델의 자율성으로 인해 `DROP TABLE`과 같은 파괴적 쿼리 전송 위험이 있습니다. 인프라 단에서 에이전트에게 할당하는 DB 연결 쿼리 토큰은 엄격한 조회(Read-Only) 권한만 갖도록 Role-Based Access Control(RBAC) 체계를 구성해야 합니다.

---

## 5. 메모리 DB 통합 아키텍처 및 데이터 스키마 단위

현대적인 에이전트 메모리는 단일 DB가 아닌 목적에 맞는 다중 스키마를 혼합하여 사용합니다. 각 저장소 계층의 전형적인 데이터 구성(Schema)은 다음과 같습니다.

### 5.1. Session DB 스냅샷 구조 (LangGraph 기반 예시)
RDBMS 기반 세션 상태 추적기는 주로 타임스탬프 기반 트리 구조를 사용합니다.
- `thread_id` (String): 각 대화 세션의 고유 식별자.
- `checkpoint_id` (String): 분기를 위한 타임스탬프 기반 고유 ID.
- `parent_checkpoint_id` (String): 이전 상태의 ID (Time-travel 및 롤백 지원).
- `checkpoint` (JSONB / BLOB): 에이전트의 내부 State 전체 직렬화 객체 (메시지 배열, 함수 호출 상태 등).
- `metadata` (JSONB): 검색 필터링을 위한 속성(유저 ID, 테넌트 구분 등).

### 5.2. Vector DB 레코드 구조 (일반적인 Semantic DB 예시)
단일 세션을 넘어 영구적으로 남아야 하는 요약된 팩트(Fact)의 저장 구조입니다.
- `id` (UUID): 메모리 파편의 고유 식별값.
- `vector` (Array[Float]): 텍스트를 숫자로 변환한 고차원 배열 (예: 1536차원).
- `text` (String): 추출된 팩트 원문 (예: *"사용자는 Python 언어를 선호함"*).
- `metadata` (JSON): RAG 쿼리 필터링 및 권한 제어를 위한 필수 하이브리드 필드 (`user_id`, `agent_id`, `session_source`, `created_at` 등).

### 5.3. 통합 메모리 플랫폼 (Memory-as-a-Service 적용 사례)
최근 인프라 트렌드는 개발자가 직접 RDBMS, Vector DB, Graph DB 복잡도를 관리하지 않도록, 모든 것을 내재화하여 하나의 통합 API로 제공하는 전용 메모리 플랫폼을 도입하는 것입니다.

- **Zep (Zep Analytics):** 내부적으로 1개의 PostgreSQL(pgvector 활성화)을 사용하여 Chat History 테이블, Document Vector 테이블, Entity 테이블(Graph)을 통합 관리합니다. 대화가 수신되면 비동기 백그라운드 프로세스가 RDBMS 기본 저장, 요약 벡터 추출, 인물/객체 추출을 단일 인프라에서 알아서 수행합니다.
- **Mem0:** LLM 자체를 중간 DB 제어 계층에 삽입하여, 새로운 발화가 들어왔을 때 기존 저장소 내용을 어떻게 수정할지(Create, Update, Delete)를 스스로 판단해 DB에 CRUD 트랜잭션을 전송하는 추상화 레이어를 제공합니다.
