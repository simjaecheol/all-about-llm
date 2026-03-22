---
layout: default
title: Advanced RAG
parent: RAG
nav_order: 10
---

# Advanced RAG (심화 RAG)

초기 RAG(Naive RAG)는 문서를 단순 청크(Chunk)로 쪼개고 한 번의 벡터 검색 후 LLM에 넘기는 선형 구조를 가졌습니다. 하지만 이는 노이즈 문맥, 잘못된 검색, 복잡한 쿼리에 대한 낮은 정확도라는 한계에 부딪혔습니다. 이를 극복하기 위해 제안된 것이 **Advanced RAG (심화 RAG)** 패러다임입니다.

## 1. Modular RAG (모듈형 RAG)

정보를 단순히 검색하는 것을 넘어 전처리(Pre-retrieval)와 후처리(Post-retrieval) 단계를 모듈화하여 정밀도를 높이는 구조입니다.

- **Query Routing (쿼리 라우팅)**: 사용자의 질문 의도를 파악하여 적절한 데이터베이스(벡터 DB, 관계형 DB, 웹 검색 등)로 요청을 분기합니다.
- **Query Rewriting (쿼리 재작성)**: 사용자의 애매한 질문을 LLM을 통해 명확한 검색용 쿼리로 변환하거나 다중 쿼리로 분할(Multi-Query)하여 검색 정확도를 높입니다.
- **Hybrid Search & Re-ranking**: 벡터 검색과 키워드(BM25) 검색을 결합하고, 검색된 문서들을 문맥적 관련성 기준으로 재정렬(Re-ranking)하여 최상위 컨텍스트만 선별합니다.
- **Context Distillation (문맥 압축)**: 불필요한 노이즈를 제거하여 LLM의 Context Window를 효율적으로 사용하고 토큰 비용을 절감합니다.

## 2. Agentic RAG (에이전틱 RAG) & Self-RAG

에이전트(Agent)가 RAG 시스템의 두뇌 역할을 하여 단방향 프로세스를 능동적인 자가-수정(Self-Correcting) 루프로 바꿉니다.

- **Agentic RAG**: 
  - 외부 툴을 호출하고 다단계(Multi-step) 검색이 필요한지 에이전트가 스스로 판단합니다.
  - 필요한 문서를 모두 모을 때까지 반복해서 검색 쿼리를 생성하고 정보를 검증하는 '연구원'처럼 능동적으로 동작합니다. (예: LangChain, LlamaIndex 기반 에이전트 워크플로우)
- **Self-RAG (Self-Reflective RAG)**: 
  - 모델 스스로 검색 결과가 유용한지(Relevance), 자신의 답변이 검색된 문서에 의해 지지되는지(Supported)를 평가하는 성찰 토큰(Reflection Tokens)을 활용합니다. 환각(Hallucination) 현상을 대폭으로 감소시킵니다.

## 3. GraphRAG

단순 텍스트의 유사도를 넘어서 데이터 간의 관계를 바탕으로 문맥을 추론하기 위해 **지식 그래프(Knowledge Graph)**를 결합한 기술입니다 (예: Microsoft Research 발표 패러다임).

- **Entity-Relationship 추출**: 문서 전체에서 주요 엔티티(인물, 조직, 개념)와 관계를 추출하여 거대한 네트워크 구조를 형성합니다.
- **계층적 커뮤니티 파악**: 그래프 구조상 밀집된 정보를 군집화(Community)하여, 전체 맥락을 아우르는 복합적이고 추상적인 질문(Multi-hop 질의)에 대한 답변 능력이 Naive RAG 대비 크게 뛰어납니다.
