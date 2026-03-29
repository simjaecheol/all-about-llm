---
layout: default
title: RAG
has_children: true
nav_order: 61
---

# RAG (Retrieval-Augmented Generation)

RAG(검색 증강 생성)는 대규모 언어 모델(LLM)의 환각(Hallucination) 현상과 최신 정보 부족 등 한계를 보완하기 위해 외부 지식 소스에서 관련 정보를 검색하고, 이를 기반으로 더 정확하고 신뢰성 있는 답변을 생성하는 핵심 AI 기술입니다.

최근 RAG는 단순한 인덱싱-검색-생성의 선형적인 구조(Naive RAG)에서 벗어나, 지식 그래프를 활용하거나 멀티모달 데이터를 처리하고, 자율 에이전트가 주도하는 지능형 파이프라인(Advanced & Agentic RAG)으로 진화하고 있습니다.

이 문서는 RAG를 구성하는 기본 요소부터 최신 고급 패러다임까지 폭넓게 다루는 안내서입니다.

## 기본 구성 요소 (Naive RAG)

RAG의 기본적인 데이터 흐름과 뼈대를 구성하는 핵심 요소들입니다.

- **[1. 임베딩 (Embedding)](./embedding.md)**: 텍스트와 데이터를 의미를 담은 숫자 벡터로 변환하는 기술
- **[2. 검색 (Retrieval)](./retrieval.md)**: 사용자의 질문 의도와 가장 관련성 높은 정보를 찾아내는 과정 및 기법
- **[3. 벡터 데이터베이스 (Vector DB)](./vector_db.md)**: 대규모 벡터 데이터를 효율적으로 저장, 관리, 검색하는 시스템

## 최신 RAG 패러다임 (Advanced & Beyond)

초기 RAG의 한계를 극복하고 복잡한 추론과 데이터 처리를 가능하게 하는 최신 모델 아키텍처들입니다.

- **[Advanced RAG (심화 RAG)](./advanced_rag.md)**: 
  - **Modular RAG / 구조적 최적화**: 쿼리 재작성, 라우팅, Re-ranking, Context 압축 등 모듈화된 기법
  - **GraphRAG**: 지식 그래프(Knowledge Graph)를 활용하여 엔티티 간의 복잡한 관계와 문맥을 이해하는 기법
  - **Agentic RAG & Self-RAG**: 자율 AI 에이전트 체인을 도입하여 다단계 문맥 검색 및 자가 피드백을 수행하는 프레임워크
- **[Multimodal RAG (멀티모달 RAG)](./multimodal_rag.md)**: 텍스트뿐만 아니라 이미지, 비디오, 오디오 등 여러 모달리티 데이터를 통합 임베딩하여 다차원적인 검색과 답변을 생성하는 차세대 RAG