---
title: 임베딩
parent: RAG
nav_order: 2
---

# 임베딩

## 개요

RAG(Retrieval-Augmented Generation) 시스템에서 임베딩은 핵심적인 역할을 합니다. 질문과 문서를 같은 벡터 공간으로 변환하여 의미적 유사도 기반 검색을 가능하게 하고, 정확한 문서 검색을 통해 답변의 품질을 향상시킵니다.

## RAG 시스템에서의 임베딩 역할

### 1. RAG 시스템 구조

**기본 RAG 흐름:**
```
질문 → 질문 임베딩 → 벡터 DB 검색 → 관련 문서 → 답변 생성
```

**임베딩의 핵심 역할:**
- 질문과 문서를 같은 벡터 공간으로 변환
- 의미적 유사도 기반 검색
- 정확한 문서 검색을 통한 답변 품질 향상

### 2. 임베딩의 중요성

**RAG에서 임베딩이 중요한 이유:**
- **정확한 검색**: 의미적으로 유사한 문서를 찾기 위해
- **효율적인 검색**: 벡터 유사도로 빠른 검색 가능
- **확장성**: 대용량 문서 집합에서도 효율적 처리

## 문서 임베딩 (Document Embedding)

### 1. 청크 단위 임베딩

**문서를 청크로 분할하여 임베딩하는 이유:**
- 긴 문서는 검색 정확도가 떨어짐
- 관련 정보가 포함된 적절한 크기의 청크로 분할
- 각 청크가 독립적으로 검색 가능

```python
def create_document_embeddings(documents, embedding_model):
    """문서를 청크 단위로 임베딩"""
    embeddings = []
    
    for doc in documents:
        # 문서를 청크로 분할
        chunks = split_document(doc, chunk_size=512)
        
        for chunk in chunks:
            # 청크를 임베딩
            embedding = embedding_model.encode(chunk)
            embeddings.append({
                'text': chunk,
                'embedding': embedding,
                'metadata': {'source': doc['source']}
            })
    
    return embeddings

# 예시
documents = [
    {"text": "인공지능은 컴퓨터가 인간의 지능을 모방하는 기술입니다.", "source": "AI_guide"},
    {"text": "머신러닝은 데이터로부터 패턴을 학습하는 방법입니다.", "source": "ML_intro"}
]

embeddings = create_document_embeddings(documents, embedding_model)
```

### 2. 청크 분할 전략

#### 고정 길이 청크
```python
def split_fixed_length(text, chunk_size=512, overlap=50):
    """고정 길이로 청크 분할"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# 예시
long_text = "매우 긴 문서 내용..."
chunks = split_fixed_length(long_text, chunk_size=512, overlap=50)
```

#### 의미 기반 청크
```python
def split_semantic_chunks(text, embedding_model):
    """의미 기반으로 청크 분할"""
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        
        # 청크의 의미적 일관성 확인
        if len(current_chunk) >= 3:  # 최소 3문장
            chunk_text = " ".join(current_chunk)
            # 의미적 일관성 검사 (예: 임베딩 유사도)
            if is_semantically_coherent(chunk_text, embedding_model):
                chunks.append(chunk_text)
                current_chunk = []
    
    # 남은 문장들 처리
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def is_semantically_coherent(text, embedding_model):
    """텍스트의 의미적 일관성 검사"""
    # 간단한 구현: 문장 수와 길이 기반
    sentences = split_into_sentences(text)
    return len(sentences) >= 2 and len(text) <= 1000
```

### 3. 임베딩 최적화 기법

#### 벡터 정규화
```python
def optimize_embeddings_for_retrieval(embeddings, method="normalize"):
    """검색을 위한 임베딩 최적화"""
    if method == "normalize":
        # L2 정규화로 벡터 길이를 1로 조정
        for emb in embeddings:
            emb['embedding'] = normalize_vector(emb['embedding'])
    
    elif method == "quantize":
        # 양자화로 메모리 사용량 감소
        for emb in embeddings:
            emb['embedding'] = quantize_vector(emb['embedding'])
    
    return embeddings

def normalize_vector(vector):
    """벡터 정규화"""
    import numpy as np
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
```

#### 메타데이터 추가
```python
def add_metadata_to_embeddings(embeddings, documents):
    """임베딩에 메타데이터 추가"""
    for emb in embeddings:
        # 소스 문서 정보
        emb['metadata']['source'] = emb.get('source', 'unknown')
        
        # 청크 정보
        emb['metadata']['chunk_id'] = generate_chunk_id(emb['text'])
        emb['metadata']['chunk_length'] = len(emb['text'])
        
        # 임베딩 정보
        emb['metadata']['embedding_dim'] = len(emb['embedding'])
        emb['metadata']['created_at'] = datetime.now().isoformat()
    
    return embeddings
```

## 질문 임베딩 (Query Embedding)

### 1. 질문 전처리

**질문 전처리의 중요성:**
- 검색 정확도 향상
- 노이즈 제거
- 질문 유형 파악

```python
def preprocess_query(query):
    """질문 전처리"""
    # 1. 불필요한 공백 제거
    query = query.strip()
    
    # 2. 특수문자 정리
    query = clean_special_chars(query)
    
    # 3. 질문 유형 파악
    question_type = classify_question(query)
    
    return query, question_type

def classify_question(query):
    """질문 유형 분류"""
    question_words = ["무엇", "어떻게", "왜", "언제", "어디", "누가"]
    
    for word in question_words:
        if word in query:
            return word
    
    return "일반"

def clean_special_chars(text):
    """특수문자 정리"""
    import re
    # 불필요한 특수문자 제거
    text = re.sub(r'[^\w\s가-힣]', '', text)
    return text
```

### 2. 질문 임베딩 생성

```python
def create_query_embedding(query, embedding_model):
    """질문 임베딩 생성"""
    # 질문 전처리
    processed_query, question_type = preprocess_query(query)
    
    # 임베딩 생성
    query_embedding = embedding_model.encode(processed_query)
    
    return {
        'query': query,
        'processed_query': processed_query,
        'embedding': query_embedding,
        'question_type': question_type
    }

# 예시
query = "인공지능이란 무엇인가요?"
query_embedding = create_query_embedding(query, embedding_model)
```

### 3. 질문 확장 (Query Expansion)

**질문 확장의 목적:**
- 검색 범위 확대
- 관련 문서 검색률 향상
- 다양한 표현 방식 고려

```python
def expand_query(query, embedding_model, top_k=3):
    """질문 확장"""
    # 1. 원본 질문 임베딩
    original_embedding = embedding_model.encode(query)
    
    # 2. 유사한 질문 찾기 (사전 구축된 질문 DB에서)
    similar_queries = find_similar_queries(original_embedding, question_db, top_k)
    
    # 3. 확장된 질문 생성
    expanded_queries = [query] + [q['text'] for q in similar_queries]
    
    return expanded_queries

def find_similar_queries(query_embedding, question_db, top_k):
    """유사한 질문 찾기"""
    similarities = []
    
    for q in question_db:
        similarity = cosine_similarity(query_embedding, q['embedding'])
        similarities.append({
            'text': q['text'],
            'similarity': similarity
        })
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]
```

## 관련 문서

- **[LLM에서의 임베딩 활용](../LLM/embedding.md)**: LLM 내부에서 임베딩이 어떻게 작동하는지와 토큰-임베딩 관계에 대한 기본 개념
- **[임베딩 모델](./embedding_model.md)**: RAG 성능에 영향을 미치는 다양한 임베딩 모델 소개
- **[검색 (Retrieval)](./retrieval.md)**: 임베딩 벡터를 활용한 검색 기법 소개
