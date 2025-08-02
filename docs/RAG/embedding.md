---
title: 임베딩
parent: RAG
nav_order: 1
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

## 벡터 유사도 검색

### 1. 코사인 유사도

**가장 일반적으로 사용되는 유사도 측정 방법:**

```python
def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    import numpy as np
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

def search_similar_documents(query_embedding, document_embeddings, top_k=5):
    """유사한 문서 검색"""
    similarities = []
    
    for doc in document_embeddings:
        similarity = cosine_similarity(
            query_embedding['embedding'], 
            doc['embedding']
        )
        similarities.append({
            'document': doc,
            'similarity': similarity
        })
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_k]

# 예시
query = "머신러닝이란?"
query_embedding = create_query_embedding(query, embedding_model)
similar_docs = search_similar_documents(query_embedding, document_embeddings)
```

### 2. 고급 검색 기법

#### 밀집 검색 (Dense Retrieval)
```python
def dense_search(query_embedding, document_embeddings):
    """밀집 검색 (임베딩 기반)"""
    return search_similar_documents(query_embedding, document_embeddings)
```

#### 희소 검색 (Sparse Retrieval)
```python
def sparse_search(query, document_embeddings, method="bm25"):
    """희소 검색 (키워드 기반)"""
    if method == "bm25":
        return bm25_search(query, document_embeddings)
    elif method == "tfidf":
        return tfidf_search(query, document_embeddings)

def bm25_search(query, documents):
    """BM25 검색"""
    # BM25 알고리즘 구현
    # 키워드 기반 검색으로 의미적 검색 보완
    pass
```

#### 하이브리드 검색
```python
def hybrid_search(query_embedding, document_embeddings, weight=0.7):
    """하이브리드 검색"""
    # 밀집 검색
    dense_results = dense_search(query_embedding, document_embeddings)
    
    # 희소 검색
    sparse_results = sparse_search(query_embedding['query'], document_embeddings)
    
    # 결과 결합
    return combine_results(dense_results, sparse_results, weight)

def combine_results(dense_results, sparse_results, weight=0.7):
    """결과 결합"""
    combined = []
    
    for dense in dense_results:
        for sparse in sparse_results:
            if dense['document']['text'] == sparse['document']['text']:
                combined_score = weight * dense['similarity'] + (1-weight) * sparse['similarity']
                combined.append({
                    'document': dense['document'],
                    'score': combined_score
                })
    
    return sorted(combined, key=lambda x: x['score'], reverse=True)
```

### 3. 검색 최적화

#### 임베딩 캐싱
```python
class EmbeddingCache:
    def __init__(self):
        self.cache = {}
    
    def get_embedding(self, text, embedding_model):
        """캐시에서 임베딩 가져오기"""
        if text in self.cache:
            return self.cache[text]
        
        # 캐시에 없으면 새로 생성
        embedding = embedding_model.encode(text)
        self.cache[text] = embedding
        return embedding
    
    def clear_cache(self):
        """캐시 정리"""
        self.cache.clear()

# 사용 예시
cache = EmbeddingCache()
embedding = cache.get_embedding("질문 텍스트", embedding_model)
```

#### 배치 처리
```python
def batch_embedding_search(queries, document_embeddings, embedding_model, batch_size=32):
    """배치 처리로 검색 속도 향상"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        
        # 배치로 임베딩 생성
        batch_embeddings = embedding_model.encode(batch_queries)
        
        # 각 질문에 대해 검색
        for j, query_embedding in enumerate(batch_embeddings):
            query_results = search_similar_documents(
                {'embedding': query_embedding}, 
                document_embeddings
            )
            results.append(query_results)
    
    return results
```

## 임베딩 모델 선택

### 1. RAG에 특화된 임베딩 모델

| 모델 | 특징 | RAG 활용도 |
|------|------|------------|
| **Sentence-BERT** | 문장 임베딩에 특화 | 높음 |
| **DPR** | 검색에 최적화 | 매우 높음 |
| **USE** | 다국어 지원 | 중간 |
| **BERT** | 범용적 | 중간 |

### 2. 도메인별 모델 선택

```python
# 도메인별 권장 임베딩 모델
domain_models = {
    "일반 텍스트": "sentence-transformers/all-MiniLM-L6-v2",
    "의료": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "법률": "nlpaueb/legal-bert-base-uncased",
    "코드": "microsoft/codebert-base",
    "다국어": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

def select_embedding_model(domain):
    """도메인에 맞는 임베딩 모델 선택"""
    return domain_models.get(domain, domain_models["일반 텍스트"])
```

### 3. 성능 비교

```python
# RAG 성능 비교
rag_performance = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "retrieval_accuracy": 0.85,
        "speed": "fast",
        "memory": "low"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "retrieval_accuracy": 0.92,
        "speed": "medium",
        "memory": "medium"
    },
    "DPR": {
        "retrieval_accuracy": 0.95,
        "speed": "slow",
        "memory": "high"
    }
}
```

## 실무 최적화

### 1. 메모리 효율성

```python
def optimize_embedding_memory(embeddings, method="quantize"):
    """임베딩 메모리 최적화"""
    
    if method == "quantize":
        # 32비트 → 8비트 양자화
        for emb in embeddings:
            emb['embedding'] = quantize_to_int8(emb['embedding'])
    
    elif method == "compress":
        # 압축 알고리즘 적용
        for emb in embeddings:
            emb['embedding'] = compress_vector(emb['embedding'])
    
    return embeddings

def quantize_to_int8(vector):
    """벡터를 8비트로 양자화"""
    import numpy as np
    # -128 ~ 127 범위로 정규화
    normalized = (vector - vector.min()) / (vector.max() - vector.min())
    quantized = (normalized * 255 - 128).astype(np.int8)
    return quantized
```

### 2. 검색 속도 최적화

```python
def optimize_search_speed(document_embeddings, method="index"):
    """검색 속도 최적화"""
    
    if method == "index":
        # 벡터 인덱스 구축 (FAISS, Annoy 등)
        return build_vector_index(document_embeddings)
    
    elif method == "approximate":
        # 근사 검색 알고리즘
        return approximate_search(document_embeddings)
    
    elif method == "cache":
        # 검색 결과 캐싱
        return cache_search_results(document_embeddings)

def build_vector_index(embeddings):
    """벡터 인덱스 구축"""
    import faiss
    
    # 벡터들을 numpy 배열로 변환
    vectors = np.array([emb['embedding'] for emb in embeddings])
    
    # FAISS 인덱스 구축
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 내적 기반 인덱스
    index.add(vectors.astype('float32'))
    
    return index
```

### 3. 검색 품질 향상

```python
def improve_search_quality(query_embedding, document_embeddings, methods=["rerank"]):
    """검색 품질 향상"""
    
    # 1단계: 초기 검색
    initial_results = search_similar_documents(query_embedding, document_embeddings, top_k=20)
    
    if "rerank" in methods:
        # 2단계: 재순위화
        reranked_results = rerank_results(query_embedding, initial_results)
        return reranked_results[:5]
    
    return initial_results[:5]

def rerank_results(query_embedding, initial_results):
    """검색 결과 재순위화"""
    # 더 정교한 모델로 재순위화
    # 예: Cross-Encoder 사용
    reranked = []
    
    for result in initial_results:
        # Cross-Encoder로 점수 재계산
        new_score = cross_encoder_score(query_embedding['query'], result['document']['text'])
        reranked.append({
            'document': result['document'],
            'similarity': new_score
        })
    
    return sorted(reranked, key=lambda x: x['similarity'], reverse=True)
```

## 결론

RAG 시스템에서 임베딩은 질문과 문서를 같은 벡터 공간으로 변환하여 의미적 검색을 가능하게 하는 핵심 기술입니다. 적절한 임베딩 모델 선택과 최적화는 RAG 시스템의 성능에 직접적인 영향을 미칩니다.

### 핵심 포인트

1. **문서 임베딩**: 청크 단위로 문서를 임베딩하여 벡터 DB 구축
2. **질문 임베딩**: 사용자 질문을 임베딩하여 검색
3. **벡터 검색**: 코사인 유사도 기반 문서 검색
4. **최적화**: 메모리, 속도, 검색 품질의 균형

이러한 이해를 바탕으로 RAG 시스템을 구축할 때 적절한 임베딩 모델을 선택하고 검색 성능을 최적화할 수 있습니다.

---

## 관련 문서

- **[LLM에서의 임베딩 활용](../LLM/embedding.md)**: LLM 내부에서 임베딩이 어떻게 작동하는지와 토큰-임베딩 관계에 대한 기본 개념
