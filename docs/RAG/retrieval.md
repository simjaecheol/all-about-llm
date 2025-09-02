---
title: 검색 (Retrieval)
parent: RAG
nav_order: 4
---

# 검색 (Retrieval)

검색(Retrieval)은 RAG 시스템의 핵심 구성 요소로, 사용자의 질문과 가장 관련성이 높은 문서를 벡터 데이터베이스에서 찾아내는 과정입니다. 검색의 정확도가 LLM이 생성하는 답변의 품질을 결정합니다.

## 벡터 유사도 검색

### 1. 코사인 유사도 (Cosine Similarity)

벡터 검색에서 가장 널리 사용되는 유사도 측정 방법입니다. 두 벡터가 이루는 각도의 코사인 값을 이용하여 방향성의 유사도를 측정합니다.

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
```

## 고급 검색 기법

단순한 벡터 유사도 검색을 넘어, 검색 품질을 높이기 위한 다양한 고급 기법들이 사용됩니다.

### 1. 어휘 검색 (Lexical Search)과 희소 검색 (Sparse Retrieval)

어휘 검색은 사용자의 질의에 포함된 **키워드**가 문서에 얼마나 자주 나타나는지를 기반으로 관련성을 계산하는 전통적인 정보 검색 방식입니다. 대표적으로 **TF-IDF**와 **BM25** 알고리즘이 있습니다. 이러한 방식은 벡터 공간에서 키워드의 존재 여부를 희소(sparse) 벡터로 표현하기 때문에 희소 검색이라고도 불립니다.

-   **장점**: 키워드가 명확하고 정확하게 일치해야 하는 경우 (예: 특정 제품명, 인명, 코드 변수명 검색) 매우 효과적입니다. 계산 비용이 비교적 저렴하고 구현이 간단합니다.
-   **단점**: 동의어, 유의어 등 의미는 같지만 형태가 다른 단어를 처리하지 못합니다. 문맥이나 의미적 유사성을 파악할 수 없습니다.

#### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF는 특정 단어가 특정 문서 내에서 얼마나 중요한지를 나타내는 통계적 수치입니다. "단어 빈도(TF)"와 "역 문서 빈도(IDF)"를 곱한 값으로, 문서 내에서 자주 나타나지만 다른 문서에서는 잘 나타나지 않는 단어일수록 높은 가중치를 부여받습니다.

**TF-IDF의 구성 요소:**

1.  **TF (Term Frequency, 단어 빈도)**: 한 문서 내에서 특정 단어가 얼마나 자주 등장하는지를 나타냅니다. 값이 높을수록 해당 문서에서 중요한 단어일 가능성이 있습니다.
    -   `TF(t, d) = (문서 d에 단어 t가 나타난 횟수) / (문서 d의 전체 단어 수)`

2.  **IDF (Inverse Document Frequency, 역 문서 빈도)**: 전체 문서 집합에서 특정 단어가 얼마나 희귀하게 등장하는지를 나타냅니다. 모든 문서에 공통적으로 나타나는 단어(예: 조사, 관사)의 중요도를 낮추고, 특정 문서에만 집중적으로 나타나는 단어의 중요도를 높입니다.
    -   `IDF(t, D) = log( (전체 문서 수) / (단어 t를 포함한 문서 수 + 1) )`
    -   *분모에 1을 더하는 것은 특정 단어가 전체 문서에 없는 경우 0으로 나누는 것을 방지하기 위함입니다.*

**TF-IDF 점수 계산:**
이 두 값을 곱하여 최종 TF-IDF 점수를 계산합니다.
- `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

```python
# scikit-learn을 사용한 TF-IDF 구현 예시
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "자연어 처리는 인공지능의 한 분야입니다.",
    "인공지능 기술은 빠르게 발전하고 있습니다.",
    "TF-IDF는 정보 검색과 텍스트 마이닝에서 사용됩니다."
]

# TF-IDF 벡터라이저 생성
tfidf_vectorizer = TfidfVectorizer()

# 문서들을 TF-IDF 벡터로 변환
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 단어 목록과 각 단어의 TF-IDF 점수 확인
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names)
print(tfidf_matrix.toarray())
```

#### BM25 (Best Match 25)

BM25는 현재 가장 널리 사용되는 어휘 검색 알고리즘 중 하나로, TF-IDF를 개선한 모델입니다. 문서 길이를 고려하여 점수를 정규화하고, 단어 빈도가 점수에 미치는 영향을 조절하여 정확도를 높였습니다.

**BM25 점수 계산의 핵심 요소:**

1.  **TF (Term Frequency)**: 특정 단어가 문서에 얼마나 자주 나타나는지 빈도를 측정합니다. 하지만 단어가 무한정 많이 나온다고 점수가 계속 높아지지 않도록 값을 조절합니다.
2.  **IDF (Inverse Document Frequency)**: 특정 단어가 전체 문서 집합에서 얼마나 희귀한지를 측정합니다. 모든 문서에 공통으로 나타나는 단어(예: a, the, 은, 는)는 낮은 가중치를, 특정 문서에만 나타나는 단어는 높은 가중치를 부여받습니다.
3.  **문서 길이 정규화 (Document Length Normalization)**: 문서가 길수록 특정 단어가 포함될 확률이 높으므로, 문서 길이를 고려하여 짧은 문서에 부당한 페널티가 가지 않도록 점수를 보정합니다.

**BM25의 파라미터:**

-   `k1`: TF 값의 영향력을 조절하는 파라미터입니다. 값이 높을수록 TF의 중요도가 높아집니다. (일반적으로 1.2 ~ 2.0 사용)
-   `b`: 문서 길이에 따른 정규화 강도를 조절하는 파라미터입니다. 0에 가까울수록 문서 길이의 영향을 적게 받고, 1에 가까울수록 많이 받습니다. (일반적으로 0.75 사용)

```python
# BM25 라이브러리 (e.g., rank_bm25)를 사용한 구현
from rank_bm25 import BM25Okapi

def bm25_search(query, documents, k1=1.5, b=0.75):
    """BM25 알고리즘을 이용한 희소 검색"""
    
    # 1. 문서들을 토큰화하여 코퍼스 생성
    corpus = [doc['text'].split(" ") for doc in documents]
    bm25 = BM25Okapi(corpus)
    
    # 2. 질의어 토큰화
    tokenized_query = query.split(" ")
    
    # 3. BM25 점수 계산
    doc_scores = bm25.get_scores(tokenized_query)
    
    # 4. 점수와 문서 매핑 후 정렬
    results = [{'document': doc, 'similarity': score} for doc, score in zip(documents, doc_scores)]
    return sorted(results, key=lambda x: x['similarity'], reverse=True)
```

### 2. 하이브리드 검색 (Hybrid Search)

-   **개념**: 밀집 검색(Dense Retrieval)과 희소 검색(Sparse Retrieval)의 장점을 결합한 방식입니다.
-   **장점**: 의미적 유사성과 키워드 일치를 모두 고려하여 검색 정확도를 극대화합니다.
-   **구현**: 각 검색 방식의 점수를 합산하거나, Reciprocal Rank Fusion (RRF)와 같은 알고리즘으로 순위를 결합합니다.

```python
def hybrid_search(query_embedding, document_embeddings, weight=0.7):
    """하이브리드 검색"""
    # 밀집 검색
    dense_results = search_similar_documents(query_embedding, document_embeddings)
    
    # 희소 검색
    sparse_results = sparse_search(query_embedding['query'], document_embeddings)
    
    # 결과 결합 (간단한 가중치 합)
    final_scores = {}
    for res in dense_results:
        final_scores[res['document']['text']] = res['similarity'] * weight
        
    for res in sparse_results:
        text = res['document']['text']
        if text in final_scores:
            final_scores[text] += res['similarity'] * (1 - weight)
        else:
            final_scores[text] = res['similarity'] * (1 - weight)
            
    # 점수 기반으로 정렬된 문서 반환
    # ...
```

### 3. 재순위화 (Re-ranking)

-   **개념**: 1단계 검색(Retrieval)에서 상위 K개의 문서를 가져온 후, 더 정교한 모델(Re-ranker)을 사용하여 순위를 재조정하는 방식입니다.
-   **장점**: 검색 속도와 정확도의 균형을 맞출 수 있습니다. Cross-Encoder와 같이 계산 비용이 높지만 정확한 모델을 2단계에서 사용하여 전체적인 품질을 높입니다.
-   **모델**: Cohere Rerank, Cross-Encoders 등

```python
def rerank_results(query, initial_results, reranker_model):
    """검색 결과 재순위화"""
    # Cross-Encoder 모델을 사용하여 재순위화
    pairs = [(query, result['document']['text']) for result in initial_results]
    scores = reranker_model.predict(pairs)
    
    for result, score in zip(initial_results, scores):
        result['similarity'] = score # 새로운 점수로 갱신
        
    return sorted(initial_results, key=lambda x: x['similarity'], reverse=True)
```

## 검색 최적화

### 1. 임베딩 캐싱 (Embedding Caching)

자주 사용되는 질문이나 문서의 임베딩 결과를 저장해두고 재사용하여, 반복적인 임베딩 계산 비용을 줄입니다.

### 2. 배치 처리 (Batch Processing)

여러 개의 질문을 모아 한 번에 임베딩하고 검색을 수행하여, GPU와 같은 하드웨어 자원을 효율적으로 사용하고 처리 속도를 높입니다.
