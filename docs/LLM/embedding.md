---
title: Embedding
parent: LLM이란 무엇인가?
nav_order: 5
---

# Embedding

## 개요

임베딩(Embedding)은 토큰을 고차원 벡터로 변환하는 과정으로, LLM이 텍스트의 의미를 수학적으로 표현할 수 있게 해주는 핵심 기술입니다. 토큰화된 텍스트를 모델이 이해할 수 있는 숫자 형태로 변환하는 첫 번째 단계이며, RAG(Retrieval-Augmented Generation) 시스템에서도 중요한 역할을 합니다.

## 토큰과 임베딩의 관계

### 1. 처리 과정

**텍스트 → 토큰 → 임베딩 → 모델 처리**

```python
# 전체 과정 예시
text = "안녕하세요"
tokens = ["안녕", "하세요"]  # 토크나이저
token_ids = [101, 102]      # 토큰 ID
embeddings = [[0.1, 0.3, -0.2, ...], [0.4, -0.1, 0.5, ...]]  # 임베딩 벡터
```

### 2. 임베딩의 역할

**토큰의 한계:**
- 토큰은 단순한 ID 번호
- 의미적 관계를 표현할 수 없음
- 유사한 단어들이 전혀 다른 ID를 가짐

**임베딩의 해결책:**
- 의미적으로 유사한 토큰들이 유사한 벡터를 가짐
- 벡터 간 거리로 의미적 유사도 계산 가능
- 수학적 연산으로 의미 조작 가능

### 3. 임베딩 공간의 특징

```python
# 임베딩 공간에서의 의미적 관계
king_vector = embeddings["king"]
queen_vector = embeddings["queen"]
man_vector = embeddings["man"]
woman_vector = embeddings["woman"]

# 의미적 관계: king - man + woman ≈ queen
result = king_vector - man_vector + woman_vector
# result와 queen_vector가 유사함
```

## 임베딩 모델의 종류

### 1. 정적 임베딩 (Static Embedding)

**특징:**
- 학습 후 고정된 임베딩
- 단어의 의미가 문맥과 무관하게 일정
- 빠른 처리 속도

**대표 모델:**

#### Word2Vec
```python
# Word2Vec 예시
from gensim.models import Word2Vec

# 학습
sentences = [["나는", "학교에", "갔다"], ["그는", "회사에", "갔다"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 임베딩 추출
embedding = model.wv["학교"]
```

#### GloVe
```python
# GloVe 예시
# 전역적 단어-단어 동시발생 행렬을 사용
# Word2Vec보다 더 정확한 의미 표현
```

### 2. 문맥 임베딩 (Contextual Embedding)

**특징:**
- 문맥에 따라 동적으로 변하는 임베딩
- 같은 단어도 문맥에 따라 다른 벡터
- 더 정확한 의미 표현

**대표 모델:**

#### BERT 임베딩
```python
# BERT 임베딩 예시
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I love artificial intelligence"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 문맥 임베딩 (마지막 층)
contextual_embeddings = outputs.last_hidden_state
```

#### GPT 임베딩
```python
# GPT 임베딩 예시
from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

text = "I love artificial intelligence"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# GPT 임베딩
gpt_embeddings = outputs.last_hidden_state
```

## 임베딩의 수학적 기반

### 1. 임베딩 행렬

```python
# 임베딩 행렬 구조
embedding_matrix = {
    "안녕": [0.1, 0.3, -0.2, 0.5, ...],  # 512차원 벡터
    "하세요": [0.4, -0.1, 0.5, 0.2, ...],
    "반갑습니다": [0.2, 0.6, -0.3, 0.1, ...],
    # ... 수만 개의 토큰
}

# 임베딩 차원
embedding_dim = 512  # 일반적인 임베딩 차원
vocab_size = 50000   # 어휘집 크기
```

### 2. 임베딩 계산

```python
def get_embedding(token_id, embedding_matrix):
    """토큰 ID로부터 임베딩 벡터 추출"""
    return embedding_matrix[token_id]

def compute_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    import numpy as np
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 예시
token_id = 101
embedding = get_embedding(token_id, embedding_matrix)
similarity = compute_similarity(embedding1, embedding2)
```

### 3. 임베딩 학습

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, token_ids):
        return self.embedding(token_ids)

# 학습 과정
embedding_layer = EmbeddingLayer(vocab_size=50000, embedding_dim=512)
optimizer = torch.optim.Adam(embedding_layer.parameters())

for batch in training_data:
    token_ids, targets = batch
    embeddings = embedding_layer(token_ids)
    loss = compute_loss(embeddings, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## LLM에서의 임베딩 활용

### 1. LLM 내부 임베딩 처리

**LLM에서 임베딩의 역할:**
- 토큰을 모델이 이해할 수 있는 벡터로 변환
- 문맥 정보를 고려한 의미적 표현
- 모델의 학습과 추론에 핵심적인 역할

### 2. 임베딩 레이어 (Embedding Layer)

#### 기본 임베딩 레이어
```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, token_ids):
        return self.embedding(token_ids)

# 사용 예시
embedding_layer = EmbeddingLayer(vocab_size=50000, embedding_dim=512)
token_ids = torch.tensor([[101, 102, 103]])
embeddings = embedding_layer(token_ids)
```

#### 위치 인코딩과 결합
```python
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device)
        
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)
        
        return token_embeddings + position_embeddings
```

### 3. 임베딩 학습과 최적화

#### 임베딩 학습 과정
```python
def train_embeddings(model, training_data, epochs=10):
    """임베딩 학습"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in training_data:
            token_ids, targets = batch
            
            # 임베딩 생성
            embeddings = model.embedding_layer(token_ids)
            
            # 모델 통과
            outputs = model(embeddings)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 임베딩 최적화 기법
```python
def optimize_embeddings_for_llm(embeddings, method="normalize"):
    """LLM을 위한 임베딩 최적화"""
    if method == "normalize":
        # L2 정규화
        for emb in embeddings:
            emb = normalize_vector(emb)
    
    elif method == "dropout":
        # 드롭아웃으로 과적합 방지
        dropout = nn.Dropout(0.1)
        embeddings = dropout(embeddings)
    
    return embeddings

def normalize_vector(vector):
    """벡터 정규화"""
    import numpy as np
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
```

### 4. 임베딩 시각화와 분석

#### 임베딩 시각화
```python
def visualize_embeddings(embeddings, labels, method="tsne"):
    """임베딩 시각화"""
    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    
    # 시각화
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.show()
```

#### 임베딩 분석
```python
def analyze_embeddings(embeddings, vocab):
    """임베딩 분석"""
    # 1. 유사한 단어 찾기
    def find_similar_words(word, top_k=5):
        word_idx = vocab.get(word, -1)
        if word_idx == -1:
            return []
        
        word_embedding = embeddings[word_idx]
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            if i != word_idx:
                similarity = cosine_similarity(word_embedding, embedding)
                similarities.append((vocab.idx_to_token[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # 2. 임베딩 공간 분석
    def analyze_embedding_space():
        # 평균 벡터 길이
        avg_length = np.mean([np.linalg.norm(emb) for emb in embeddings])
        
        # 벡터 간 평균 거리
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        return {
            'avg_length': avg_length,
            'avg_distance': avg_distance,
            'vocab_size': len(embeddings)
        }
    
    return find_similar_words, analyze_embedding_space
```

## 임베딩 모델 선택 가이드

### 1. 작업별 임베딩 모델

| 작업 유형 | 권장 모델 | 특징 |
|-----------|-----------|------|
| **일반 텍스트** | BERT, RoBERTa | 문맥 이해 우수 |
| **다국어** | mBERT, XLM-R | 다국어 지원 |
| **도메인 특화** | BioBERT, SciBERT | 특정 도메인에 최적화 |
| **검색 최적화** | DPR, Sentence-BERT | 검색 성능에 특화 |
| **코드** | CodeBERT, GraphCodeBERT | 코드 이해에 특화 |

### 2. 성능 비교

```python
# 임베딩 모델 성능 비교
embedding_models = {
    "BERT-base": {
        "dimension": 768,
        "speed": "medium",
        "accuracy": "high",
        "memory": "medium"
    },
    "Sentence-BERT": {
        "dimension": 768,
        "speed": "fast",
        "accuracy": "high",
        "memory": "medium"
    },
    "USE": {
        "dimension": 512,
        "speed": "very_fast",
        "accuracy": "medium",
        "memory": "low"
    }
}
```

### 3. 실무 고려사항

#### 메모리 효율성
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

#### 속도 최적화
```python
def optimize_embedding_speed(embedding_model, method="batch"):
    """임베딩 속도 최적화"""
    
    if method == "batch":
        # 배치 처리로 GPU 활용도 향상
        return batch_embedding(embedding_model)
    
    elif method == "cache":
        # 캐싱으로 중복 계산 방지
        return cached_embedding(embedding_model)
    
    elif method == "approximate":
        # 근사 알고리즘으로 속도 향상
        return approximate_embedding(embedding_model)

def batch_embedding(embedding_model):
    """배치 임베딩"""
    def batch_encode(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        return embeddings
    
    return batch_encode
```

## 결론

임베딩은 LLM의 핵심 구성 요소로, 토큰을 의미 있는 벡터로 변환하여 모델이 텍스트를 이해할 수 있게 해줍니다. 적절한 임베딩 모델 선택과 최적화는 LLM의 성능과 효율성에 직접적인 영향을 미칩니다.

### 핵심 포인트

1. **토큰-임베딩 관계**: 토큰을 의미 있는 벡터로 변환하는 과정
2. **임베딩 모델 종류**: 정적 임베딩과 문맥 임베딩의 차이
3. **LLM 내부 활용**: 모델의 학습과 추론에 핵심적인 역할
4. **실무 최적화**: 메모리, 속도, 정확도의 균형 고려

이러한 이해를 바탕으로 LLM을 구축할 때 적절한 임베딩 모델을 선택하고 최적화할 수 있습니다.

---

## 관련 문서

- **[RAG에서의 임베딩 활용](../RAG/embedding.md)**: RAG 시스템에서 임베딩을 활용한 문서 검색과 질문 답변 시스템 구축 방법
