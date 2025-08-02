---
title: Attention
parent: LLM이란 무엇인가?
nav_order: 1
---

# Attention

## 개요

Attention 메커니즘은 Transformer 아키텍처의 핵심 구성 요소로, LLM(Large Language Model)이 입력 시퀀스의 다양한 부분에 집중하여 관련성 높은 정보를 추출할 수 있게 해주는 메커니즘입니다.

## Attention의 기본 개념

### 1. Attention이란?

Attention은 다음과 같은 핵심 아이디어를 기반으로 합니다:
- **Query(쿼리)**: 현재 위치에서 찾고자 하는 정보
- **Key(키)**: 각 위치가 가지고 있는 정보의 특징
- **Value(값)**: 각 위치의 실제 정보

### 2. Attention 계산 과정

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

여기서:
- `Q`: Query 행렬
- `K`: Key 행렬  
- `V`: Value 행렬
- `d_k`: Key의 차원
- `√d_k`: 스케일링 팩터 (gradient vanishing 방지)

## Attention의 종류

### 1. Self-Attention

자기 자신의 모든 위치에 대해 attention을 계산하는 방식입니다.

**특징:**
- 입력 시퀀스 내의 모든 토큰 간의 관계를 학습
- 병렬 처리 가능
- 긴 시퀀스에서도 효과적

### 2. Multi-Head Attention

여러 개의 attention head를 병렬로 사용하는 방식입니다.

**장점:**
- 서로 다른 관점에서 정보를 학습
- 모델의 표현력 향상
- 다양한 패턴 인식 가능

### 3. Cross-Attention

서로 다른 시퀀스 간의 attention을 계산하는 방식입니다.

**사용 사례:**
- 인코더-디코더 구조
- 번역 모델
- 질문-답변 시스템

## Attention의 수학적 이해

### 1. Attention Score 계산

```python
# 의사 코드
def attention_scores(query, key, value):
    # 1. Query와 Key의 유사도 계산
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 2. 스케일링
    scores = scores / math.sqrt(d_k)
    
    # 3. Softmax 적용
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. Value와 가중 평균
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### 2. Positional Encoding

Attention은 위치 정보를 고려하지 않기 때문에 positional encoding이 필요합니다.

```python
# Sinusoidal Positional Encoding
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

## LLM 서비스에서의 Attention 활용

### 1. 메모리 효율성

**문제점:**
- Attention의 계산 복잡도: O(n²)
- 긴 시퀀스에서 메모리 사용량 급증

**해결책:**
- **Sparse Attention**: 일부 위치만 attention 계산
- **Sliding Window Attention**: 제한된 범위 내에서만 attention
- **Flash Attention**: 메모리 효율적인 attention 구현

### 2. 추론 최적화

**기법들:**
- **KV Cache**: 이전 토큰의 Key, Value를 캐싱
- **Grouped Query Attention**: 여러 head를 그룹화하여 메모리 절약
- **Multi-Query Attention**: 하나의 Key, Value를 여러 Query가 공유

### 3. 실제 구현 예시

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear 변환 및 head 분할
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # 2. Attention 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # 3. Head 결합 및 출력
        context = context.view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights
```

## 성능 최적화 기법

### 1. Flash Attention

메모리 효율성을 크게 향상시키는 attention 구현 방식입니다.

**장점:**
- 메모리 사용량: O(n) → O(√n)
- 계산 속도 향상
- 긴 시퀀스 처리 가능

### 2. Sparse Attention

전체 토큰 쌍이 아닌 일부만 attention을 계산하는 방식입니다.

**패턴들:**
- **Local Attention**: 인접한 토큰들만 attention
- **Strided Attention**: 일정 간격으로 attention
- **Global Attention**: 특정 토큰은 모든 토큰과 attention

### 3. Linear Attention

Attention을 선형 복잡도로 근사하는 방식입니다.

**핵심 아이디어:**
- Kernel trick을 사용하여 복잡도 감소
- O(n²) → O(n)으로 개선

## 실무에서의 고려사항

### 1. 메모리 관리

```python
# 메모리 효율적인 attention 구현
def efficient_attention(query, key, value, chunk_size=1024):
    batch_size, seq_len, d_model = query.shape
    output = torch.zeros_like(query)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        chunk_query = query[:, i:end_i]
        
        # 청크 단위로 attention 계산
        scores = torch.matmul(chunk_query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        chunk_output = torch.matmul(attention_weights, value)
        
        output[:, i:end_i] = chunk_output
    
    return output
```

### 2. 배치 처리 최적화

```python
# 배치 크기와 시퀀스 길이에 따른 메모리 사용량 계산
def estimate_memory_usage(batch_size, seq_len, d_model, num_heads):
    # Attention 행렬 크기
    attention_matrix_size = batch_size * num_heads * seq_len * seq_len * 4  # float32
    
    # KV 캐시 크기
    kv_cache_size = batch_size * seq_len * d_model * 2 * 4  # Key + Value
    
    total_memory_mb = (attention_matrix_size + kv_cache_size) / (1024 * 1024)
    
    return total_memory_mb
```

### 3. 추론 최적화

```python
class OptimizedAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # KV 캐시 초기화
        self.k_cache = None
        self.v_cache = None
        
    def forward(self, query, key, value, use_cache=True):
        if use_cache and self.k_cache is not None:
            # 캐시된 KV와 새로운 KV 결합
            key = torch.cat([self.k_cache, key], dim=1)
            value = torch.cat([self.v_cache, value], dim=1)
        
        # Attention 계산
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        # KV 캐시 업데이트
        if use_cache:
            self.k_cache = key
            self.v_cache = value
            
        return output
```

## 결론

Attention 메커니즘은 LLM의 핵심 구성 요소로, 효과적인 구현과 최적화가 LLM 서비스의 성능을 크게 좌우합니다. 메모리 효율성, 계산 속도, 그리고 정확성의 균형을 고려한 설계가 중요합니다.

### 주요 포인트

1. **기본 개념 이해**: Query, Key, Value의 역할과 관계
2. **수학적 기반**: Attention score 계산과 softmax의 역할
3. **최적화 기법**: Flash Attention, Sparse Attention 등
4. **실무 적용**: 메모리 관리, 배치 처리, 추론 최적화
5. **확장성 고려**: 긴 시퀀스 처리와 메모리 효율성

이러한 이해를 바탕으로 LLM 서비스를 구축할 때 적절한 attention 구현을 선택하고 최적화할 수 있습니다.
