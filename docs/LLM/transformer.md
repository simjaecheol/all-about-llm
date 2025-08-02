---
title: Transformer
parent: LLM이란 무엇인가?
nav_order: 2
---

# Transformer

## 개요

Transformer는 2017년 "Attention Is All You Need" 논문에서 제안된 신경망 아키텍처로, 순환 신경망(RNN)과 컨볼루션 신경망(CNN) 없이 오직 Attention 메커니즘만을 사용하여 시퀀스 데이터를 처리하는 혁신적인 모델입니다. 현재 대부분의 LLM(Large Language Model)들이 이 Transformer 아키텍처를 기반으로 하고 있습니다.

## Transformer의 핵심 구성 요소

### 1. 전체 아키텍처

```
Input Embedding + Positional Encoding
           ↓
    Encoder Stack (N layers)
           ↓
    Decoder Stack (N layers)
           ↓
    Output Linear + Softmax
```

### 2. 주요 구성 요소와 역할

#### Input Embedding (입력 임베딩)
**역할**: 텍스트를 숫자로 변환하는 첫 번째 단계
- **목적**: 단어나 토큰을 고차원 벡터로 변환
- **작동 방식**: 각 단어를 고유한 벡터로 매핑
- **예시**: "안녕하세요" → [0.1, 0.3, -0.2, ...] (512차원 벡터)

#### Positional Encoding (위치 인코딩)
**역할**: 단어의 순서 정보를 모델에 제공
- **목적**: Attention은 위치 정보를 고려하지 않기 때문에 별도로 추가
- **작동 방식**: 각 위치에 고유한 패턴을 더함
- **예시**: 첫 번째 단어는 sin(1), 두 번째 단어는 sin(2) 패턴 추가

#### Encoder (인코더)
**역할**: 입력 텍스트를 이해하고 분석하는 부분

**Multi-Head Self-Attention (다중 헤드 자기 주의)**
- **목적**: 문장 내 모든 단어 간의 관계를 동시에 분석
- **작동 방식**: 각 단어가 다른 모든 단어를 "주목"하여 관계 학습
- **예시**: "나는 학교에 갔다"에서 "학교"가 "갔다"와의 관계를 학습
- **장점**: 병렬 처리로 모든 관계를 동시에 계산

**Feed-Forward Network (전연결 신경망)**
- **목적**: 각 단어의 의미를 더 깊이 이해하고 변환
- **작동 방식**: 각 단어를 독립적으로 처리하여 새로운 표현 생성
- **예시**: "학교" → "교육기관", "학습장소" 등 다양한 의미로 확장

**Layer Normalization (층 정규화)**
- **목적**: 학습을 안정화하고 속도를 향상
- **작동 방식**: 각 층의 출력을 정규화하여 값의 범위를 조정
- **예시**: 너무 큰 값이나 작은 값을 적절한 범위로 조정

**Residual Connection (잔차 연결)**
- **목적**: 그래디언트 흐름을 개선하여 깊은 네트워크 학습 가능
- **작동 방식**: 입력을 출력에 더해서 정보 손실 방지
- **예시**: 원본 정보 + 변환된 정보 = 더 풍부한 표현

#### Decoder (디코더)
**역할**: 이해한 내용을 바탕으로 새로운 텍스트를 생성하는 부분

**Masked Multi-Head Self-Attention (마스킹된 다중 헤드 자기 주의)**
- **목적**: 미래 정보를 숨기고 과거 정보만 사용하여 학습
- **작동 방식**: 각 단어는 자신보다 앞에 있는 단어들만 참조
- **예시**: "나는 학교에"까지 보고 "갔다"를 예측
- **이유**: 실제 사용 시에는 미래 정보가 없기 때문

**Multi-Head Cross-Attention (다중 헤드 교차 주의)**
- **목적**: 인코더에서 이해한 정보를 디코더에서 활용
- **작동 방식**: 디코더의 각 단어가 인코더의 모든 정보를 참조
- **예시**: 번역 시 원문(인코더)의 정보를 바탕으로 번역문(디코더) 생성

**Feed-Forward Network**
- **목적**: 인코더와 동일하게 각 단어의 의미를 깊이 이해
- **차이점**: 인코더의 정보도 함께 고려하여 처리

#### Output Linear + Softmax (출력 변환)
**역할**: 모델의 내부 표현을 실제 단어로 변환
- **Linear**: 고차원 벡터를 어휘 크기만큼의 벡터로 변환
- **Softmax**: 확률 분포로 변환하여 가장 가능성 높은 단어 선택
- **예시**: [0.1, 0.7, 0.2] → "학교" (70% 확률)

## Transformer의 수학적 기반

### 1. Self-Attention 계산

```python
def self_attention(x, W_q, W_k, W_v):
    # 1. Query, Key, Value 계산
    Q = x @ W_q  # (seq_len, d_k)
    K = x @ W_k  # (seq_len, d_k)
    V = x @ W_v  # (seq_len, d_v)
    
    # 2. Attention Score 계산
    scores = Q @ K.T / sqrt(d_k)
    
    # 3. Softmax 적용
    attention_weights = softmax(scores, dim=-1)
    
    # 4. Value와 가중 평균
    output = attention_weights @ V
    
    return output, attention_weights
```

### 2. Multi-Head Attention

```python
def multi_head_attention(x, num_heads, d_model):
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    
    # 각 head별로 attention 계산
    heads = []
    for h in range(num_heads):
        W_q_h = W_q[h * d_k:(h + 1) * d_k]
        W_k_h = W_k[h * d_k:(h + 1) * d_k]
        W_v_h = W_v[h * d_v:(h + 1) * d_v]
        
        head_output, _ = self_attention(x, W_q_h, W_k_h, W_v_h)
        heads.append(head_output)
    
    # 모든 head 결합
    concat_heads = torch.cat(heads, dim=-1)
    output = concat_heads @ W_o
    
    return output
```

### 3. Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

## Transformer의 작동 원리

### 1. 인코더 (Encoder)

#### 입력 처리
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 1. Self-Attention + Residual Connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual Connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

#### Feed-Forward Network
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
```

### 2. 디코더 (Decoder)

#### Masked Self-Attention
```python
def create_causal_mask(seq_len):
    """디코더에서 미래 토큰을 가리는 마스크 생성"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Cross-Attention
        cross_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## Transformer의 핵심 특징

### 1. 병렬 처리 가능성

**RNN vs Transformer의 차이점:**

**RNN (순환 신경망)의 한계:**
- **문제점**: 단어를 하나씩 순서대로 처리해야 함
- **예시**: "나는 학교에 갔다"를 처리할 때
  1. "나는" 처리 → 결과 저장
  2. "학교에" 처리 → 이전 결과와 함께 처리
  3. "갔다" 처리 → 이전 모든 결과와 함께 처리
- **단점**: 병렬 처리 불가능, 긴 문장에서 정보 손실

**Transformer의 혁신:**
- **해결책**: 모든 단어를 동시에 처리
- **예시**: "나는 학교에 갔다"의 모든 단어를 한 번에 분석
- **장점**: 병렬 처리로 속도 향상, 모든 단어 간 관계 동시 학습

```python
# RNN의 순차적 처리 (느림)
def rnn_forward(x):
    h = torch.zeros(hidden_size)
    outputs = []
    for t in range(seq_len):
        h = rnn_cell(x[t], h)  # 한 번에 하나씩만 처리
        outputs.append(h)
    return torch.stack(outputs)

# Transformer의 병렬 처리 (빠름)
def transformer_forward(x):
    # 모든 토큰을 동시에 처리
    attention_output = self_attention(x, x, x)
    return attention_output
```

### 2. 긴 시퀀스 처리 능력

**Transformer의 장점:**
- **직접 연결**: 모든 단어가 다른 모든 단어와 직접 연결
- **예시**: 100번째 단어가 1번째 단어의 정보를 직접 활용 가능
- **긴 거리 의존성**: 먼 거리에 있는 단어들도 관계 학습 가능
- **병렬 처리**: 모든 관계를 동시에 계산하여 속도 향상

**실제 예시:**
- **짧은 문장**: "나는 학교에 갔다" → 모든 단어 간 관계 학습
- **긴 문장**: "나는 어제 친구와 함께 서울에 있는 대학교에 갔다" → "나는"과 "갔다"의 관계도 직접 학습

**단점과 해결책:**
- **문제**: O(n²) 메모리 복잡도로 긴 문장에서 메모리 부족
- **해결**: Sparse Attention, Sliding Window 등 최적화 기법 사용

### 3. Attention 가중치의 해석 가능성

**Transformer의 특별한 장점:**
- **투명성**: 어떤 단어가 어떤 단어를 주목하는지 확인 가능
- **이해 가능성**: 모델의 판단 근거를 시각적으로 확인
- **디버깅**: 잘못된 예측의 원인을 분석 가능

**실제 활용 예시:**
```python
def visualize_attention(attention_weights, tokens):
    """Attention 가중치 시각화"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues')
    plt.title('어떤 단어가 어떤 단어를 주목하는지 시각화')
    plt.show()

# 예시: "나는 학교에 갔다"에서 "갔다"가 "학교"를 많이 주목하는 것을 확인
```

**해석 예시:**
- **번역**: "I love you" → "나는 당신을 사랑합니다"
  - "사랑합니다"가 "love"를 주목하는 것을 확인
- **질문 답변**: "학교는 어디에 있나요?" → "서울에 있습니다"
  - "서울"이 "어디에"를 주목하는 것을 확인

## LLM에서의 Transformer 활용

Transformer는 다양한 방식으로 활용되어 여러 종류의 LLM을 만들 수 있습니다. 각각의 특징과 용도를 이해해보겠습니다.

### 1. 인코더 전용 모델 (BERT 계열)
**역할**: 텍스트를 이해하고 분석하는 모델

**특징:**
- **구조**: 인코더만 사용 (디코더 없음)
- **목적**: 텍스트 분류, 질문 답변, 감정 분석 등
- **작동 방식**: 입력 텍스트를 깊이 이해하여 의미를 추출

**실제 활용 예시:**
- **감정 분석**: "이 영화는 정말 재미있다" → 긍정 (90% 확률)
- **질문 답변**: "서울의 수도는?" → "서울"
- **텍스트 분류**: 뉴스 기사 → 정치, 경제, 스포츠 등 분류

```python
class BERTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 인코더만 사용 (디코더 없음)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding(x)
        
        # 인코더를 통과하여 텍스트 이해
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
            
        return self.classifier(x)
```

### 2. 디코더 전용 모델 (GPT 계열)
**역할**: 텍스트를 생성하는 모델

**특징:**
- **구조**: 디코더만 사용 (인코더 없음)
- **목적**: 텍스트 생성, 대화, 코드 작성 등
- **작동 방식**: 이전 단어들을 보고 다음 단어를 예측

**실제 활용 예시:**
- **텍스트 생성**: "오늘 날씨가" → "오늘 날씨가 좋다"
- **대화**: "안녕하세요" → "안녕하세요! 무엇을 도와드릴까요?"
- **코드 작성**: "def calculate" → "def calculate_sum(a, b): return a + b"

```python
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 디코더만 사용 (인코더 없음)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding(x)
        
        # 디코더를 통과하여 텍스트 생성
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, x, mask, mask)  # Self-attention only
            
        return self.output_projection(x)
```

### 3. 인코더-디코더 모델 (T5 계열)
**역할**: 입력을 이해하고 새로운 출력을 생성하는 모델

**특징:**
- **구조**: 인코더와 디코더 모두 사용
- **목적**: 번역, 요약, 질문 답변 등
- **작동 방식**: 인코더가 입력을 이해하고, 디코더가 새로운 출력 생성

**실제 활용 예시:**
- **번역**: "I love you" → "나는 당신을 사랑합니다"
- **요약**: 긴 기사 → 핵심 내용만 요약
- **질문 답변**: 질문 + 문서 → 정확한 답변

```python
class T5LikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 인코더와 디코더 모두 사용
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 1단계: 인코더가 입력을 이해
        src_emb = self.embedding(src) + self.positional_encoding(src)
        for encoder_layer in self.encoder_layers:
            src_emb = encoder_layer(src_emb, src_mask)
        
        # 2단계: 디코더가 이해된 정보를 바탕으로 출력 생성
        tgt_emb = self.embedding(tgt) + self.positional_encoding(tgt)
        for decoder_layer in self.decoder_layers:
            tgt_emb = decoder_layer(tgt_emb, src_emb, src_mask, tgt_mask)
            
        return self.output_projection(tgt_emb)
```

### 모델 선택 가이드

**언제 어떤 모델을 사용할까?**

1. **BERT 계열 (인코더 전용)**
   - 텍스트 분류가 필요할 때
   - 감정 분석, 스팸 탐지 등
   - 예시: 고객 리뷰의 긍정/부정 분석

2. **GPT 계열 (디코더 전용)**
   - 텍스트 생성이 필요할 때
   - 대화, 코드 작성, 창작 등
   - 예시: 챗봇, AI 작가

3. **T5 계열 (인코더-디코더)**
   - 입력을 다른 형태로 변환할 때
   - 번역, 요약, 질문 답변 등
   - 예시: 한국어 → 영어 번역

## 성능 최적화 기법

### 1. 메모리 효율성

```python
class MemoryEfficientTransformer(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.attention = MultiHeadAttention(d_model, num_heads)
        
    def forward(self, x):
        seq_len = x.size(1)
        output = torch.zeros_like(x)
        
        # 청크 단위로 처리하여 메모리 사용량 감소
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_i]
            
            # 청크에 대한 attention 계산
            chunk_output, _ = self.attention(chunk, x, x)
            output[:, i:end_i] = chunk_output
            
        return output
```

### 2. 추론 최적화

```python
class OptimizedTransformer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.kv_cache = {}
        
    def forward(self, x, use_cache=True):
        if use_cache:
            # KV 캐시를 활용한 효율적인 추론
            return self._cached_forward(x)
        else:
            return self._standard_forward(x)
            
    def _cached_forward(self, x):
        # 캐시된 Key, Value를 활용하여 중복 계산 방지
        pass
```

## 실무 적용 시 고려사항

### 1. 모델 크기와 성능의 균형

```python
def calculate_model_size(d_model, num_layers, num_heads, vocab_size):
    """모델 크기 계산"""
    # Embedding
    embedding_params = vocab_size * d_model
    
    # Transformer layers
    layer_params = num_layers * (
        4 * d_model * d_model +  # Q, K, V, O projections
        2 * d_model * (4 * d_model) +  # Feed-forward
        4 * d_model  # Layer norms
    )
    
    total_params = embedding_params + layer_params
    return total_params
```

### 2. 학습 안정성

```python
class StableTransformer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        
        # 그래디언트 클리핑
        self.grad_clip = 1.0
        
    def forward(self, x):
        # Residual connection with proper scaling
        residual = x
        x = self.norm(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + 0.1 * attn_output  # 작은 스케일링
        
        return x
```

### 3. 배치 처리 최적화

```python
def optimize_batch_processing(batch_size, seq_len, d_model):
    """배치 처리 최적화"""
    # 메모리 사용량 계산
    attention_memory = batch_size * seq_len * seq_len * 4  # float32
    embedding_memory = batch_size * seq_len * d_model * 4
    
    total_memory_mb = (attention_memory + embedding_memory) / (1024 * 1024)
    
    # 배치 크기 조정
    if total_memory_mb > 8000:  # 8GB 제한
        optimal_batch_size = int(8000 * 1024 * 1024 / (seq_len * seq_len * 4))
        return optimal_batch_size
    
    return batch_size
```

## 결론

Transformer는 LLM의 기반이 되는 혁신적인 아키텍처로, Attention 메커니즘을 통해 시퀀스 데이터를 효과적으로 처리할 수 있습니다. 병렬 처리 가능성, 긴 시퀀스 처리 능력, 그리고 해석 가능성 등이 주요 장점입니다.

### 핵심 포인트

1. **아키텍처 이해**: 인코더-디코더 구조와 각 구성 요소의 역할
2. **수학적 기반**: Self-Attention, Multi-Head Attention의 계산 과정
3. **실무 적용**: 메모리 효율성, 추론 최적화, 배치 처리
4. **모델 변형**: BERT, GPT, T5 등 다양한 모델 구조
5. **성능 최적화**: 메모리 관리, 학습 안정성, 배치 처리 최적화

### 초보자를 위한 요약

**Transformer란 무엇인가요?**
- 텍스트를 이해하고 생성하는 AI 모델의 핵심 구조
- 마치 사람이 문장을 읽고 이해하는 방식과 유사
- 각 단어가 다른 단어들과 어떤 관계가 있는지 동시에 분석

**왜 중요한가요?**
- 이전 모델들보다 훨씬 정확하고 빠름
- 긴 문장도 잘 처리할 수 있음
- 현재 대부분의 AI 언어 모델이 이 구조를 사용

**어떻게 작동하나요?**
1. **입력**: "안녕하세요" → 숫자로 변환
2. **이해**: 각 단어가 다른 단어와 어떤 관계인지 분석
3. **생성**: 이해한 내용을 바탕으로 새로운 텍스트 생성

**실제 활용 예시:**
- **번역**: 한국어 → 영어
- **대화**: 질문에 대한 답변
- **작성**: 코드, 글, 시 등

이러한 이해를 바탕으로 LLM 서비스를 구축할 때 적절한 Transformer 구현을 선택하고 최적화할 수 있습니다.
