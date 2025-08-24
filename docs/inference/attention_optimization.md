---
title: Attention 최적화 방법
parent: 추론(Inference)
nav_order: 4
---

# Attention 최적화 방법 (Attention Optimization)

LLM 추론에서 attention 메커니즘을 최적화하여 메모리 사용량을 줄이고 속도를 향상시키는 방법들을 소개합니다.

## 4.1 Sparse Attention

### TidalDecode

**개념**
- 위치 지속적 sparse attention을 통한 빠르고 정확한 LLM 디코딩
- 토큰 선택의 공간적 일관성 활용

**특징**
- ✅ 2.1배 지연 시간 감소
- ✅ 공간적 일관성 유지
- ✅ 정확도 보존
- ❌ 구현 복잡도

**핵심 아이디어**
```python
def tidal_decode_attention(query, key, value, mask=None, sparsity_ratio=0.8):
    # 1. 위치 기반 sparsity 패턴 생성
    seq_len = query.size(1)
    sparsity_mask = generate_spatial_sparsity_mask(seq_len, sparsity_ratio)
    
    # 2. Sparse attention 계산
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 3. Sparsity 적용
    attention_scores = attention_scores * sparsity_mask
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax 및 출력 계산
    attention_probs = torch.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probs, value)
    
    return output

def generate_spatial_sparsity_mask(seq_len, sparsity_ratio):
    """위치 기반 sparse attention 마스크 생성"""
    mask = torch.ones(seq_len, seq_len)
    
    # 대각선 주변 영역만 유지
    for i in range(seq_len):
        start = max(0, i - int(seq_len * (1 - sparsity_ratio) // 2))
        end = min(seq_len, i + int(seq_len * (1 - sparsity_ratio) // 2))
        mask[i, :start] = 0
        mask[i, end:] = 0
    
    return mask
```

### FlexPrefill

**개념**
- 각 입력과 attention 헤드의 특정 요구사항에 맞게 동적으로 조정되는 sparse attention
- Jensen-Shannon divergence를 사용한 쿼리 인식 sparse 패턴 결정

**특징**
- ✅ 동적 sparse 패턴
- ✅ 입력별 최적화
- ✅ 헤드별 특성 고려
- ❌ 계산 오버헤드

**구현 예시**
```python
def flex_prefill_attention(query, key, value, input_characteristics):
    # 1. 입력 특성 분석
    complexity_score = analyze_input_complexity(input_characteristics)
    
    # 2. 헤드별 sparse 패턴 결정
    head_patterns = []
    for head_idx in range(query.size(1)):
        pattern = determine_head_sparsity_pattern(
            head_idx, complexity_score, query[:, head_idx]
        )
        head_patterns.append(pattern)
    
    # 3. 동적 sparse attention 계산
    outputs = []
    for head_idx in range(query.size(1)):
        head_query = query[:, head_idx:head_idx+1]
        head_key = key[:, head_idx:head_idx+1]
        head_value = value[:, head_idx:head_idx+1]
        
        # 헤드별 sparse 패턴 적용
        sparse_output = compute_sparse_attention(
            head_query, head_key, head_value, head_patterns[head_idx]
        )
        outputs.append(sparse_output)
    
    return torch.cat(outputs, dim=1)

def determine_head_sparsity_pattern(head_idx, complexity, query_vector):
    """헤드별 sparse 패턴 결정"""
    # Jensen-Shannon divergence 기반 패턴 선택
    if complexity > 0.7:  # 복잡한 입력
        return "local_window"  # 로컬 윈도우 패턴
    elif complexity > 0.3:  # 중간 복잡도
        return "strided"  # 스트라이드 패턴
    else:  # 단순한 입력
        return "global"  # 글로벌 패턴
```

### Star Attention

**개념**
- 두 단계 블록-sparse 근사로 긴 시퀀스에서 효율적인 추론
- 여러 호스트에서 attention을 분할하면서 통신 오버헤드 최소화

**특징**
- ✅ 최대 11배 속도 향상
- ✅ 97-100% 정확도 유지
- ✅ 분산 처리 지원
- ❌ 구현 복잡도
- ❌ 통신 오버헤드

**논문**
- "Star Attention: Efficient Attention for Long Sequences" (2024)

**구현 예시**
```python
def star_attention(query, key, value, block_size=64, num_blocks=8):
    batch_size, num_heads, seq_len, head_dim = query.size()
    
    # 1. 시퀀스를 블록으로 분할
    num_blocks_seq = (seq_len + block_size - 1) // block_size
    
    # 2. 블록별 attention 계산
    block_outputs = []
    for block_idx in range(num_blocks_seq):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, seq_len)
        
        # 현재 블록의 query
        block_query = query[:, :, start_idx:end_idx, :]
        
        # 관련 블록들 선택 (Star 패턴)
        relevant_blocks = select_relevant_blocks(block_idx, num_blocks_seq, num_blocks)
        
        # 선택된 블록들의 key, value
        relevant_keys = []
        relevant_values = []
        for rel_block in relevant_blocks:
            rel_start = rel_block * block_size
            rel_end = min(rel_start + block_size, seq_len)
            relevant_keys.append(key[:, :, rel_start:rel_end, :])
            relevant_values.append(value[:, :, rel_start:rel_end, :])
        
        # Concatenate
        block_key = torch.cat(relevant_keys, dim=2)
        block_value = torch.cat(relevant_values, dim=2)
        
        # Attention 계산
        attention_scores = torch.matmul(block_query, block_key.transpose(-2, -1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        block_output = torch.matmul(attention_probs, block_value)
        
        block_outputs.append(block_output)
    
    # 3. 블록 출력들을 결합
    output = torch.cat(block_outputs, dim=2)
    return output

def select_relevant_blocks(current_block, total_blocks, num_relevant):
    """Star 패턴에 따른 관련 블록 선택"""
    relevant = [current_block]
    
    # 대각선 방향으로 관련 블록 선택
    for i in range(1, num_relevant):
        # 위쪽 블록
        up_block = current_block - i
        if up_block >= 0:
            relevant.append(up_block)
        
        # 아래쪽 블록
        down_block = current_block + i
        if down_block < total_blocks:
            relevant.append(down_block)
    
    return sorted(list(set(relevant)))
```

## 4.2 KV Cache 최적화

### LOOK-M

**개념**
- 다중모달 긴 컨텍스트 추론을 위한 KV 캐시 최적화
- 텍스트 우선 방법과 병합 전략 통합

**특징**
- ✅ 80% 메모리 사용량 감소
- ✅ 다중모달 지원
- ✅ 긴 컨텍스트 최적화
- ❌ 구현 복잡도

**구현 예시**
```python
class LOOKMCache:
    def __init__(self, max_cache_size, text_priority_ratio=0.7):
        self.max_cache_size = max_cache_size
        self.text_priority_ratio = text_priority_ratio
        self.text_cache = {}
        self.multimodal_cache = {}
    
    def add_to_cache(self, key, value, modality='text'):
        if modality == 'text':
            # 텍스트는 우선순위가 높음
            if len(self.text_cache) >= self.max_cache_size * self.text_priority_ratio:
                self._evict_least_important_text()
            self.text_cache[key] = value
        else:
            # 멀티모달 데이터
            if len(self.multimodal_cache) >= self.max_cache_size * (1 - self.text_priority_ratio):
                self._evict_least_important_multimodal()
            self.multimodal_cache[key] = value
    
    def _evict_least_important_text(self):
        # LRU 기반 텍스트 캐시 제거
        oldest_key = min(self.text_cache.keys(), key=lambda k: self.text_cache[k]['timestamp'])
        del self.text_cache[oldest_key]
    
    def _evict_least_important_multimodal(self):
        # 중요도 기반 멀티모달 캐시 제거
        least_important = min(
            self.multimodal_cache.keys(),
            key=lambda k: self.multimodal_cache[k]['importance']
        )
        del self.multimodal_cache[least_important]
    
    def get_from_cache(self, key):
        if key in self.text_cache:
            return self.text_cache[key]['value']
        elif key in self.multimodal_cache:
            return self.multimodal_cache[key]['value']
        return None
```

### QuantSpec

**개념**
- 계층적 4비트 양자화된 KV 캐시를 사용한 자기 추측 디코딩
- 90% 이상의 높은 수용률 유지하면서 2.5배 속도 향상

**특징**
- ✅ 2.5배 속도 향상
- ✅ 90% 이상 수용률
- ✅ 메모리 효율적
- ❌ 양자화 품질 손실

**구현 예시**
```python
class QuantSpecCache:
    def __init__(self, cache_size, quantization_bits=4):
        self.cache_size = cache_size
        self.quantization_bits = quantization_bits
        self.cache = {}
        self.quantized_cache = {}
    
    def quantize_tensor(self, tensor):
        """텐서를 4비트로 양자화"""
        # Min-max 양자화
        min_val = tensor.min()
        max_val = tensor.max()
        
        # 양자화 스케일 계산
        scale = (max_val - min_val) / (2**self.quantization_bits - 1)
        
        # 양자화
        quantized = torch.round((tensor - min_val) / scale)
        quantized = torch.clamp(quantized, 0, 2**self.quantization_bits - 1)
        
        return quantized, scale, min_val
    
    def dequantize_tensor(self, quantized, scale, min_val):
        """양자화된 텐서를 원래 스케일로 복원"""
        return quantized * scale + min_val
    
    def add_to_cache(self, key, value):
        if len(self.cache) >= self.cache_size:
            self._evict_oldest()
        
        # 원본 텐서 저장
        self.cache[key] = value
        
        # 양자화된 텐서 저장
        quantized, scale, min_val = self.quantize_tensor(value)
        self.quantized_cache[key] = {
            'quantized': quantized,
            'scale': scale,
            'min_val': min_val
        }
    
    def get_from_cache(self, key, use_quantized=True):
        if key not in self.cache:
            return None
        
        if use_quantized and key in self.quantized_cache:
            # 양자화된 텐서 반환
            quant_data = self.quantized_cache[key]
            return self.dequantize_tensor(
                quant_data['quantized'],
                quant_data['scale'],
                quant_data['min_val']
            )
        else:
            # 원본 텐서 반환
            return self.cache[key]
```

### Round Attention

**개념**
- 상위 k개의 관련 라운드의 KV 캐시를 선택적으로 처리
- 54%-82% 메모리 사용량 감소

**특징**
- ✅ 54%-82% 메모리 절약
- ✅ 관련성 기반 선택
- ✅ 품질 유지
- ❌ 라운드 선택 복잡도

**구현 예시**
```python
def round_attention(query, key, value, round_importance_scores, top_k_rounds=5):
    batch_size, num_heads, seq_len, head_dim = query.size()
    
    # 1. 라운드별 중요도 계산
    round_scores = compute_round_importance(round_importance_scores)
    
    # 2. 상위 k개 라운드 선택
    top_rounds = torch.topk(round_scores, k=top_k_rounds, dim=-1)
    selected_rounds = top_rounds.indices
    
    # 3. 선택된 라운드의 KV 캐시만 사용
    selected_keys = []
    selected_values = []
    
    for round_idx in selected_rounds[0]:  # batch 차원
        round_start = round_idx * seq_len // len(round_importance_scores)
        round_end = (round_idx + 1) * seq_len // len(round_importance_scores)
        
        selected_keys.append(key[:, :, round_start:round_end, :])
        selected_values.append(value[:, :, round_start:round_end, :])
    
    # 4. 선택된 KV로 attention 계산
    if selected_keys:
        key_concat = torch.cat(selected_keys, dim=2)
        value_concat = torch.cat(selected_values, dim=2)
        
        attention_scores = torch.matmul(query, key_concat.transpose(-2, -1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value_concat)
    else:
        # 선택된 라운드가 없는 경우 기본 처리
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
    
    return output

def compute_round_importance(round_scores):
    """라운드별 중요도 계산"""
    # 시간적 가중치 적용 (최근 라운드가 더 중요)
    temporal_weights = torch.exp(torch.arange(len(round_scores)) * 0.1)
    weighted_scores = round_scores * temporal_weights
    
    # 정규화
    normalized_scores = weighted_scores / weighted_scores.sum()
    
    return normalized_scores
```

### SCOPE

**개념**
- prefill과 decoding 단계에서 KV 캐시 최적화를 별도로 수행
- 단계별 특성에 맞는 최적화 전략 적용

**특징**
- ✅ 단계별 최적화
- ✅ 효율적인 메모리 관리
- ✅ 성능 향상
- ❌ 구현 복잡도

## 성능 비교 및 벤치마크

### 메모리 사용량 비교

| 방법 | 메모리 절약 | 속도 향상 | 품질 유지 | 구현 난이도 |
|------|-------------|-----------|-----------|-------------|
| TidalDecode | 20-30% | 2.1x | ✅ | ⭐⭐⭐ |
| FlexPrefill | 30-50% | 1.5-2x | ✅ | ⭐⭐⭐⭐ |
| Star Attention | 40-60% | 11x | ✅ | ⭐⭐⭐⭐⭐ |
| LOOK-M | 80% | 1.5x | ✅ | ⭐⭐⭐⭐ |
| QuantSpec | 75% | 2.5x | ⚠️ | ⭐⭐⭐ |
| Round Attention | 54-82% | 1.3x | ✅ | ⭐⭐⭐⭐ |
| SCOPE | 40-60% | 1.5-2x | ✅ | ⭐⭐⭐⭐ |

### 사용 시나리오별 권장 방법

#### 긴 시퀀스 처리
- **Star Attention**: 최대 속도 향상
- **TidalDecode**: 균형잡힌 접근

#### 메모리 제약 환경
- **LOOK-M**: 최대 메모리 절약
- **QuantSpec**: 속도와 메모리 절약

#### 다중모달 응용
- **LOOK-M**: 다중모달 최적화
- **FlexPrefill**: 동적 최적화

#### 실시간 시스템
- **TidalDecode**: 빠른 응답
- **Round Attention**: 효율적인 캐시 관리

## 구현 고려사항

### 하드웨어 요구사항
- **GPU**: Star Attention, TidalDecode
- **CPU**: QuantSpec, Round Attention
- **혼합**: LOOK-M, FlexPrefill

### 메모리 관리
- **캐시 크기 조정**: 사용 가능한 메모리에 맞춤
- **LRU/LFU 정책**: 효율적인 캐시 교체
- **압축률 조정**: 품질과 메모리 절약의 균형

### 품질 보장
- **양자화 품질 모니터링**: 품질 손실 추적
- **fallback 전략**: 최적화 실패 시 기본 방법 사용
- **지속적인 평가**: 성능 지표 모니터링
