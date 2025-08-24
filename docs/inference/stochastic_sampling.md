---
title: 확률적 샘플링 방법
parent: 추론(Inference)
nav_order: 2
---

# 확률적 샘플링 방법 (Stochastic Sampling)

확률적 샘플링은 모델의 확률 분포에서 토큰을 무작위로 선택하는 방법으로, 다양성과 창의성을 제공합니다.

## 2.1 기본 샘플링 전략

### Temperature Sampling

**개념**
- softmax 분포에서 온도 매개변수로 무작위성 조절
- 가장 기본적이고 널리 사용되는 확률적 샘플링 방법

**특징**
- ✅ 구현이 간단
- ✅ 무작위성 조절 가능
- ✅ 창의적 출력 생성
- ❌ 일관성 부족
- ❌ 품질 예측 어려움

**온도별 특성**
- **낮은 온도 (0에 가까움)**: 예측 가능하고 보수적인 출력
- **높은 온도 (1 이상)**: 창의적이고 다양한 출력
- **중간 온도 (0.7-0.9)**: 균형잡힌 출력

**구현 예시**
```python
def temperature_sampling(model, input_ids, max_length, temperature=1.0):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        
        # softmax로 확률 분포 생성
        probs = torch.softmax(logits, dim=-1)
        
        # 확률에 따라 토큰 샘플링
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return input_ids
```

**온도별 비교**
```python
# 다양한 온도에서의 샘플링
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
for temp in temperatures:
    result = temperature_sampling(model, input_ids, max_length, temp)
    print(f"Temperature {temp}: {tokenizer.decode(result)}")
```

### Top-k Sampling

**개념**
- 상위 k개의 가능한 토큰에서만 샘플링
- 고정된 k값으로 후보 집합 제한

**특징**
- ✅ 구현이 간단
- ✅ 무작위성 제한
- ✅ 품질 향상
- ❌ 적응성 부족
- ❌ k값 선택의 어려움

**구현 예시**
```python
def top_k_sampling(model, input_ids, max_length, k=50):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # 상위 k개 토큰 선택
        top_k_logits, top_k_indices = torch.topk(logits, k)
        
        # 선택된 토큰들에 대해서만 softmax 적용
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # 샘플링
        selected_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[0, selected_idx]
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### Top-p (Nucleus) Sampling

**개념**
- 누적 확률이 p를 초과하는 최소 토큰 집합에서 샘플링
- 동적으로 후보 집합 크기 조절

**특징**
- ✅ top-k보다 더 유연하고 적응적
- ✅ 모델의 확신도에 따른 동적 조절
- ✅ 품질과 다양성의 균형
- ❌ 구현이 약간 복잡

**구현 예시**
```python
def top_p_sampling(model, input_ids, max_length, p=0.9):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # 확률 분포 계산
        probs = torch.softmax(logits, dim=-1)
        
        # 확률을 내림차순으로 정렬
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 누적 확률 계산
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # p를 초과하는 토큰들 제거
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        # 유효한 토큰들만 선택
        valid_indices = sorted_indices[~sorted_indices_to_remove]
        valid_probs = sorted_probs[~sorted_indices_to_remove]
        
        # 정규화된 확률로 샘플링
        normalized_probs = valid_probs / valid_probs.sum()
        selected_idx = torch.multinomial(normalized_probs, num_samples=1)
        next_token = valid_indices[selected_idx]
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

## 2.2 고급 샘플링 방법

### Min-p Sampling

**개념**
- 상위 토큰의 확률에 따라 샘플링 임계값을 동적 조정
- 모델의 확신도에 기반한 적응적 토큰 선택

**특징**
- ✅ 높은 온도에서도 일관성과 다양성 균형 유지
- ✅ 모델의 확신도에 따른 적응적 조절
- ✅ 창의성과 품질의 균형
- ❌ 구현 복잡도 증가

**논문**
- "Min-p Sampling for Creative and Coherent LLM Outputs" (2024)

**구현 예시**
```python
def min_p_sampling(model, input_ids, max_length, min_p=0.1):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # 확률 분포 계산
        probs = torch.softmax(logits, dim=-1)
        
        # 최소 확률 임계값 적용
        min_prob = probs.max() * min_p
        valid_mask = probs >= min_prob
        
        if valid_mask.sum() == 0:
            valid_mask = probs >= probs.max()
        
        # 유효한 토큰들만 선택
        valid_probs = probs * valid_mask
        valid_probs = valid_probs / valid_probs.sum()
        
        # 샘플링
        next_token = torch.multinomial(valid_probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return input_ids
```

### Typical Sampling

**개념**
- 통계적으로 "전형적인" 토큰 선택
- 너무 예측 가능하거나 희귀하지 않은 토큰 선택

**특징**
- ✅ 자연스러운 텍스트 생성
- ✅ 극단적 확률 방지
- ✅ 일관성 향상
- ❌ 구현 복잡도

**핵심 아이디어**
```python
def typical_sampling(model, input_ids, max_length, typical_p=0.9):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # 확률 분포 계산
        probs = torch.softmax(logits, dim=-1)
        
        # 엔트로피 계산
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # typicality 점수 계산
        log_probs = torch.log(probs + 1e-10)
        typicality = torch.abs(log_probs + entropy)
        
        # typical_p에 해당하는 토큰들 선택
        sorted_typicality, sorted_indices = torch.sort(typicality)
        cutoff_idx = int(typical_p * len(sorted_typicality))
        
        valid_indices = sorted_indices[:cutoff_idx]
        valid_probs = probs[valid_indices]
        valid_probs = valid_probs / valid_probs.sum()
        
        # 샘플링
        selected_idx = torch.multinomial(valid_probs, num_samples=1)
        next_token = valid_indices[selected_idx]
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### Mirostat Sampling

**개념**
- top-k 토큰에서 샘플링하는 동안 텍스트의 perplexity 비율을 직접 제어
- 목표 surprise 값을 유지하기 위해 피드백 기반으로 k값 조정

**특징**
- ✅ 일정한 perplexity 유지
- ✅ 일관된 품질 보장
- ✅ 피드백 기반 적응
- ❌ 구현 복잡도
- ❌ 계산 오버헤드

**논문**
- "mirostat: a neural text decoding algorithm" (2020)

**구현 예시**
```python
def mirostat_sampling(model, input_ids, max_length, target_surprise=1.0, learning_rate=0.1):
    k = 50  # 초기 k값
    
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # top-k 샘플링
        top_k_logits, top_k_indices = torch.topk(logits, k)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # 샘플링
        selected_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[0, selected_idx]
        
        # surprise 계산 (선택된 토큰의 확률)
        selected_prob = top_k_probs[0, selected_idx]
        surprise = -torch.log(selected_prob)
        
        # k값 조정
        error = target_surprise - surprise
        k = max(1, min(100, k + learning_rate * error))
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### η-Sampling

**개념**
- 엔트로피 의존적 임계값 아래의 확률을 가진 단어를 잘라내기

**특징**
- ✅ 엔트로피 기반 적응적 샘플링
- ✅ 자연스러운 확률 분포 유지
- ❌ 구현 복잡도

## 샘플링 방법 비교

### 성능 비교표

| 방법 | 속도 | 품질 | 다양성 | 일관성 | 구현 난이도 |
|------|------|------|--------|--------|-------------|
| Temperature | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Top-k | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Top-p | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Min-p | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Typical | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Mirostat | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

### 권장 사용 시나리오

- **Temperature**: 창의적 글쓰기, 브레인스토밍
- **Top-k/Top-p**: 일반적인 텍스트 생성
- **Min-p**: 높은 온도에서도 일관성 필요한 경우
- **Typical**: 자연스러운 대화, 스토리텔링
- **Mirostat**: 일정한 품질이 요구되는 경우

### 하이브리드 접근법

여러 샘플링 방법을 조합하여 사용할 수 있습니다:

```python
def hybrid_sampling(model, input_ids, max_length, method='top_p', **kwargs):
    if method == 'top_p':
        return top_p_sampling(model, input_ids, max_length, **kwargs)
    elif method == 'min_p':
        return min_p_sampling(model, input_ids, max_length, **kwargs)
    elif method == 'typical':
        return typical_sampling(model, input_ids, max_length, **kwargs)
    elif method == 'mirostat':
        return mirostat_sampling(model, input_ids, max_length, **kwargs)
    else:
        return temperature_sampling(model, input_ids, max_length, **kwargs)
```
