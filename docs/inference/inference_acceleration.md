---
title: 추론 가속화 방법
parent: 추론(Inference)
nav_order: 3
---

# 추론 가속화 방법 (Inference Acceleration)

LLM 추론 속도를 향상시키는 다양한 방법들을 소개합니다. 이 방법들은 품질을 유지하면서 추론 속도를 크게 향상시킬 수 있습니다.

## 3.1 Speculative Decoding 계열

### Traditional Speculative Decoding

**개념**
- 작은 드래프트 모델이 여러 토큰을 제안하고, 대상 모델이 병렬로 검증
- 품질 손실 없이 2-3배 속도 향상

**작동 원리**
1. **드래프트 생성**: 작은 모델이 γ개의 토큰을 순차적으로 생성
2. **병렬 검증**: 대상 모델이 모든 드래프트 토큰을 병렬로 검증
3. **수용/거부**: 검증된 토큰들을 수용하고 거부된 토큰부터 재생성

**특징**
- ✅ 품질 손실 없음
- ✅ 2-3배 속도 향상
- ✅ 기존 모델과 호환
- ❌ 드래프트 모델 필요
- ❌ 메모리 사용량 증가

**논문**
- "Fast Inference from Transformers via Speculative Decoding" (2022)

**구현 예시**
```python
def speculative_decoding(target_model, draft_model, input_ids, max_length, gamma=4):
    for _ in range(max_length):
        # 1. 드래프트 모델로 γ개 토큰 생성
        draft_tokens = []
        current_ids = input_ids.clone()
        
        for _ in range(gamma):
            outputs = draft_model(current_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            draft_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
        
        # 2. 대상 모델로 병렬 검증
        extended_ids = torch.cat([input_ids] + draft_tokens, dim=-1)
        target_outputs = target_model(extended_ids)
        
        # 3. 토큰별 확률 계산
        target_probs = torch.softmax(target_outputs.logits, dim=-1)
        draft_probs = torch.softmax(draft_model(extended_ids).logits, dim=-1)
        
        # 4. 수용/거부 결정
        accept_mask = torch.rand_like(target_probs) < (target_probs / draft_probs)
        
        # 수용된 토큰들 찾기
        accepted_count = 0
        for i in range(gamma):
            if accept_mask[0, -gamma+i]:
                accepted_count += 1
            else:
                break
        
        # 수용된 토큰들 추가
        if accepted_count > 0:
            input_ids = torch.cat([input_ids] + draft_tokens[:accepted_count], dim=-1)
        
        # 거부된 경우 새로운 토큰 생성
        if accepted_count < gamma:
            outputs = target_model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### Self-Speculative Decoding

**개념**
- 대상 LLM의 중간 레이어를 건너뛰어 드래프트 모델로 사용
- 추가 매개변수나 훈련 없이 plug-and-play 솔루션

**특징**
- ✅ 추가 모델 불필요
- ✅ plug-and-play 솔루션
- ✅ 메모리 효율적
- ❌ 모델 아키텍처 의존적

**논문**
- "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding" (2024)

**구현 예시**
```python
def self_speculative_decoding(model, input_ids, max_length, skip_layers=[6, 12, 18]):
    for _ in range(max_length):
        # 1. 중간 레이어 건너뛰기로 드래프트 생성
        draft_outputs = model(input_ids, skip_layers=skip_layers)
        draft_token = torch.argmax(draft_outputs.logits[:, -1, :], dim=-1)
        
        # 2. 전체 모델로 검증
        full_outputs = model(input_ids)
        full_token = torch.argmax(full_outputs.logits[:, -1, :], dim=-1)
        
        # 3. 토큰 비교 및 수용
        if draft_token == full_token:
            input_ids = torch.cat([input_ids, draft_token.unsqueeze(0)], dim=-1)
        else:
            input_ids = torch.cat([input_ids, full_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### SWIFT (On-the-Fly Self-Speculative)

**개념**
- 추론 중에 건너뛸 중간 레이어를 적응적으로 선택
- 1.3x-1.6x 속도 향상 달성

**특징**
- ✅ 적응적 레이어 선택
- ✅ 동적 최적화
- ✅ 효율적인 속도 향상
- ❌ 구현 복잡도

**논문**
- "SWIFT: On-the-Fly Self-Speculative Decoding" (2024)

## 3.2 병렬 디코딩 방법

### Medusa Decoding

**개념**
- LLM에 여러 디코딩 헤드를 추가하여 여러 후속 토큰을 병렬로 예측
- 트리 기반 attention 메커니즘 사용

**특징**
- ✅ 별도의 드래프트 모델 없음
- ✅ 2배 속도 향상
- ✅ 트리 기반 attention
- ❌ 모델 수정 필요
- ❌ 추가 훈련 필요

**논문**
- "Medusa: Simple LLM Inference Acceleration Framework" (2024)

**구현 예시**
```python
class MedusaHead(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)
        ])
    
    def forward(self, hidden_states):
        outputs = []
        for head in self.heads:
            outputs.append(head(hidden_states))
        return torch.stack(outputs, dim=1)

def medusa_decoding(model, input_ids, max_length, medusa_head):
    for _ in range(max_length):
        # 1. 메인 모델 forward pass
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states[-1]
        
        # 2. Medusa 헤드로 병렬 토큰 예측
        medusa_outputs = medusa_head(hidden_states[:, -1:])
        
        # 3. 트리 기반 attention으로 토큰 선택
        tree_attention = compute_tree_attention(medusa_outputs)
        
        # 4. 최적 경로 선택
        selected_tokens = select_optimal_path(tree_attention)
        
        input_ids = torch.cat([input_ids, selected_tokens], dim=-1)
    
    return input_ids
```

### Blockwise Parallel Decoding

**개념**
- 여러 시간 단계에 대한 예측을 병렬로 수행
- 상수 개수의 mask-predict 사이클로 디코딩

**특징**
- ✅ 병렬 처리
- ✅ 일정한 디코딩 시간
- ❌ 품질 손실 가능성

**논문**
- "Blockwise Parallel Decoding for Deep Autoregressive Models" (2018)

### ParaDecode

**개념**
- 보조 모델이나 원본 모델 매개변수 변경 없이 병렬 토큰 처리
- 중간 레이어 표현을 사용한 토큰 예측

**특징**
- ✅ 모델 수정 불필요
- ✅ 효율적인 병렬 처리
- ❌ 구현 복잡도

## 3.3 기타 가속화 방법

### Model-free Speculative Decoding

**개념**
- n-gram 토큰 맵을 사용하여 드래프트 토큰 생성
- 별도의 드래프트 모델 없이 효율적인 추론

**특징**
- ✅ 드래프트 모델 불필요
- ✅ n-gram 기반 빠른 생성
- ✅ 메모리 효율적
- ❌ n-gram 품질 의존적

**구현 예시**
```python
def ngram_speculative_decoding(model, input_ids, max_length, ngram_map, gamma=4):
    for _ in range(max_length):
        # 1. n-gram 기반 드래프트 토큰 생성
        context = input_ids[-3:].tolist()  # 3-gram 컨텍스트
        draft_tokens = ngram_map.get(tuple(context), [])
        
        if len(draft_tokens) >= gamma:
            draft_tokens = draft_tokens[:gamma]
        else:
            # 부족한 경우 모델로 생성
            remaining = gamma - len(draft_tokens)
            for _ in range(remaining):
                outputs = model(input_ids)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                draft_tokens.append(next_token)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        # 2. 모델 검증 및 수용
        for token in draft_tokens:
            outputs = model(input_ids)
            predicted_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            
            if token == predicted_token:
                input_ids = torch.cat([input_ids, token.unsqueeze(0)], dim=-1)
            else:
                input_ids = torch.cat([input_ids, predicted_token.unsqueeze(0)], dim=-1)
                break
    
    return input_ids
```

### NoMAD-Attention

**개념**
- CPU에서 곱셈-덧셈 없는 attention 계산
- SIMD 레지스터의 초고속 검색 활용

**특징**
- ✅ CPU 최적화
- ✅ 곱셈-덧셈 연산 제거
- ✅ SIMD 활용
- ❌ 하드웨어 의존적

## 성능 비교 및 벤치마크

### 속도 향상 비교

| 방법 | 속도 향상 | 품질 유지 | 구현 난이도 | 메모리 사용량 |
|------|-----------|-----------|-------------|---------------|
| Traditional Speculative | 2-3x | ✅ | ⭐⭐⭐ | 중간 |
| Self-Speculative | 1.5-2x | ✅ | ⭐⭐ | 낮음 |
| SWIFT | 1.3-1.6x | ✅ | ⭐⭐⭐ | 낮음 |
| Medusa | 2x | ✅ | ⭐⭐⭐⭐ | 중간 |
| Blockwise | 1.5-2x | ⚠️ | ⭐⭐⭐ | 낮음 |
| ParaDecode | 1.5-2x | ✅ | ⭐⭐⭐⭐ | 낮음 |
| N-gram Speculative | 1.5-2x | ⚠️ | ⭐⭐ | 매우 낮음 |

### 사용 시나리오별 권장 방법

#### 실시간 대화 시스템
- **Self-Speculative Decoding**: 빠른 응답, 품질 유지
- **SWIFT**: 적응적 최적화

#### 배치 처리
- **Traditional Speculative**: 높은 품질, 안정적 성능
- **Medusa**: 지속적인 속도 향상

#### 제한된 리소스 환경
- **N-gram Speculative**: 메모리 효율적
- **ParaDecode**: 모델 수정 불필요

#### 고품질 요구사항
- **Traditional Speculative**: 품질 보장
- **Self-Speculative**: 균형잡힌 접근

## 구현 고려사항

### 하드웨어 요구사항
- **GPU**: Traditional Speculative, Medusa
- **CPU**: NoMAD-Attention, N-gram Speculative
- **혼합**: Self-Speculative, SWIFT

### 메모리 최적화
- **KV Cache 관리**: 효율적인 메모리 사용
- **배치 크기 조정**: 메모리와 속도의 균형
- **그래디언트 체크포인팅**: 메모리 절약

### 품질 보장
- **검증 메커니즘**: 품질 손실 방지
- **fallback 전략**: 실패 시 안전한 복구
- **모니터링**: 지속적인 품질 추적
