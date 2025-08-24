---
title: 결정론적 디코딩 방법
parent: 추론(Inference)
nav_order: 1
---

# 결정론적 디코딩 방법 (Deterministic Decoding)

결정론적 디코딩은 각 단계에서 가장 확실한 토큰을 선택하는 방법으로, 일관된 결과를 보장하지만 다양성은 제한됩니다.

## 1.1 기본 방법들

### Greedy Search (그리디 서치)

**개념**
- 각 단계에서 가장 높은 확률의 토큰을 선택
- 가장 빠르고 단순한 디코딩 방법

**특징**
- ✅ 가장 빠른 추론 속도
- ✅ 구현이 쉽고 예측 가능한 출력
- ✅ 메모리 사용량 최소
- ❌ 품질이 낮고 반복적인 텍스트 생성 가능
- ❌ 다양성 부족

**사용 시나리오**
- 번역, 요약 등 정확성이 중요한 작업
- 빠른 응답이 필요한 실시간 시스템
- 일관된 출력이 요구되는 경우

**구현 예시**
```python
def greedy_search(model, input_ids, max_length):
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    return input_ids
```

### Beam Search (빔 서치)

**개념**
- 각 단계에서 상위 k개의 후보 시퀀스를 유지
- 전역적으로 최적화된 출력 생성

**특징**
- ✅ 전역적으로 최적화된 출력
- ✅ 번역 및 요약에 효과적
- ✅ 품질 향상
- ❌ 계산 비용이 높음
- ❌ 메모리 사용량 증가

**논문**
- Harpy Speech Recognition System (1976)에서 최초 도입
- "Beam Search Strategies for Neural Machine Translation" (2017)

**구현 예시**
```python
def beam_search(model, input_ids, max_length, beam_size=5):
    beam_scores = torch.zeros(beam_size)
    beam_tokens = input_ids.unsqueeze(0).repeat(beam_size, 1)
    
    for _ in range(max_length):
        candidates = []
        for i in range(beam_size):
            outputs = model(beam_tokens[i])
            top_k = torch.topk(outputs.logits[:, -1, :], beam_size)
            for j in range(beam_size):
                score = beam_scores[i] + top_k.values[0, j]
                tokens = torch.cat([beam_tokens[i], top_k.indices[0, j:j+1]])
                candidates.append((score, tokens))
        
        # 상위 beam_size개 선택
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam_scores = torch.tensor([c[0] for c in candidates[:beam_size]])
        beam_tokens = torch.stack([c[1] for c in candidates[:beam_size]])
    
    return beam_tokens[0]
```

### Diverse Beam Search

**개념**
- 표준 beam search의 변형으로 다양성을 증진
- 그룹 간 다양성을 최대화하는 다양성 항 포함

**특징**
- ✅ 다양성 증진
- ✅ 품질 유지
- ❌ 계산 복잡도 증가

## 1.2 대조적 방법들

### Contrastive Search

**개념**
- 모델의 등방성 표현 공간을 기반으로 한 새로운 디코딩 방법
- 이전 컨텍스트에서 더 구별되는 토큰을 선택

**특징**
- ✅ 인간 수준의 성능에 근접
- ✅ 일관성과 다양성의 균형
- ✅ 반복성 감소

**논문**
- "Contrastive Search Is What You Need For Neural Text Generation" (2022)

**핵심 아이디어**
```python
def contrastive_search(model, input_ids, max_length, alpha=0.6):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # 확률 분포
        probs = torch.softmax(logits, dim=-1)
        
        # 대조적 점수 계산
        contrastive_scores = alpha * torch.log(probs) - (1 - alpha) * diversity_score
        
        next_token = torch.argmax(contrastive_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return input_ids
```

### Contrastive Decoding

**개념**
- 큰 LLM과 작은 보조 모델 간의 확률 차이를 최대화
- 원치 않는 행동을 제거하기 위해 설계

**특징**
- ✅ 품질 향상
- ✅ 원치 않는 출력 방지
- ❌ 두 모델 필요

### Frustratingly Simple Decoding

**개념**
- 현재 접두사를 기반으로 구성된 보조 안티-LM과 LLM 간의 대비를 활용

**특징**
- ✅ 구현이 간단
- ✅ 효과적인 품질 향상
- ❌ 보조 모델 필요

## 사용 가이드

### 선택 기준

| 방법 | 속도 | 품질 | 다양성 | 메모리 |
|------|------|------|--------|--------|
| Greedy | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| Beam Search | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Contrastive | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 권장 사용 시나리오

- **Greedy Search**: 빠른 응답이 필요한 경우
- **Beam Search**: 번역, 요약 등 품질이 중요한 경우
- **Contrastive Search**: 창의성과 품질을 모두 원하는 경우
- **Diverse Beam**: 다양한 후보가 필요한 경우
