---
title: 품질 및 제어 방법
parent: 추론(Inference)
nav_order: 5
---

# 품질 및 제어 방법 (Quality and Control Methods)

LLM 추론에서 출력 품질을 향상시키고 생성 과정을 제어하는 다양한 방법들을 소개합니다.

## 5.1 품질 향상

### Minimum Bayes Risk (MBR) Decoding

**개념**
- 유틸리티 함수에 대해 모델 분포에서 가장 높은 기대 유틸리티를 가진 가설 출력
- confidence 기반 pruning으로 계산 효율성 개선

**특징**
- ✅ 품질 향상
- ✅ confidence 기반 선택
- ✅ 계산 효율성
- ❌ 구현 복잡도
- ❌ 유틸리티 함수 정의 필요

**핵심 아이디어**
```python
def mbr_decoding(model, input_ids, max_length, num_candidates=100, utility_function=None):
    if utility_function is None:
        utility_function = default_utility_function
    
    # 1. 후보 시퀀스 생성
    candidates = generate_candidates(model, input_ids, max_length, num_candidates)
    
    # 2. 각 후보에 대한 기대 유틸리티 계산
    expected_utilities = []
    for candidate in candidates:
        utility = compute_expected_utility(candidate, candidates, model, utility_function)
        expected_utilities.append(utility)
    
    # 3. 최고 기대 유틸리티를 가진 후보 선택
    best_candidate_idx = torch.argmax(torch.tensor(expected_utilities))
    best_candidate = candidates[best_candidate_idx]
    
    return best_candidate

def compute_expected_utility(candidate, all_candidates, model, utility_function):
    """후보의 기대 유틸리티 계산"""
    total_utility = 0.0
    
    for other_candidate in all_candidates:
        # 모델이 other_candidate를 생성할 확률
        probability = compute_generation_probability(model, other_candidate)
        
        # candidate와 other_candidate 간의 유틸리티
        utility = utility_function(candidate, other_candidate)
        
        total_utility += probability * utility
    
    return total_utility

def default_utility_function(candidate1, candidate2):
    """기본 유틸리티 함수 (BLEU 스코어 기반)"""
    # 간단한 BLEU 스코어 계산
    return compute_bleu_score(candidate1, candidate2)

def compute_generation_probability(model, sequence):
    """시퀀스 생성 확률 계산"""
    logits = model(sequence[:-1]).logits
    probs = torch.softmax(logits, dim=-1)
    
    # 각 토큰의 확률을 곱하여 전체 시퀀스 확률 계산
    sequence_prob = 1.0
    for i, token in enumerate(sequence[1:]):
        token_prob = probs[i, token].item()
        sequence_prob *= token_prob
    
    return sequence_prob
```

### Look-back Decoding

**개념**
- Kullback-Leibler divergence를 활용하여 현재와 과거 디코딩 단계 간 분포 거리 추적
- 반복적인 구문과 주제 편향 자동 예측 및 제거

**특징**
- ✅ 반복성 감소
- ✅ 주제 편향 방지
- ✅ 자동 품질 개선
- ❌ 계산 오버헤드
- ❌ 메모리 사용량

**논문**
- "Look-back Decoding: Open-ended Text Generation" (2022)

**구현 예시**
```python
def look_back_decoding(model, input_ids, max_length, lookback_window=10, divergence_threshold=0.1):
    generated_tokens = []
    current_ids = input_ids.clone()
    
    for step in range(max_length):
        # 1. 현재 단계의 토큰 분포 계산
        outputs = model(current_ids)
        current_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        
        # 2. 과거 단계들과의 분포 거리 계산
        if len(generated_tokens) >= lookback_window:
            divergence_scores = []
            for i in range(lookback_window):
                past_step = len(generated_tokens) - lookback_window + i
                past_probs = get_past_probability_distribution(past_step)
                
                # KL divergence 계산
                kl_div = compute_kl_divergence(current_probs, past_probs)
                divergence_scores.append(kl_div)
            
            # 3. 평균 divergence가 임계값을 초과하면 토큰 선택 조정
            avg_divergence = torch.mean(torch.tensor(divergence_scores))
            if avg_divergence < divergence_threshold:
                # 다양성을 높이기 위해 temperature 조정
                adjusted_probs = adjust_probabilities_for_diversity(current_probs)
            else:
                adjusted_probs = current_probs
        else:
            adjusted_probs = current_probs
        
        # 4. 토큰 선택
        next_token = torch.multinomial(adjusted_probs, num_samples=1)
        generated_tokens.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=-1)
    
    return current_ids

def compute_kl_divergence(p, q):
    """KL divergence 계산"""
    # 0으로 나누기 방지
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    
    kl_div = torch.sum(p_safe * torch.log(p_safe / q_safe))
    return kl_div

def adjust_probabilities_for_diversity(probs, diversity_factor=1.5):
    """다양성을 높이기 위한 확률 조정"""
    # 엔트로피 기반 다양성 조정
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    # 높은 엔트로피를 가진 토큰들의 확률을 증가
    adjusted_probs = probs * (1 + diversity_factor * entropy)
    
    # 정규화
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    return adjusted_probs
```

### Reflection-Window Decoding

**개념**
- 언어 모델이 계속하기 전에 최근 텍스트를 일시 정지하고 수정할 수 있는 능력 제공
- 선택적 개선을 통한 자기 반성적 텍스트 생성

**특징**
- ✅ 자기 반성적 생성
- ✅ 품질 자동 개선
- ✅ 일관성 향상
- ❌ 생성 시간 증가
- ❌ 복잡한 제어 로직

**구현 예시**
```python
def reflection_window_decoding(model, input_ids, max_length, reflection_interval=20, reflection_threshold=0.3):
    generated_tokens = []
    current_ids = input_ids.clone()
    
    for step in range(max_length):
        # 1. 토큰 생성
        outputs = model(current_ids)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        # 2. 반성 윈도우 도달 시 품질 평가 및 수정
        if step > 0 and step % reflection_interval == 0:
            # 최근 텍스트 품질 평가
            recent_text = current_ids[-reflection_interval:]
            quality_score = evaluate_text_quality(recent_text)
            
            # 품질이 임계값 이하인 경우 수정
            if quality_score < reflection_threshold:
                current_ids = improve_text_quality(model, current_ids, reflection_interval)
    
    return current_ids

def evaluate_text_quality(text_tokens):
    """텍스트 품질 평가"""
    # 간단한 품질 메트릭 (반복성, 일관성 등)
    repetition_score = compute_repetition_score(text_tokens)
    coherence_score = compute_coherence_score(text_tokens)
    
    # 종합 품질 점수
    quality_score = (repetition_score + coherence_score) / 2
    return quality_score

def improve_text_quality(model, current_ids, window_size):
    """텍스트 품질 개선"""
    # 최근 윈도우 제거
    base_ids = current_ids[:-window_size]
    
    # 개선된 텍스트 생성
    improved_tokens = []
    for _ in range(window_size):
        outputs = model(torch.cat([base_ids, torch.tensor(improved_tokens)], dim=-1))
        
        # 품질을 고려한 토큰 선택
        logits = outputs.logits[:, -1, :]
        
        # 반복성 방지를 위한 페널티 적용
        if improved_tokens:
            last_token = improved_tokens[-1]
            logits[0, last_token] -= 2.0  # 반복 토큰 페널티
        
        next_token = torch.argmax(logits, dim=-1)
        improved_tokens.append(next_token.item())
    
    # 개선된 텍스트로 교체
    improved_ids = torch.cat([base_ids, torch.tensor(improved_tokens)], dim=-1)
    return improved_ids
```

## 5.2 제어된 생성

### DExperts

**개념**
- "전문가" LM과 "안티 전문가" LM을 사전 훈련된 언어 모델과 결합
- 전문가 곱 방식으로 디코딩 시점 제어 생성

**특징**
- ✅ 특정 도메인 전문성 향상
- ✅ 원치 않는 행동 제거
- ✅ 유연한 제어
- ❌ 여러 모델 필요
- ❌ 훈련 비용

**구현 예시**
```python
class DExperts:
    def __init__(self, base_model, expert_model, anti_expert_model, alpha=0.5):
        self.base_model = base_model
        self.expert_model = expert_model
        self.anti_expert_model = anti_expert_model
        self.alpha = alpha  # 전문가 가중치
    
    def generate(self, input_ids, max_length):
        generated_tokens = []
        current_ids = input_ids.clone()
        
        for _ in range(max_length):
            # 1. 각 모델의 로짓 계산
            base_logits = self.base_model(current_ids).logits[:, -1, :]
            expert_logits = self.expert_model(current_ids).logits[:, -1, :]
            anti_expert_logits = self.anti_expert_model(current_ids).logits[:, -1, :]
            
            # 2. DExperts 로짓 계산
            dexperts_logits = (
                base_logits + 
                self.alpha * expert_logits - 
                (1 - self.alpha) * anti_expert_logits
            )
            
            # 3. 토큰 선택
            next_token = torch.argmax(dexperts_logits, dim=-1)
            generated_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        return current_ids
    
    def set_expert_weight(self, alpha):
        """전문가 가중치 조정"""
        self.alpha = alpha

# 사용 예시
dexperts = DExperts(
    base_model=base_lm,
    expert_model=expert_lm,
    anti_expert_model=anti_expert_lm,
    alpha=0.7  # 전문가 모델에 더 높은 가중치
)

# 전문가 모델이 강조된 생성
dexperts.set_expert_weight(0.8)
expert_output = dexperts.generate(input_ids, max_length=100)

# 안티 전문가 모델이 강조된 생성
dexperts.set_expert_weight(0.3)
anti_expert_output = dexperts.generate(input_ids, max_length=100)
```

### Confidence-based Decoding

**개념**
- 활성화 기반 confidence 보정 및 안내된 디코딩
- 높은 confidence로 진실한 답변 추출

**특징**
- ✅ 진실성 향상
- ✅ confidence 기반 제어
- ✅ 안전한 생성
- ❌ confidence 계산 오버헤드
- ❌ 보정 모델 필요

**구현 예시**
```python
def confidence_based_decoding(model, input_ids, max_length, confidence_threshold=0.8):
    generated_tokens = []
    current_ids = input_ids.clone()
    
    for _ in range(max_length):
        # 1. 모델 출력 및 confidence 계산
        outputs = model(current_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # 2. Confidence 계산
        confidence = compute_confidence(probs)
        
        # 3. Confidence가 임계값 이하인 경우 fallback 전략
        if confidence < confidence_threshold:
            # 더 보수적인 토큰 선택
            next_token = select_conservative_token(probs, confidence)
        else:
            # 일반적인 토큰 선택
            next_token = torch.argmax(logits, dim=-1)
        
        generated_tokens.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=-1)
    
    return current_ids

def compute_confidence(probs):
    """확률 분포의 confidence 계산"""
    # 여러 confidence 메트릭 조합
    max_prob = torch.max(probs)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    # 높은 최대 확률과 낮은 엔트로피 = 높은 confidence
    confidence = max_prob * (1 - entropy / 10)  # 정규화
    return confidence.item()

def select_conservative_token(probs, confidence):
    """낮은 confidence에서 보수적인 토큰 선택"""
    # 상위 k개 토큰에서만 선택
    k = max(3, int(10 * confidence))  # confidence에 따라 k 조정
    
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # 보수적인 선택 (높은 확률 토큰 선호)
    conservative_probs = top_k_probs ** 2  # 확률 제곱으로 보수성 강화
    conservative_probs = conservative_probs / conservative_probs.sum()
    
    selected_idx = torch.multinomial(conservative_probs, num_samples=1)
    return top_k_indices[0, selected_idx]
```

## 5.3 품질 평가 및 모니터링

### 품질 메트릭

**텍스트 품질 지표**
- **BLEU Score**: 기계 번역 품질
- **ROUGE Score**: 요약 품질
- **Perplexity**: 언어 모델 성능
- **Repetition Rate**: 반복성 측정
- **Coherence Score**: 일관성 측정

**구현 예시**
```python
class QualityMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_bleu_score(self, candidate, reference):
        """BLEU 스코어 계산"""
        from nltk.translate.bleu_score import sentence_bleu
        
        return sentence_bleu([reference], candidate)
    
    def compute_repetition_rate(self, text_tokens, window_size=10):
        """반복률 계산"""
        if len(text_tokens) < window_size:
            return 0.0
        
        repetitions = 0
        total_windows = len(text_tokens) - window_size + 1
        
        for i in range(total_windows):
            window = text_tokens[i:i+window_size]
            # 다른 위치에서 동일한 윈도우가 있는지 확인
            for j in range(i+1, total_windows):
                other_window = text_tokens[j:j+window_size]
                if window == other_window:
                    repetitions += 1
        
        return repetitions / total_windows
    
    def compute_coherence_score(self, text_tokens, model):
        """일관성 점수 계산"""
        # 연속된 토큰들 간의 확률 계산
        if len(text_tokens) < 2:
            return 1.0
        
        total_prob = 0.0
        for i in range(1, len(text_tokens)):
            context = text_tokens[:i]
            target = text_tokens[i]
            
            outputs = model(torch.tensor([context]))
            probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            token_prob = probs[0, target].item()
            
            total_prob += token_prob
        
        return total_prob / (len(text_tokens) - 1)
    
    def evaluate_text_quality(self, text_tokens, reference_tokens=None, model=None):
        """종합 품질 평가"""
        quality_scores = {}
        
        # 반복률
        quality_scores['repetition_rate'] = self.compute_repetition_rate(text_tokens)
        
        # 일관성 (모델이 제공된 경우)
        if model is not None:
            quality_scores['coherence'] = self.compute_coherence_score(text_tokens, model)
        
        # BLEU 스코어 (참조가 제공된 경우)
        if reference_tokens is not None:
            quality_scores['bleu'] = self.compute_bleu_score(text_tokens, reference_tokens)
        
        # 종합 품질 점수
        if 'coherence' in quality_scores:
            quality_scores['overall'] = (
                (1 - quality_scores['repetition_rate']) * 0.4 +
                quality_scores['coherence'] * 0.6
            )
        else:
            quality_scores['overall'] = 1 - quality_scores['repetition_rate']
        
        return quality_scores
```

## 성능 비교 및 사용 가이드

### 품질 향상 방법 비교

| 방법 | 품질 향상 | 속도 영향 | 구현 난이도 | 메모리 사용량 |
|------|-----------|-----------|-------------|---------------|
| MBR Decoding | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 중간 |
| Look-back | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 낮음 |
| Reflection-Window | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 중간 |
| DExperts | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 높음 |
| Confidence-based | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 낮음 |

### 사용 시나리오별 권장 방법

#### 고품질 요구사항
- **MBR Decoding**: 최고 품질, 계산 리소스 충분
- **DExperts**: 특정 도메인 전문성 필요

#### 실시간 시스템
- **Confidence-based**: 빠른 응답, 안전성 중요
- **Look-back**: 균형잡힌 접근

#### 반복성 방지
- **Look-back**: 자동 반복성 감소
- **Reflection-Window**: 적극적 품질 개선

#### 안전성 중시
- **Confidence-based**: confidence 기반 제어
- **DExperts**: 원치 않는 행동 방지

## 구현 고려사항

### 계산 효율성
- **배치 처리**: 여러 후보를 동시에 평가
- **캐싱**: 중복 계산 방지
- **조기 종료**: 품질 임계값 도달 시 조기 완료

### 메모리 관리
- **점진적 평가**: 긴 시퀀스를 청크 단위로 평가
- **스트리밍**: 실시간 품질 모니터링
- **압축**: 중간 결과 압축 저장

### 품질 보장
- **다중 메트릭**: 단일 메트릭의 한계 극복
- **적응적 임계값**: 컨텍스트에 따른 동적 조정
- **사용자 피드백**: 인간 평가와의 연동
