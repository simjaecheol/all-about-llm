---
title: Parameters
parent: LLM이란 무엇인가?
nav_order: 6
---

# Parameters

## 개요

LLM(Large Language Model)의 파라미터는 모델의 출력을 제어하는 핵심 요소입니다. 이러한 파라미터들을 조정함으로써 생성되는 텍스트의 품질, 다양성, 일관성을 세밀하게 제어할 수 있습니다.

## 주요 파라미터 종류

### 1. Temperature (온도)

**정의:** 텍스트 생성의 무작위성을 제어하는 파라미터 (0~2 범위)

**작동 원리:**
```python
def apply_temperature(logits, temperature):
    """Temperature 적용"""
    # logits를 temperature로 나누어 확률 분포 조정
    adjusted_logits = logits / temperature
    # softmax 적용
    probabilities = softmax(adjusted_logits)
    return probabilities

# 예시
logits = [2.0, 1.0, 0.5, -1.0]
temperature_0_1 = apply_temperature(logits, 0.1)  # 매우 확실한 선택
temperature_1_0 = apply_temperature(logits, 1.0)  # 기본 확률
temperature_2_0 = apply_temperature(logits, 2.0)  # 매우 무작위적
```

**Temperature 값별 특징:**

| Temperature | 특징 | 사용 사례 |
|-------------|------|-----------|
| **0.1 이하** | 매우 확실한 선택, 일관된 출력 | 사실 기반 답변, 코드 생성 |
| **0.3-0.7** | 균형잡힌 창의성 | 일반적인 대화, 문서 작성 |
| **0.8-1.2** | 창의적이지만 논리적 | 창작, 스토리텔링 |
| **1.5 이상** | 매우 창의적, 예측 불가능 | 실험적 창작, 아이디어 생성 |

**실제 예시:**
```python
# 같은 프롬프트, 다른 temperature
prompt = "인공지능의 미래는"

# Temperature 0.1: 일관된 답변
response_low = "인공지능의 미래는 매우 밝습니다. 기술 발전으로..."
# Temperature 1.0: 균형잡힌 답변  
response_medium = "인공지능의 미래는 복잡합니다. 긍정적 측면과..."
# Temperature 2.0: 창의적인 답변
response_high = "인공지능의 미래는 마치 우주를 탐험하는 것과 같습니다..."
```

### 2. Top-p (Nucleus Sampling)

**정의:** 누적 확률이 p가 될 때까지의 토큰들만 고려하는 샘플링 방법

**작동 원리:**
```python
def nucleus_sampling(probabilities, p=0.9):
    """Top-p (Nucleus) 샘플링"""
    # 확률을 내림차순으로 정렬
    sorted_probs = sorted(probabilities, reverse=True)
    cumulative_probs = []
    cumulative = 0
    
    for prob in sorted_probs:
        cumulative += prob
        cumulative_probs.append(cumulative)
        if cumulative >= p:
            break
    
    # p 이하의 확률을 가진 토큰들만 선택
    valid_tokens = [i for i, prob in enumerate(probabilities) 
                   if prob >= sorted_probs[len(cumulative_probs)-1]]
    
    return valid_tokens

# 예시
probabilities = [0.4, 0.3, 0.2, 0.1]
top_p_tokens = nucleus_sampling(probabilities, p=0.8)
# 결과: [0, 1] (누적 확률 0.7이 0.8 이하)
```

**Top-p 값별 특징:**

| Top-p | 특징 | 사용 사례 |
|-------|------|-----------|
| **0.1-0.3** | 매우 집중된 선택 | 정확한 정보 제공 |
| **0.5-0.8** | 균형잡힌 다양성 | 일반적인 대화 |
| **0.9-1.0** | 높은 다양성 | 창의적 글쓰기 |

### 3. Top-k

**정의:** 확률이 높은 상위 k개의 토큰만 고려하는 샘플링 방법

**작동 원리:**
```python
def top_k_sampling(probabilities, k=50):
    """Top-k 샘플링"""
    # 상위 k개 토큰의 인덱스 찾기
    top_k_indices = np.argsort(probabilities)[-k:]
    
    # 상위 k개 토큰의 확률만 유지
    filtered_probs = np.zeros_like(probabilities)
    filtered_probs[top_k_indices] = probabilities[top_k_indices]
    
    # 정규화
    filtered_probs = filtered_probs / np.sum(filtered_probs)
    
    return filtered_probs

# 예시
probabilities = [0.4, 0.3, 0.2, 0.1]
top_k_probs = top_k_sampling(probabilities, k=2)
# 결과: [0.57, 0.43, 0, 0] (상위 2개만 유지)
```

**Top-k 값별 특징:**

| Top-k | 특징 | 사용 사례 |
|-------|------|-----------|
| **1-10** | 매우 보수적 | 사실 확인, 정확한 답변 |
| **20-50** | 균형잡힌 | 일반적인 대화 |
| **100+** | 창의적 | 창작 활동 |

### 4. Frequency Penalty

**정의:** 이미 사용된 토큰의 확률을 감소시키는 파라미터

**작동 원리:**
```python
def apply_frequency_penalty(logits, generated_tokens, penalty=0.1):
    """Frequency Penalty 적용"""
    # 생성된 토큰들의 빈도 계산
    token_counts = {}
    for token in generated_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    # 빈도에 따른 페널티 적용
    for token_id, count in token_counts.items():
        if token_id < len(logits):
            logits[token_id] -= penalty * count
    
    return logits

# 예시
logits = [2.0, 1.0, 0.5, -1.0]
generated_tokens = [0, 0, 1]  # 토큰 0이 2번, 토큰 1이 1번 사용됨
penalized_logits = apply_frequency_penalty(logits, generated_tokens, penalty=0.1)
# 결과: [1.8, 0.9, 0.5, -1.0] (토큰 0은 -0.2, 토큰 1은 -0.1 페널티)
```

**Frequency Penalty 값별 특징:**

| 값 | 특징 | 사용 사례 |
|----|------|-----------|
| **0.0** | 반복 허용 | 단순한 반복이 필요한 경우 |
| **0.1-0.5** | 적당한 반복 방지 | 일반적인 텍스트 생성 |
| **0.5-1.0** | 강한 반복 방지 | 창의적 글쓰기 |
| **1.0+** | 매우 강한 반복 방지 | 실험적 창작 |

### 5. Presence Penalty

**정의:** 특정 토큰이 한 번이라도 사용되었으면 그 토큰의 확률을 감소시키는 파라미터

**작동 원리:**
```python
def apply_presence_penalty(logits, generated_tokens, penalty=0.1):
    """Presence Penalty 적용"""
    # 사용된 토큰들의 집합
    used_tokens = set(generated_tokens)
    
    # 사용된 토큰들에 페널티 적용
    for token_id in used_tokens:
        if token_id < len(logits):
            logits[token_id] -= penalty
    
    return logits

# 예시
logits = [2.0, 1.0, 0.5, -1.0]
generated_tokens = [0, 2]  # 토큰 0과 2가 사용됨
penalized_logits = apply_presence_penalty(logits, generated_tokens, penalty=0.1)
# 결과: [1.9, 1.0, 0.4, -1.0] (토큰 0과 2에 -0.1 페널티)
```

**Presence Penalty 값별 특징:**

| 값 | 특징 | 사용 사례 |
|----|------|-----------|
| **0.0** | 토큰 재사용 허용 | 반복이 필요한 경우 |
| **0.1-0.3** | 적당한 토큰 다양성 | 일반적인 텍스트 |
| **0.3-0.7** | 높은 토큰 다양성 | 창의적 글쓰기 |
| **0.7+** | 매우 높은 다양성 | 실험적 창작 |

### 6. Repetition Penalty

**정의:** 반복 패턴을 방지하기 위해 연속적으로 사용된 토큰에 페널티를 적용하는 파라미터

**참고:** OpenAI는 공식적으로 Repetition Penalty 파라미터를 제공하지 않습니다. 대신 Frequency Penalty와 Presence Penalty를 조합하여 사용합니다. Repetition Penalty는 주로 오픈소스 모델(Hugging Face 등)에서 사용됩니다.

**작동 원리:**
```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.1):
    """Repetition Penalty 적용"""
    # 최근 N개 토큰에서 반복 패턴 찾기
    recent_tokens = generated_tokens[-10:]  # 최근 10개 토큰
    
    # 반복 패턴 감지
    repetition_patterns = find_repetition_patterns(recent_tokens)
    
    # 반복된 토큰들에 페널티 적용
    for token_id in repetition_patterns:
        if token_id < len(logits):
            # penalty > 1.0이면 확률 감소, < 1.0이면 확률 증가
            logits[token_id] /= penalty
    
    return logits

def find_repetition_patterns(tokens, min_length=2):
    """반복 패턴 찾기"""
    patterns = set()
    
    for length in range(min_length, len(tokens)//2 + 1):
        for i in range(len(tokens) - length + 1):
            pattern = tuple(tokens[i:i+length])
            
            # 패턴이 반복되는지 확인
            if tokens.count(pattern) > 1:
                patterns.update(pattern)
    
    return patterns

# 예시
logits = [2.0, 1.0, 0.5, -1.0]
generated_tokens = [0, 1, 0, 1, 0, 1]  # 0,1 패턴 반복
penalized_logits = apply_repetition_penalty(logits, generated_tokens, penalty=1.2)
# 결과: [1.67, 0.83, 0.5, -1.0] (토큰 0과 1에 1.2로 나누는 페널티)
```

**Repetition Penalty 값별 특징:**

| 값 | 특징 | 사용 사례 |
|----|------|-----------|
| **1.0** | 페널티 없음 | 반복 허용 |
| **1.1-1.3** | 약한 반복 방지 | 일반적인 텍스트 |
| **1.3-1.5** | 강한 반복 방지 | 창의적 글쓰기 |
| **1.5+** | 매우 강한 반복 방지 | 실험적 창작 |

**N-gram 반복 방지:**

N-gram은 연속된 N개의 토큰을 의미하며, `no_repeat_ngram_size` 파라미터로 특정 n-gram이 반복되는 것을 방지합니다.

**작동 원리:**
```python
def apply_ngram_penalty(logits, generated_tokens, ngram_size=3):
    """N-gram 반복 방지"""
    # 최근 생성된 토큰들
    recent_tokens = generated_tokens[-ngram_size+1:]
    
    # 금지할 n-gram 패턴들 찾기
    forbidden_ngrams = set()
    
    # 전체 텍스트에서 n-gram 패턴 찾기
    for i in range(len(generated_tokens) - ngram_size + 1):
        ngram = tuple(generated_tokens[i:i+ngram_size])
        
        # 이 n-gram이 이미 사용되었는지 확인
        if ngram in forbidden_ngrams:
            # 마지막 토큰을 금지 목록에 추가
            forbidden_ngrams.add(tuple(generated_tokens[i+1:i+ngram_size]))
        else:
            forbidden_ngrams.add(ngram)
    
    # 금지된 토큰들의 확률을 0으로 설정
    for forbidden_token in forbidden_ngrams:
        if len(forbidden_token) == 1:  # 단일 토큰
            token_id = forbidden_token[0]
            if token_id < len(logits):
                logits[token_id] = float('-inf')  # 확률 0으로 설정
    
    return logits

# 예시
logits = [2.0, 1.0, 0.5, -1.0]
generated_tokens = [0, 1, 2, 0, 1, 2]  # [0,1,2] 패턴 반복
penalized_logits = apply_ngram_penalty(logits, generated_tokens, ngram_size=3)
# 결과: [2.0, 1.0, 0.5, -inf] (토큰 2가 금지됨)
```

**N-gram 크기별 특징:**

| N-gram 크기 | 특징 | 사용 사례 |
|-------------|------|-----------|
| **1** | 단일 토큰 반복 방지 | 기본적인 반복 방지 |
| **2** | 2-gram 반복 방지 | 단어 수준 반복 방지 |
| **3** | 3-gram 반복 방지 | 구문 수준 반복 방지 |
| **4+** | 긴 구문 반복 방지 | 문장 수준 반복 방지 |

**OpenAI vs 오픈소스 모델 비교:**

```python
# OpenAI API (N-gram 반복 방지 없음)
openai_params = {
    "temperature": 0.7,
    "top_p": 0.8,
    "frequency_penalty": 0.1,  # 반복 방지용
    "presence_penalty": 0.1    # 반복 방지용
}

# 오픈소스 모델 (Hugging Face 등)
huggingface_params = {
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.1,  # 직접적인 반복 방지
    "no_repeat_ngram_size": 3   # 3-gram 반복 방지
}
```

**실제 사용 예시:**
```python
# OpenAI API 사용 시 (Repetition Penalty 대체)
def openai_repetition_control(prompt, max_tokens=100):
    """OpenAI API에서 반복 방지"""
    params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "frequency_penalty": 0.3,  # 반복 사용된 토큰에 페널티
        "presence_penalty": 0.2,   # 사용된 토큰에 페널티
        "max_tokens": max_tokens
    }
    return generate_with_openai(prompt, **params)

# Hugging Face 사용 시
def huggingface_repetition_control(prompt, max_tokens=100):
    """Hugging Face에서 반복 방지"""
    params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.2,  # 직접적인 반복 페널티
        "no_repeat_ngram_size": 3,  # 3-gram 반복 방지
        "max_new_tokens": max_tokens
    }
    return generate_with_huggingface(prompt, **params)

# N-gram 크기별 사용 예시
def generate_with_ngram_control(prompt, ngram_size=3):
    """N-gram 크기에 따른 반복 방지"""
    params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": ngram_size,  # n-gram 크기 조정
        "max_new_tokens": 200
    }
    
    # n-gram 크기별 특징
    ngram_descriptions = {
        1: "단일 토큰 반복 방지",
        2: "단어 수준 반복 방지", 
        3: "구문 수준 반복 방지",
        4: "문장 수준 반복 방지"
    }
    
    print(f"N-gram 크기 {ngram_size}: {ngram_descriptions[ngram_size]}")
    return generate_with_huggingface(prompt, **params)
```

### 7. Max Tokens

**정의:** 생성할 수 있는 최대 토큰 수

**작동 원리:**
```python
def limit_max_tokens(generated_tokens, max_tokens=100):
    """Max Tokens 제한"""
    if len(generated_tokens) >= max_tokens:
        return generated_tokens[:max_tokens]
    return generated_tokens

# 예시
generated_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
limited_tokens = limit_max_tokens(generated_tokens, max_tokens=5)
# 결과: [1, 2, 3, 4, 5]
```

**Max Tokens 값별 특징:**

| 값 | 특징 | 사용 사례 |
|----|------|-----------|
| **10-50** | 짧은 답변 | 간단한 질문, 요약 |
| **100-300** | 중간 길이 | 일반적인 대화, 설명 |
| **500-1000** | 긴 답변 | 상세한 설명, 에세이 |
| **1000+** | 매우 긴 답변 | 긴 문서, 창작물 |

## 파라미터 조합 전략

### 1. 정확한 정보 제공
```python
# 정확한 정보를 위한 파라미터 설정
accurate_params = {
    "temperature": 0.1,      # 낮은 무작위성
    "top_p": 0.3,           # 집중된 선택
    "top_k": 10,            # 상위 10개만 고려
    "frequency_penalty": 0.0, # 반복 허용
    "presence_penalty": 0.0,  # 토큰 재사용 허용
    "repetition_penalty": 1.0, # 반복 허용 (오픈소스 모델)
    "max_tokens": 200        # 적당한 길이
}
```

### 2. 창의적 글쓰기
```python
# 창의적 글쓰기를 위한 파라미터 설정
creative_params = {
    "temperature": 0.8,      # 높은 창의성
    "top_p": 0.9,           # 다양한 선택
    "top_k": 100,           # 많은 옵션 고려
    "frequency_penalty": 0.5, # 반복 방지
    "presence_penalty": 0.3,  # 토큰 다양성
    "repetition_penalty": 1.3, # 강한 반복 방지 (오픈소스 모델)
    "max_tokens": 500        # 충분한 길이
}
```

### 3. 대화형 챗봇
```python
# 대화형 챗봇을 위한 파라미터 설정
chatbot_params = {
    "temperature": 0.7,      # 자연스러운 대화
    "top_p": 0.8,           # 균형잡힌 다양성
    "top_k": 50,            # 적당한 옵션
    "frequency_penalty": 0.2, # 약간의 반복 방지
    "presence_penalty": 0.1,  # 자연스러운 다양성
    "repetition_penalty": 1.1, # 약한 반복 방지 (오픈소스 모델)
    "max_tokens": 150        # 대화에 적합한 길이
}
```

## 실무 적용 예시

### 1. 코드 생성
```python
def generate_code_prompt(prompt, language="python"):
    """코드 생성을 위한 파라미터 설정"""
    params = {
        "temperature": 0.1,      # 정확한 코드
        "top_p": 0.2,           # 확실한 선택
        "top_k": 20,            # 제한된 옵션
        "frequency_penalty": 0.0, # 반복 허용 (변수명 등)
        "presence_penalty": 0.0,  # 토큰 재사용 허용
        "repetition_penalty": 1.0, # 반복 허용 (오픈소스 모델)
        "max_tokens": 300        # 충분한 코드 길이
    }
    
    return generate_text(prompt, **params)

# 예시
code_prompt = "Python으로 피보나치 수열을 계산하는 함수를 작성해주세요."
code = generate_code_prompt(code_prompt)
```

### 2. 창작 글쓰기
```python
def generate_creative_text(prompt, style="story"):
    """창작 글쓰기를 위한 파라미터 설정"""
    params = {
        "temperature": 0.9,      # 높은 창의성
        "top_p": 0.95,          # 매우 다양한 선택
        "top_k": 150,           # 많은 옵션
        "frequency_penalty": 0.7, # 강한 반복 방지
        "presence_penalty": 0.5,  # 높은 토큰 다양성
        "repetition_penalty": 1.4, # 강한 반복 방지 (오픈소스 모델)
        "max_tokens": 800        # 긴 창작물
    }
    
    return generate_text(prompt, **params)

# 예시
story_prompt = "미래 도시에서 일어나는 로봇과 인간의 우정에 대한 이야기를 써주세요."
story = generate_creative_text(story_prompt)
```

### 3. 요약 생성
```python
def generate_summary(text, max_length=100):
    """요약 생성을 위한 파라미터 설정"""
    params = {
        "temperature": 0.3,      # 일관된 요약
        "top_p": 0.5,           # 핵심 정보 집중
        "top_k": 30,            # 제한된 옵션
        "frequency_penalty": 0.1, # 약간의 반복 방지
        "presence_penalty": 0.0,  # 중요 단어 재사용 허용
        "repetition_penalty": 1.1, # 약한 반복 방지 (오픈소스 모델)
        "max_tokens": max_length  # 요약 길이 제한
    }
    
    prompt = f"다음 텍스트를 {max_length}자 이내로 요약해주세요:\n\n{text}"
    return generate_text(prompt, **params)
```

## 파라미터 튜닝 가이드

### 1. 단계별 튜닝 과정
```python
def tune_parameters(base_prompt, target_style):
    """파라미터 튜닝 과정"""
    
    # 1단계: 기본 설정으로 시작
    base_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 50,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 200
    }
    
    # 2단계: 목적에 따라 조정
    if target_style == "accurate":
        base_params["temperature"] = 0.1
        base_params["top_p"] = 0.3
        base_params["top_k"] = 10
    
    elif target_style == "creative":
        base_params["temperature"] = 0.9
        base_params["top_p"] = 0.95
        base_params["frequency_penalty"] = 0.5
    
    # 3단계: 결과 평가 및 미세 조정
    return base_params
```

### 2. A/B 테스트
```python
def ab_test_parameters(prompt, param_sets):
    """파라미터 A/B 테스트"""
    results = []
    
    for i, params in enumerate(param_sets):
        result = generate_text(prompt, **params)
        results.append({
            "param_set": i,
            "params": params,
            "result": result,
            "quality_score": evaluate_quality(result)
        })
    
    # 최적 파라미터 선택
    best_result = max(results, key=lambda x: x["quality_score"])
    return best_result
```

## 결론

LLM 파라미터는 모델의 출력을 세밀하게 제어하는 강력한 도구입니다. 각 파라미터의 특성을 이해하고 목적에 맞게 조합하여 사용하면 원하는 품질의 텍스트를 생성할 수 있습니다.

### 핵심 포인트

1. **Temperature**: 무작위성과 창의성 제어
2. **Top-p/Top-k**: 선택 가능한 토큰 범위 제어
3. **Frequency/Presence Penalty**: 반복과 다양성 제어 (OpenAI)
4. **Repetition Penalty**: 직접적인 반복 패턴 방지 (오픈소스 모델)
5. **N-gram 반복 방지**: 특정 길이의 토큰 시퀀스 반복 방지
6. **Max Tokens**: 출력 길이 제어
7. **파라미터 조합**: 목적에 맞는 최적 설정 필요

이러한 이해를 바탕으로 LLM을 효과적으로 활용하여 원하는 결과를 얻을 수 있습니다.
