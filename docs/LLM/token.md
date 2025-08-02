---
title: Token
parent: LLM이란 무엇인가?
nav_order: 3
---

# Token

## 개요

토큰(Token)은 LLM이 텍스트를 처리하는 가장 기본적인 단위입니다. 인간이 단어나 문장 단위로 텍스트를 이해하는 것과 달리, LLM은 토큰이라는 더 작은 단위로 텍스트를 분해하여 처리합니다. 토크나이저(Tokenizer)는 이러한 텍스트를 토큰으로 변환하는 핵심 도구입니다.

## 토큰이란?

### 1. 토큰의 정의

**토큰(Token)**은 LLM이 이해할 수 있는 텍스트의 최소 단위입니다.

**예시:**
```
텍스트: "안녕하세요, 반갑습니다!"
토큰: ["안녕", "하세요", ",", "반갑", "습니다", "!"]
```

### 2. 토큰의 특징

- **고정된 어휘집**: 모델이 학습할 때 사용한 토큰들의 집합
- **숫자 인덱스**: 각 토큰은 고유한 숫자로 매핑
- **벡터 변환**: 토큰은 임베딩을 통해 벡터로 변환

### 3. 토큰화의 필요성

**왜 토큰화가 필요한가?**

1. **다양한 언어 지원**: 영어, 한국어, 중국어 등 모든 언어를 통일된 방식으로 처리
2. **알 수 없는 단어 처리**: 학습 시 보지 못한 새로운 단어도 처리 가능
3. **효율적인 처리**: 텍스트를 일정한 크기의 단위로 나누어 처리

## 토크나이저의 종류

### 1. 단어 기반 토크나이저 (Word-based Tokenizer)

**특징:**
- 공백을 기준으로 단어를 분리
- 가장 직관적이고 이해하기 쉬움
- 어휘집 크기가 매우 클 수 있음

**예시:**
```python
# 영어 예시
text = "I love artificial intelligence"
tokens = ["I", "love", "artificial", "intelligence"]

# 한국어 예시 (공백 기준)
text = "나는 인공지능을 좋아합니다"
tokens = ["나는", "인공지능을", "좋아합니다"]
```

**장단점:**
- **장점**: 직관적이고 이해하기 쉬움
- **단점**: 어휘집이 매우 크고, 새로운 단어 처리 어려움

### 2. 문자 기반 토크나이저 (Character-based Tokenizer)

**특징:**
- 각 문자를 개별 토큰으로 처리
- 어휘집 크기가 매우 작음
- 모든 텍스트를 처리할 수 있음

**예시:**
```python
text = "Hello"
tokens = ["H", "e", "l", "l", "o"]

text = "안녕하세요"
tokens = ["안", "녕", "하", "세", "요"]
```

**장단점:**
- **장점**: 어휘집이 작고, 새로운 텍스트 처리 가능
- **단점**: 토큰 수가 많아지고, 의미 정보 손실

### 3. 서브워드 토크나이저 (Subword Tokenizer)

**특징:**
- 단어를 더 작은 단위로 분해
- 자주 사용되는 패턴을 학습
- 알 수 없는 단어도 처리 가능

**대표적인 서브워드 토크나이저:**

#### BPE (Byte Pair Encoding)
```python
# BPE 예시
text = "artificial intelligence"
# 학습 과정에서 자주 나타나는 패턴을 찾아 토큰화
tokens = ["art", "ificial", "intel", "ligence"]
```

#### WordPiece
```python
# WordPiece 예시 (BERT에서 사용)
text = "artificial intelligence"
tokens = ["art", "##ificial", "intel", "##ligence"]
# ##은 서브워드임을 나타내는 표시
```

#### SentencePiece
```python
# SentencePiece 예시 (다국어 지원)
text = "안녕하세요 Hello"
tokens = ["▁안녕", "하세요", "▁Hello"]
# ▁은 단어 시작을 나타내는 표시
```

## 주요 토크나이저 비교

### 1. GPT 계열 (BPE)
**특징:**
- 영어 중심으로 설계
- 웹 크롤링 데이터로 학습
- 대소문자 구분

**예시:**
```python
# GPT 토크나이저
text = "Hello, world!"
tokens = ["Hello", ",", "Ġworld", "!"]
# Ġ는 공백 다음에 오는 토큰을 나타냄
```

### 2. BERT 계열 (WordPiece)
**특징:**
- 영어 중심이지만 다국어 지원
- 대소문자 구분
- 서브워드 표시로 ## 사용

**예시:**
```python
# BERT 토크나이저
text = "artificial intelligence"
tokens = ["art", "##ificial", "intel", "##ligence"]
```

### 3. T5 계열 (SentencePiece)
**특징:**
- 다국어 지원
- 언어 구분 없이 통합 처리
- 단어 시작 표시로 ▁ 사용

**예시:**
```python
# T5 토크나이저
text = "안녕하세요 Hello"
tokens = ["▁안녕", "하세요", "▁Hello"]
```

## 토크나이저의 작동 원리

### 1. 학습 과정

**BPE 학습 예시:**
```python
# 1단계: 초기 어휘집 생성
vocab = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}

# 2단계: 텍스트를 문자 단위로 분리
text = "artificial intelligence"
chars = ["a", "r", "t", "i", "f", "i", "c", "i", "a", "l", " ", "i", "n", "t", "e", "l", "l", "i", "g", "e", "n", "c", "e"]

# 3단계: 가장 자주 나타나는 쌍을 찾아 병합
# "ar" + "t" = "art"
# "intel" + "ligence" = "intelligence"

# 4단계: 최종 토큰
tokens = ["art", "ificial", "intelligence"]
```

### 2. 토큰화 과정

```python
def tokenize_text(text, tokenizer):
    # 1. 텍스트 전처리
    text = preprocess(text)
    
    # 2. 토큰화
    tokens = tokenizer.encode(text)
    
    # 3. 토큰 ID로 변환
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return token_ids

# 예시
text = "안녕하세요"
token_ids = [101, 102, 103, 104, 105]  # 실제 ID는 다를 수 있음
```

### 3. 역토큰화 과정

```python
def detokenize_text(token_ids, tokenizer):
    # 1. ID를 토큰으로 변환
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # 2. 토큰을 텍스트로 결합
    text = tokenizer.convert_tokens_to_string(tokens)
    
    return text

# 예시
token_ids = [101, 102, 103, 104, 105]
text = "안녕하세요"
```

## 토큰화의 실제 영향

### 1. 토큰 수와 비용

**토큰 수 계산:**
```python
def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)

# 예시
text = "안녕하세요, 반갑습니다!"
token_count = count_tokens(text, tokenizer)  # 예: 8개 토큰
```

**비용 계산:**
```python
def calculate_cost(token_count, price_per_1k_tokens):
    cost = (token_count / 1000) * price_per_1k_tokens
    return cost

# 예시
token_count = 1000
price_per_1k = 0.002  # $0.002 per 1K tokens
cost = calculate_cost(token_count, price_per_1k)  # $0.002
```

### 2. 언어별 토큰 효율성

**영어 vs 한국어 비교:**
```python
# 영어
english_text = "Hello, how are you?"
english_tokens = ["Hello", ",", "Ġhow", "Ġare", "Ġyou", "?"]  # 6개 토큰

# 한국어
korean_text = "안녕하세요, 어떻게 지내세요?"
korean_tokens = ["안녕", "하세요", ",", "어떻게", "지내세요", "?"]  # 6개 토큰

# 같은 의미지만 토큰 수가 다를 수 있음
```

### 3. 토큰 제한과 처리

**컨텍스트 길이 제한:**
```python
def check_context_length(text, tokenizer, max_tokens=4096):
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    if token_count > max_tokens:
        # 토큰 수가 제한을 초과하면 잘라내기
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
        return text, token_count
    else:
        return text, token_count

# 예시
long_text = "매우 긴 텍스트..."
truncated_text, count = check_context_length(long_text, tokenizer)
```

## 토큰 수 계산과 분석

### 1. 토큰 수 계산 방법

**기본 토큰 수 계산:**
```python
def count_tokens(text, tokenizer):
    """텍스트의 토큰 수 계산"""
    tokens = tokenizer.encode(text)
    return len(tokens)

def count_tokens_detailed(text, tokenizer):
    """상세한 토큰 정보 반환"""
    tokens = tokenizer.encode(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(tokens),
        "character_count": len(text),
        "tokens_per_character": len(tokens) / len(text) if text else 0
    }

# 예시
text = "안녕하세요, 반갑습니다!"
result = count_tokens_detailed(text, tokenizer)
print(f"토큰 수: {result['token_count']}")
print(f"문자 수: {result['character_count']}")
print(f"문자당 토큰 수: {result['tokens_per_character']:.2f}")
```

### 2. 다양한 텍스트 유형별 토큰 수 분석

```python
def analyze_token_usage_by_type():
    """텍스트 유형별 토큰 사용량 분석"""
    
    sample_texts = {
        "영어": "Hello, how are you today?",
        "한국어": "안녕하세요, 오늘 날씨가 좋네요.",
        "코드": "def hello_world(): print('Hello, World!')",
        "숫자": "1234567890",
        "특수문자": "!@#$%^&*()",
        "이모지": "안녕하세요 😊 반갑습니다 👋"
    }
    
    results = {}
    for text_type, text in sample_texts.items():
        tokens = tokenizer.encode(text)
        results[text_type] = {
            "text": text,
            "token_count": len(tokens),
            "character_count": len(text),
            "efficiency": len(text) / len(tokens) if tokens else 0
        }
    
    return results

# 분석 결과 예시
analysis = analyze_token_usage_by_type()
for text_type, data in analysis.items():
    print(f"{text_type}: {data['token_count']} 토큰 ({data['character_count']} 문자)")
```

### 3. 토큰 수 예측 모델

```python
def predict_token_count(text, language="korean"):
    """언어별 토큰 수 예측"""
    
    # 언어별 평균 토큰 비율
    token_ratios = {
        "korean": 0.8,    # 한국어: 문자당 약 0.8 토큰
        "english": 0.4,   # 영어: 문자당 약 0.4 토큰
        "chinese": 1.2,   # 중국어: 문자당 약 1.2 토큰
        "code": 0.6,      # 코드: 문자당 약 0.6 토큰
        "mixed": 0.7      # 혼합: 문자당 약 0.7 토큰
    }
    
    estimated_tokens = int(len(text) * token_ratios.get(language, 0.7))
    return estimated_tokens

def estimate_cost_by_tokens(token_count, model="gpt-3.5-turbo"):
    """토큰 수에 따른 비용 추정"""
    
    # 모델별 토큰당 비용 (USD)
    costs_per_1k_tokens = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03}
    }
    
    model_costs = costs_per_1k_tokens.get(model, {"input": 0.0015, "output": 0.002})
    
    # 입력 토큰 비용 (예상)
    input_cost = (token_count * model_costs["input"]) / 1000
    
    # 출력 토큰 비용 (예상 - 입력의 50%로 가정)
    output_cost = (token_count * 0.5 * model_costs["output"]) / 1000
    
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": token_count,
        "estimated_output_tokens": int(token_count * 0.5),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# 사용 예시
text = "인공지능에 대해 자세히 설명해주세요."
estimated_tokens = predict_token_count(text, "korean")
cost_estimate = estimate_cost_by_tokens(estimated_tokens, "gpt-3.5-turbo")

print(f"예상 토큰 수: {estimated_tokens}")
print(f"예상 비용: ${cost_estimate['total_cost']:.4f}")
```

### 4. 토큰 수 최적화 도구

```python
class TokenOptimizer:
    def __init__(self, tokenizer, target_tokens=1000):
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
    
    def optimize_text(self, text):
        """텍스트를 목표 토큰 수에 맞게 최적화"""
        current_tokens = len(self.tokenizer.encode(text))
        
        if current_tokens <= self.target_tokens:
            return text, current_tokens
        
        # 1. 불필요한 공백 제거
        optimized = re.sub(r'\s+', ' ', text).strip()
        
        # 2. 반복되는 표현 제거
        optimized = self._remove_redundant_expressions(optimized)
        
        # 3. 문장 단위로 자르기
        if len(self.tokenizer.encode(optimized)) > self.target_tokens:
            optimized = self._truncate_by_sentences(optimized)
        
        final_tokens = len(self.tokenizer.encode(optimized))
        return optimized, final_tokens
    
    def _remove_redundant_expressions(self, text):
        """반복되는 표현 제거"""
        # 간단한 예시: 연속된 동일한 단어 제거
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
            prev_word = word
        
        return ' '.join(cleaned_words)
    
    def _truncate_by_sentences(self, text):
        """문장 단위로 자르기"""
        sentences = text.split('.')
        result = ""
        
        for sentence in sentences:
            test_text = result + sentence + "."
            if len(self.tokenizer.encode(test_text)) <= self.target_tokens:
                result = test_text
            else:
                break
        
        return result.strip()

# 사용 예시
optimizer = TokenOptimizer(tokenizer, target_tokens=100)
long_text = "매우 긴 텍스트 내용..."
optimized_text, token_count = optimizer.optimize_text(long_text)
print(f"최적화된 토큰 수: {token_count}")
```

### 5. 배치 토큰 수 계산

```python
def calculate_batch_tokens(texts, tokenizer):
    """여러 텍스트의 토큰 수를 배치로 계산"""
    results = []
    total_tokens = 0
    
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        total_tokens += token_count
        
        results.append({
            "index": i,
            "text": text[:50] + "..." if len(text) > 50 else text,
            "token_count": token_count,
            "cumulative_tokens": total_tokens
        })
    
    return {
        "individual_results": results,
        "total_tokens": total_tokens,
        "average_tokens": total_tokens / len(texts) if texts else 0
    }

# 배치 처리 예시
texts = [
    "첫 번째 텍스트입니다.",
    "두 번째 텍스트입니다.",
    "세 번째 텍스트입니다."
]

batch_result = calculate_batch_tokens(texts, tokenizer)
print(f"총 토큰 수: {batch_result['total_tokens']}")
print(f"평균 토큰 수: {batch_result['average_tokens']:.1f}")
```

## 실무에서의 고려사항

### 1. 토크나이저 선택

**언어별 권장 토크나이저:**
```python
tokenizer_recommendations = {
    "영어": "GPT, BERT 토크나이저",
    "한국어": "KoBERT, KoGPT 토크나이저", 
    "다국어": "mT5, XLM-R 토크나이저",
    "코드": "CodeGPT, CodeBERT 토크나이저"
}
```

### 2. 토큰 효율성 최적화

```python
def optimize_token_usage(text, tokenizer):
    # 1. 불필요한 공백 제거
    text = text.strip()
    
    # 2. 반복되는 패턴 제거
    text = remove_redundant_patterns(text)
    
    # 3. 토큰 수 확인
    token_count = len(tokenizer.encode(text))
    
    return text, token_count

# 예시
original_text = "   안녕하세요   반갑습니다   "
optimized_text, count = optimize_token_usage(original_text, tokenizer)
```

### 3. 토큰 예산 관리

```python
class TokenBudget:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.used_tokens = 0
    
    def add_tokens(self, token_count):
        self.used_tokens += token_count
        return self.used_tokens <= self.max_tokens
    
    def get_remaining_tokens(self):
        return max(0, self.max_tokens - self.used_tokens)

# 사용 예시
budget = TokenBudget(max_tokens=4096)
text = "안녕하세요"
tokens = tokenizer.encode(text)

if budget.add_tokens(len(tokens)):
    print(f"토큰 추가 성공. 남은 토큰: {budget.get_remaining_tokens()}")
else:
    print("토큰 제한 초과!")
```

## 결론

토큰과 토크나이저는 LLM의 핵심 구성 요소로, 텍스트를 모델이 이해할 수 있는 형태로 변환하는 역할을 합니다. 적절한 토크나이저 선택과 토큰 효율성 관리는 LLM 서비스의 성능과 비용에 직접적인 영향을 미칩니다.

### 핵심 포인트

1. **토큰의 중요성**: LLM이 텍스트를 처리하는 기본 단위
2. **토크나이저의 종류**: 단어, 문자, 서브워드 기반 토크나이저
3. **언어별 특성**: 영어, 한국어 등 언어에 따른 토크나이저 차이
4. **실무 고려사항**: 토큰 수 제한, 비용 관리, 효율성 최적화

이러한 이해를 바탕으로 LLM 서비스를 구축할 때 적절한 토크나이저를 선택하고 토큰 사용을 최적화할 수 있습니다.
