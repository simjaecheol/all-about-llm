---
title: Encoder-Decoder 모델 구조
parent: LLM이란 무엇인가?
nav_order: 3
---

# Encoder-Decoder 모델 구조

## 개요

Transformer 아키텍처를 기반으로 한 LLM은 크게 세 가지 구조로 나뉩니다: **Encoder Only**, **Decoder Only**, **Encoder-Decoder**. 각 구조는 특정한 목적과 장단점을 가지고 있으며, 다양한 자연어 처리 작업에 활용됩니다.

## 1. Encoder Only 모델

### 구조와 특징

**기본 구조:**
```
Input Text → Embedding → Encoder Layers → Output Head
```

**핵심 특징:**
- **양방향 Attention**: 각 단어가 문장의 모든 단어를 참조
- **문맥 이해**: 전체 문장의 맥락을 고려한 의미 추출
- **고정 길이 출력**: 입력 길이와 관계없이 일정한 크기의 벡터 생성

### 대표 모델들

#### BERT (Bidirectional Encoder Representations from Transformers)
**특징:**
- **마스킹 방식**: MLM (Masked Language Modeling)
- **학습 방법**: 문장의 15% 단어를 가리고 예측
- **장점**: 깊은 문맥 이해, 다양한 다운스트림 작업에 적합

**실제 활용:**
```python
# BERT 예시
text = "나는 [MASK]에 갔다"
# 모델이 [MASK] 위치에 "학교"를 예측
```

#### RoBERTa (Robustly Optimized BERT)
**BERT 개선점:**
- **더 큰 배치 크기**: 256 → 8,000
- **더 긴 시퀀스**: 512 → 512+ 토큰
- **더 많은 데이터**: 16GB → 160GB
- **동적 마스킹**: 매 에포크마다 다른 패턴

#### ALBERT (A Lite BERT)
**경량화 기법:**
- **Factorized Embedding**: 임베딩 차원 축소
- **Cross-layer Parameter Sharing**: 층 간 가중치 공유
- **Inter-sentence Coherence Loss**: 문장 간 일관성 학습

### Encoder Only 모델의 장단점

**장점:**
- **깊은 문맥 이해**: 양방향으로 모든 단어 관계 학습
- **다양한 작업 적합**: 분류, 추출, 분석 등
- **안정적인 성능**: 일관된 결과 제공

**단점:**
- **생성 능력 제한**: 텍스트 생성에는 부적합
- **고정 길이 출력**: 가변 길이 생성 어려움
- **단일 작업**: 한 번에 하나의 작업만 수행

### 실제 활용 사례

```python
# 감정 분석
text = "이 영화는 정말 재미있다"
# BERT → [긍정: 0.85, 부정: 0.15]

# 개체명 인식
text = "김철수는 서울에서 태어났다"
# BERT → [PERSON: 김철수, LOCATION: 서울]

# 질문 답변
question = "서울의 수도는?"
context = "서울은 대한민국의 수도이다"
# BERT → "서울"
```

## 2. Decoder Only 모델

### 구조와 특징

**기본 구조:**
```
Input Text → Embedding → Decoder Layers → Next Token Prediction
```

**핵심 특징:**
- **단방향 Attention**: 각 단어가 이전 단어들만 참조
- **자기회귀 생성**: 이전 토큰을 바탕으로 다음 토큰 예측
- **가변 길이 출력**: 입력에 따라 다양한 길이의 텍스트 생성

### 대표 모델들

#### GPT (Generative Pre-trained Transformer)
**특징:**
- **자기회귀 학습**: 다음 단어 예측
- **대화형 생성**: 자연스러운 대화 가능
- **창작 능력**: 글, 시, 코드 등 다양한 창작

**학습 방식:**
```python
# GPT 학습 예시
text = "나는 학교에 갔다"
# 학습 시퀀스:
# "나는" → "학교에" 예측
# "나는 학교에" → "갔다" 예측
```

#### GPT-2
**GPT 개선점:**
- **더 큰 모델**: 117M → 1.5B 파라미터
- **더 다양한 데이터**: 웹 크롤링 데이터 활용
- **제로샷 학습**: 특별한 파인튜닝 없이 다양한 작업 수행

#### GPT-3
**혁신적 특징:**
- **거대한 규모**: 175B 파라미터
- **Few-shot 학습**: 몇 개의 예시만으로 새로운 작업 학습
- **다양한 도메인**: 코드, 수학, 창작 등

### Decoder Only 모델의 장단점

**장점:**
- **자연스러운 생성**: 인간과 유사한 텍스트 생성
- **창작 능력**: 새로운 내용 창작 가능
- **대화형 인터페이스**: 자연스러운 대화 가능

**단점:**
- **문맥 이해 제한**: 단방향으로 인한 문맥 제한
- **사실성 부족**: 가짜 정보 생성 가능
- **제어 어려움**: 특정 목적에 맞는 출력 제어 어려움

### 실제 활용 사례

```python
# 텍스트 생성
prompt = "오늘 날씨가"
# GPT → "오늘 날씨가 좋다"

# 대화
user = "안녕하세요"
# GPT → "안녕하세요! 무엇을 도와드릴까요?"

# 코드 작성
prompt = "def calculate"
# GPT → "def calculate_sum(a, b): return a + b"

# 창작
prompt = "봄날의 아침"
# GPT → "봄날의 아침, 햇살이 따뜻하게..."
```

## 3. Encoder-Decoder 모델

### 구조와 특징

**기본 구조:**
```
Input Text → Encoder → Encoded Representation → Decoder → Output Text
```

**핵심 특징:**
- **2단계 처리**: 인코더가 이해, 디코더가 생성
- **Cross-Attention**: 디코더가 인코더 정보 활용
- **변환 작업**: 입력을 다른 형태로 변환

### 대표 모델들

#### T5 (Text-to-Text Transfer Transformer)
**특징:**
- **통합 프레임워크**: 모든 NLP 작업을 텍스트-텍스트 변환으로 통합
- **다양한 작업**: 번역, 요약, 질문 답변 등
- **스케일링**: 다양한 크기의 모델 제공

**작업 예시:**
```python
# 번역
input = "translate English to German: I love you"
output = "Ich liebe dich"

# 요약
input = "summarize: 긴 기사 내용..."
output = "핵심 요약"

# 질문 답변
input = "question: 서울의 수도는? context: 서울은 대한민국의 수도이다"
output = "서울"
```

#### BART (Bidirectional and Auto-Regressive Transformers)
**특징:**
- **노이즈 제거**: 손상된 텍스트를 원본으로 복원
- **다양한 노이즈**: 토큰 마스킹, 삭제, 순서 변경 등
- **생성에 특화**: 요약, 대화 등 생성 작업에 우수

#### mT5 (Multilingual T5)
**다국어 지원:**
- **101개 언어**: 다양한 언어 동시 학습
- **언어 간 지식 전이**: 고품질 언어에서 저품질 언어로 지식 전이
- **통합 모델**: 하나의 모델로 모든 언어 처리

### Encoder-Decoder 모델의 장단점

**장점:**
- **정확한 변환**: 입력을 정확히 다른 형태로 변환
- **다양한 작업**: 번역, 요약, 질문 답변 등
- **제어 가능**: 특정 목적에 맞는 출력 생성

**단점:**
- **복잡한 구조**: 인코더와 디코더 모두 필요
- **학습 어려움**: 두 모듈을 동시에 학습해야 함
- **메모리 사용량**: 두 배의 모델 크기

### 실제 활용 사례

```python
# 번역
input = "I love you"
# T5 → "나는 당신을 사랑합니다"

# 요약
input = "긴 뉴스 기사..."
# T5 → "핵심 내용 요약"

# 질문 답변
input = "질문: 인공지능이란? 문서: AI에 대한 설명..."
# T5 → "정확한 답변"

# 코드 변환
input = "Python: def hello(): print('Hello')"
# T5 → "JavaScript: function hello() { console.log('Hello'); }"
```

## 모델 비교 및 선택 가이드

### 작업별 모델 선택

| 작업 유형 | 권장 모델 | 이유 |
|-----------|-----------|------|
| **텍스트 분류** | Encoder Only | 깊은 문맥 이해로 정확한 분류 |
| **감정 분석** | Encoder Only | 양방향 문맥으로 정확한 감정 파악 |
| **개체명 인식** | Encoder Only | 문맥을 고려한 정확한 개체 식별 |
| **텍스트 생성** | Decoder Only | 자연스러운 텍스트 생성 |
| **대화 시스템** | Decoder Only | 대화형 인터페이스에 적합 |
| **번역** | Encoder-Decoder | 정확한 언어 간 변환 |
| **요약** | Encoder-Decoder | 핵심 내용 추출 및 재구성 |
| **질문 답변** | Encoder-Decoder | 정확한 답변 생성 |

### 성능 비교

```python
# 모델별 성능 비교 예시
def compare_models(task_type):
    if task_type == "classification":
        # Encoder Only 모델이 우수
        return "BERT, RoBERTa"
    elif task_type == "generation":
        # Decoder Only 모델이 우수
        return "GPT, GPT-2, GPT-3"
    elif task_type == "translation":
        # Encoder-Decoder 모델이 우수
        return "T5, BART, mT5"
```

### 실무 적용 시 고려사항

#### 1. 리소스 제약
```python
# 모델 크기별 메모리 사용량
model_sizes = {
    "BERT-base": "110M parameters",
    "GPT-2": "117M-1.5B parameters", 
    "T5-base": "220M parameters",
    "GPT-3": "175B parameters"
}

# 하드웨어 요구사항
hardware_requirements = {
    "BERT-base": "8GB GPU",
    "GPT-2": "8-16GB GPU",
    "T5-base": "16GB GPU",
    "GPT-3": "Multiple A100 GPUs"
}
```

#### 2. 추론 속도
```python
# 모델별 추론 속도 (토큰/초)
inference_speed = {
    "BERT-base": "1000-2000 tokens/sec",
    "GPT-2": "500-1000 tokens/sec", 
    "T5-base": "800-1500 tokens/sec"
}
```

#### 3. 정확도 vs 속도 트레이드오프
```python
# 작업별 정확도 vs 속도
tradeoffs = {
    "실시간 대화": "속도 우선 → GPT 계열",
    "정확한 분석": "정확도 우선 → BERT 계열", 
    "번역 서비스": "균형 → T5 계열"
}
```

## 결론

각 모델 구조는 고유한 장단점을 가지고 있으며, 작업의 특성에 따라 적절한 모델을 선택하는 것이 중요합니다.

### 핵심 포인트

1. **Encoder Only**: 깊은 문맥 이해, 분류/분석 작업에 적합
2. **Decoder Only**: 자연스러운 생성, 대화/창작 작업에 적합  
3. **Encoder-Decoder**: 정확한 변환, 번역/요약 작업에 적합

### 실무 적용 가이드

- **리소스 제약**: 모델 크기와 하드웨어 요구사항 고려
- **작업 특성**: 정확도 vs 속도 트레이드오프 고려
- **확장성**: 향후 작업 확장 가능성 고려

이러한 이해를 바탕으로 LLM 서비스 구축 시 적절한 모델 구조를 선택하고 최적화할 수 있습니다.
