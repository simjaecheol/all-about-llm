---
title: Pre-Training
parent: LLM 학습 방법
nav_order: 1
---

# Pre-Training

## 개요

Pre-Training(사전 학습)은 대규모 언어 모델(LLM)의 기초를 다지는 핵심 단계입니다. 이 단계에서는 방대한 양의 텍스트 데이터를 사용하여 모델이 언어의 패턴, 문법, 의미를 학습하도록 합니다. Pre-Training을 통해 얻은 모델을 Foundation Model이라고 하며, 이는 다양한 다운스트림 작업에 적용할 수 있는 기반 모델이 됩니다.

## Pre-Training의 목적

### 1. 언어 이해 능력 습득
- **어휘 학습**: 대규모 코퍼스에서 단어와 구문의 의미를 학습
- **문법 패턴 인식**: 문장 구조와 문법적 규칙을 내재적으로 학습
- **의미적 표현**: 단어와 문장의 의미적 관계를 이해

### 2. 일반적인 지식 습득
- **세계 지식**: 다양한 도메인의 지식을 암묵적으로 학습
- **추론 능력**: 논리적 사고와 추론 패턴 학습
- **상식**: 인간의 상식적 지식과 경험 학습

## Pre-Training 방법론

### 1. Masked Language Modeling (MLM)

MLM은 BERT에서 처음 도입된 방법으로, 문장에서 일정 비율의 토큰을 마스킹하고 이를 예측하는 방식입니다.

#### 작동 원리
```python
# MLM 예시
original_text = "The cat sat on the mat"
masked_text = "The [MASK] sat on the [MASK]"
# 모델이 [MASK] 위치의 토큰을 예측해야 함
```

#### 마스킹 전략
- **15% 마스킹**: 전체 토큰의 15%를 마스킹
- **80% [MASK]**: 마스킹된 토큰의 80%를 [MASK] 토큰으로 교체
- **10% 랜덤 토큰**: 10%는 다른 랜덤 토큰으로 교체
- **10% 원본 유지**: 10%는 원본 토큰 그대로 유지

#### 장점
- **양방향 문맥**: 좌우 문맥을 모두 활용하여 예측
- **깊은 이해**: 문맥을 통한 의미적 이해 능력 향상
- **다양한 태스크**: 분류, NER, QA 등에 효과적

#### 단점
- **생성 제한**: 텍스트 생성에는 부적합
- **사전 학습과 파인튜닝 불일치**: 실제 태스크와 다른 형태

#### 구현 예시
```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The [MASK] sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 마스킹된 위치의 예측
masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
logits = outputs.logits
mask_logits = logits[0, masked_index, :]
predicted_token_ids = torch.argmax(mask_logits, dim=-1)
```

### 2. Causal Language Modeling (CLM)

CLM은 GPT 시리즈에서 사용하는 방법으로, 이전 토큰들을 기반으로 다음 토큰을 예측하는 방식입니다.

#### 작동 원리
```python
# CLM 예시
text = "The cat sat on the mat"
# 모델은 각 위치에서 다음 토큰을 예측
# "The" → "cat"
# "The cat" → "sat"
# "The cat sat" → "on"
# ...
```

#### 학습 방식
- **자기회귀적 학습**: 각 위치에서 다음 토큰을 예측
- **Attention 마스킹**: 미래 토큰에 대한 정보 차단
- **시퀀스 길이**: 일반적으로 512~2048 토큰

#### 장점
- **자연스러운 생성**: 텍스트 생성에 최적화
- **일관성**: 사전 학습과 생성 태스크의 일관성
- **확장성**: 긴 시퀀스 생성 가능

#### 단점
- **단방향 제약**: 미래 정보를 활용할 수 없음
- **문맥 제한**: 특정 위치에서의 문맥 이해 제한

#### 구현 예시
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "The cat sat on"
inputs = tokenizer(text, return_tensors="pt")

# 다음 토큰 예측
with torch.no_grad():
    outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    predicted_text = tokenizer.decode(next_token)
```

#### 손실 함수
```python
def causal_lm_loss(logits, labels):
    # 시프트: 입력과 타겟을 한 위치씩 이동
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Cross-entropy 손실 계산
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))
    return loss
```

### 3. Prefix Language Modeling (PLM)

PLM은 T5에서 사용하는 방법으로, 프리픽스(입력)와 타겟(출력)을 분리하여 학습하는 방식입니다.

#### 작동 원리
```python
# PLM 예시
# 번역 태스크
prefix = "translate English to French: "
input_text = "Hello world"
target_text = "Bonjour le monde"

# 요약 태스크
prefix = "summarize: "
input_text = "Long article text..."
target_text = "Summary of the article"
```

#### 구조적 특징
- **인코더-디코더**: Transformer의 인코더-디코더 구조 활용
- **태스크 프리픽스**: 각 태스크를 구분하는 프리픽스 사용
- **통합 학습**: 다양한 태스크를 하나의 모델에서 학습

#### 태스크 프리픽스 예시
```python
task_prefixes = {
    "translation": "translate English to French: ",
    "summarization": "summarize: ",
    "question_answering": "question: ",
    "sentiment_analysis": "sentiment: ",
    "text_generation": "generate: "
}
```

#### 장점
- **다양한 태스크**: 하나의 모델로 다양한 태스크 처리
- **유연성**: 새로운 태스크 추가 용이
- **효율성**: 태스크별 모델 불필요

#### 단점
- **복잡성**: 인코더-디코더 구조의 복잡성
- **메모리**: 더 많은 메모리 요구사항

#### 구현 예시
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# 번역 태스크
prefix = "translate English to French: "
input_text = "Hello world"
full_input = prefix + input_text

inputs = tokenizer(full_input, return_tensors="pt")
target_text = "Bonjour le monde"
labels = tokenizer(target_text, return_tensors="pt").input_ids

# 학습
outputs = model(input_ids=inputs.input_ids, labels=labels)
loss = outputs.loss
```

### 4. 방법론 비교

| 특성 | MLM | CLM | PLM |
|------|-----|-----|-----|
| **문맥 방향** | 양방향 | 단방향 | 양방향 |
| **주요 모델** | BERT, RoBERTa | GPT 시리즈 | T5, BART |
| **적합한 태스크** | 이해 태스크 | 생성 태스크 | 통합 태스크 |
| **생성 능력** | 제한적 | 우수 | 우수 |
| **이해 능력** | 우수 | 보통 | 우수 |
| **구조** | 인코더만 | 디코더만 | 인코더-디코더 |

### 5. 최신 발전 동향

#### 1. 통합 방법론
- **UniLM**: MLM과 CLM을 결합한 통합 모델
- **BART**: 인코더-디코더 구조의 MLM
- **T5**: 모든 태스크를 텍스트-텍스트로 통합

#### 2. 효율성 개선
- **ELECTRA**: 생성자-판별자 구조로 효율성 향상
- **DeBERTa**: 분해된 어텐션으로 성능 향상
- **ALBERT**: 파라미터 공유로 모델 크기 감소

#### 3. 특화 방법론
- **SpanBERT**: 연속된 토큰 마스킹
- **StructBERT**: 구조적 정보 활용
- **CodeBERT**: 코드 특화 마스킹 전략

## 데이터셋 구성

### 1. 웹 텍스트 데이터
- **Common Crawl**: 웹에서 수집된 대규모 텍스트 데이터
- **Wikipedia**: 구조화된 지식 데이터
- **Books**: 문학 작품과 교육 자료

### 2. 코드 데이터
- **GitHub**: 오픈소스 코드 저장소
- **Stack Overflow**: 프로그래밍 관련 질문과 답변

### 3. 다국어 데이터
- **다양한 언어**: 영어 외 다양한 언어의 텍스트
- **번역 데이터**: 병렬 코퍼스를 통한 다국어 이해

## 학습 과정

### 1. 토큰화 (Tokenization)
```python
# 예시: BPE 토큰화
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Hello, world!")
```

### 2. 배치 구성
- **시퀀스 길이**: 일반적으로 512~2048 토큰
- **배치 크기**: GPU 메모리에 따라 조정 (보통 32~128)
- **그래디언트 누적**: 대용량 배치 효과를 위한 기법

### 3. 손실 함수
```python
# Causal LM 손실 계산
def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))
    return loss
```

## 최적화 기법

### 1. 학습률 스케줄링
- **Warmup**: 초기 학습률을 점진적으로 증가
- **Cosine Annealing**: 학습률을 코사인 함수로 감소
- **Linear Decay**: 선형적으로 학습률 감소

### 2. 정규화 기법
- **Layer Normalization**: 각 레이어의 출력 정규화
- **Dropout**: 과적합 방지를 위한 랜덤 드롭아웃
- **Weight Decay**: 가중치 정규화

### 3. 메모리 최적화
- **Gradient Checkpointing**: 메모리 사용량 감소
- **Mixed Precision**: FP16을 사용한 학습 속도 향상
- **ZeRO**: 분산 학습을 위한 메모리 최적화

## 평가 지표

### 1. Perplexity
- **정의**: 모델이 다음 토큰을 예측할 때의 불확실성
- **계산**: $PPL = \exp(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i))$

### 2. Zero-shot 성능
- **정의**: 특별한 파인튜닝 없이 다양한 태스크 수행
- **평가**: 언어 모델링, 질문 답변, 요약 등

### 3. 다운스트림 태스크 성능
- **GLUE**: 자연어 이해 태스크 벤치마크
- **SuperGLUE**: 더 어려운 NLU 태스크
- **MMLU**: 다중 분야 언어 이해 평가

## 주요 모델 사례

### 1. GPT 시리즈
- **GPT-1**: 117M 파라미터, BooksCorpus 데이터셋
- **GPT-2**: 1.5B 파라미터, WebText 데이터셋
- **GPT-3**: 175B 파라미터, 대규모 웹 데이터

### 2. BERT 시리즈
- **BERT-Base**: 110M 파라미터, MLM + NSP
- **BERT-Large**: 340M 파라미터, 더 깊은 구조
- **RoBERTa**: BERT의 개선된 학습 방법

### 3. T5 시리즈
- **T5-Base**: 220M 파라미터, 통합 텍스트-텍스트 모델
- **T5-Large**: 770M 파라미터
- **T5-11B**: 11B 파라미터, 대규모 모델

## 도전 과제

### 1. 계산 비용
- **GPU 시간**: 수개월의 학습 시간
- **전력 소비**: 환경적 영향
- **하드웨어 요구사항**: 고성능 GPU 클러스터

### 2. 데이터 품질
- **편향**: 학습 데이터의 사회적 편향
- **유해 콘텐츠**: 부적절한 내용 필터링
- **저작권**: 데이터 사용 권한

### 3. 환경적 영향
- **탄소 발자국**: 대규모 학습의 환경 비용
- **에너지 효율성**: 효율적인 학습 방법 연구

## 향후 방향

### 1. 효율적 학습
- **Parameter-Efficient Fine-tuning**: 적은 파라미터로 효율적 학습
- **Knowledge Distillation**: 작은 모델로 지식 전이
- **Pruning**: 불필요한 파라미터 제거

### 2. 지속 가능성
- **Green AI**: 환경 친화적 AI 연구
- **재사용 가능한 모델**: 다양한 태스크에 적용 가능한 모델

### 3. 윤리적 고려사항
- **투명성**: 모델의 의사결정 과정 설명
- **공정성**: 편향 없는 모델 개발
- **책임성**: AI 시스템의 책임 있는 개발

## 결론

Pre-Training은 현대 LLM의 핵심 구성 요소로, 모델이 언어와 지식을 이해하는 기초를 제공합니다. 대규모 데이터와 계산 자원을 투입하여 얻은 Foundation Model은 다양한 다운스트림 태스크에 적용할 수 있는 강력한 기반을 제공합니다. 하지만 계산 비용, 환경적 영향, 윤리적 고려사항 등 여러 도전 과제가 있으며, 지속 가능하고 책임 있는 AI 개발을 위한 연구가 계속 진행되고 있습니다.
