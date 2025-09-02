---
title: 임베딩 모델
parent: RAG
nav_order: 3
---

# 임베딩 모델 (Embedding Model)

RAG 시스템의 성능은 어떤 임베딩 모델을 사용하느냐에 따라 크게 달라집니다. 임베딩 모델은 텍스트를 의미적으로 유사한 벡터 공간으로 변환하는 역할을 하며, 검색 정확도와 효율성에 직접적인 영향을 미칩니다.

## RAG에 특화된 임베딩 모델

| 모델 | 특징 | RAG 활용도 |
|------|------|------------|
| **Sentence-BERT** | 문장 임베딩에 특화되어 문장 간의 의미적 유사성을 잘 포착합니다. | 높음 |
| **DPR (Dense Passage Retrieval)** | 검색 작업에 특화된 모델로, 질문과 문서 쌍으로 학습하여 검색 정확도를 극대화합니다. | 매우 높음 |
| **Contriever** | 비지도 학습 방식으로 학습된 밀집 검색 모델로, 라벨링된 데이터 없이도 좋은 성능을 보입니다. | 높음 |
| **BGE (BAAI General Embedding)** | 다국어 지원과 높은 성능을 목표로 개발된 모델로, 다양한 검색 태스크에서 우수한 성능을 보입니다. | 매우 높음 |
| **Cohere Embed** | 다국어와 도메인 특화 기능을 제공하는 상용 임베딩 모델입니다. | 높음 |

## 임베딩 모델 선택 기준

1.  **성능 (Performance)**: MTEB (Massive Text Embedding Benchmark)와 같은 벤치마크 점수를 참고하여 검색 정확도가 높은 모델을 선택합니다.
2.  **도메인 특화 (Domain-Specific)**: 의료, 법률, 금융 등 특정 도메인의 데이터로 학습된 모델이 해당 분야에서 더 좋은 성능을 보입니다.
3.  **다국어 지원 (Multilingual Support)**: 여러 언어를 처리해야 하는 경우, 다국어를 지원하는 모델을 선택해야 합니다.
4.  **속도와 크기 (Speed & Size)**: 모델의 크기가 작고 추론 속도가 빠를수록 실제 서비스에 적용하기 용이합니다.
5.  **라이선스 (License)**: 상업적 이용이 가능한 라이선스인지 확인해야 합니다.

### 도메인별 권장 임베딩 모델

```python
# 도메인별 권장 임베딩 모델
domain_models = {
    "일반 텍스트": "sentence-transformers/all-MiniLM-L6-v2",
    "의료": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "법률": "nlpaueb/legal-bert-base-uncased",
    "코드": "microsoft/codebert-base",
    "다국어": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

def select_embedding_model(domain):
    """도메인에 맞는 임베딩 모델 선택"""
    return domain_models.get(domain, domain_models["일반 텍스트"])
```

## 임베딩 모델 파인튜닝 (Fine-tuning)

기성 임베딩 모델이 특정 도메인이나 태스크에서 만족스러운 성능을 내지 못할 경우, 파인튜닝을 통해 성능을 개선할 수 있습니다.

-   **목적**: 특정 데이터셋에 대한 검색 정확도 향상
-   **방법**: 질문-문서 쌍 데이터를 이용하여 모델을 추가 학습
-   **주의사항**: 과적합(Overfitting)을 방지하고, 파인튜닝 후 성능 평가가 중요합니다.

```python
# Sentence Transformers 라이브러리를 이용한 파인튜닝 예시
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 모델 로드
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# 학습 데이터 생성
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

# 데이터로더 및 손실 함수 정의
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 모델 학습
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
```
