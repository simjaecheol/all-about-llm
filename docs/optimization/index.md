---
layout: default
title: LLM 최적화
nav_order: 30
has_children: true
---

# LLM 최적화 (Optimization)

LLM 최적화는 모델의 성능(정확도)을 최대한 유지하면서, 추론 속도를 높이고 메모리 사용량을 줄이는 모든 기술을 의미합니다. 특히 상용 서비스 단계에서는 비용 절감과 사용자 경험(응답 속도) 개선을 위해 필수적인 과정입니다.

## 주요 최적화 영역

LLM 최적화는 크게 세 가지 방향으로 나뉩니다.

### 1. 모델 경량화 (Model Compression)
모델의 파라미터 크기 자체를 줄여 VRAM 사용량을 낮추는 기술입니다.
- **양자화 (Quantization):** 가중치의 정밀도를 낮추어 용량을 줄이는 기법 (AWQ, GPTQ, GGUF 등)
- **가지치기 (Pruning):** 중요도가 낮은 가중치를 제거하여 계산량을 줄이는 기법
- **지식 증류 (Knowledge Distillation):** 거대 모델(Teacher)의 지식을 작은 모델(Student)에게 전수하는 기법

### 2. 추론 가속 (Inference Acceleration)
추론 과정에서의 병목 현상을 해결하고 계산 효율을 극대화하는 기술입니다.
- **어텐션 최적화:** Flash Attention, PagedAttention (vLLM)
- **디코딩 전략:** 투기적 디코딩 (Speculative Decoding)
- **메모리 최적화:** KV 캐시 최적화 및 압축

### 3. 구조적 최적화 (Architectural Optimization)
모델 설계 단계에서부터 효율성을 고려하는 방식입니다.
- **MoE (Mixture of Experts):** 필요한 전문가 레이어만 활성화하여 계산 효율 증대
- **GQA (Grouped Query Attention):** Multi-head Attention의 효율적 변형

---

## 최적화 기법별 비교

| 구분 | 주요 기술 | 핵심 효과 | 적용 시점 |
| :--- | :--- | :--- | :--- |
| **양자화** | AWQ, GPTQ, GGUF | VRAM 절감, 속도 향상 | 학습 후 (PTQ) |
| **추론 엔진** | vLLM, TensorRT-LLM | 처리량(Throughput) 증대 | 서비스 배포 시 |
| **알고리즘** | Speculative Decoding | 응답 시간(Latency) 단축 | 추론 실행 시 |

각 하위 페이지에서 각 기법의 상세 원리와 최신 트렌드를 확인하실 수 있습니다.