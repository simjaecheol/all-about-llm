---
layout: default
title: LLM-Specific Benchmarks
parent: Evaluation Benchmarks
nav_order: 2
---

# LLM 특화 벤치마크 (LLM-Specific Benchmarks)

## 1. HELM (Holistic Evaluation of Language Models)

HELM은 언어 모델을 종합적으로 평가하는 벤치마크입니다.

```python
def load_helm_benchmarks():
    """HELM 벤치마크"""
    helm = {
        "HELM": {
            "description": "Holistic Evaluation of Language Models",
            "tasks": ["Question Answering", "Summarization", "Translation", "Reasoning"],
            "metrics": ["accuracy", "robustness", "fairness", "efficiency"]
        }
    }
    return helm
```

**평가 영역:**
- **Question Answering**: 질문 답변 능력
- **Summarization**: 텍스트 요약 능력
- **Translation**: 번역 능력
- **Reasoning**: 추론 능력

**평가 지표:**
- **Accuracy**: 정확도
- **Robustness**: 견고성
- **Fairness**: 공정성
- **Efficiency**: 효율성

## 2. BigBench (Beyond the Imitation Game)

BigBench는 다양한 언어 이해 및 추론 작업을 포함하는 벤치마크입니다.

```python
def load_bigbench_benchmarks():
    """BigBench 벤치마크"""
    bigbench = {
        "BigBench": {
            "description": "Beyond the Imitation Game",
            "tasks": ["Language Understanding", "Reasoning", "Creativity"],
            "metrics": ["accuracy", "diversity", "creativity"]
        }
    }
    return bigbench
```

**주요 영역:**
- **Language Understanding**: 언어 이해
- **Reasoning**: 논리적 추론
- **Creativity**: 창의성

## 3. AlpacaEval

AlpacaEval은 지시사항 따르기 능력을 평가하는 벤치마크입니다.

```python
def load_alpacaeval_benchmarks():
    """AlpacaEval 벤치마크"""
    alpacaeval = {
        "AlpacaEval": {
            "description": "Evaluation for Instruction Following",
            "tasks": ["Instruction Following", "Task Completion"],
            "metrics": ["win_rate", "human_preference"]
        }
    }
    return alpacaeval
```

**평가 영역:**
- **Instruction Following**: 지시사항 따르기
- **Task Completion**: 작업 완성도

**평가 지표:**
- **Win Rate**: 승률
- **Human Preference**: 인간 선호도
