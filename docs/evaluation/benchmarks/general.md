---
layout: default
title: General Benchmarks
parent: Evaluation Benchmarks
nav_order: 1
---

# 일반적인 벤치마크 (General Benchmarks)

## 1. GLUE (General Language Understanding Evaluation)

GLUE는 자연어 이해 작업을 위한 표준 벤치마크입니다.

```python
def load_benchmark_datasets():
    """벤치마크 데이터셋 로드"""
    benchmarks = {
        "GLUE": {
            "description": "General Language Understanding Evaluation",
            "tasks": ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE"],
            "metrics": ["accuracy", "f1", "pearson", "spearman"]
        }
    }
    return benchmarks
```

**주요 작업:**
- **CoLA**: 언어 수용성 판단
- **SST-2**: 감정 분석
- **MRPC**: 문장 유사성 판단
- **QQP**: 질문 유사성 판단
- **STS-B**: 의미적 텍스트 유사성
- **MNLI**: 자연어 추론
- **QNLI**: 질문-답변 자연어 추론
- **RTE**: 텍스트 함의 관계

## 2. SuperGLUE

SuperGLUE는 GLUE보다 더 어려운 자연어 이해 작업을 포함합니다.

```python
def load_superglue_benchmarks():
    """SuperGLUE 벤치마크"""
    superglue = {
        "SuperGLUE": {
            "description": "More challenging NLU tasks",
            "tasks": ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"],
            "metrics": ["accuracy", "f1", "exact_match"]
        }
    }
    return superglue
```

**주요 작업:**
- **BoolQ**: 예/아니오 질문 답변
- **CB**: 약한 지도학습 자연어 추론
- **COPA**: 인과관계 추론
- **MultiRC**: 다중 문장 독해
- **ReCoRD**: 독해 기반 질문 답변
- **WiC**: 문맥에서의 단어 의미
- **WSC**: Winograd 스키마 챌린지

## 3. MMLU (Massive Multitask Language Understanding)

MMLU는 다양한 학문 분야에 대한 지식을 테스트하는 벤치마크입니다.

```python
def load_mmlu_benchmarks():
    """MMLU 벤치마크"""
    mmlu = {
        "MMLU": {
            "description": "Massive Multitask Language Understanding",
            "tasks": ["STEM", "Humanities", "Social Sciences", "Other"],
            "metrics": ["accuracy"]
        }
    }
    return mmlu
```

**주요 분야:**
- **STEM**: 과학, 기술, 공학, 수학
- **Humanities**: 인문학
- **Social Sciences**: 사회과학
- **Other**: 기타 분야
