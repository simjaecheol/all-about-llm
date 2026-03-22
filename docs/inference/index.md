---
title: 추론(Inference)
nav_order: 7
---

# 추론(Inference)

LLM 추론에서 사용되는 다양한 디코딩 전략들을 체계적으로 분류하고 소개합니다. 각 방법은 품질, 속도, 메모리 사용량, 창의성 간의 서로 다른 트레이드오프를 제공합니다.

## 📚 디코딩 전략 카테고리

### 1. [결정론적 디코딩 방법](deterministic_decoding.md)
**일관된 결과를 보장하지만 다양성은 제한되는 방법들**

- **Greedy Search**: 가장 빠르고 단순한 디코딩
- **Beam Search**: 전역적으로 최적화된 출력 생성
- **Contrastive Search**: 모델의 등방성 표현 공간 활용
- **Diverse Beam Search**: 다양성을 증진하는 beam search 변형

**사용 시나리오**: 번역, 요약 등 정확성이 중요한 작업

### 2. [확률적 샘플링 방법](stochastic_sampling.md)
**모델의 확률 분포에서 토큰을 무작위로 선택하는 방법들**

- **Temperature Sampling**: 온도 매개변수로 무작위성 조절
- **Top-k/Top-p Sampling**: 상위 확률 토큰에서 샘플링
- **Min-p Sampling**: 모델의 확신도에 기반한 적응적 토큰 선택
- **Typical Sampling**: 통계적으로 "전형적인" 토큰 선택
- **Mirostat Sampling**: perplexity 비율을 직접 제어

**사용 시나리오**: 창의적 글쓰기, 브레인스토밍, 자연스러운 대화

### 3. [추론 가속화 방법](inference_acceleration.md)
**품질을 유지하면서 추론 속도를 크게 향상시키는 방법들**

- **Speculative Decoding**: 작은 드래프트 모델과 병렬 검증
- **Self-Speculative Decoding**: 대상 모델의 중간 레이어 활용
- **Medusa Decoding**: 여러 디코딩 헤드로 병렬 토큰 예측
- **Blockwise Parallel Decoding**: 여러 시간 단계를 병렬로 처리

**사용 시나리오**: 실시간 대화, 대화형 시스템, 빠른 응답이 필요한 경우

### 4. [Attention 최적화 방법](attention_optimization.md)
**Attention 메커니즘을 최적화하여 메모리와 속도 향상**

- **Sparse Attention**: TidalDecode, FlexPrefill, Star Attention
- **KV Cache 최적화**: LOOK-M, QuantSpec, Round Attention
- **메모리 절약**: 최대 80% 메모리 사용량 감소
- **속도 향상**: 최대 11배 속도 향상

**사용 시나리오**: 긴 시퀀스 처리, 메모리 제약 환경, 다중모달 응용

### 5. [품질 및 제어 방법](quality_control.md)
**출력 품질을 향상시키고 생성 과정을 제어하는 방법들**

- **MBR Decoding**: 최고 기대 유틸리티를 가진 출력 선택
- **Look-back Decoding**: KL divergence를 활용한 반복성 감소
- **Reflection-Window**: 자기 반성적 텍스트 생성
- **DExperts**: 전문가와 안티 전문가 모델 결합
- **Confidence-based**: 활성화 기반 confidence 보정

**사용 시나리오**: 고품질 요구사항, 반복성 방지, 안전성 중시

### 6. [최신 트렌드 및 새로운 접근법](emerging_trends.md)
**2024-2025년의 최신 연구 동향과 혁신적인 방법들**

- **Chain-of-Thought Decoding**: 구조화된 탐색을 통한 추론 향상
- **Adaptive Decoding**: 엔트로피 기반 confidence 메트릭
- **Semantic Uncertainty Analysis**: 의미적 불확실성 조사
- **Multi-Objective Decoding**: 다중 목표 동시 최적화
- **Context-Aware Mechanisms**: 입력별 동적 attention 패턴

**사용 시나리오**: 연구 및 개발, 프로덕션 시스템, 특수 목적

### 7. [구조적 출력 및 제약된 생성](structured_output.md)
**특정 스키마나 규칙을 따르는 정형 데이터 생성 방법들**

- **JSON Mode**: 모델의 출력 형식을 JSON으로 강제
- **Constrained Decoding**: FSM, Regex 등을 이용한 토큰 단위 제약
- **JSON Schema / Pydantic**: 스키마 기반 데이터 추출 및 검증
- **Grammar-based Generation**: CFG 등을 활용한 프로그래밍 언어 생성

**사용 시나리오**: 데이터 추출, API 응답 생성, 코드 생성, 에이전트 도구 호출

## 🎯 선택 가이드

### 기본 응용
- **Greedy/Beam Search**: 번역, 요약 등 정확성이 중요한 작업
- **Temperature Sampling**: 창의적 글쓰기, 브레인스토밍

### 구조 및 정밀도
- **Structured Output**: JSON, XML 등 정형 데이터가 필요한 모든 작업
- **JSON Schema**: 강력한 타입 안정성이 필요한 API 연동
- **Grammar-based**: 코드 생성이나 특정 문법 준수가 필수적인 경우

### 품질 중시
- **Contrastive Search**: 일관성과 품질이 모두 중요한 작업
- **Min-p Sampling**: 높은 온도에서도 일관성 필요한 경우
- **MBR Decoding**: 최고 품질이 요구되는 경우

### 속도 중시
- **Speculative Decoding**: 실시간 대화, 대화형 시스템
- **Medusa**: 추가 모델 없이 빠른 생성 필요한 경우
- **Star Attention**: 긴 시퀀스에서 최대 속도 향상

### 메모리 제약
- **LOOK-M**: 다중모달 긴 컨텍스트에서 메모리 절약
- **QuantSpec**: 양자화를 통한 메모리 효율성
- **Round Attention**: 관련성 기반 선택적 캐시 처리

### 창의성과 다양성
- **Typical Sampling**: 자연스러운 텍스트 생성
- **Mirostat**: 일정한 perplexity 유지
- **Adaptive Decoding**: 컨텍스트에 따른 동적 조정

## 🔬 구현 고려사항

### 하드웨어 요구사항
- **GPU**: Star Attention, TidalDecode, Traditional Speculative
- **CPU**: NoMAD-Attention, N-gram Speculative
- **혼합**: Self-Speculative, SWIFT, LOOK-M

### 메모리 관리
- **KV Cache 최적화**: 효율적인 메모리 사용
- **배치 크기 조정**: 메모리와 속도의 균형
- **압축 및 양자화**: 품질 유지하면서 메모리 절약

### 품질 보장
- **검증 메커니즘**: 품질 손실 방지
- **fallback 전략**: 실패 시 안전한 복구
- **지속적인 모니터링**: 성능 지표 추적

## 📊 성능 비교 요약

| 카테고리 | 품질 | 속도 | 메모리 효율성 | 구현 난이도 |
|----------|------|------|---------------|-------------|
| 결정론적 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 확률적 샘플링 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 추론 가속화 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Attention 최적화 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 품질 제어 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 최신 트렌드 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 구조적 출력 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🚀 시작하기

1. **기본 디코딩 방법**부터 시작하여 [결정론적 디코딩](deterministic_decoding.md) 학습
2. **확률적 샘플링**의 다양한 옵션을 [확률적 샘플링 방법](stochastic_sampling.md)에서 확인
3. **속도 향상**이 필요한 경우 [추론 가속화 방법](inference_acceleration.md) 참조
4. **메모리 최적화**를 위해 [Attention 최적화 방법](attention_optimization.md) 학습
5. **품질 향상**을 위해 [품질 및 제어 방법](quality_control.md) 적용
6. **최신 동향**을 [최신 트렌드](emerging_trends.md)에서 확인

각 방법은 특정 요구사항에 맞는 적절한 디코딩 전략을 선택하는 데 도움이 됩니다. 프로젝트의 목표와 제약사항을 고려하여 최적의 방법을 선택하세요.
