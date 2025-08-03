---
title: PEFT
parent: LLM 학습 방법
nav_order: 3
---

# PEFT (Parameter Efficient Fine Tuning)

## 개요

PEFT(Parameter-Efficient Fine-Tuning)는 대규모 언어 모델의 효율적인 적응을 위한 핵심 기술입니다. 전체 모델 파라미터를 업데이트하는 대신, 일부 파라미터만 학습하여 계산 비용과 메모리 사용량을 대폭 줄이면서도 유사한 성능을 달성합니다.

## PEFT 도입 배경

### 기존 Full Fine-tuning의 한계

1. **막대한 계산 비용**: GPT-3 175B 모델의 경우 모든 1,750억 개의 파라미터를 독립적으로 fine-tuning하는 것은 비용 면에서 비현실적
2. **저장 공간 문제**: 각 downstream task마다 전체 모델 크기와 동일한 체크포인트 저장 필요
3. **Catastrophic Forgetting**: 새로운 작업을 학습할 때 이전에 학습한 지식을 잊어버리는 현상
4. **메모리 요구사항**: 대형 모델 fine-tuning 시 일반적인 하드웨어로는 불가능한 수준의 GPU 메모리 필요

## PEFT의 장점

### 1. 효율성 (Efficiency)
- **파라미터 절약**: 전체 파라미터의 0.01%-0.1%만 학습하여 Full Fine-tuning과 유사한 성능 달성
- **메모리 최적화**: LoRA 적용 시 GPU 메모리 요구량을 3배 감소
- **학습 속도 향상**: 적은 파라미터 업데이트로 인한 빠른 학습 속도

### 2. 저장 공간 절약 (Storage Efficiency)
- **체크포인트 크기**: Full Fine-tuning 시 40GB → PEFT 적용 시 몇 MB
- **다중 작업 지원**: 하나의 기본 모델에 여러 작업별 어댑터 모듈 추가 가능

### 3. Catastrophic Forgetting 완화
- **지식 보존**: 기존 pre-trained weights를 고정함으로써 원래 학습한 지식 유지
- **안정성**: 새로운 작업 학습 시에도 기존 성능 유지

## 대표적인 PEFT 방법론

### 1. LoRA (Low-Rank Adaptation)

**핵심 원리**: 큰 가중치 행렬의 업데이트를 두 개의 작은 low-rank 행렬의 곱으로 분해

```python
# LoRA 구현 예제
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# LoRA 설정
lora_config = LoraConfig(
    r=16,                    # rank (낮을수록 파라미터 수 감소)
    lora_alpha=32,          # scaling parameter
    task_type="CAUSAL_LM",  # 작업 유형
    target_modules=["q_proj", "v_proj"]  # 적용할 레이어
)

# PEFT 모델 생성
model = get_peft_model(model, lora_config)
```

**성능**: GPT-3 175B에서 파라미터 수를 10,000배 감소시키면서도 동등한 성능 달성

### 2. Adapter

**핵심 원리**: Transformer의 각 레이어에 작은 bottleneck 모듈 삽입

**특징**:
- 전체 파라미터의 3.6%만 학습하여 Full Fine-tuning 대비 0.4% 성능 차이
- 각 작업별로 독립적인 어댑터 모듈 관리 가능

### 3. Prefix Tuning

**핵심 원리**: 입력 시퀀스 앞에 학습 가능한 연속적 벡터(prefix) 추가

```python
# Prefix Tuning 예제
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # prefix 길이
    encoder_hidden_size=768
)
```

**장점**: 모델 구조 변경 없이 입력만 수정하여 효율적 적응

### 4. Prompt Tuning

**핵심 원리**: 입력 임베딩에 학습 가능한 "soft prompt" 추가

**특징**: 모델 크기가 클수록 Full Fine-tuning에 근접한 성능 달성

### 5. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**핵심 원리**: 내부 활성화 값을 학습된 벡터로 스케일링

**특징**:
- 극도로 적은 파라미터 (전체의 0.01%) 사용
- T-Few 레시피에서 GPT-3보다 우수한 성능 달성

### 6. QLoRA (Quantized LoRA)

**핵심 원리**: 4-bit 양자화와 LoRA 결합

**혁신 기술**:
- 4-bit NormalFloat (NF4) 데이터 타입
- Double Quantization으로 메모리 사용량 추가 감소
- 65B 파라미터 모델을 48GB GPU에서 학습 가능

## PEFT vs Full Fine-tuning 성능 비교

| 방법론 | 파라미터 비율 | 메모리 사용량 | 성능 | 적용 시나리오 |
|-------|--------------|-------------|------|-------------|
| Full Fine-tuning | 100% | 높음 | 최고 | 복잡한 도메인 특화 작업 |
| LoRA | 0.1-1% | 낮음 | 유사 | 일반적인 적응 작업 |
| IA³ | 0.01% | 매우 낮음 | 유사 | Few-shot 학습 |
| Adapter | 3.6% | 낮음 | 유사 | 다중 작업 환경 |
| Prefix Tuning | 0.1% | 낮음 | 유사 | 생성 작업 |

## 오픈소스 프로젝트에서의 PEFT 활용

### 1. Hugging Face PEFT Library

가장 널리 사용되는 PEFT 구현체로, 다양한 방법론을 통합 제공:

```python
# 기본 사용법
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32)
model = get_peft_model(model, peft_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# 출력: trainable params: 294,912 || all params: 117,577,728 || trainable%: 0.25
```

### 2. Microsoft LoRA Implementation

원본 LoRA 논문의 공식 구현체:

```python
import loralib as lora

# 기존 Linear 레이어를 LoRA로 교체
layer = lora.Linear(in_features, out_features, r=16)

# LoRA 파라미터만 학습 가능하도록 설정
lora.mark_only_lora_as_trainable(model)

# LoRA 파라미터만 저장
torch.save(lora.lora_state_dict(model), checkpoint_path)
```

## 최신 연구 동향

### 1. 고급 PEFT 방법론
- **AdaLoRA**: 적응적 예산 할당을 통한 LoRA 개선
- **RoSA**: Low-rank와 sparse 컴포넌트 결합
- **HydraLoRA**: 비대칭 구조의 LoRA 프레임워크

### 2. 다중 모달 확장
- **Context-PEFT**: 다중 모달, 다중 작업을 위한 프레임워크
- **VL-Adapter**: Vision-Language 작업용 파라미터 효율적 방법론

### 3. 메모리 최적화 연구
- **Local LoRA**: 청크 단위 순차 학습으로 메모리 요구사항 완전 분리
- **HSplitLoRA**: 분할 학습과 LoRA 결합으로 이질적 환경 대응

## 실무 적용 가이드

### 1. 방법론 선택 기준

| 상황 | 추천 방법론 | 이유 |
|------|-----------|------|
| 제한된 자원 | IA³, Prompt Tuning | 극소 파라미터 사용 |
| 일반적 적응 | LoRA | 성능-효율성 균형 |
| 다중 작업 | Adapter | 작업별 모듈 독립성 |
| 생성 작업 | Prefix Tuning | 생성 최적화 |

### 2. 하드웨어 요구사항

**최소 사양**:
- GPU: NVIDIA A100 80GB (권장)
- RAM: 32GB 이상 (대형 모델은 64GB)
- 저장공간: 50GB 이상

**메모리 사용량 예시**:
- T0_3B Full Fine-tuning: 47.14GB GPU
- T0_3B + LoRA: 14.4GB GPU (67% 절약)
- T0_3B + LoRA + DeepSpeed: 9.8GB GPU (79% 절약)

## 결론

PEFT는 대규모 언어 모델의 효율적 활용을 위한 필수 기술로 자리잡았습니다. **LoRA**가 가장 널리 사용되는 방법론이지만, 작업의 특성과 자원 제약에 따라 **Adapter**, **Prefix Tuning**, **IA³** 등 다양한 선택지가 있습니다.

특히 **QLoRA**와 같은 양자화 결합 기술은 일반 소비자 하드웨어에서도 대형 모델 학습을 가능하게 하여, PEFT의 민주화에 크게 기여하고 있습니다. 향후에는 **다중 모달 확장**, **자동 하이퍼파라미터 최적화**, **메모리 효율성 극대화** 방향으로 연구가 진행될 것으로 예상됩니다.

LLM 서비스 개발을 계획하는 조직이라면, 자원 효율성과 성능의 균형을 고려하여 적절한 PEFT 방법론을 선택하고, Hugging Face PEFT와 같은 검증된 오픈소스 도구를 활용할 것을 권장합니다.
