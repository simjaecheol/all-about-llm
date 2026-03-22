---
layout: default
title: VLM (Vision-Language Models)
parent: 멀티모달
nav_order: 1
---

# VLM (Vision-Language Models)

VLM(Vision-Language Models)은 시각적 정보(이미지, 비디오 등)와 텍스트(자연어)를 동시에 이해하고 연관 짓도록 설계된 모델입니다. 

## 최근 트렌드 (2024-2025)

1. **Instruction Tuning(명령어 튜닝)과 추론 능력 강화**: 초기 VLM이 이미지 내 객체 인식이나 간단한 캡셔닝에 머물렀다면, 최근에는 사용자의 복잡한 지시(Instruction)를 따르고 시각적 문맥을 기반으로 논리적으로 추론하는 능력이 크게 향상되었습니다.
2. **효율적 어댑터와 파라미터 미세조정 (LoRA)**: 거대한 Foundation 모델을 처음부터 학습시키기보다 이미 학습된 LLM에 시각적 능력을 부여하기 위해 LoRA 같은 효율적 미세조정 기법이 주로 사용됩니다.
3. **MoE(Mixture-of-Experts) 도입**: 효율성과 성능을 높이기 위해 VLM에도 MoE 아키텍처가 도입되어(예: Llama 4, Kimi-VL), 입력에 따라 모델의 특정 파라미터만 활성화합니다.
4. **소규모 모델(Smol Models)과 엣지 디바이스**: Phi-4 Multimodal, DeepSeek-VL2처럼 크기는 작지만 성능이 뛰어난 '소규모 고효율 모델'들이 엣지 디바이스 구동을 위해 많이 연구되고 있습니다.
5. **VLA (Vision-Language-Action)**: 이미지와 텍스트를 받아 로봇의 행동 제어 명령어로 변환하는 로보틱스 분야로의 확장이 눈에 띕니다.

## 대표적인 모델

- **GPT-4V / GPT-4.1**: 강력한 일반화 성능과 벤치마크 점수, 차트/문서 분석(OCR), 시각적 수학 문제 풀이 등에서 뛰어난 성능을 보입니다.
- **Gemini Series (Google)**: 텍스트, 이미지, 짧은 비디오를 동시에 이해하여 정보 간의 복잡한 추론 능력을 갖추고 있습니다.
- **Molmo (Allen AI)**: 1B, 7B, 72B 등 다양한 가중치를 제공하며, Proprietary 모델에 필적하는 성능을 자랑합니다. 특히 이미지 내 특정 위치를 정확히 가리키는 Pointing 능력이 우수합니다.
- **Llama 3.2 Vision (Meta)**: 향상된 OCR 능력과 오픈 데이터 기반 시각적 질문 응답(VQA) 등 강력한 오픈소스 모델로, 최대 128K 컨텍스트를 지원합니다.
- **Qwen 2.5 VL (Alibaba)**: 비디오 입력 처리와 다국어 처리에 강점을 가지며, 시각적 정보의 지역화(Localization) 능력이 우수합니다.
- **DeepSeek-VL2**: 강력한 과학/수학적 추론 능력을 갖춘 작고 효율적인 혼합 오픈 모델입니다.
