---
layout: default
title: Mistral & Mixtral (Mistral AI)
parent: 오픈 웨이트 모델 (Open-Weight Models)
nav_order: 5
---

# Mistral & Mixtral (Mistral AI)

## 개요
프랑스의 Mistral AI에서 개발한 Mistral 시리즈는 효율적인 아키텍처와 뛰어난 성능으로 유럽의 AI 혁신을 주도하고 있습니다. 특히 Mixtral 8x7B를 통해 MoE(Mixture of Experts) 모델의 대중화를 이끌었습니다.

## 주요 시리즈 및 특징

### 1. Mistral 7B
*   **SWA (Sliding Window Attention)**: 긴 문맥을 효율적으로 처리하기 위한 윈도우 어텐션 기법 도입.
*   **성능**: 출시 당시 7B 파라미터 모델 중 압도적인 1위를 기록하며 '작지만 강력한 모델'의 기준 제시.

### 2. Mixtral 8x7B / 8x22B (MoE)
*   **Mixtral 8x7B**: 8개의 전문가 모델을 활용한 MoE 구조. Llama 2 70B와 대등하거나 그 이상의 성능을 훨씬 빠른 속도로 달성.
*   **Mixtral 8x22B**: 더욱 확장된 전문가 모델군으로 더 복잡한 추론과 지식 처리 가능.

### 3. Mistral Large & Pixtral
*   **Mistral Large**: 유료 모델 수준의 지능을 갖춘 거대 모델.
*   **Pixtral**: 이미지와 텍스트를 동시에 이해하는 멀티모달 모델.

## 기술적 강점
*   **효율적인 MoE**: 전문가 선택(Router) 로직의 효율성을 높여 적은 활성 파라미터로도 높은 지능 구현.
*   **간결한 아키텍처**: 불필요한 복잡성을 제거하여 추론 속도가 매우 빠름.
*   **개방성**: 개발자 친화적인 라이선스와 명확한 기술 문서를 통해 커뮤니티의 지지도가 높음.

## 평가 및 벤치마크
Mistral 모델들은 유럽 언어 지원 능력이 매우 뛰어나며, 전반적인 상식과 지시 이행 능력에서 균형 잡힌 성능을 보여줍니다. 특히 로컬 서버나 엣지 디바이스에서 고성능을 필요로 하는 환경에서 최우선으로 고려되는 모델입니다.
