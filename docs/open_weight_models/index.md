---
layout: default
title: 오픈 웨이트 모델 (Open-Weight Models)
has_children: true
nav_order: 160
---

# 오픈 웨이트 모델 (Open-Weight Models)

## 개요

오픈 웨이트 모델은 가중치(Weights)가 공개되어 사용자가 자신의 로컬 인프라나 클라우드에서 직접 호스팅하고 미세 조정(Fine-tuning)할 수 있는 LLM을 의미합니다. 최근 Meta의 Llama, Alibaba의 Qwen, DeepSeek 등 성능이 뛰어난 오픈 웨이트 모델들이 등장하며 AI 생태계의 민주화를 이끌고 있습니다.

## 주요 오픈 웨이트 모델 시리즈

이 섹션에서는 현재 LLM 생태계를 주도하고 있는 주요 오픈 웨이트 모델들의 특징, 아키텍처, 벤치마크 성능을 정리합니다.

### 모델별 상세 정보

- [Llama (Meta)](./llama.md) - 오픈 웨이트 모델의 표준을 정립한 시리즈
- [DeepSeek](./deepseek.md) - 압도적인 비용 효율성과 성능을 자랑하는 추론 특화 모델
- [Qwen (Alibaba Cloud)](./qwen.md) - 강력한 코딩 및 멀티모달 능력을 갖춘 모델
- [GLM (Zhipu AI)](./glm.md) - 시스템 엔지니어링 및 에이전트 성능에 특화된 모델
- [Mistral & Mixtral (Mistral AI)](./mistral.md) - 효율적인 MoE 아키텍처의 선구자
- [Kimi (Moonshot AI)](./kimi.md) - 초장문 맥락과 자율 에이전틱 AI의 강자
- [Yi (01.AI)](./yi.md) - 우수한 중-영 이중 언어 성능과 실전 벤치마크 상위권 모델
- [기타 주요 중국 모델](./others_chinese.md) - InternLM, MiniCPM, Baichuan 등 특화 모델군

## 오픈 웨이트 모델의 가치

1.  **데이터 프라이버시**: 기업 내부 데이터를 외부 API로 전송하지 않고 로컬에서 처리 가능.
2.  **커스터마이징**: 특정 도메인에 맞게 미세 조정(Fine-tuning)이 자유로움.
3.  **비용 절감**: 대규모 추론 시 API 호출 비용 대비 효율적인 인프라 운영 가능.
4.  **연구 및 개발**: 모델의 내부 동작을 분석하고 새로운 아키텍처를 실험하는 데 필수적.
