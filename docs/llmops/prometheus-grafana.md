---
layout: default
title: Prometheus & Grafana
parent: LLMOps
nav_order: 4
---

# Prometheus & Grafana를 활용한 LLM 운영 모니터링

LLM 애플리케이션이 PoC(Proof of Concept) 단계를 넘어 실제 서비스되는 운영(Production) 환경으로 진입하게 되면, 트래픽을 감당하는 인프라의 안정성과 비용 효율성을 모니터링하는 것이 필수적입니다. Langfuse나 LangSmith 같은 도구가 에이전트의 사고 흐름(Trace)과 결과물의 품질(Evaluation)을 중점적으로 추적한다면, **Prometheus & Grafana 조합은 하드웨어와 인프라스트럭처 레벨의 핵심 지표들을 관찰**하는 데 강점을 지닙니다.

## 운영 환경 모니터링의 필요성
운영 단계에서는 모델의 추론 한계나 메모리 초과로 인한 지연과 장애가 수익과 직결됩니다. 복수의 GPU가 동시에 작동하는 상황에서 vLLM, Hugging Face TGI, Triton Inference Server와 같은 서빙(Serving) 프레임워크들은 병렬 처리 효율을 높이기 위해 복잡한 스케줄링 로직(예: Continuous Batching, KV Cache PagedAttention)을 사용합니다. 이를 시각화하여 병목 현상을 해결하는 것이 주된 목적입니다.

---

## 핵심 모니터링 지표 (Key Metrics)

LLM 모니터링을 위해 보통 다음 지표들을 대시보드에서 주로 추적합니다.

### 1. 사용자 경험 및 성능 (Performance)
* **TTFT (Time To First Token)**: 사용자가 프롬프트를 보낸 후 첫 번째 토큰이 생성되기까지 걸린 시간입니다. 서비스 반응형 속도(Perceived Latency)를 나타내는 핵심 지표입니다.
* **TPOT (Time Per Output Token)**: 한 토큰이 생성되고 다음 토큰이 생성되기까지 걸린 시간의 평균입니다. 스트리밍 속도를 대변합니다.
* **e2e Latency**: 프롬프트 입력부터 전체 응답이 생성되기까지의 총 지연 시간입니다. (주로 p50, p95, p99 백분위수 활용)
* **Throughput**: 초당 처리되는 요청의 수(Requests/sec) 및 초당 생성되는 토큰 수(Tokens/sec)를 나타내며, 모델의 처리 용량을 표시합니다.

### 2. 하드웨어 및 리소스 상태 (Resource & Hardware)
* **GPU Memory Utilization & Temperature**: GPU VRAM이 얼마나 사용되고 있는지, 과열되지 않는지 체크합니다.
* **KV Cache Usage** (가장 중요): 모델이 Attention 연산을 위해 저장해둔 키-값 캐시의 사용량입니다. 이 공간이 포화 상태(Saturation)에 다다르면 더 이상 새 요청을 받지 못하거나 메모리 오류(OOM)가 발생합니다.
* **Queue Length**: 서빙 프레임워크가 처리하지 못하고 대기열(Queue)에 쌓인 요청의 수입니다. 이 수치가 높으면 스케일 아웃(Scale-out)이 필요하다는 명확한 신호입니다.

### 3. 비용 및 사용 효율 (Cost & Usage)
* **Token Usage Tracking**: 모델에 주입되는 Prompt Token과 생성되는 Generation Token 비율. 많은 상용 API들은 토큰 기반 과금이므로 이는 곧 운영 비용과 직결됩니다.
* **Cache Hit Rate**: 만약 Prompt Cache 등을 사용한다면 얼마나 캐시를 효과적으로 활용하고 있는지 파악하여 중복 연산을 줄입니다.

### 4. 안정성 (Reliability)
* **Error Rate**: OOM, Timeout 등의 원인으로 실패한 요청의 빈도를 파악합니다.

---

## 아키텍처 통합 방안 (Integration Workflow)

전형적인 아키텍처 구성은 다음과 같이 동작합니다.

1. **서빙 프레임워크 (Exporters)**
    * 대부분의 모던 LLM 서빙 프레임워크(vLLM, TGI, Ollama 등)는 기본적으로 `/metrics` 엔드포인트를 노출하여 Prometheus가 긁어갈 수 있는 데이터 형식(Prometheus-compatible format)을 제공합니다.
    * 만약 제공하지 않는 경우(예: 외부 상용 API 래핑), Python으로 직접 Custom Exporter를 작성할 수 있습니다.
2. **Prometheus를 통한 Metric 수집 (Scraping)**
    * 주기적으로 시스템 및 서빙 서버의 상태 데이터를 수집(Pull)하여 시계열 데이터베이스 공간에 저장합니다.
3. **Grafana 대시보드 시각화 (Visualization)**
    * Prometheus를 Data Source로 설정한 뒤 시계열 데이터(PromQL) 구문을 사용해 그래프를 그립니다.
    * vLLM Dashboard 등 사전 제작된 템플릿들을 Grafana 오픈소스 커뮤니티에서 쉽게 가져와 수정할 수 있습니다.
