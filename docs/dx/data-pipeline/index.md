---
layout: default
title: 데이터 파이프라인 (Data Pipeline)
parent: 디지털 전환 (DX)
has_children: true
nav_order: 4
---

# 데이터 파이프라인 (Data Pipeline)

## 개요
데이터 파이프라인은 파편화된 원천 데이터를 신뢰할 수 있는 정보로 변환하여 AI 모델이나 분석 시스템에 공급하는 '혈관'입니다. 설계 방식에 따라 시스템의 유연성과 적시성이 결정됩니다. DX에서 확보한 강력한 데이터 파이프라인은 AX로 나아가기 위한 가장 핵심적인 인프라입니다.

---

## 1. 데이터 파이프라인 설계 원칙

### 1) ETL (Extract, Transform, Load) vs ELT (Extract, Load, Transform)
데이터를 처리하는 '순서'의 변화는 현대 데이터 인프라(Modern Data Stack)의 가장 큰 특징입니다.

| 구분 | **ETL (전통적 방식)** | **ELT (현대적 방식)** |
| :--- | :--- | :--- |
| **순서** | 추출 → **가공** → 적재 | 추출 → 적재 → **가공** |
| **처리 위치** | 별도의 가공 엔진 (Spark, Python 등) | 대상 저장소 (Snowflake, BigQuery 등) |
| **장점** | 민감 정보 마스킹 후 적재 가능, 저장 비용 절감 | 처리 속도 빠름, 원본 데이터 보존으로 유연성 확보 |
| **단점** | 가공 로직 변경 시 전체 파이프라인 재실행 필요 | 클라우드 웨어하우스 비용 발생 가능 |
| **주요 도구** | Informatica, Talend, Custom Python Scripts | **dbt**, Airbyte, Fivetran |

*   **AX로의 연결**: ELT 방식은 원본 데이터를 그대로 저장(Bronze Layer)해두기 때문에, 나중에 AI 모델의 학습 요건이 바뀌더라도 파이프라인을 처음부터 다시 구축할 필요 없이 SQL(dbt) 수정만으로 대응이 가능하다는 큰 장점이 있습니다.

### 2) 데이터 처리 모드: Batch vs Streaming
데이터가 생성된 후 시스템에 반영되기까지의 '시간적 간격(Latency)'에 따른 구분입니다.

*   **배치 처리 (Batch Processing)**:
    *   **특징**: 일정 주기(시간, 일, 주)마다 대량의 데이터를 한꺼번에 처리합니다.
    *   **용도**: 일일 결산 리포트, 대규모 모델 학습 데이터 준비.
    *   **도구**: Apache Airflow, Cron.
*   **스트리밍 처리 (Streaming/Real-time)**:
    *   **특징**: 데이터가 생성되는 즉시(또는 수초 내) 처리합니다.
    *   **용도**: 실시간 이상 감지, 금융 거래 모니터링, 실시간 AI 추천.
    *   **도구**: Apache Kafka, Flink, Spark Streaming.

### 3) 하이브리드 아키텍처
최신 시스템은 분석의 정확도(Batch)와 즉각적인 반응성(Streaming)을 모두 잡기 위해 두 방식을 혼합하여 사용합니다.
*   **람다 아키텍처 (Lambda Architecture)**: 배치 레이어와 스피드 레이어를 독립적으로 운영하여 정확성과 실시간성을 모두 확보.
*   **카파 아키텍처 (Kappa Architecture)**: 모든 데이터를 스트림으로 간주하여 단일 파이프라인으로 단순화 (Kafka 중심).

---

## 주요 학습 로드맵
향후 다음 항목들을 상세히 다룰 예정입니다.

1. **[데이터 아키텍처 (계층화)](./architecture)**: 메달리온 아키텍처 및 데이터 레이크하우스
2. **[워크플로우 오케스트레이션](./orchestration)**: Apache Airflow 및 현대적 도구들
3. **[데이터 가공 및 품질 관리](./transformation-quality)**: dbt와 데이터 리니지
4. **[AI/AX 특화 파이프라인](./ai-specialized)**: Vector Pipeline 및 Feature Store
