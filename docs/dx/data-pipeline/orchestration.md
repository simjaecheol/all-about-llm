---
layout: default
title: 워크플로우 오케스트레이션 (Orchestration)
parent: 데이터 파이프라인 (Data Pipeline)
nav_order: 2
---

# 워크플로우 오케스트레이션 (Orchestration)

## 1. 개요
데이터 파이프라인이 복잡해짐에 따라 수백 개의 작업(Task) 간의 선후행 관계(의존성), 성공/실패 시 재시도(Retry), 스케줄링 등을 체계적으로 관리해야 합니다. 이를 수행하는 도구가 **워크플로우 오케스트레이터**입니다.

## 2. Apache Airflow
현재 업계 표준으로 자리 잡은 오픈소스 오케스트레이션 플랫폼입니다.

### 핵심 개념: DAG (Directed Acyclic Graph)
Airflow는 작업의 흐름을 **방향성 비순환 그래프(DAG)**로 표현합니다.
- **Directed (방향성)**: 작업 간의 순서가 명확해야 함.
- **Acyclic (비순환)**: 루프(Cycle)가 없어야 하며, 무한 반복을 방지함.
- **Workflow as Code**: 모든 워크플로우를 Python 코드로 작성하여 버전 관리(Git)와 테스트가 가능합니다.

### Airflow DAG 작성 예시 (Python)
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def extract():
    # 데이터 추출 로직
    pass

default_args = {
    "owner": "data_eng",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_pipeline_v1",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    task_extract = PythonOperator(task_id="extract", python_callable=extract)
    # ... 추가 태스크 정의 및 연결
```

## 3. 안정적인 파이프라인을 위한 설계 원칙
- **멱등성 (Idempotency)**: 동일한 데이터를 대상으로 파이프라인을 여러 번 실행해도 결과가 항상 같아야 합니다. (장애 복구의 핵심)
- **원자성 (Atomicity)**: 하나의 작업은 완전히 성공하거나 완전히 실패해야 하며, 어설픈 중간 상태를 남겨선 안 됩니다.
- **단일 책임 원칙**: 하나의 Task는 하나의 논리적 작업(예: 데이터 추출만, 또는 변환만)만 수행해야 합니다.

## 4. 대안 도구 비교
- **Dagster**: 데이터 자산(Data Assets) 중심의 설계를 강조하며, 로컬 테스트가 매우 용이함.
- **Prefect**: 동적 워크플로우에 강점이 있으며, 설정보다 코드(Pythonic)를 중시함.
- **Mage.ai**: 저코드(Low-code) UI와 노트북 환경을 결합하여 빠른 개발 지원.
