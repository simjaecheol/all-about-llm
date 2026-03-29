---
layout: default
title: 인프라 배포 및 운영 전략 (Kubernetes & Docker)
parent: AI Transformation 인프라 (AX Infra)
nav_order: 6
---

# 인프라 배포 및 운영 전략 (Kubernetes & Docker)

## 개요

앞서 다룬 5대 에이전틱 인프라(AI 서빙, 벡터 DB, 관측, 오케스트레이션, 실행 샌드박스)의 컴포넌트들은 각각의 역할과 요구 자원(GPU, 메모리, 스토리지)이 다릅니다. 이들을 안정적으로 연결하고 언제든 스케일업/오토스케일링을 수행할 수 있도록 엔터프라이즈 환경에서는 **Docker 컨테이너**와 **Kubernetes (Helm Chart)** 기반의 클라우드 네이티브 배포 전략이 필수적입니다.

---

## 1. 초기/소규모 배포 전략: Docker Compose

PoC 직후, 빠르게 팀 레벨에서 자체 호스팅(Self-hosting) 인프라를 통합 관리하려면 `docker-compose.yml`을 기반으로 한 단일 노드(혹은 소규모 단일 서버) 셋업이 효과적입니다.

### 장점
- 설정 파일 하나로 여러 인프라(예: Langfuse 백엔드 + PostgreSQL + Redis + LiteLLM Proxy)를 묶어서 즉시 배포 가능.
- 초기 구축 시간 절약. 네트워크 브릿지(Bridge Network)를 통해 내부 컨테이너 통신 용이.

### 단점
- 단일 서버의 한계로 인해, vLLM을 통한 대규모 추론 트래픽이 몰릴 경우 시스템의 오토스케일링이 사실상 불가능함.
- 장애 대응(Failover) 로직이 없어, 서버 자체가 다운되면 에이전트 인프라 전체가 붕괴됨.

---

## 2. 프로덕션 환경의 표준: Kubernetes (K8s) & Helm Chart

LLM의 무거운 연산 비용(GPU)과 다중 에이전트 노선의 복잡도를 모두 제어하려면 결국 마이크로서비스 아키텍처(MSA) 관점의 Kubernetes 클러스터 운영이 불가피합니다. 대부분의 에이전트 인프라 프로젝트들은 프로덕션 관리를 위한 **공식 Helm Chart**를 제공하고 있습니다.

### 인프라별 K8s/Helm Chart 관리 특징

#### A. 인퍼런스 서빙 (vLLM / LiteLLM)
- **vLLM Deployment**: 추론 처리량 유지를 위해 적절한 노드 톨러레이션(Toleration)과 nodeSelector 등을 K8s에 설정하여, 트래픽 폭주 시 GPU가 할당된 Node로 파드(Pod)를 동적 스케일아웃(Scale-out).
- **LiteLLM**: LiteLLM 마스터 노드를 K8s 상에서 다중 파드로 확장하여 엔드포인트 요청을 안전하게 라우팅함.

#### B. 메모리 및 벡터 데이터베이스 (Milvus / Qdrant)
- 대규모 벡터 DB는 State(영속성)가 매우 중요하므로 StatefulSet 및 분산 스토리지(Persistent Volumes) 정책과 함께 배포됨.
- **Milvus Helm Chart**: Proxy 노드, 쿼리 노드(Query Node), 데이터 노드(Data Node) 등을 각각 분리하여 독립적으로 오토스케일링할 수 있도록 매우 큰 규모의 분산 컴포넌트를 Helm으로 한 번에 관리함.

#### C. 관측성 시스템 (Langfuse)
- 애플리케이션 로그, 에이전트 Span 파싱 등으로 인해 쏟아지는 막대한 쓰기 부하를 감당하기 위해 DB(PostgreSQL), 메모리 캐시(Redis), 백엔드 서버 등을 Helm 밸류값 수정으로 세밀하게 분산 배포.

#### D. 오케스트레이션 엔진 (Temporal)
- 워크플로우 이벤트들을 무결성 있게 기억하고 실행하기 위한 `Temporal Web`, `Frontend`, `History`, `Matching` 서비스 등을 Helm으로 구성.
- 각 구성요소(Workers 포함)를 필요에 맞춰 파드 리소스를 튜닝(HPA 설정)하는 것이 프로덕션의 핵심.

---

## 3. 대표적인 인프라스트럭처 레시피 (Infrastructure as Code)

성공적인 배포를 위해서는 개별적인 CLI 명령이 아니라 IaC(Infrastructure as Code) 기반 접근이 요구됩니다.
- Terraform과 Helm을 결합하여, `terraform apply` 단 한 번으로 클라우드 인프라(AWS EKS 등) 생성부터 `LiteLLM+Langfuse+Milvus` K8s 패키지 로드까지 완결되게 구축.
- **GitOps (ArgoCD / Flux)** 연동을 통해 쿠버네티스 인프라를 Git으로 영구 추적하고 장애 시 롤백 프로세스를 자동화.
