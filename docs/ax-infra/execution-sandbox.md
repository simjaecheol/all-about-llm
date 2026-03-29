---
layout: default
title: 실행 샌드박스 (Execution Sandbox)
parent: AI Transformation 인프라 (AX Infra)
nav_order: 5
---

# 실행 샌드박스 (Execution Sandbox)

## 개요

에이전트는 프롬프트를 넘어서 '도구(Tool)'를 사용하여 실제로 시스템에 영향을 미칠 수 있습니다. 특히 에이전트가 데이터 분석을 위해 **직접 코드를 작성하고 실행**하거나(Code Interpreter), 사용자의 시스템 커맨드를 직접 제어하게 될 때, 기업 입장에서는 심각한 보안 및 시스템 파괴 위협에 노출됩니다.

따라서 에이전트의 코드가 프로덕션 코드나 중요 DB를 건드리지 않도록, 에이전트 전용의 격리되고 일회용인 물리/논리적 영역인 **실행 샌드박스(Execution Sandbox)** 인프라 구축이 반드시 필요합니다.

---

## 1. 샌드박싱의 핵심 목표

1. **격리 (Isolation)**: AI가 작성한 악성 코드나 버그가 스크립트가 실행되는 호스트(Host) 시스템의 파일이나 네트워크로 탈출하지 못하게 차단.
2. **제어 (Control)**: CPU, 메모리 자원에 제한을 두어 무한 루프(Infinite Loop)나 폭주(Out of Memory) 시 강제 종료.
3. **일회성 환경 (Ephemeral/Stateless)**: 에이전트 단위 도구 실행이 끝난 후, 컨테이너나 VM 환경을 즉시 파기.

---

## 2. 대표적인 샌드박싱 스택

자체 구축(Self-hosted)하는 인프라 방식과 SaaS 형태의 API를 사용하는 방식이 있습니다.

### [E2B (클라우드 샌드박스)](https://e2b.dev/)
- AI 에이전트를 위해 특별히 고안된(Agent-Native) 클라우드 환경 실행 API.
- 모델이 생성한 Python 코드를 가상 머신(Firecracker microVM 기반) 위에서 안전하게 실행시키고 반환값만 돌려줌. "AI 개발자/데이터분석가" 에이전트를 만들 때 표준화된 인프라로 자리잡고 있음.

### [Docker 기반 자가 호스팅](../agent/index.md) (Self-Hosted Contariner)
- 전통적이고 가장 보편적인 방법.
- 사용자의 프롬프트 요청 시 백그라운드에서 임시 Docker Container를 생성(`docker run`)하여, 컨테이너 안에서만 툴 코드를 구동.
- 관리가 복잡할 수 있으나 보안 통제가 중요한 Enterprise 온프레미스 환경에서 주로 사용.

### [WebAssembly (WASM)](https://webassembly.org/)
- 브라우저나 에지(Edge) 인프라 위에서 구동되는 고도의 경량 격리 기술.
- Docker 대비 구동 속도(밀리초 단위)가 압도적으로 빠르며, 로컬 환경에서 에이전트를 구동할 때 호스트 OS를 보호하는 가장 강력한 보안 기반(Boundary) 메커니즘 제공.
