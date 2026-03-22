---
layout: default
title: Perplexica
parent: 오픈소스 프로젝트
nav_order: 6
---

# Perplexica

Perplexica는 Perplexity AI의 강력한 오픈소스 대안으로, 개인정보 보호와 사용자 맞춤형 설정을 극대화한 **AI 기반 메타 검색 엔진**입니다.

## 주요 특징

- **익명 검색 (SearXNG 기반)**: 245개 이상의 검색 서비스를 통합하는 SearXNG를 사용하여 사용자의 신원을 노출하지 않고 웹 데이터를 수집합니다.
- **로컬 LLM 지원**: Ollama 등을 통해 Llama 3, Mixtral 등 오픈소스 모델을 로컬에서 구동하여 100% 오프라인/프라이빗 검색 환경을 구축할 수 있습니다.
- **다양한 포커스 모드 (Focus Modes)**:
    - **Academic**: 학술 논문 및 연구 자료 검색 최적화
    - **Writing Assistant**: 웹 검색 없이 글쓰기 보조에 집중
    - **YouTube/Reddit**: 특정 커뮤니티 내의 토론과 영상 정보 추출
    - **Wolfram Alpha**: 복잡한 계산 및 데이터 분석 수행
- **코파일럿 모드 (Copilot Mode)**: 질문을 분석하여 최적의 검색 쿼리를 생성하고, 상위 결과 페이지를 직접 방문하여 깊이 있는 답변과 정확한 출처를 제공합니다.

## 기술적 장점

- **모듈형 아키텍처**: OpenAI, Anthropic, Google Gemini뿐만 아니라 Groq나 로컬 LLM을 자유롭게 선택하여 구성할 수 있습니다.
- **개인정보 보호**: 외부 서버에 검색 기록이 남지 않도록 설계되어 있어 보안이 중요한 연구나 비즈니스 용도로 적합합니다.
- **완전 오픈소스**: MIT 라이선스로 누구나 자유롭게 커스터마이징하고 배포할 수 있습니다.
