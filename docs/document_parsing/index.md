---
layout: default
---
title: 문서 파싱 (Document Parsing)
has_children: true
nav_order: 60
---
# 문서 파싱 (Document Parsing)

LLM 및 RAG 기반 시스템의 응답 품질은 입력되는 데이터의 질에 크게 의존합니다 (Garbage In, Garbage Out). 따라서 원본 문서를 텍스트로 얼마나 정확하고 일관되게 변환하느냐가 전체 시스템 성능을 좌우하는 핵심 요소입니다.

이 문서 파싱 카테고리에서는 문서의 형식과 특징에 따라 최적의 정보를 추출해 내는 전략을 소개합니다.

## 카테고리 안내

- **[PDF 및 복합 문서 파싱](./pdf_parsing.md)**: 텍스트 및 스캔된 PDF, 표(Table)와 수식을 보존하며 파싱하는 전략
- **[Office 및 웹 문서 파싱](./office_and_web.md)**: Word, PPT, Excel, HTML 문서에서 핵심 정보 추출 기법
- **[문서 내 이미지 데이터 처리](./image_processing.md)**: 문서 속에 포함된 이미지, 차트, 다이어그램을 다루는 방법 (OCR, VLM 캡셔닝, 멀티모달 임베딩)
