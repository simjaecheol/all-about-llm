---
layout: default
title: Office 및 웹 문서 파싱
parent: 문서 파싱 (Document Parsing)
nav_order: 2
---

# Office 문서 및 웹(HTML) 파싱

## 1. Office 문서 파싱 (Word, Excel, PowerPoint)

기업용 데이터를 다룰 때 가장 흔하게 접하는 형식입니다. Office 파일들은 내부적으로 압축된 XML 기반 형식(.docx, .xlsx, .pptx)을 가지므로 전용 라이브러리로 접근하는 것이 유리합니다.

- **Word (.docx)**: `python-docx`
  - 단락(Paragraph)과 표(Table) 개체를 분류하여 파싱할 수 있습니다.
  - 굵은 글씨, 글자 크기나 헤딩 정보 같은 스타일 메타데이터를 활용하여 문서의 논리적 덩어리(Chunk) 경계를 설정하면 유리합니다.
- **Excel (.xlsx, .csv)**: `pandas`, `openpyxl`
  - 2차원 표 데이터를 다루는 것이 핵심입니다. 셀들을 단순히 한 줄 텍스트로 이어붙이면 LLM이 열(Column)과 행(Row)의 논리적 관계를 잃어버리므로, Markdown Table, CSV, 또는 JSON 포맷의 문자열로 직렬화(Serialization)하여 LLM에 제공하는 것이 강력히 권장됩니다.
- **PowerPoint (.pptx)**: `python-pptx`
  - 각 슬라이드별로 텍스트 상자, 도형 내 텍스트, 화자 노트를 개별적으로 추출할 수 있습니다. 문서의 분할(Chunking)을 슬라이드 단위로 진행하는 것이 가장 자연스럽고 문맥 보존에 좋습니다.

## 2. 웹 문서 (HTML) 파싱

크롤링을 통한 웹 문서는 레이아웃과 네비게이션 태그가 너무 많아 노이즈가 심하므로, 본문 텍스트만 깨끗하게 걷어내는 전처리 과정이 필수입니다.

- **BeautifulSoup**: HTML DOM 트리를 순회하여 광고 스크립트 태그나 주변 네비게이션용 `<nav>`, `<aside>`, `<script>` 등을 선제적으로 제거하고 핵심 `<p>`, `<h1>` 등의 본문만 추출합니다.
- **Markdownify**: HTML 코드를 보기 편한 일반 Markdown으로 직관적으로 변환해 주는 도구입니다. 링크(`<a href>`), 표(`<table>`), 목록 구조를 유지하면서 불필요한 시각적 HTML 요소를 날릴 때 매우 유용합니다.
- **Selenium / Playwright**: 자바스크립트 렌더링이 완전히 수행되어야만 알맹이가 보이는 동적(Dynamic) 웹페이지를 파싱할 때, 브라우저 엔진 기반 렌더링 후 DOM을 추출하기 위해 제한적으로 사용합니다.
