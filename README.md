# ALL ABOUT LLM

LLM(Large Language Model)에 대한 모든 것을 다루는 레포입니다.

## 목적

LLM 기술이 발전함에 따라 LLM을 활용한 서비스들이 많이 출시되고 있습니다. 이 레포는 LLM 서비스를 구축하기 위해 알아두면 좋은 지식들을 누구나 쉽게 얻을 수 있도록 돕기 위해 만들어졌습니다.

## 목차

- [LLM](./docs/LLM/)
- [Prompt Engineering](./docs/prompt_engineering/)
- [RAG](./docs/RAG/)
- [Agent](./docs/agent/)
- [Serving](./docs/serving/)
- [Inference](./docs/inference/)
- [Optimization](./docs/optimization/)
- [Training Framework](./docs/training_framework/)
- [Training Methods](./docs/training_methods/)
- [Multi Modal](./docs/multi_modal/)
- [Tool Call](./docs/tool_call/)
- [Context Engineering](./docs/context_engineering/)
- [Open Source Project](./docs/open_source_project/)
- [Reasoning](./docs/reasoning/)

## 개발 환경 설정

### Python 코드 포맷팅 (Black)

이 프로젝트는 [Black](https://black.readthedocs.io/)을 사용하여 Python 코드를 자동으로 포맷팅합니다.

#### 자동 설정 (권장)
```bash
python setup_black.py
```

#### 수동 설정
1. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. pre-commit hooks 설치:
   ```bash
   pre-commit install
   ```

3. VS Code 확장 프로그램 설치:
   - Python
   - Black Formatter
   - isort
   - Flake8

#### 사용법
- **자동 포맷팅**: VS Code에서 파일 저장 시 자동 적용
- **수동 포맷팅**: `black <파일명>` 또는 `black .`
- **Git commit 시**: pre-commit hooks가 자동으로 실행

