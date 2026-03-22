---
title: 개발 가이드
parent: Tool Call
nav_order: 4
---

# 개발자를 위한 실무 적용 가이드

## 🚀 OpenAI Tool Call 실습 가이드

이 실습에서는 OpenAI의 Tool Call(함수 호출) 기능을 실제로 구현하고 테스트해볼 수 있습니다. Tool Call은 LLM이 외부 도구나 함수를 호출해서 더 정확하고 유용한 응답을 생성할 수 있게 하는 핵심 기능입니다.

### 📚 개요

Tool Call은 LLM이 외부 도구나 API와 상호작용하여 자신의 한계를 극복하고 더 정확하고 유용한 응답을 생성할 수 있게 하는 핵심 기능입니다.

### 🚀 빠른 시작

#### 1. 필수 라이브러리 설치

```bash
pip install openai python-dotenv requests
```

#### 2. API 키 설정

**방법 1: 환경변수 설정**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**방법 2: .env 파일 사용**
1. 프로젝트 폴더에 `.env` 파일 생성
2. 다음 내용 입력:
```
OPENAI_API_KEY=your-api-key-here
```

#### 3. API 키 발급 방법
1. [OpenAI Platform](https://platform.openai.com) 접속
2. 로그인 후 API Keys 메뉴로 이동
3. "Create new secret key" 클릭하여 키 생성
4. 생성된 키를 안전한 곳에 보관

### 📁 파일 구성

- `simple-tool-example.py`: 기본적인 Tool Call 예제
- `openai-tool-example.py`: 완전한 실습용 코드 (고급 기능 포함)
- `env-example.txt`: 환경변수 설정 예제

### 🎯 실습 단계

#### 단계 1: 기본 예제 실행

```bash
python simple-tool-example.py
```

이 예제는 다음 기능을 포함합니다:
- 날씨 조회 함수
- 계산 함수
- 기본적인 Tool Call 워크플로우

#### 단계 2: 고급 예제 실행

```bash
python openai-tool-example.py
```

고급 예제의 주요 기능:
- 다양한 도구 함수 (날씨, 계산, 시간, 위키피디아 검색)
- 병렬 Tool Call 처리
- 오류 처리 및 보안 고려사항
- 시뮬레이션 모드 (API 키 없이도 테스트 가능)

### 🔧 핵심 개념 이해

#### Tool Call 워크플로우

1. **사용자 입력**: "서울 날씨를 알려주세요"
2. **모델 분석**: GPT가 날씨 함수가 필요하다고 판단
3. **함수 호출 요청**: 함수명과 매개변수를 JSON으로 반환
4. **함수 실행**: 개발자가 실제 함수를 실행
5. **결과 전달**: 함수 결과를 모델에게 다시 전달
6. **최종 응답**: 모델이 결과를 바탕으로 자연어 응답 생성

#### 함수 스키마 구조

```python
{
    "type": "function",
    "function": {
        "name": "함수명",
        "description": "함수 설명",
        "parameters": {
            "type": "object",
            "properties": {
                "매개변수명": {
                    "type": "string",
                    "description": "매개변수 설명"
                }
            },
            "required": ["필수_매개변수"]
        }
    }
}
```

### 💡 실습 아이디어

#### 기본 실습
1. 새로운 함수 추가 (예: 환율 조회, 뉴스 검색)
2. 함수 매개변수 수정 및 테스트
3. 다양한 사용자 입력으로 테스트

#### 고급 실습
1. 실제 API 연동 (날씨 API, 주식 API 등)
2. 데이터베이스 연동
3. 파일 시스템 조작 함수
4. 웹 스크래핑 함수

### 🛡️ 보안 고려사항

#### API 키 보안
- 환경변수나 .env 파일 사용
- 코드에 API 키 직접 입력 금지
- GitHub 등에 업로드 시 .env 파일 제외

#### 함수 실행 보안
- `eval()` 사용 시 입력 검증 필수
- 허용된 문자/명령어만 실행
- 시스템 명령어 실행 시 권한 제한
- 사용자 입력 검증 및 필터링

### 🐛 문제 해결

#### 자주 발생하는 오류

1. **"Incorrect API key provided"**
   - API 키가 올바르게 설정되었는지 확인
   - 환경변수 또는 .env 파일 확인

2. **"You exceeded your current quota"**
   - OpenAI API 사용량 한도 초과
   - 결제 정보 확인 또는 플랜 업그레이드

3. **함수 실행 오류**
   - 함수 스키마와 실제 함수 매개변수 일치 확인
   - JSON 파싱 오류 확인

### 📖 추가 학습 자료

#### 공식 문서
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

#### 활용 예제
- 챗봇에 실시간 정보 검색 기능 추가
- 업무 자동화 시스템 구축
- 데이터 분석 보조 도구 개발

### 🎓 다음 단계

이 실습을 완료한 후에는:
1. MCP (Model Context Protocol) 학습
2. LangChain과의 통합
3. 프로덕션 환경 배포
4. 모니터링 및 로깅 시스템 구축

---

**실습을 통해 Tool Call의 강력함을 직접 경험해보세요! 🚀**

---

## 🎯 OpenAI Tool Call 실습용 완전 동작 코드 예제

OpenAI Client를 사용한 실제 동작하는 Tool Call 실습 예제를 준비했습니다. 이 예제들은 LLM을 잘 모르는 분들도 쉽게 따라할 수 있도록 구성되었습니다.

### 📁 제공된 파일들

- **`simple-tool-example.py`**: 기본적인 Tool Call 예제
  - 날씨 조회와 계산 함수만 포함하여 초보자도 쉽게 이해할 수 있습니다
  - Tool Call의 핵심 개념을 이해하기 위한 간단한 예제입니다

- **`openai-tool-example.py`**: 완전한 실습용 코드 (고급 기능 포함)
  - 4가지 도구 함수(날씨, 계산, 시간, 위키피디아)를 포함한 고급 예제입니다
  - 병렬 Tool Call 처리, 오류 처리, 보안 고려사항을 모두 포함하고 있으며
  - API 키 없이도 시뮬레이션 모드로 테스트할 수 있습니다

- **`env-example.txt`**: 환경변수 설정 예제
  - API 키를 안전하게 관리하기 위한 .env 파일 예제입니다

### 🚀 빠른 시작 방법

#### 1. 환경 설정
```bash
# 필수 라이브러리 설치
pip install openai python-dotenv requests

# API 키 설정 (둘 중 하나 선택)
export OPENAI_API_KEY="your-api-key-here"
# 또는 .env 파일에 OPENAI_API_KEY=your-api-key-here
```

#### 2. 기본 예제 실행
```bash
python simple-tool-example.py
```

#### 3. 고급 예제 실행  
```bash
python openai-tool-example.py
```

### 🎯 실습 예제의 주요 특징

#### **완전 동작 코드**
- 실제 OpenAI API 호출
- 오류 처리 및 예외 상황 대응
- API 키가 없어도 시뮬레이션 모드로 테스트 가능

#### **다양한 Tool 함수**
- **날씨 조회**: 도시별 날씨 정보 반환
- **수학 계산**: 안전한 수식 계산 (보안 검증 포함)
- **시간 조회**: 현재 시간 및 타임스탬프
- **정보 검색**: 위키피디아 시뮬레이션

#### **실무 적용 가능**
- JSON 스키마 정의 방법
- 병렬 Tool Call 처리
- 함수 매핑 및 실행 로직
- 보안 고려사항 (입력 검증, API 키 보호)

### 💡 학습 포인트

#### **Tool Call 워크플로우 이해**
1. 사용자 입력 → 2. 모델 분석 → 3. 함수 호출 요청 → 4. 함수 실행 → 5. 결과 전달 → 6. 최종 응답

#### **실제 구현 시 고려사항**
- **함수 스키마 설계**: 명확한 이름, 설명, 매개변수 정의
- **오류 처리**: 함수 실행 실패, 타임아웃, 잘못된 입력 처리
- **보안**: `eval()` 사용 시 입력 검증, API 키 보호, 권한 제한

#### **확장 가능성**
- 실제 API 연동 (날씨 API, 주식 API, 뉴스 API)
- 데이터베이스 연동
- 파일 시스템 조작
- 웹 스크래핑 및 자동화

### 🛡️ 현대적 개발 보안 및 관리 기법 (2026)

이 실습 예제들을 통해 Tool Call의 개념부터 실제 구현까지 단계적으로 학습할 수 있으며, 특히 2026년 기준 표준이 된 **타입 안전성(Type-Safety)**과 **에이전틱 거버넌스** 패턴을 익힐 수 있습니다.

#### 1. 타입 안전한 도구 정의 (PydanticAI 활용)

JSON 스키마를 수동으로 작성하는 대신, Python 타입 힌트를 사용하여 런타임 오류를 방지하고 자동 재시도를 활성화합니다.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# 1. 입력 스키마 정의
class WeatherInput(BaseModel):
    city: str = Field(description="도시 이름")
    unit: str = Field(default="celsius", description="온도 단위")

# 2. 에이전트 및 도구 설정
agent = Agent('openai:gpt-4o')

@agent.tool
def get_weather(ctx: RunContext[None], input_data: WeatherInput) -> str:
    # input_data는 Pydantic에 의해 이미 검증된 상태임
    return f"{input_data.city}의 날씨는 20도입니다."
```

#### 2. Human-in-the-loop (HITL) 패턴

고위험 도구(결제, 데이터 삭제, 메일 발송 등)는 실행 전 사람의 승인을 받는 단계를 명시적으로 설계해야 합니다.

- **Checkpoint**: 도구 실행 직전 상태를 저장하고 대기합니다.
- **Review UI**: 사용자에게 에이전트가 생성한 인자와 실행 의도를 명확히 표시합니다.
- **Override**: 사용자가 인자를 직접 수정하거나 실행을 거부할 수 있도록 구현합니다.

#### 3. 검증 기반 자동 재시도 (Validation Retries)

모델이 잘못된 형식의 인자를 생성했을 때, 오류 메시지를 다시 모델에게 전달하여 스스로 수정하게 합니다. PydanticAI와 같은 현대적 프레임워크는 이를 기본적으로 지원합니다.

---

## 🔧 성능 최적화 및 운영 팁

### 효율적인 도구 설계

**On-demand Tool Discovery**:
- 도구가 수백 개인 경우, 모든 스키마를 컨텍스트에 넣지 마세요.
- 사용자 질문과 유사한 도구 명세만 동적으로 로드(Semantic Search)하여 토큰을 절약하세요.

**Parallel Execution Strategy**:
- 독립적인 I/O 작업(여러 도시 날씨 조회 등)은 반드시 병렬로 처리하여 응답 속도를 최적화하세요.

### 모니터링 및 관측 가능성 (Observability)

- **Traces**: 도구 호출 전 '사고(Thinking)' 과정부터 결과 반환까지의 전체 경로를 추적하세요.
- **Cost Analysis**: 도구 호출로 인한 토큰 소비량과 실행 비용을 실시간으로 집계하세요.
- **Success Rate**: 특정 도구의 실패율이 높다면 프롬프트(Description) 또는 스키마를 수정해야 합니다.

## 실제 구현 예시 (현대적 접근)

### Python (PydanticAI)
```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def add(a: int, b: int) -> int:
    return a + b

# 실행 시 타입 검증 및 에러 교정이 자동으로 수행됨
result = agent.run_sync('123 더하기 456은?')
print(result.data)
```

## 테스트 및 디버깅

### 도구 테스트 전략

**단위 테스트**:
- 각 도구 함수가 다양한 입력(Edge cases)에 대해 올바른 출력을 내는지 독립적으로 테스트하세요.

**에이전트 시뮬레이션**:
- 모델이 특정 상황에서 올바른 도구를 선택하는지 시나리오 테스트를 수행하세요.

## 배포 및 운영

### 거버넌스 및 규정 준수 (EU AI Act 등)

- **Audit Logs**: 에이전트가 내린 모든 도구 호출 결정과 그 근거(Reasoning trace)를 불변 로그로 기록하세요.
- **Access Control**: 도구별 실행 권한을 최소화 원칙에 따라 관리하세요.
