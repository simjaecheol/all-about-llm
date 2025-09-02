# OpenAI Tool Call 실습 예제
# 작성일: 2025년 8월 24일
# 실제 동작하는 완전한 코드 예제

import json
import os
from datetime import datetime

from openai import OpenAI

# ========================================
# 1. 환경 설정
# ========================================

# OpenAI API 키 설정 (환경변수 사용 권장)
# export OPENAI_API_KEY="your-api-key-here" 또는 .env 파일 사용
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 실제 사용 시 API 키 설정 필요

# ========================================
# 2. Tool Functions 정의
# ========================================


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """현재 날씨 정보를 반환하는 시뮬레이션 함수"""
    # 실제 구현에서는 날씨 API (OpenWeatherMap 등) 사용
    weather_data = {
        "seoul": {"temperature": 15, "condition": "맑음", "humidity": 60},
        "busan": {"temperature": 18, "condition": "흐림", "humidity": 70},
        "tokyo": {"temperature": 12, "condition": "비", "humidity": 80},
        "new york": {"temperature": 8, "condition": "눈", "humidity": 45},
        "london": {"temperature": 5, "condition": "안개", "humidity": 85},
    }

    location_lower = location.lower()
    for city in weather_data:
        if city in location_lower:
            data = weather_data[city]
            return json.dumps(
                {
                    "location": location,
                    "temperature": data["temperature"],
                    "unit": unit,
                    "condition": data["condition"],
                    "humidity": data["humidity"],
                },
                ensure_ascii=False,
            )

    return json.dumps(
        {"location": location, "error": "해당 지역의 날씨 정보를 찾을 수 없습니다."},
        ensure_ascii=False,
    )


def calculate_expression(expression: str) -> str:
    """수학 표현식을 계산하는 함수 (안전한 계산)"""
    try:
        # 보안을 위해 허용된 문자만 사용
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return json.dumps(
                {
                    "expression": expression,
                    "error": "허용되지 않은 문자가 포함되어 있습니다.",
                },
                ensure_ascii=False,
            )

        # eval 대신 더 안전한 방법 사용 (실제로는 ast.literal_eval 등 권장)
        result = eval(expression)
        return json.dumps(
            {"expression": expression, "result": result}, ensure_ascii=False
        )
    except Exception as e:
        return json.dumps(
            {"expression": expression, "error": f"계산 오류: {str(e)}"},
            ensure_ascii=False,
        )


def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """현재 시간을 반환하는 함수"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return json.dumps(
            {
                "timezone": timezone,
                "current_time": current_time,
                "timestamp": datetime.now().timestamp(),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": f"시간 조회 오류: {str(e)}"}, ensure_ascii=False)


def search_wikipedia(query: str, language: str = "ko") -> str:
    """Wikipedia 검색 함수 (시뮬레이션)"""
    # 실제로는 Wikipedia API 사용
    mock_results = {
        "python": "Python은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어입니다.",
        "ai": "인공지능(AI)은 기계가 인간과 같은 지능적 행동을 보이도록 하는 기술입니다.",
        "openai": "OpenAI는 2015년에 설립된 AI 연구 회사로, GPT 시리즈를 개발했습니다.",
        "서울": "서울특별시는 대한민국의 수도이자 최대 도시입니다.",
    }

    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower or query in key:
            return json.dumps(
                {"query": query, "summary": value, "language": language},
                ensure_ascii=False,
            )

    return json.dumps(
        {
            "query": query,
            "summary": f"'{query}'에 대한 정보를 찾을 수 없습니다.",
            "language": language,
        },
        ensure_ascii=False,
    )


# ========================================
# 3. Tool 스키마 정의
# ========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "특정 지역의 현재 날씨 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "도시 이름 (예: 서울, 부산, 도쿄)",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "온도 단위",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "수학 표현식을 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수학 표현식 (예: '2 + 3 * 4')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 시간을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "시간대 (기본값: Asia/Seoul)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 키워드"},
                    "language": {
                        "type": "string",
                        "enum": ["ko", "en"],
                        "description": "검색 언어 (기본값: ko)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ========================================
# 4. 함수 실행 매핑
# ========================================

available_functions = {
    "get_current_weather": get_current_weather,
    "calculate_expression": calculate_expression,
    "get_current_time": get_current_time,
    "search_wikipedia": search_wikipedia,
}

# ========================================
# 5. Tool Call 실행 함수
# ========================================


def run_conversation(user_message: str):
    """Tool Call을 포함한 대화 실행"""
    print(f"🔵 사용자 입력: {user_message}")
    print("-" * 50)

    # 1단계: 초기 메시지와 도구 목록 전송
    messages = [{"role": "user", "content": user_message}]

    try:
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4",  # 또는 "gpt-3.5-turbo"
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 모델이 자동으로 도구 사용 결정
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        print(f"🤖 모델 응답: {response_message.content}")

        # 2단계: Tool Call 처리
        if response_message.tool_calls:
            print("⚡ Tool Call 실행:")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  📞 함수 호출: {function_name}")
                print(f"  📝 매개변수: {function_args}")

                # 함수 실행
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    print(f"  ✅ 실행 결과: {function_response}")

                    # 함수 실행 결과를 메시지에 추가
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                else:
                    print(f"  ❌ 함수를 찾을 수 없음: {function_name}")

            # 3단계: 함수 실행 결과를 바탕으로 최종 응답 생성
            print("\n🔄 최종 응답 생성 중...")
            final_response = client.chat.completions.create(
                model="gpt-4", messages=messages
            )

            final_content = final_response.choices[0].message.content
            print(f"🎯 최종 응답:\n{final_content}")

        else:
            print("ℹ️  Tool Call이 필요하지 않은 일반 응답입니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("💡 API 키가 설정되어 있는지 확인해주세요.")


# ========================================
# 6. 실습 예제 실행
# ========================================


def main():
    """실습 예제 메인 함수"""
    print("🎯 OpenAI Tool Call 실습 예제")
    print("=" * 50)

    # 예제 실행을 위한 테스트 시나리오
    test_scenarios = [
        "서울의 현재 날씨를 알려주세요",
        "25 * 4 + 10을 계산해주세요",
        "현재 시간이 몇 시인가요?",
        "Python에 대해 설명해주세요",
        "부산 날씨와 현재 시간을 둘 다 알려주세요",  # 멀티 tool call 예제
    ]

    print("API 키가 설정되지 않은 경우 시뮬레이션 모드로 실행됩니다.\n")

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("⚠️  실제 API 호출을 위해서는 OPENAI_API_KEY 환경변수를 설정해주세요.")
        print("📝 시뮬레이션 모드에서 함수 기능만 테스트합니다.\n")

        # 함수 직접 테스트
        print("🔧 함수 직접 테스트:")
        print(f"날씨: {get_current_weather('서울')}")
        print(f"계산: {calculate_expression('25 * 4 + 10')}")
        print(f"시간: {get_current_time()}")
        print(f"검색: {search_wikipedia('Python')}")
        return

    # 실제 API 호출 테스트
    print("🚀 실제 OpenAI API 호출 테스트:\n")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 테스트 {i}/{len(test_scenarios)}")
        run_conversation(scenario)
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
