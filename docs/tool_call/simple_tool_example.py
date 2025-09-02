# OpenAI Tool Call 실습용 간단한 예제
# 기본적인 function calling 데모

import json

from openai import OpenAI

# 1. OpenAI 클라이언트 초기화
client = OpenAI(api_key="your-api-key-here")  # 실제 사용 시 환경변수나 .env 파일 사용


# 2. 도구 함수 정의
def get_weather(location):
    """날씨 정보를 반환하는 시뮬레이션 함수"""
    weather_info = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
        "대구": "맑음, 16°C",
    }
    return weather_info.get(location, f"{location}의 날씨 정보가 없습니다.")


def calculate(expression):
    """간단한 계산 함수"""
    try:
        result = eval(expression)  # 실제로는 더 안전한 방법 사용 권장
        return f"{expression} = {result}"
    except Exception:
        return "계산할 수 없는 식입니다."


# 3. 함수 매핑
functions_map = {"get_weather": get_weather, "calculate": calculate}

# 4. 도구 스키마 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 날씨를 알려줍니다",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "도시 이름"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 계산을 수행합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수학 표현식",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# 5. 대화 함수
def chat_with_tools(user_input):
    """도구를 사용한 대화 함수"""
    print(f"사용자: {user_input}")

    # 첫 번째 API 호출
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # 도구 호출이 있는 경우
    if message.tool_calls:
        print("🔧 도구 사용 중...")

        # 메시지 히스토리에 추가
        messages = [{"role": "user", "content": user_input}, message]

        # 각 도구 호출 처리
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"  함수: {function_name}")
            print(f"  인자: {arguments}")

            # 함수 실행
            if function_name in functions_map:
                result = functions_map[function_name](**arguments)
                print(f"  결과: {result}")

                # 결과를 메시지에 추가
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    }
                )

        # 두 번째 API 호출 (최종 답변 생성)
        final_response = client.chat.completions.create(
            model="gpt-4", messages=messages
        )

        print(f"AI: {final_response.choices[0].message.content}")

    else:
        print(f"AI: {message.content}")


# 6. 테스트 실행
if __name__ == "__main__":
    print("=== OpenAI Tool Call 기본 예제 ===\n")

    # 테스트 시나리오
    test_cases = [
        "서울 날씨 어때?",
        "10 + 5 * 2를 계산해줘",
        "부산 날씨와 20 - 3을 둘 다 알려줘",
        "안녕하세요!",  # 도구가 필요 없는 일반 대화
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        try:
            chat_with_tools(test)
        except Exception as e:
            print(f"오류: {e}")
            print("API 키를 확인하세요!")
        print("-" * 30)
