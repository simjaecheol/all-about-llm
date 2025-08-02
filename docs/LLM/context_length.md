---
title: Context Length
parent: LLM이란 무엇인가?
nav_order: 7
---

# Context Length

## 개요

Context Length(컨텍스트 길이)는 LLM이 한 번에 처리할 수 있는 입력 텍스트의 최대 길이를 의미합니다. 이는 모델의 성능과 사용 가능한 메모리에 직접적인 영향을 미치는 핵심 제약사항입니다.

## Context Length vs Max Tokens

### 1. 개념적 차이

**Context Length:**
- **정의**: 모델이 입력으로 받을 수 있는 최대 토큰 수
- **범위**: 입력 프롬프트 + 시스템 메시지 + 대화 히스토리
- **제약**: 모델 아키텍처에 의해 결정되는 하드 리미트

**Max Tokens:**
- **정의**: 모델이 생성할 수 있는 최대 토큰 수
- **범위**: 출력 텍스트만 해당
- **제약**: Context Length 내에서 설정 가능한 소프트 리미트

```python
# Context Length와 Max Tokens의 관계
def calculate_token_usage(prompt, max_tokens, context_length=4096):
    """토큰 사용량 계산"""
    prompt_tokens = count_tokens(prompt)
    available_tokens = context_length - prompt_tokens
    
    # Max Tokens는 사용 가능한 토큰 수를 초과할 수 없음
    actual_max_tokens = min(max_tokens, available_tokens)
    
    return {
        "prompt_tokens": prompt_tokens,
        "available_tokens": available_tokens,
        "max_tokens": actual_max_tokens,
        "total_tokens": prompt_tokens + actual_max_tokens
    }

# 예시
prompt = "인공지능에 대해 설명해주세요."
result = calculate_token_usage(prompt, max_tokens=1000, context_length=4096)
# 결과: {"prompt_tokens": 15, "available_tokens": 4081, "max_tokens": 1000, "total_tokens": 1015}
```

### 2. 실제 사용 예시

```python
# Context Length 제한 확인
def check_context_limits(prompt, context_length=4096):
    """컨텍스트 길이 제한 확인"""
    prompt_tokens = count_tokens(prompt)
    
    if prompt_tokens > context_length:
        # 프롬프트가 너무 긴 경우 처리
        truncated_prompt = truncate_prompt(prompt, context_length)
        return {
            "status": "truncated",
            "original_tokens": prompt_tokens,
            "truncated_tokens": count_tokens(truncated_prompt),
            "truncated_prompt": truncated_prompt
        }
    else:
        return {
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "available_tokens": context_length - prompt_tokens
        }

def truncate_prompt(prompt, max_tokens):
    """프롬프트를 토큰 제한에 맞게 자르기"""
    tokens = tokenize(prompt)
    if len(tokens) > max_tokens:
        # 뒤에서부터 자르기 (최신 정보 유지)
        truncated_tokens = tokens[-max_tokens:]
        return detokenize(truncated_tokens)
    return prompt
```

## 주요 모델별 Context Length (2025년 기준)

### 1. 초대형 컨텍스트 모델 (1M+ 토큰)

| 모델 | Context Length | 특징 |
|------|----------------|------|
| **Magic.dev LTM-2-Mini** | 100,000,000 tokens | 현재 가장 큰 컨텍스트 윈도우 |
| **Meta Llama 4 Scout** | 10,000,000 tokens | 단일 GPU에서 실행 가능 |
| **Meta Llama 4 Maverick** | 1,000,000 tokens | 멀티모달 기능 지원 |
| **OpenAI GPT-4.1 (전체 시리즈)** | 1,000,000 tokens | API에서만 제공, ChatGPT는 32K 제한 |
| **Google Gemini 2.5 Pro** | 1,000,000 tokens | 최대 64,000 출력 토큰 |
| **Google Gemini 1.5 Pro** | 1,000,000 tokens | 실험적으로 2,000,000 토큰까지 |
| **Google Gemini 2.0 Pro** | 1,000,000 tokens | 멀티모달 지원 |
| **xAI Grok 3** | 1,000,000 tokens | 최신 모델 |

### 2. 대형 컨텍스트 모델 (128K-500K 토큰)

| 모델 | Context Length | 특징 |
|------|----------------|------|
| **OpenAI o3/o3-mini** | 200,000 tokens | 추론 모델 |
| **OpenAI GPT-4o** | 128,000 tokens | 멀티모달 지원 |
| **Meta Llama 3.1 (8B/70B/405B)** | 128,000 tokens | Llama 3 대비 16배 증가 |
| **Anthropic Claude Enterprise** | 500,000 tokens | 엔터프라이즈 전용 |
| **Anthropic Claude 4 (Enterprise)** | 500,000 tokens | 고성능 엔터프라이즈 |
| **Anthropic Claude 3.7 Sonnet** | 200,000 tokens | 최신 Sonnet 모델 |
| **Anthropic Claude 3.5 Sonnet** | 200,000 tokens | 안정적인 성능 |
| **DeepSeek-V3** | 128,000 tokens | 32,768 출력 토큰 |
| **DeepSeek-R1** | 128,000 tokens | 최대 64K 출력 |
| **Mistral Large 2** | 128,000 tokens | 최신 대형 모델 |
| **Mistral Small 3.1** | 128,000 tokens | Apache 2.0 라이선스 |

### 3. 중형 컨텍스트 모델 (8K-32K 토큰)

| 모델 | Context Length | 특징 |
|------|----------------|------|
| **Meta Llama 3** | 8,000 tokens | 기본 모델 |
| **Mistral 7B v0.2** | 8,192 tokens | 오픈소스 모델 |
| **DeepSeek (레거시)** | 32,000 tokens | 이전 버전 |
| **Mistral 7B v0.1** | 4,096 tokens | 슬라이딩 윈도우 |

### 4. 소형 컨텍스트 모델 (1K-4K 토큰)

| 모델 | Context Length | 특징 |
|------|----------------|------|
| **Google Gemini Nano** | 1,024 tokens | 온디바이스 모델 |
| **Meta Llama 2** | 4,096 tokens | 오픈소스 모델 |
| **Mistral 7B v0.1** | 4,096 tokens | 슬라이딩 윈도우 |

### 2. 컨텍스트 윈도우 크기별 활용 사례

**초대형 컨텍스트 (1M+ 토큰):**
- **1M 토큰 = 약 750,000 단어 = 약 1,500페이지**
- 전체 코드베이스 분석 (50,000줄 코드)
- 여러 권의 소설 동시 처리
- 장시간 대화 기록 유지
- 대규모 문서 분석

**대형 컨텍스트 (128K-500K 토큰):**
- **128K 토큰 = 약 96,000 단어 = 약 300페이지**
- 긴 기술 문서 분석
- 법률 계약서 검토
- 연구 논문 요약
- 멀티턴 대화 유지

**중형 컨텍스트 (8K-32K 토큰):**
- 일반적인 대화형 AI 서비스
- 짧은 문서 요약
- 코드 생성 및 디버깅

**소형 컨텍스트 (1K-4K 토큰):**
- 온디바이스 AI 서비스
- 실시간 대화
- 간단한 텍스트 처리

### 3. 토큰 수 계산 예시

```python
def estimate_context_usage(text_type, content):
    """컨텍스트 사용량 추정"""
    estimates = {
        "short_conversation": {
            "description": "짧은 대화",
            "tokens_per_message": 50,
            "example": "안녕하세요, 날씨가 좋네요."
        },
        "medium_article": {
            "description": "중간 길이 기사",
            "tokens_per_paragraph": 150,
            "example": "인공지능 기술의 발전으로..."
        },
        "long_document": {
            "description": "긴 문서",
            "tokens_per_page": 500,
            "example": "기술 보고서나 논문"
        },
        "mega_document": {
            "description": "초대형 문서",
            "tokens_per_page": 500,
            "example": "전체 코드베이스, 여러 권의 책"
        }
    }
    
    return estimates[text_type]

# 실제 사용 예시
conversation = [
    "안녕하세요!",
    "오늘 날씨가 정말 좋네요.",
    "인공지능에 대해 궁금한 것이 있어요.",
    "GPT 모델이 어떻게 작동하는지 설명해주세요."
]

total_tokens = sum([count_tokens(msg) for msg in conversation])
print(f"대화 토큰 수: {total_tokens}")
```

## Context Length 최적화 전략

### 1. 프롬프트 압축

```python
def compress_prompt(prompt, target_tokens=3000):
    """프롬프트 압축"""
    current_tokens = count_tokens(prompt)
    
    if current_tokens <= target_tokens:
        return prompt
    
    # 1. 불필요한 공백 제거
    compressed = re.sub(r'\s+', ' ', prompt)
    
    # 2. 반복되는 내용 제거
    compressed = remove_redundant_content(compressed)
    
    # 3. 핵심 정보만 추출
    if count_tokens(compressed) > target_tokens:
        compressed = extract_key_points(compressed, target_tokens)
    
    return compressed

def extract_key_points(text, max_tokens):
    """핵심 포인트만 추출"""
    sentences = split_into_sentences(text)
    key_sentences = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens <= max_tokens:
            key_sentences.append(sentence)
            current_tokens += sentence_tokens
        else:
            break
    
    return ' '.join(key_sentences)
```

### 2. 대화 히스토리 관리

```python
class ConversationManager:
    def __init__(self, max_context_length=4096):
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.system_message = ""
    
    def add_message(self, role, content):
        """메시지 추가"""
        message = {"role": role, "content": content}
        self.conversation_history.append(message)
        
        # 컨텍스트 길이 확인
        self._check_context_length()
    
    def _check_context_length(self):
        """컨텍스트 길이 확인 및 조정"""
        total_tokens = self._calculate_total_tokens()
        
        if total_tokens > self.max_context_length:
            # 가장 오래된 메시지부터 제거
            while total_tokens > self.max_context_length and len(self.conversation_history) > 1:
                removed = self.conversation_history.pop(1)  # 시스템 메시지 제외
                total_tokens = self._calculate_total_tokens()
    
    def _calculate_total_tokens(self):
        """전체 토큰 수 계산"""
        all_content = self.system_message + " " + " ".join([
            msg["content"] for msg in self.conversation_history
        ])
        return count_tokens(all_content)
    
    def get_conversation(self):
        """현재 대화 반환"""
        return [{"role": "system", "content": self.system_message}] + self.conversation_history

# 사용 예시
manager = ConversationManager(max_context_length=4096)
manager.system_message = "당신은 도움이 되는 AI 어시스턴트입니다."

manager.add_message("user", "안녕하세요!")
manager.add_message("assistant", "안녕하세요! 무엇을 도와드릴까요?")
manager.add_message("user", "인공지능에 대해 설명해주세요.")

conversation = manager.get_conversation()
```

### 3. 청크 단위 처리

```python
def process_long_document(document, chunk_size=3000, overlap=200):
    """긴 문서를 청크 단위로 처리"""
    chunks = []
    
    # 문서를 청크로 분할
    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        chunks.append(chunk)
    
    results = []
    for i, chunk in enumerate(chunks):
        prompt = f"다음 문서의 {i+1}번째 부분을 요약해주세요:\n\n{chunk}"
        
        # 각 청크 처리
        result = process_chunk(prompt)
        results.append(result)
    
    # 결과 통합
    return combine_results(results)

def process_chunk(prompt):
    """개별 청크 처리"""
    # 컨텍스트 길이 확인
    if count_tokens(prompt) > 4000:  # 안전 마진
        prompt = truncate_prompt(prompt, 4000)
    
    return generate_response(prompt)
```

## 실무 적용 예시

### 1. 문서 요약 시스템

```python
def summarize_long_document(document, max_context_length=4096):
    """긴 문서 요약 시스템"""
    
    # 1. 문서 길이 확인
    doc_tokens = count_tokens(document)
    
    if doc_tokens <= max_context_length * 0.8:  # 80% 이하
        # 한 번에 처리 가능
        prompt = f"다음 문서를 요약해주세요:\n\n{document}"
        return generate_summary(prompt)
    
    else:
        # 청크 단위로 처리
        return process_in_chunks(document, max_context_length)

def process_in_chunks(document, max_length):
    """청크 단위 처리"""
    chunks = split_document_into_chunks(document, max_length * 0.7)
    summaries = []
    
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    
    # 요약들을 다시 요약
    combined_summary = " ".join(summaries)
    if count_tokens(combined_summary) > max_length * 0.8:
        return summarize_long_document(combined_summary, max_length)
    
    return combined_summary
```

### 2. 대화형 챗봇

```python
class SmartChatbot:
    def __init__(self, max_context_length=4096):
        self.max_context_length = max_context_length
        self.conversation_manager = ConversationManager(max_context_length)
        self.memory_bank = []  # 중요 정보 저장
    
    def respond(self, user_message):
        """사용자 메시지에 응답"""
        
        # 1. 컨텍스트 길이 확인
        current_tokens = self._get_current_tokens()
        
        # 2. 컨텍스트가 너무 길면 압축
        if current_tokens > self.max_context_length * 0.9:
            self._compress_conversation()
        
        # 3. 메시지 추가
        self.conversation_manager.add_message("user", user_message)
        
        # 4. 응답 생성
        response = self._generate_response()
        self.conversation_manager.add_message("assistant", response)
        
        return response
    
    def _compress_conversation(self):
        """대화 압축"""
        # 중요 정보 추출
        important_info = self._extract_important_info()
        
        # 대화 히스토리 초기화
        self.conversation_manager.conversation_history = []
        
        # 중요 정보를 시스템 메시지에 추가
        self.conversation_manager.system_message += f"\n\n중요 정보: {important_info}"
    
    def _extract_important_info(self):
        """중요 정보 추출"""
        # 간단한 키워드 추출 예시
        keywords = ["이름", "선호도", "목표", "문제"]
        extracted_info = []
        
        for msg in self.conversation_manager.conversation_history:
            for keyword in keywords:
                if keyword in msg["content"]:
                    extracted_info.append(f"{keyword}: {msg['content']}")
        
        return "; ".join(extracted_info[:3])  # 상위 3개만
```

### 3. 코드 리뷰 시스템

```python
def review_code_with_context(code, context_length=4096):
    """컨텍스트를 고려한 코드 리뷰"""
    
    # 코드 토큰 수 계산
    code_tokens = count_tokens(code)
    
    if code_tokens > context_length * 0.7:
        # 긴 코드는 부분별로 리뷰
        return review_code_in_parts(code, context_length)
    else:
        # 한 번에 리뷰
        prompt = f"다음 코드를 리뷰해주세요:\n\n{code}"
        return generate_code_review(prompt)

def review_code_in_parts(code, max_length):
    """코드를 부분별로 리뷰"""
    functions = extract_functions(code)
    reviews = []
    
    for func in functions:
        if count_tokens(func) <= max_length * 0.6:
            review = review_single_function(func)
            reviews.append(review)
        else:
            # 함수가 너무 길면 더 작은 단위로 분할
            sub_reviews = review_large_function(func, max_length)
            reviews.extend(sub_reviews)
    
    return combine_code_reviews(reviews)
```

## 성능 최적화 팁

### 1. 토큰 사용량 모니터링

```python
def monitor_token_usage(prompt, response, context_length=4096):
    """토큰 사용량 모니터링"""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
    total_tokens = prompt_tokens + response_tokens
    
    usage_percentage = (total_tokens / context_length) * 100
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "usage_percentage": usage_percentage,
        "remaining_tokens": context_length - total_tokens,
        "efficiency": "good" if usage_percentage < 80 else "warning"
    }
```

### 2. 동적 컨텍스트 관리

```python
def adaptive_context_management(prompt, target_response_length=500):
    """적응적 컨텍스트 관리"""
    
    # 예상 응답 길이에 따른 컨텍스트 조정
    estimated_response_tokens = target_response_length * 1.5  # 안전 마진
    
    available_tokens = 4096 - estimated_response_tokens
    max_prompt_tokens = available_tokens * 0.8  # 80% 사용
    
    if count_tokens(prompt) > max_prompt_tokens:
        # 프롬프트 압축
        compressed_prompt = compress_prompt(prompt, max_prompt_tokens)
        return compressed_prompt
    
    return prompt
```

## 주요 제한사항 및 고려사항

### 1. ChatGPT vs API 차이

```python
def check_api_vs_chatgpt_limits():
    """API와 ChatGPT의 컨텍스트 제한 차이"""
    limits = {
        "gpt-4.1": {
            "api_context_length": 1_000_000,
            "chatgpt_context_length": 32_768,
            "difference": "API에서만 전체 컨텍스트 윈도우 활용 가능"
        },
        "claude": {
            "api_context_length": 500_000,
            "chatgpt_context_length": 200_000,
            "difference": "엔터프라이즈 버전에서 더 큰 컨텍스트"
        }
    }
    return limits
```

### 2. 출력 토큰 제한

```python
def get_output_token_limits():
    """모델별 출력 토큰 제한"""
    output_limits = {
        "gemini_2.5_pro": {
            "context_length": 1_000_000,
            "output_tokens": 64_000,
            "ratio": "6.4%"
        },
        "o3_models": {
            "context_length": 200_000,
            "output_tokens": 100_000,
            "ratio": "50%"
        },
        "deepseek_v3": {
            "context_length": 128_000,
            "output_tokens": 32_768,
            "ratio": "25.6%"
        }
    }
    return output_limits
```

### 3. 비용 고려사항

```python
def calculate_context_cost(context_length, model="gemini_2.5_pro"):
    """컨텍스트 길이에 따른 비용 계산"""
    
    # 모델별 1M 토큰당 비용 (USD)
    costs_per_1m_tokens = {
        "gemini_2.5_pro": 1.25,
        "gpt-4.1": 2.50,
        "claude_enterprise": 3.00,
        "o3_models": 2.00
    }
    
    cost_per_1m = costs_per_1m_tokens.get(model, 2.00)
    total_cost = (context_length / 1_000_000) * cost_per_1m
    
    return {
        "context_length": context_length,
        "cost_per_1m_tokens": cost_per_1m,
        "total_cost": total_cost,
        "cost_efficiency": "gemini_2.5_pro가 가장 경제적"
    }
```

## 2025년 트렌드

### 1. 컨텍스트 윈도우 확장 경쟁

- **2024년 초**: 8K-32K 토큰이 표준
- **2025년**: 1M 토큰이 새로운 기준
- **Magic.dev**: 1억 토큰 모델로 새로운 한계 제시

### 2. 멀티모달 통합

```python
def multimodal_context_usage():
    """멀티모달 컨텍스트 활용"""
    capabilities = {
        "text": "긴 문서, 코드, 대화",
        "image": "이미지 분석, OCR, 시각적 이해",
        "audio": "음성 인식, 음성 합성",
        "video": "동영상 분석, 프레임별 처리"
    }
    return capabilities
```

### 3. 온디바이스 vs 클라우드

```python
def device_vs_cloud_context():
    """온디바이스 vs 클라우드 컨텍스트 비교"""
    comparison = {
        "cloud_models": {
            "context_length": "1M+ tokens",
            "examples": ["Gemini 2.5 Pro", "GPT-4.1", "Claude Enterprise"],
            "use_cases": "대규모 문서 분석, 전체 코드베이스 처리"
        },
        "on_device_models": {
            "context_length": "1K-8K tokens",
            "examples": ["Gemini Nano", "Llama 4 Scout"],
            "use_cases": "실시간 대화, 개인정보 보호가 중요한 경우"
        }
    }
    return comparison
```

## 결론

Context Length는 LLM 사용에서 가장 중요한 제약사항 중 하나입니다. 2025년 현재, 컨텍스트 윈도우의 크기가 AI 애플리케이션의 활용 범위를 크게 넓히고 있으며, 특히 기업용 문서 분석, 코드 리뷰, 장기간 대화 시스템 등에서 혁신적인 변화를 가져오고 있습니다.

### 핵심 포인트

1. **Context Length vs Max Tokens**: 입력 제한 vs 출력 제한의 차이
2. **모델별 차이**: 2025년 기준 1M 토큰이 새로운 표준
3. **최적화 전략**: 압축, 청크 처리, 히스토리 관리
4. **실무 적용**: 문서 요약, 챗봇, 코드 리뷰 등
5. **성능 모니터링**: 토큰 사용량 추적과 적응적 관리
6. **비용 효율성**: Gemini 2.5 Pro가 가장 경제적
7. **멀티모달 통합**: 텍스트, 이미지, 오디오, 비디오 동시 처리

이러한 이해를 바탕으로 컨텍스트 길이 제한을 효과적으로 관리하여 LLM의 성능을 최적화할 수 있습니다.
