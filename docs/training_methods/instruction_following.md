---
title: Instruction Following
parent: LLM 학습 방법
nav_order: 2
---

# Instruction Following

## 개요

Instruction Following(지시사항 따르기)은 사전 학습이 완료된 Foundation Model이 사용자의 지시사항을 이해하고 적절히 수행할 수 있도록 학습하는 단계입니다. 이 단계에서는 모델이 자연어로 된 지시사항을 받아들이고, 해당 지시사항에 맞는 응답을 생성하도록 학습합니다.

## Instruction Following의 중요성

### 1. 사용자 의도 이해
- **명시적 지시사항**: 사용자가 명확히 요청한 작업 수행
- **암묵적 의도**: 사용자의 의도를 파악하여 적절한 응답 생성
- **맥락 이해**: 대화의 맥락을 고려한 응답 생성

### 2. 실용성 향상
- **사용자 친화적**: 복잡한 프롬프트 엔지니어링 없이도 원하는 결과 획득
- **일관성**: 동일한 지시사항에 대해 일관된 응답 제공
- **안전성**: 유해하거나 부적절한 요청에 대한 적절한 대응

## Instruction Following 학습 방법

### 1. Supervised Fine-tuning (SFT)

SFT는 인간이 작성한 고품질 지시사항-응답 쌍을 사용하여 모델을 학습시키는 방법입니다.

#### 데이터 구성
```python
# 지시사항-응답 쌍 예시
instruction_data = [
    {
        "instruction": "다음 텍스트를 요약해주세요:",
        "input": "인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다...",
        "output": "인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다."
    },
    {
        "instruction": "다음 문장의 감정을 분석해주세요:",
        "input": "오늘은 정말 좋은 날씨네요!",
        "output": "긍정적인 감정입니다. 날씨에 대한 만족감이 표현되어 있습니다."
    }
]
```

#### 학습 과정
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def format_instruction_data(instruction, input_text, output):
    """지시사항 데이터 포맷팅"""
    prompt = f"### 지시사항:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n{output}"
    return prompt

def train_sft(model, tokenizer, instruction_data):
    """SFT 학습"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for batch in instruction_data:
        # 데이터 포맷팅
        formatted_text = format_instruction_data(
            batch["instruction"], 
            batch["input"], 
            batch["output"]
        )
        
        # 토큰화
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True)
        labels = inputs["input_ids"].clone()
        
        # 지시사항 부분은 손실 계산에서 제외
        instruction_end = formatted_text.find("### 응답:")
        instruction_tokens = tokenizer(formatted_text[:instruction_end], 
                                     return_tensors="pt")["input_ids"]
        
        # 응답 부분만 손실 계산
        labels[:, :instruction_tokens.shape[1]] = -100
        
        # 순전파
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Prompt Engineering 기법

#### 1) Chain-of-Thought (CoT)
```python
# CoT 프롬프트 예시
cot_prompt = """
다음 문제를 단계별로 생각해보세요:

문제: 15개의 사과가 있고, 3명이 나누어 가지려고 합니다. 각 사람이 몇 개씩 가질 수 있을까요?

단계별 해결:
1. 총 사과 개수: 15개
2. 나누는 사람 수: 3명
3. 각 사람이 가질 수 있는 사과 개수 = 15 ÷ 3 = 5개

답: 각 사람이 5개씩 가질 수 있습니다.
"""
```

#### 2) Few-shot Learning
```python
# Few-shot 프롬프트 예시
few_shot_prompt = """
다음 예시를 참고하여 질문에 답하세요:

예시 1:
질문: 파이썬에서 리스트를 정렬하는 방법은?
답: sort() 메서드나 sorted() 함수를 사용합니다.

예시 2:
질문: 자바스크립트에서 배열을 정렬하는 방법은?
답: sort() 메서드를 사용합니다.

질문: HTML에서 링크를 만드는 태그는?
답: <a> 태그를 사용합니다.
"""
```

#### 3) Role-based Prompting
```python
# 역할 기반 프롬프트
role_prompts = {
    "expert": "당신은 해당 분야의 전문가입니다. 전문적이고 정확한 답변을 제공하세요.",
    "teacher": "당신은 친절한 선생님입니다. 이해하기 쉽게 설명해주세요.",
    "assistant": "당신은 도움이 되는 AI 어시스턴트입니다. 유용한 정보를 제공하세요.",
    "creative": "당신은 창의적인 작가입니다. 독창적이고 흥미로운 답변을 제공하세요."
}
```

### 3. Instruction Following 데이터 구성

Instruction Following Tuning에서 사용되는 데이터는 기존 NLP Task와 근본적으로 다른 구조를 가집니다. 기존에는 단순히 입력-출력 쌍이었다면, Instruction Following에서는 명시적인 지시사항(Instruction)이 추가되어 3요소 구조를 가집니다.

#### 1) 기본 데이터 구조

```python
# 기존 NLP Task 구조
traditional_nlp_data = [
    {
        "input": "Hello world",
        "output": "안녕하세요 세계"
    }
]

# Instruction Following 구조
instruction_following_data = [
    {
        "instruction": "다음 영어 문장을 한국어로 번역해주세요:",
        "input": "Hello world",
        "output": "안녕하세요 세계"
    }
]
```

#### 2) 번역 Task의 Instruction Variance

번역 Task에서 Instruction의 다양한 변형을 통해 Robustness를 높이는 방법:

```python
translation_instruction_variants = [
    # 기본 번역 지시사항
    "다음 영어 문장을 한국어로 번역해주세요:",
    "영어를 한국어로 번역해주세요:",
    "이 문장을 한국어로 바꿔주세요:",
    "한국어로 번역해주세요:",
    
    # 더 구체적인 지시사항
    "다음 영어 텍스트를 자연스러운 한국어로 번역해주세요:",
    "영어 문장을 한국어로 옮겨주세요:",
    "이 영어 표현을 한국어로 바꿔주세요:",
    
    # 문체별 지시사항
    "다음 영어 문장을 공식적인 한국어로 번역해주세요:",
    "영어를 친근한 한국어로 번역해주세요:",
    "이 영어를 비즈니스 한국어로 번역해주세요:",
    
    # 도메인별 지시사항
    "다음 영어 문장을 기술 문서용 한국어로 번역해주세요:",
    "영어를 문학적 한국어로 번역해주세요:",
    "이 영어를 의학용 한국어로 번역해주세요:",
    
    # 품질 관련 지시사항
    "다음 영어를 정확하고 자연스러운 한국어로 번역해주세요:",
    "영어를 의미가 명확한 한국어로 번역해주세요:",
    "이 영어를 읽기 쉬운 한국어로 번역해주세요:"
]

# 실제 데이터 구성 예시
translation_data = []
for instruction in translation_instruction_variants:
    translation_data.extend([
        {
            "instruction": instruction,
            "input": "Hello world",
            "output": "안녕하세요 세계"
        },
        {
            "instruction": instruction,
            "input": "Good morning",
            "output": "좋은 아침입니다"
        },
        {
            "instruction": instruction,
            "input": "Thank you very much",
            "output": "정말 감사합니다"
        }
    ])
```

#### 3) 요약 Task의 Instruction Variance

```python
summarization_instruction_variants = [
    # 기본 요약 지시사항
    "다음 텍스트를 요약해주세요:",
    "이 글의 핵심을 간단히 정리해주세요:",
    "다음 내용을 요약해주세요:",
    
    # 길이별 지시사항
    "다음 텍스트를 한 문장으로 요약해주세요:",
    "이 글을 3문장으로 요약해주세요:",
    "다음 내용을 짧게 요약해주세요:",
    "이 텍스트를 자세히 요약해주세요:",
    
    # 관점별 지시사항
    "다음 기사를 객관적으로 요약해주세요:",
    "이 글을 독자의 관점에서 요약해주세요:",
    "다음 내용을 전문가 관점에서 요약해주세요:",
    
    # 형식별 지시사항
    "다음 텍스트를 글머리 기호로 요약해주세요:",
    "이 내용을 질문-답변 형식으로 요약해주세요:",
    "다음 글을 핵심 키워드로 요약해주세요:",
    
    # 목적별 지시사항
    "다음 기사를 초등학생이 이해할 수 있게 요약해주세요:",
    "이 내용을 비즈니스 요약으로 정리해주세요:",
    "다음 텍스트를 학술 요약으로 작성해주세요:"
]
```

#### 4) 감정 분석 Task의 Instruction Variance

```python
sentiment_analysis_instruction_variants = [
    # 기본 감정 분석
    "다음 문장의 감정을 분석해주세요:",
    "이 텍스트의 감정을 판단해주세요:",
    "다음 내용의 감정을 분석해주세요:",
    
    # 세분화된 감정 분석
    "다음 문장의 감정을 긍정/부정/중립으로 분류해주세요:",
    "이 텍스트의 감정을 1-5점 척도로 평가해주세요:",
    "다음 내용의 감정을 구체적으로 분석해주세요:",
    
    # 감정 유형별 분석
    "다음 문장에서 기쁨, 슬픔, 분노 중 어떤 감정이 나타나는지 분석해주세요:",
    "이 텍스트의 감정을 기쁨/슬픔/분노/놀람/두려움으로 분류해주세요:",
    "다음 내용의 감정을 세밀하게 분석해주세요:",
    
    # 맥락별 분석
    "다음 문장의 감정을 맥락을 고려하여 분석해주세요:",
    "이 텍스트의 감정을 화자의 의도를 고려하여 분석해주세요:",
    "다음 내용의 감정을 상황을 고려하여 분석해주세요:"
]
```

#### 5) 코드 생성 Task의 Instruction Variance

```python
code_generation_instruction_variants = [
    # 기본 코드 생성
    "파이썬으로 피보나치 수열을 계산하는 함수를 작성해주세요:",
    "피보나치 수열을 계산하는 파이썬 함수를 만들어주세요:",
    "파이썬으로 피보나치 함수를 구현해주세요:",
    
    # 상세한 요구사항
    "파이썬으로 피보나치 수열을 계산하는 함수를 작성해주세요. 재귀를 사용하지 말고 반복문을 사용해주세요:",
    "피보나치 수열을 계산하는 파이썬 함수를 만들어주세요. 메모이제이션을 사용해주세요:",
    "파이썬으로 피보나치 함수를 구현해주세요. 제너레이터를 사용해주세요:",
    
    # 성능 관련 지시사항
    "파이썬으로 효율적인 피보나치 수열 계산 함수를 작성해주세요:",
    "피보나치 수열을 빠르게 계산하는 파이썬 함수를 만들어주세요:",
    "파이썬으로 메모리 효율적인 피보나치 함수를 구현해주세요:",
    
    # 문서화 관련 지시사항
    "파이썬으로 피보나치 수열을 계산하는 함수를 작성해주세요. 주석을 포함해주세요:",
    "피보나치 수열을 계산하는 파이썬 함수를 만들어주세요. docstring을 포함해주세요:",
    "파이썬으로 피보나치 함수를 구현해주세요. 사용 예시도 포함해주세요:"
]
```

#### 6) 데이터 품질 관리 및 검증

```python
def validate_instruction_data(data):
    """Instruction Following 데이터 검증"""
    validated_data = []
    
    for item in data:
        # 필수 필드 확인
        if not all(key in item for key in ["instruction", "input", "output"]):
            continue
            
        # Instruction 품질 검사
        if len(item["instruction"]) < 5 or len(item["instruction"]) > 200:
            continue
            
        # Input 품질 검사 (빈 값 허용)
        if item["input"] is None:
            item["input"] = ""
            
        # Output 품질 검사
        if len(item["output"]) < 3 or len(item["output"]) > 2000:
            continue
            
        # 유해한 내용 필터링
        if contains_harmful_content(item["instruction"]) or \
           contains_harmful_content(item["input"]) or \
           contains_harmful_content(item["output"]):
            continue
            
        validated_data.append(item)
    
    return validated_data

def create_instruction_variants(base_instruction, variants):
    """기본 지시사항에 다양한 변형 생성"""
    instruction_data = []
    
    for variant in variants:
        # 기본 지시사항과 변형을 결합
        combined_instruction = f"{base_instruction} {variant}"
        
        # 데이터에 추가
        instruction_data.append({
            "instruction": combined_instruction,
            "input": "sample input",
            "output": "sample output"
        })
    
    return instruction_data
```

#### 7) 실제 데이터셋 구성 예시

```python
# Alpaca 스타일 데이터 구성
alpaca_style_data = [
    {
        "instruction": "다음 영어 문장을 한국어로 번역해주세요:",
        "input": "The weather is beautiful today.",
        "output": "오늘 날씨가 아름답습니다."
    },
    {
        "instruction": "다음 텍스트를 요약해주세요:",
        "input": "인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다. 머신러닝과 딥러닝을 포함하며, 다양한 분야에서 활용되고 있습니다.",
        "output": "인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술로, 머신러닝과 딥러닝을 포함하며 다양한 분야에서 활용됩니다."
    },
    {
        "instruction": "다음 문장의 감정을 분석해주세요:",
        "input": "오늘 정말 좋은 날이에요!",
        "output": "긍정적인 감정입니다. 날씨나 상황에 대한 만족감과 기쁨이 표현되어 있습니다."
    }
]

# Vicuna 스타일 데이터 구성 (대화형)
vicuna_style_data = [
    {
        "instruction": "사용자: 다음 영어 문장을 한국어로 번역해주세요.\n\nThe weather is beautiful today.",
        "input": "",
        "output": "오늘 날씨가 아름답습니다."
    },
    {
        "instruction": "사용자: 다음 텍스트를 요약해주세요.\n\n인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다...",
        "input": "",
        "output": "인공지능은 컴퓨터가 인간의 지능을 모방하여 학습하고 추론하는 기술로, 머신러닝과 딥러닝을 포함하며 다양한 분야에서 활용됩니다."
    }
]
```

#### 8) Instruction Robustness 향상 전략

```python
def enhance_instruction_robustness(base_data):
    """지시사항 Robustness 향상"""
    enhanced_data = []
    
    for item in base_data:
        # 1. 다양한 표현 방식
        instruction_variants = generate_instruction_variants(item["instruction"])
        
        # 2. 다양한 어조와 스타일
        tone_variants = generate_tone_variants(item["instruction"])
        
        # 3. 다양한 복잡도
        complexity_variants = generate_complexity_variants(item["instruction"])
        
        # 4. 다양한 언어 (다국어 지원)
        language_variants = generate_language_variants(item["instruction"])
        
        # 모든 변형을 데이터에 추가
        all_variants = instruction_variants + tone_variants + complexity_variants + language_variants
        
        for variant in all_variants:
            enhanced_data.append({
                "instruction": variant,
                "input": item["input"],
                "output": item["output"]
            })
    
    return enhanced_data

def generate_instruction_variants(instruction):
    """지시사항 변형 생성"""
    variants = [
        instruction,
        instruction.replace("해주세요", "해주세요."),
        instruction.replace("해주세요", "해주세요!"),
        instruction.replace("해주세요", "해주세요~"),
        instruction.replace("다음", "이"),
        instruction.replace("이", "다음"),
        instruction.replace("주세요", "주세요."),
        instruction.replace("주세요", "주세요!"),
    ]
    return variants
```

### 4. 학습 최적화 기법

#### 1) 데이터 품질 관리
```python
def validate_instruction_data(data):
    """지시사항 데이터 검증"""
    valid_data = []
    
    for item in data:
        # 필수 필드 확인
        if not all(key in item for key in ["instruction", "output"]):
            continue
            
        # 응답 길이 확인 (너무 짧거나 긴 응답 제외)
        if len(item["output"]) < 10 or len(item["output"]) > 1000:
            continue
            
        # 유해한 내용 필터링
        if contains_harmful_content(item["instruction"]) or contains_harmful_content(item["output"]):
            continue
            
        valid_data.append(item)
    
    return valid_data

def contains_harmful_content(text):
    """유해한 내용 검사"""
    harmful_keywords = ["폭력", "차별", "혐오", "불법"]
    return any(keyword in text for keyword in harmful_keywords)
```

#### 2) 학습률 스케줄링
```python
from transformers import get_cosine_schedule_with_warmup

def setup_optimizer_and_scheduler(model, num_training_steps):
    """옵티마이저와 스케줄러 설정"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Warmup과 cosine annealing 스케줄
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler
```

#### 3) 평가 지표
```python
def evaluate_instruction_following(model, tokenizer, test_data):
    """지시사항 따르기 성능 평가"""
    model.eval()
    results = []
    
    for item in test_data:
        # 입력 구성
        prompt = f"### 지시사항:\n{item['instruction']}\n\n### 입력:\n{item.get('input', '')}\n\n### 응답:\n"
        
        # 생성
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]
        
        # 평가 (BLEU, ROUGE, 인간 평가 등)
        score = calculate_similarity(response, item["output"])
        results.append({
            "instruction": item["instruction"],
            "expected": item["output"],
            "generated": response,
            "score": score
        })
    
    return results
```

### 5. 주요 도전 과제

#### 1) 데이터 품질
- **일관성**: 동일한 지시사항에 대한 일관된 응답 보장
- **다양성**: 다양한 유형의 지시사항과 응답 패턴
- **품질**: 인간 수준의 고품질 응답 생성

#### 2) 안전성과 윤리
```python
def safety_check(instruction, response):
    """안전성 검사"""
    safety_issues = []
    
    # 유해한 내용 검사
    if contains_harmful_content(response):
        safety_issues.append("유해한 내용 포함")
    
    # 편향 검사
    if contains_bias(response):
        safety_issues.append("편향된 내용 포함")
    
    # 개인정보 검사
    if contains_personal_info(response):
        safety_issues.append("개인정보 포함")
    
    return safety_issues
```

#### 3) 일반화 능력
- **새로운 지시사항**: 학습하지 않은 새로운 유형의 지시사항 처리
- **도메인 적응**: 다양한 분야의 지시사항에 대한 적응
- **언어 다양성**: 다국어 지시사항 처리

### 6. 최신 연구 동향

#### 1) Self-Instruct
```python
def self_instruct_generation(model, seed_instructions):
    """자기 지시사항 생성"""
    generated_instructions = []
    
    for seed in seed_instructions:
        # 기존 지시사항을 기반으로 새로운 지시사항 생성
        prompt = f"다음과 같은 지시사항과 유사한 새로운 지시사항을 생성해주세요:\n{seed}"
        
        response = generate_response(model, prompt)
        generated_instructions.append(response)
    
    return generated_instructions
```

#### 2) Instruction Tuning
```python
def instruction_tuning(model, instruction_data, epochs=3):
    """지시사항 튜닝"""
    for epoch in range(epochs):
        for batch in instruction_data:
            # 지시사항 형식으로 데이터 구성
            formatted_input = format_instruction(batch)
            
            # 학습
            loss = train_step(model, formatted_input, batch["output"])
            
            # 검증
            if should_validate():
                validate_performance(model, validation_data)
```

#### 3) Multi-task Learning
```python
def multi_task_instruction_learning(model, task_datasets):
    """다중 태스크 지시사항 학습"""
    for task_name, dataset in task_datasets.items():
        # 태스크별 지시사항 프리픽스 추가
        task_prefix = f"[{task_name}] "
        
        for item in dataset:
            item["instruction"] = task_prefix + item["instruction"]
        
        # 태스크별 학습
        train_task(model, dataset)
```

### 7. 평가 방법

#### 1) 자동 평가
```python
def automatic_evaluation(generated_responses, reference_responses):
    """자동 평가 지표"""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    bleu_scores = []
    rouge_scores = []
    
    for gen, ref in zip(generated_responses, reference_responses):
        # BLEU 점수
        bleu = sentence_bleu([ref.split()], gen.split())
        bleu_scores.append(bleu)
        
        # ROUGE 점수
        rouge = Rouge()
        scores = rouge.get_scores(gen, ref)
        rouge_scores.append(scores[0]['rouge-1']['f'])
    
    return {
        'bleu': np.mean(bleu_scores),
        'rouge': np.mean(rouge_scores)
    }
```

#### 2) 인간 평가
```python
def human_evaluation_criteria():
    """인간 평가 기준"""
    criteria = {
        "정확성": "지시사항을 정확히 수행했는가?",
        "완성도": "응답이 완전하고 충분한가?",
        "유용성": "응답이 실제로 유용한가?",
        "안전성": "응답이 안전하고 적절한가?",
        "자연스러움": "응답이 자연스럽고 인간다운가?"
    }
    return criteria
```

### 8. 실제 적용 사례

#### 1) ChatGPT
- **InstructGPT**: 인간 피드백을 활용한 지시사항 학습
- **RLHF**: 강화학습을 통한 지시사항 따르기 개선
- **Safety**: 안전성과 윤리적 고려사항 반영

#### 2) Claude
- **Constitutional AI**: 헌법적 원칙을 기반으로 한 안전한 지시사항 따르기
- **Self-critique**: 모델 스스로의 비판적 사고를 통한 품질 향상

#### 3) LLaMA
- **Alpaca**: Stanford의 52K 지시사항 데이터셋을 활용한 학습
- **Vicuna**: 대화형 지시사항 따르기 특화

## 결론

Instruction Following은 Foundation Model을 실제 사용 가능한 AI 시스템으로 만드는 핵심 단계입니다. 고품질의 지시사항 데이터, 효과적인 학습 방법, 그리고 안전성과 윤리적 고려사항을 모두 포함한 종합적인 접근이 필요합니다. 지속적인 연구와 개선을 통해 더욱 유용하고 안전한 AI 시스템을 개발할 수 있을 것입니다.
