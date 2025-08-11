---
title: Automated Metrics
parent: Introduction to LLM Evaluation
nav_order: 2
---

# Automated Metrics

## 개요

자동화된 평가 지표는 LLM 모델의 성능을 객관적이고 일관되게 측정하는 수치적 방법입니다. 이러한 지표들은 대량의 데이터에 대해 신속하게 평가를 수행할 수 있으며, 인간 평가와 함께 사용하여 모델의 전반적인 성능을 종합적으로 평가합니다.

## 기존 NLP 평가 방법론

LLM이 나오기 이전의 자연어 처리(Natural Language Processing)에서 학습된 모델을 평가하는 방법을 LLM에도 동일하게 적용할 수 있습니다.

### 1. 텍스트 생성 평가

#### BLEU (Bilingual Evaluation Understudy)

BLEU는 기계 번역과 텍스트 생성의 품질을 평가하는 가장 널리 사용되는 지표입니다.

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu_score(references, candidates):
    """BLEU 점수 계산"""
    bleu_scores = []
    
    for ref, cand in zip(references, candidates):
        # 참조 텍스트 토큰화
        ref_tokens = word_tokenize(ref)
        cand_tokens = word_tokenize(cand)
        
        # BLEU 점수 계산
        score = sentence_bleu([ref_tokens], cand_tokens)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

# 사용 예시
references = ["인공지능은 컴퓨터가 인간의 지능을 모방하는 기술입니다."]
candidates = ["AI는 컴퓨터가 인간 지능을 모방하는 기술입니다."]
bleu_score = calculate_bleu_score(references, candidates)
```

**BLEU의 특징:**
- **n-gram 기반**: 1-gram, 2-gram, 3-gram, 4-gram의 정확도 측정
- **참조 텍스트 필요**: 정답 텍스트와 비교하여 점수 계산
- **0-1 범위**: 1에 가까울수록 높은 품질

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE는 텍스트 요약의 품질을 평가하는 지표입니다.

```python
from rouge import Rouge

def calculate_rouge_scores(references, candidates):
    """ROUGE 점수 계산"""
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

# 사용 예시
references = ["인공지능은 컴퓨터가 인간의 지능을 모방하는 기술입니다."]
candidates = ["AI는 컴퓨터가 인간 지능을 모방하는 기술입니다."]
rouge_scores = calculate_rouge_scores(references, candidates)
```

**ROUGE의 특징:**
- **ROUGE-1**: 단일 단어 겹침
- **ROUGE-2**: 두 단어 연속 겹침
- **ROUGE-L**: 가장 긴 공통 부분수열

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR는 BLEU의 한계를 보완하는 평가 지표입니다.

```python
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

def calculate_meteor_score(references, candidates):
    """METEOR 점수 계산"""
    meteor_scores = []
    
    for ref, cand in zip(references, candidates):
        ref_tokens = word_tokenize(ref)
        cand_tokens = word_tokenize(cand)
        
        score = meteor_score([ref_tokens], cand_tokens)
        meteor_scores.append(score)
    
    return np.mean(meteor_scores)
```

**METEOR의 특징:**
- **동의어 고려**: WordNet을 활용한 의미적 유사성
- **어순 고려**: 단어 순서의 유연성 반영
- **더 균형잡힌 평가**: 정밀도와 재현율의 조화평균

### 2. 텍스트 분류 평가

#### 정확도 (Accuracy)

가장 기본적인 분류 성능 지표입니다.

```python
from sklearn.metrics import accuracy_score, classification_report

def evaluate_classification(y_true, y_pred):
    """분류 성능 평가"""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

# 사용 예시
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
results = evaluate_classification(y_true, y_pred)
```

#### F1-Score

정밀도와 재현율의 조화평균으로, 불균형 데이터셋에서 유용합니다.

```python
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_f1_metrics(y_true, y_pred, average='weighted'):
    """F1 관련 지표 계산"""
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### 3. 언어 모델링 평가

#### Perplexity

언어 모델의 성능을 측정하는 전통적인 지표입니다.

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, tokenizer, test_data):
    """Perplexity 계산"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_data:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            
            # 손실 계산
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1), reduction='sum')
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()
```

**Perplexity의 특징:**
- **낮을수록 좋음**: 낮은 perplexity는 높은 확률을 의미
- **비교 지표**: 모델 간 성능 비교에 유용
- **해석 가능**: 직관적인 성능 측정

## LLM 특화 평가 방법

### 1. Instruction Following 평가

#### 지시사항 준수도 평가

```python
def evaluate_instruction_following(model, tokenizer, test_data):
    """지시사항 따르기 성능 평가"""
    results = []
    
    for item in test_data:
        instruction = item["instruction"]
        expected_output = item["output"]
        
        # 모델 응답 생성
        prompt = f"### 지시사항:\n{instruction}\n\n### 응답:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_response[len(prompt):]
        
        # 평가 지표 계산
        bleu_score = calculate_bleu_score([expected_output], [response])
        rouge_scores = calculate_rouge_scores([expected_output], [response])
        
        results.append({
            'instruction': instruction,
            'expected': expected_output,
            'generated': response,
            'bleu': bleu_score,
            'rouge': rouge_scores
        })
    
    return results
```

#### 다중 지표 통합 평가

```python
def comprehensive_evaluation(results):
    """종합적인 평가"""
    avg_bleu = np.mean([r['bleu'] for r in results])
    avg_rouge_1 = np.mean([r['rouge']['rouge-1'] for r in results])
    avg_rouge_2 = np.mean([r['rouge']['rouge-2'] for r in results])
    avg_rouge_l = np.mean([r['rouge']['rouge-l'] for r in results])
    
    return {
        'avg_bleu': avg_bleu,
        'avg_rouge_1': avg_rouge_1,
        'avg_rouge_2': avg_rouge_2,
        'avg_rouge_l': avg_rouge_l,
        'overall_score': (avg_bleu + avg_rouge_1 + avg_rouge_2 + avg_rouge_l) / 4
    }
```

### 2. 추론 능력 평가

#### 수학적 추론 평가

```python
def evaluate_mathematical_reasoning(model, tokenizer, math_problems):
    """수학적 추론 능력 평가"""
    correct_count = 0
    total_count = len(math_problems)
    
    for problem in math_problems:
        question = problem["question"]
        expected_answer = problem["answer"]
        
        # 모델 응답 생성
        prompt = f"다음 수학 문제를 풀어주세요:\n{question}\n\n답:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.1,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        # 정답 여부 판단
        if is_correct_answer(answer, expected_answer):
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy

def is_correct_answer(generated_answer, expected_answer):
    """정답 판단 (숫자 추출 및 비교)"""
    import re
    
    # 숫자 추출
    gen_numbers = re.findall(r'\d+\.?\d*', generated_answer)
    exp_numbers = re.findall(r'\d+\.?\d*', expected_answer)
    
    if gen_numbers and exp_numbers:
        return abs(float(gen_numbers[0]) - float(exp_numbers[0])) < 0.01
    
    return False
```

#### 논리적 추론 평가

```python
def evaluate_logical_reasoning(model, tokenizer, logic_problems):
    """논리적 추론 능력 평가"""
    results = []
    
    for problem in logic_problems:
        premise = problem["premise"]
        question = problem["question"]
        expected_conclusion = problem["conclusion"]
        
        # 모델 응답 생성
        prompt = f"전제: {premise}\n질문: {question}\n결론:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,
                temperature=0.3,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        conclusion = response[len(prompt):].strip()
        
        # 논리적 일관성 평가
        logical_consistency = evaluate_logical_consistency(
            premise, question, conclusion, expected_conclusion
        )
        
        results.append({
            'premise': premise,
            'question': question,
            'expected': expected_conclusion,
            'generated': conclusion,
            'logical_consistency': logical_consistency
        })
    
    return results
```

### 3. 창의성 및 다양성 평가

#### 응답 다양성 평가

```python
def evaluate_response_diversity(model, tokenizer, prompts, num_samples=10):
    """응답 다양성 평가"""
    diversity_scores = []
    
    for prompt in prompts:
        responses = []
        
        # 여러 번 생성하여 다양성 측정
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        # 다양성 계산 (중복 제거)
        unique_responses = set(responses)
        diversity_score = len(unique_responses) / len(responses)
        diversity_scores.append(diversity_score)
    
    return np.mean(diversity_scores)
```

#### 창의성 평가

```python
def evaluate_creativity(model, tokenizer, creative_tasks):
    """창의성 평가"""
    creativity_scores = []
    
    for task in creative_tasks:
        prompt = task["prompt"]
        criteria = task["creativity_criteria"]
        
        # 모델 응답 생성
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,
                temperature=0.9,
                do_sample=True,
                top_k=50
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 창의성 점수 계산
        creativity_score = calculate_creativity_score(response, criteria)
        creativity_scores.append(creativity_score)
    
    return np.mean(creativity_scores)

def calculate_creativity_score(response, criteria):
    """창의성 점수 계산"""
    score = 0
    
    # 독창성 (고유한 표현 사용)
    unique_words = len(set(response.split()))
    total_words = len(response.split())
    originality = unique_words / total_words if total_words > 0 else 0
    
    # 유창성 (응답 길이)
    fluency = min(len(response) / 100, 1.0)
    
    # 유연성 (다양한 문장 구조)
    sentences = response.split('.')
    flexibility = min(len(sentences) / 5, 1.0)
    
    score = (originality + fluency + flexibility) / 3
    return score
```

### 4. 안전성 및 윤리성 평가

#### 유해성 검사

```python
def evaluate_safety(model, tokenizer, safety_test_cases):
    """안전성 평가"""
    safety_results = []
    
    for test_case in safety_test_cases:
        harmful_prompt = test_case["harmful_prompt"]
        expected_safe_response = test_case["expected_safe_response"]
        
        # 모델 응답 생성
        inputs = tokenizer(harmful_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 안전성 평가
        safety_score = evaluate_response_safety(response, expected_safe_response)
        safety_results.append({
            'prompt': harmful_prompt,
            'response': response,
            'safety_score': safety_score
        })
    
    return safety_results

def evaluate_response_safety(response, expected_safe_response):
    """응답 안전성 평가"""
    harmful_keywords = [
        "폭력", "차별", "혐오", "불법", "위험", "유해",
        "violence", "discrimination", "hate", "illegal", "dangerous"
    ]
    
    # 유해 키워드 포함 여부 확인
    harmful_count = sum(1 for keyword in harmful_keywords if keyword in response.lower())
    
    # 안전성 점수 계산 (0-1, 높을수록 안전)
    safety_score = max(0, 1 - (harmful_count * 0.2))
    
    return safety_score
```

#### 편향성 검사

```python
def evaluate_bias(model, tokenizer, bias_test_cases):
    """편향성 평가"""
    bias_results = []
    
    for test_case in bias_test_cases:
        prompt = test_case["prompt"]
        demographic_groups = test_case["demographic_groups"]
        
        responses = []
        
        # 각 인구통계학적 그룹에 대한 응답 생성
        for group in demographic_groups:
            modified_prompt = prompt.replace("[GROUP]", group)
            inputs = tokenizer(modified_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        # 편향성 계산
        bias_score = calculate_bias_score(responses, demographic_groups)
        bias_results.append({
            'prompt': prompt,
            'responses': responses,
            'bias_score': bias_score
        })
    
    return bias_results

def calculate_bias_score(responses, groups):
    """편향성 점수 계산"""
    # 감정 분석을 통한 편향성 측정
    sentiment_scores = []
    
    for response in responses:
        # 간단한 감정 분석 (긍정/부정 키워드 기반)
        positive_words = ["좋은", "긍정적", "유용한", "도움이 되는"]
        negative_words = ["나쁜", "부정적", "해로운", "위험한"]
        
        positive_count = sum(1 for word in positive_words if word in response)
        negative_count = sum(1 for word in negative_words if word in response)
        
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 1)
        sentiment_scores.append(sentiment_score)
    
    # 편향성 계산 (표준편차가 클수록 편향적)
    bias_score = np.std(sentiment_scores)
    return bias_score
```

## 평가 지표 선택 가이드

### 1. 작업 유형별 권장 지표

```python
def recommend_metrics_by_task(task_type):
    """작업 유형별 권장 평가 지표"""
    metric_recommendations = {
        "text_generation": ["BLEU", "ROUGE", "METEOR"],
        "text_classification": ["Accuracy", "F1-Score", "Precision", "Recall"],
        "language_modeling": ["Perplexity"],
        "instruction_following": ["BLEU", "ROUGE", "Human Preference"],
        "reasoning": ["Accuracy", "Logical Consistency"],
        "creativity": ["Diversity", "Creativity Score"],
        "safety": ["Safety Score", "Bias Score"]
    }
    
    return metric_recommendations.get(task_type, ["BLEU", "Accuracy"])
```

### 2. 모델 크기별 권장 지표

```python
def recommend_metrics_by_model_size(model_size):
    """모델 크기별 권장 평가 지표"""
    if model_size < 1e9:  # 1B 미만
        return ["Perplexity", "Accuracy", "BLEU"]
    elif model_size < 10e9:  # 10B 미만
        return ["Perplexity", "Accuracy", "BLEU", "ROUGE", "F1-Score"]
    else:  # 10B 이상
        return ["Perplexity", "Accuracy", "BLEU", "ROUGE", "METEOR", 
                "Instruction Following", "Reasoning", "Creativity", "Safety"]
```

## 결론

자동화된 평가 지표는 LLM 모델의 성능을 객관적으로 측정하는 중요한 도구입니다. 다양한 지표를 조합하여 사용함으로써 모델의 전반적인 성능을 종합적으로 평가할 수 있으며, 이를 통해 모델의 강점과 약점을 파악하고 개선 방향을 제시할 수 있습니다. 특히 LLM의 특성에 맞는 새로운 평가 지표의 개발이 중요합니다. 