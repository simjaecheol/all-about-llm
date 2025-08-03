---
title: Evaluation
parent: LLM 학습 방법
nav_order: 6
---

# Evaluation

## 개요

LLM 모델의 평가는 모델의 성능을 객관적으로 측정하고 개선 방향을 제시하는 중요한 과정입니다. 기존 NLP 평가 방법론을 기반으로 하되, LLM의 특성에 맞게 확장된 평가 방법들을 사용합니다.

## 평가 방법의 분류

### 1. 자동 평가 (Automatic Evaluation)
- **정량적 지표**: 수치로 표현 가능한 객관적 지표
- **빠른 평가**: 대량의 데이터에 대한 신속한 평가
- **일관성**: 동일한 조건에서 반복 가능한 평가

### 2. 인간 평가 (Human Evaluation)
- **정성적 평가**: 인간의 주관적 판단을 통한 평가
- **맥락 이해**: 복잡한 의미와 맥락을 고려한 평가
- **실용성**: 실제 사용 환경과 유사한 평가

## 기존 NLP 평가 방법론

LLM이 나오기 이전의 자연어 처리(Natural Language Processing)에서 학습된 모델을 평가하는 방법을 LLM에도 동일하게 적용할 수 있습니다.

### 1. 텍스트 생성 평가

#### BLEU (Bilingual Evaluation Understudy)
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

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
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

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
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

### 2. 텍스트 분류 평가

#### 정확도 (Accuracy)
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

## 인간 평가 방법

### 1. 평가 기준 설정
```python
def human_evaluation_criteria():
    """인간 평가 기준"""
    criteria = {
        "정확성": {
            "description": "모델이 정확한 정보를 제공하는가?",
            "scale": "1-5 (1: 매우 부정확, 5: 매우 정확)"
        },
        "완성도": {
            "description": "응답이 완전하고 충분한 정보를 포함하는가?",
            "scale": "1-5 (1: 매우 불완전, 5: 매우 완전)"
        },
        "유용성": {
            "description": "응답이 실제로 유용한가?",
            "scale": "1-5 (1: 매우 유용하지 않음, 5: 매우 유용)"
        },
        "자연스러움": {
            "description": "응답이 자연스럽고 인간다운가?",
            "scale": "1-5 (1: 매우 부자연스러움, 5: 매우 자연스러움)"
        },
        "안전성": {
            "description": "응답이 안전하고 적절한가?",
            "scale": "1-5 (1: 매우 위험, 5: 매우 안전)"
        }
    }
    return criteria
```

### 2. 평가 인터페이스
```python
def create_evaluation_interface(model_responses, evaluation_criteria):
    """평가 인터페이스 생성"""
    evaluation_data = []
    
    for i, response in enumerate(model_responses):
        evaluation_entry = {
            'response_id': i,
            'response': response,
            'evaluations': {}
        }
        
        for criterion, details in evaluation_criteria.items():
            evaluation_entry['evaluations'][criterion] = {
                'score': None,
                'comment': ''
            }
        
        evaluation_data.append(evaluation_entry)
    
    return evaluation_data
```

### 3. 평가 결과 분석
```python
def analyze_human_evaluation(evaluation_data):
    """인간 평가 결과 분석"""
    analysis = {}
    
    for criterion in evaluation_data[0]['evaluations'].keys():
        scores = [entry['evaluations'][criterion]['score'] 
                 for entry in evaluation_data 
                 if entry['evaluations'][criterion]['score'] is not None]
        
        if scores:
            analysis[criterion] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
    
    return analysis
```

## 벤치마크 데이터셋

### 1. 일반적인 벤치마크
```python
def load_benchmark_datasets():
    """벤치마크 데이터셋 로드"""
    benchmarks = {
        "GLUE": {
            "description": "General Language Understanding Evaluation",
            "tasks": ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE"],
            "metrics": ["accuracy", "f1", "pearson", "spearman"]
        },
        "SuperGLUE": {
            "description": "More challenging NLU tasks",
            "tasks": ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"],
            "metrics": ["accuracy", "f1", "exact_match"]
        },
        "MMLU": {
            "description": "Massive Multitask Language Understanding",
            "tasks": ["STEM", "Humanities", "Social Sciences", "Other"],
            "metrics": ["accuracy"]
        }
    }
    return benchmarks
```

### 2. LLM 특화 벤치마크
```python
def load_llm_benchmarks():
    """LLM 특화 벤치마크"""
    llm_benchmarks = {
        "HELM": {
            "description": "Holistic Evaluation of Language Models",
            "tasks": ["Question Answering", "Summarization", "Translation", "Reasoning"],
            "metrics": ["accuracy", "robustness", "fairness", "efficiency"]
        },
        "BigBench": {
            "description": "Beyond the Imitation Game",
            "tasks": ["Language Understanding", "Reasoning", "Creativity"],
            "metrics": ["accuracy", "diversity", "creativity"]
        },
        "AlpacaEval": {
            "description": "Evaluation for Instruction Following",
            "tasks": ["Instruction Following", "Task Completion"],
            "metrics": ["win_rate", "human_preference"]
        }
    }
    return llm_benchmarks
```

## 평가 자동화 및 모니터링

### 1. 지속적 평가 시스템
```python
def continuous_evaluation_pipeline(model, tokenizer, evaluation_data):
    """지속적 평가 파이프라인"""
    evaluation_results = {
        'automatic_metrics': {},
        'human_evaluation': {},
        'safety_metrics': {},
        'performance_trends': []
    }
    
    # 자동 평가 실행
    evaluation_results['automatic_metrics'] = run_automatic_evaluation(
        model, tokenizer, evaluation_data
    )
    
    # 안전성 평가
    evaluation_results['safety_metrics'] = evaluate_safety(
        model, tokenizer, evaluation_data['safety_cases']
    )
    
    # 성능 트렌드 업데이트
    evaluation_results['performance_trends'].append({
        'timestamp': datetime.now(),
        'metrics': evaluation_results['automatic_metrics']
    })
    
    return evaluation_results
```

### 2. 평가 결과 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_evaluation_results(results):
    """평가 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 자동 평가 지표
    metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    scores = [results['automatic_metrics'][m] for m in metrics]
    
    axes[0, 0].bar(metrics, scores)
    axes[0, 0].set_title('Automatic Evaluation Metrics')
    axes[0, 0].set_ylabel('Score')
    
    # 2. 인간 평가 결과
    human_metrics = list(results['human_evaluation'].keys())
    human_scores = [results['human_evaluation'][m]['mean_score'] for m in human_metrics]
    
    axes[0, 1].bar(human_metrics, human_scores)
    axes[0, 1].set_title('Human Evaluation Results')
    axes[0, 1].set_ylabel('Score')
    
    # 3. 안전성 지표
    safety_metrics = list(results['safety_metrics'].keys())
    safety_scores = [results['safety_metrics'][m] for m in safety_metrics]
    
    axes[1, 0].pie(safety_scores, labels=safety_metrics, autopct='%1.1f%%')
    axes[1, 0].set_title('Safety Metrics')
    
    # 4. 성능 트렌드
    timestamps = [r['timestamp'] for r in results['performance_trends']]
    overall_scores = [r['metrics']['overall_score'] for r in results['performance_trends']]
    
    axes[1, 1].plot(timestamps, overall_scores)
    axes[1, 1].set_title('Performance Trends')
    axes[1, 1].set_ylabel('Overall Score')
    
    plt.tight_layout()
    plt.show()
```

## 결론

LLM 모델의 평가는 기존 NLP 평가 방법론을 기반으로 하되, LLM의 특성에 맞게 확장된 종합적인 접근이 필요합니다. 자동 평가와 인간 평가를 조합하여 모델의 성능을 다각도로 측정하고, 지속적인 모니터링을 통해 모델의 개선 방향을 제시할 수 있습니다. 특히 안전성, 윤리성, 창의성 등 LLM에 특화된 평가 지표의 개발이 중요합니다.
