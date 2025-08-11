---
title: Human Evaluation
parent: Introduction to LLM Evaluation
nav_order: 3
---

# Human Evaluation

## 개요

인간 평가는 LLM 모델의 성능을 인간의 주관적 판단을 통해 평가하는 방법입니다. 자동화된 평가 지표가 측정할 수 없는 복잡한 의미, 맥락 이해, 실용성 등을 평가할 수 있으며, 실제 사용 환경과 유사한 조건에서 모델의 성능을 측정할 수 있습니다.

## 인간 평가의 중요성

### 1. 자동 평가의 한계

자동화된 평가 지표는 다음과 같은 한계가 있습니다:

- **의미적 이해 부족**: 단순한 n-gram 기반 비교로는 의미적 품질을 정확히 측정할 수 없음
- **맥락 무시**: 문맥과 상황을 고려하지 못함
- **창의성 평가 불가**: 독창적이고 혁신적인 응답을 평가하기 어려움
- **실용성 측정 불가**: 실제 사용 환경에서의 유용성을 평가할 수 없음

### 2. 인간 평가의 장점

인간 평가는 다음과 같은 장점이 있습니다:

- **맥락 이해**: 복잡한 의미와 맥락을 고려한 평가
- **실용성 평가**: 실제 사용 환경과 유사한 조건에서의 평가
- **창의성 인식**: 독창적이고 혁신적인 응답의 가치를 인식
- **윤리적 판단**: 안전성, 공정성, 편향성 등의 윤리적 측면 평가

## 평가 기준 설정

### 1. 기본 평가 기준

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

### 2. LLM 특화 평가 기준

```python
def llm_specific_criteria():
    """LLM 특화 평가 기준"""
    llm_criteria = {
        "지시사항 준수": {
            "description": "모델이 주어진 지시사항을 정확히 따르는가?",
            "scale": "1-5 (1: 전혀 따르지 않음, 5: 완벽하게 따름)"
        },
        "추론 능력": {
            "description": "모델이 논리적 추론을 통해 문제를 해결하는가?",
            "scale": "1-5 (1: 추론 없음, 5: 탁월한 추론)"
        },
        "창의성": {
            "description": "모델이 독창적이고 혁신적인 응답을 생성하는가?",
            "scale": "1-5 (1: 매우 평범함, 5: 매우 창의적)"
        },
        "일관성": {
            "description": "모델의 응답이 일관성 있게 유지되는가?",
            "scale": "1-5 (1: 매우 불일관적, 5: 매우 일관적)"
        },
        "편향성": {
            "description": "모델의 응답에 편향이 없는가?",
            "scale": "1-5 (1: 매우 편향적, 5: 매우 공정함)"
        }
    }
    return llm_criteria
```

### 3. 도메인별 평가 기준

```python
def domain_specific_criteria(domain):
    """도메인별 평가 기준"""
    domain_criteria = {
        "의료": {
            "의학적 정확성": "의학적 사실과 일치하는가?",
            "안전성": "환자에게 위험하지 않은 정보인가?",
            "전문성": "의학적 전문 지식을 반영하는가?"
        },
        "법률": {
            "법적 정확성": "법적 사실과 일치하는가?",
            "객관성": "편향 없이 객관적으로 답변하는가?",
            "책임성": "법적 책임을 회피하지 않는가?"
        },
        "교육": {
            "교육적 가치": "학습에 도움이 되는가?",
            "적절성": "학습자의 수준에 적합한가?",
            "동기부여": "학습 동기를 높이는가?"
        }
    }
    
    return domain_criteria.get(domain, {})
```

## 평가 방법

### 1. 절대 평가 (Absolute Evaluation)

절대 평가는 각 응답을 독립적으로 평가하는 방법입니다.

```python
def absolute_evaluation_interface(model_responses, evaluation_criteria):
    """절대 평가 인터페이스 생성"""
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

**절대 평가의 특징:**
- **독립적 평가**: 각 응답을 다른 응답과 비교하지 않고 독립적으로 평가
- **일관성**: 동일한 기준으로 모든 응답을 평가
- **절대적 점수**: 각 응답에 대해 절대적인 품질 점수 부여

### 2. 상대 평가 (Relative Evaluation)

상대 평가는 여러 응답을 비교하여 상대적인 품질을 평가하는 방법입니다.

```python
def relative_evaluation_interface(model_responses, evaluation_criteria):
    """상대 평가 인터페이스 생성"""
    evaluation_data = []
    
    # 응답을 쌍으로 묶어서 비교 평가
    for i in range(0, len(model_responses), 2):
        if i + 1 < len(model_responses):
            comparison_entry = {
                'comparison_id': i // 2,
                'response_a': model_responses[i],
                'response_b': model_responses[i + 1],
                'winner': None,
                'confidence': None,
                'reasoning': ''
            }
            evaluation_data.append(comparison_entry)
    
    return evaluation_data
```

**상대 평가의 특징:**
- **비교 평가**: 여러 응답을 직접 비교하여 상대적 품질 평가
- **선택적 판단**: 더 나은 응답을 선택하는 방식
- **일관성 검증**: 평가자 간 일관성 확인 가능

### 3. A/B 테스트

A/B 테스트는 두 가지 응답 중 더 나은 것을 선택하는 평가 방법입니다.

```python
def ab_test_evaluation(model_a, model_b, test_prompts, num_evaluators=10):
    """A/B 테스트 평가"""
    results = {
        'model_a_wins': 0,
        'model_b_wins': 0,
        'ties': 0,
        'detailed_results': []
    }
    
    for prompt in test_prompts:
        # 두 모델의 응답 생성
        response_a = generate_response(model_a, prompt)
        response_b = generate_response(model_b, prompt)
        
        # 평가자들에게 A/B 선택 요청
        for evaluator in range(num_evaluators):
            choice = human_choice(response_a, response_b, prompt)
            
            if choice == 'A':
                results['model_a_wins'] += 1
            elif choice == 'B':
                results['model_b_wins'] += 1
            else:
                results['ties'] += 1
            
            results['detailed_results'].append({
                'prompt': prompt,
                'evaluator': evaluator,
                'choice': choice,
                'response_a': response_a,
                'response_b': response_b
            })
    
    return results
```

## 평가 인터페이스

### 1. 웹 기반 평가 인터페이스

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

### 2. 모바일 친화적 인터페이스

```python
def create_mobile_friendly_interface(evaluation_data):
    """모바일 친화적 평가 인터페이스"""
    mobile_interface = {
        'current_response': 0,
        'total_responses': len(evaluation_data),
        'progress': 0,
        'evaluation_data': evaluation_data
    }
    
    return mobile_interface
```

### 3. 배치 평가 인터페이스

```python
def create_batch_evaluation_interface(evaluation_data, batch_size=10):
    """배치 평가 인터페이스"""
    batches = []
    
    for i in range(0, len(evaluation_data), batch_size):
        batch = evaluation_data[i:i + batch_size]
        batches.append({
            'batch_id': i // batch_size,
            'responses': batch,
            'completed': False
        })
    
    return batches
```

## 평가 결과 분석

### 1. 기본 통계 분석

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
                'max_score': np.max(scores),
                'median_score': np.median(scores)
            }
    
    return analysis
```

### 2. 평가자 간 일관성 분석

```python
def analyze_evaluator_consistency(evaluation_data, evaluator_ids):
    """평가자 간 일관성 분석"""
    consistency_analysis = {}
    
    for criterion in evaluation_data[0]['evaluations'].keys():
        evaluator_scores = {}
        
        for evaluator_id in evaluator_ids:
            evaluator_scores[evaluator_id] = []
            
            for entry in evaluation_data:
                if entry['evaluator_id'] == evaluator_id:
                    score = entry['evaluations'][criterion]['score']
                    if score is not None:
                        evaluator_scores[evaluator_id].append(score)
        
        # 평가자 간 상관관계 계산
        correlations = []
        evaluator_list = list(evaluator_scores.keys())
        
        for i in range(len(evaluator_list)):
            for j in range(i + 1, len(evaluator_list)):
                eval1, eval2 = evaluator_list[i], evaluator_list[j]
                
                if len(evaluator_scores[eval1]) > 1 and len(evaluator_scores[eval2]) > 1:
                    correlation = np.corrcoef(evaluator_scores[eval1], evaluator_scores[eval2])[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
        
        consistency_analysis[criterion] = {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'evaluator_scores': evaluator_scores
        }
    
    return consistency_analysis
```

### 3. 응답 품질 분포 분석

```python
def analyze_response_quality_distribution(evaluation_data):
    """응답 품질 분포 분석"""
    quality_distribution = {}
    
    for criterion in evaluation_data[0]['evaluations'].keys():
        scores = [entry['evaluations'][criterion]['score'] 
                 for entry in evaluation_data 
                 if entry['evaluations'][criterion]['score'] is not None]
        
        if scores:
            # 품질 등급별 분포
            quality_grades = {
                'Excellent (5)': sum(1 for s in scores if s == 5),
                'Good (4)': sum(1 for s in scores if s == 4),
                'Fair (3)': sum(1 for s in scores if s == 3),
                'Poor (2)': sum(1 for s in scores if s == 2),
                'Very Poor (1)': sum(1 for s in scores if s == 1)
            }
            
            quality_distribution[criterion] = {
                'total_responses': len(scores),
                'grade_distribution': quality_grades,
                'grade_percentages': {grade: count/len(scores)*100 
                                    for grade, count in quality_grades.items()}
            }
    
    return quality_distribution
```

## 평가 품질 향상 방법

### 1. 평가자 교육 및 가이드라인

```python
def create_evaluator_guidelines():
    """평가자 가이드라인 생성"""
    guidelines = {
        "평가 원칙": [
            "일관성 유지: 동일한 기준으로 모든 응답 평가",
            "객관성 유지: 개인적 선호도나 편견 배제",
            "맥락 고려: 응답의 전체적인 맥락과 의도 파악",
            "세밀한 관찰: 응답의 세부적인 품질 요소 검토"
        ],
        "평가 절차": [
            "응답 전체 읽기: 응답의 전체 내용을 먼저 파악",
            "기준별 평가: 각 평가 기준에 따라 체계적으로 평가",
            "점수 부여: 1-5 척도에 따라 적절한 점수 부여",
            "의견 작성: 점수에 대한 구체적인 이유나 의견 작성"
        ],
        "주의사항": [
            "성급한 판단 금지: 충분한 검토 후 평가",
            "편향성 인식: 자신의 편향성 인식 및 조정",
            "일관성 확인: 이전 평가와의 일관성 유지",
            "의문사항 기록: 평가 과정에서 발생한 의문사항 기록"
        ]
    }
    
    return guidelines
```

### 2. 평가자 간 일관성 모니터링

```python
def monitor_evaluator_consistency(evaluation_data, evaluator_ids):
    """평가자 간 일관성 모니터링"""
    consistency_report = {}
    
    for evaluator_id in evaluator_ids:
        evaluator_scores = {}
        
        for entry in evaluation_data:
            if entry['evaluator_id'] == evaluator_id:
                for criterion, eval_data in entry['evaluations'].items():
                    if criterion not in evaluator_scores:
                        evaluator_scores[criterion] = []
                    
                    if eval_data['score'] is not None:
                        evaluator_scores[criterion].append(eval_data['score'])
        
        # 평가자별 통계
        evaluator_stats = {}
        for criterion, scores in evaluator_scores.items():
            if scores:
                evaluator_stats[criterion] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'score_range': f"{min(scores)}-{max(scores)}",
                    'total_evaluations': len(scores)
                }
        
        consistency_report[evaluator_id] = evaluator_stats
    
    return consistency_report
```

### 3. 평가 품질 검증

```python
def validate_evaluation_quality(evaluation_data, quality_thresholds):
    """평가 품질 검증"""
    quality_report = {
        'overall_quality': 'Good',
        'issues_found': [],
        'recommendations': []
    }
    
    # 응답 완성도 확인
    total_responses = len(evaluation_data)
    completed_responses = sum(1 for entry in evaluation_data 
                            if all(eval_data['score'] is not None 
                                  for eval_data in entry['evaluations'].values()))
    
    completion_rate = completed_responses / total_responses
    
    if completion_rate < quality_thresholds.get('completion_rate', 0.9):
        quality_report['issues_found'].append(f"응답 완성도 낮음: {completion_rate:.2%}")
        quality_report['recommendations'].append("평가자들에게 응답 완성의 중요성 강조")
    
    # 평가자 간 일관성 확인
    for criterion in evaluation_data[0]['evaluations'].keys():
        scores = [entry['evaluations'][criterion]['score'] 
                 for entry in evaluation_data 
                 if entry['evaluations'][criterion]['score'] is not None]
        
        if scores and len(scores) > 1:
            std_score = np.std(scores)
            if std_score > quality_thresholds.get('max_std', 1.5):
                quality_report['issues_found'].append(f"{criterion} 기준 평가자 간 일관성 낮음 (표준편차: {std_score:.2f})")
                quality_report['recommendations'].append(f"{criterion} 기준에 대한 평가자 교육 강화")
    
    # 전체 품질 등급 결정
    if len(quality_report['issues_found']) == 0:
        quality_report['overall_quality'] = 'Excellent'
    elif len(quality_report['issues_found']) <= 2:
        quality_report['overall_quality'] = 'Good'
    elif len(quality_report['issues_found']) <= 4:
        quality_report['overall_quality'] = 'Fair'
    else:
        quality_report['overall_quality'] = 'Poor'
    
    return quality_report
```

## 평가 자동화 및 효율성

### 1. 스마트 배치 처리

```python
def smart_batch_processing(evaluation_data, evaluator_preferences):
    """스마트 배치 처리"""
    optimized_batches = []
    
    # 평가자별 선호도에 따른 배치 최적화
    for evaluator_id, preferences in evaluator_preferences.items():
        preferred_criteria = preferences.get('preferred_criteria', [])
        batch_size = preferences.get('preferred_batch_size', 10)
        
        # 선호하는 기준이 포함된 응답들을 우선적으로 배치
        relevant_responses = []
        for entry in evaluation_data:
            if any(criterion in preferred_criteria 
                   for criterion in entry['evaluations'].keys()):
                relevant_responses.append(entry)
        
        # 배치 생성
        for i in range(0, len(relevant_responses), batch_size):
            batch = relevant_responses[i:i + batch_size]
            optimized_batches.append({
                'batch_id': f"{evaluator_id}_{len(optimized_batches)}",
                'evaluator_id': evaluator_id,
                'responses': batch,
                'priority': 'high' if preferred_criteria else 'normal'
            })
    
    return optimized_batches
```

### 2. 실시간 품질 모니터링

```python
def real_time_quality_monitoring(evaluation_data, evaluator_ids):
    """실시간 품질 모니터링"""
    monitoring_data = {
        'timestamp': datetime.now(),
        'evaluator_status': {},
        'quality_metrics': {},
        'alerts': []
    }
    
    for evaluator_id in evaluator_ids:
        # 평가자별 진행 상황
        evaluator_entries = [entry for entry in evaluation_data 
                           if entry.get('evaluator_id') == evaluator_id]
        
        completed_count = sum(1 for entry in evaluator_entries 
                            if all(eval_data['score'] is not None 
                                  for eval_data in entry['evaluations'].values()))
        
        total_assigned = len(evaluator_entries)
        completion_rate = completed_count / total_assigned if total_assigned > 0 else 0
        
        monitoring_data['evaluator_status'][evaluator_id] = {
            'completed': completed_count,
            'total_assigned': total_assigned,
            'completion_rate': completion_rate,
            'last_activity': max([entry.get('timestamp', datetime.min) 
                                 for entry in evaluator_entries], default=datetime.min)
        }
        
        # 품질 지표 계산
        if evaluator_entries:
            scores = []
            for entry in evaluator_entries:
                for eval_data in entry['evaluations'].values():
                    if eval_data.get('score') is not None:
                        scores.append(eval_data['score'])
            
            if scores:
                monitoring_data['quality_metrics'][evaluator_id] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'score_distribution': np.bincount(scores, minlength=6)[1:6]  # 1-5 점수
                }
        
        # 알림 생성
        if completion_rate < 0.3:
            monitoring_data['alerts'].append(f"평가자 {evaluator_id}의 진행률이 낮음: {completion_rate:.1%}")
    
    return monitoring_data
```

## 결론

인간 평가는 LLM 모델의 성능을 종합적으로 평가하는 필수적인 방법입니다. 자동화된 평가 지표와 함께 사용함으로써 모델의 실제 사용 환경에서의 성능을 정확하게 측정할 수 있으며, 이를 통해 모델의 개선 방향을 제시할 수 있습니다. 특히 평가 기준의 명확화, 평가자 교육, 일관성 모니터링 등을 통해 평가의 품질을 지속적으로 향상시킬 수 있습니다. 