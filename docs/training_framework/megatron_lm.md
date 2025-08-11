---
title: Megatron LM
parent: LLM 학습 프레임워크
nav_order: 7
---

# NVIDIA Megatron-LM 학습 플랫폼 완전 분석

## 1. 개요  
Megatron-LM은 NVIDIA Applied Deep Learning Research팀이 발표한 **대규모 언어모델(LLM) 훈련**을 위한 프레임워크로, 수십억~수천억 파라미터급 모델을 GPU 클러스터에서 효율적으로 학습할 수 있게 디자인되었다.

Megatron-LM 저장소는 크게 두 부분으로 구성된다:  
- **Megatron-LM**: LLM 연구용 프레임워크(논문 코드 포함)  
- **Megatron-Core**: GPU 최적화 기법 모음집(모듈형 라이브러리)  

Megatron-Core는 Tensor Core GPU에서 FP8/FP16, FlashAttention, ZeRO, MoE 등 시스템-레벨 최적화를 제공하며, Megatron-LM과 조합해 **모델 병렬화**와 **데이터 병렬화**를 조화롭게 사용해 초대형 LLM을 훈련한다.

## 2. 핵심 기술

### 2.1 다차원 모델 병렬화  
- **텐서 병렬화(Tensor Parallelism)**: 단일 레이어를 여러 GPU로 분할 연산  
- **시퀀스 병렬화(Sequence Parallelism)**: 토큰 차원 분할로 장기 문맥 처리 효율화  
- **파이프라인 병렬화(Pipeline Parallelism)**: 레이어 그룹별로 파이프라인 분할(인터리브 스케줄링 지원)  
- **전문가 병렬화(Expert Parallelism, MoE)**: Mixture-of-Experts 구조에서 전문가별 연산 병렬화  
- **데이터 병렬화(Data Parallelism)**: 여러 GPU에 모델 복사 후 그래디언트 평균화  

이들 병렬화 기법을 조합해 **수백억~수천억 파라미터 모델**을 1000+ GPU 클러스터에서도 학습 가능하다.

### 2.2 메모리 및 연산 최적화  
- **ZeRO 옵티마이저 통합**: DeepSpeed ZeRO-1/2/3와 결합하여 옵티마이저 상태·그래디언트·활성화 메모리 분할
- **FlashAttention**: HBM↔SRAM I/O 최소화, 어텐션 연산 속도 2~3배 가속  
- **Activation Checkpointing & Recomputation**: 'selective'·'full' 모드로 활성화 메모리 축소  
- **분산 옵티마이저(Distributed Optimizer)**: 데이터 병렬 랭크 간 옵티마이저 상태 분산 저장  
- **FP8/16 혼합 정밀도**: Hopper 아키텍처 FP8 최적화 및 FP16 자동 캐스트  

### 2.3 Mixture-of-Experts (MoE)  
Megatron-Core의 MoE 구현은:  
- **Expert Parallelism**: 전문가(Experts)를 여러 워커에 분산  
- **Top-K Router & Load Balancing**: 토큰-전문가 매핑 최적화  
- **GroupedGEMM, FP8 학습**: 배치 처리량 극대화  
MoE 모델 훈련 시 최대 47% MFU 달성, 수십억 파라미터 모델의 효율적 확장 지원.

## 3. 설치 및 환경 구성

### 3.1 Docker (권장)  
```bash
docker run --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### 3.2 PyPI 설치  
```bash
# Megatron-Core + Megatron-LM 의존성
pip install megatron-core[mlm]

# 또는 Megatron-Core만
pip install megatron-core
```

### 3.3 소스 설치  
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
bash docker/common/install.sh --environment dev
```

필수 의존성: PyTorch, CUDA, cuDNN, NCCL, mamba(선택) 등.

## 4. 데이터 전처리

### 4.1 BERT  
```bash
python tools/preprocess_data.py \
  --input wiki.json \
  --output-prefix bert_cased \
  --vocab-file bert-vocab.txt \
  --tokenizer-type BertWordPieceLowerCase \
  --split-sentences
```
결과: `bert_cased_text_sentence.bin` / `.idx`.

### 4.2 GPT  
```bash
python tools/preprocess_data.py \
  --input webtext.json \
  --output-prefix gpt2 \
  --vocab-file gpt2-vocab.json \
  --merge-file gpt2-merges.txt \
  --tokenizer-type GPT2BPETokenizer \
  --append-eod
```
결과: `gpt2_text_document.bin` / `.idx`.

### 4.3 T5  
```bash
python tools/preprocess_data.py \
  --input corpus.json \
  --output-prefix t5_base \
  --vocab-file t5-vocab.txt \
  --tokenizer-type SentencePieceBPETokenizer \
  --split-sentences
```

## 5. 학습 워크플로우

### 5.1 BERT 사전학습  
```bash
bash examples/bert/train_bert_340m_distributed.sh
```
- 학습률: 선형 감쇠, Warmup 비율 조정  
- 배치: 마이크로 배치 축적으로 글로벌 배치  
- 데이터 분할: train/valid/test (949:50:1)  
- Tensor/Sequence/Data 병렬 조합 지원.

### 5.2 GPT 사전학습  
```bash
bash examples/gpt3/train_gpt3_175b_distributed.sh
```
- BPE 토크나이저, Cosine LR decay  
- 175B 모델 1024 GPU에서 학습(8×Tensor ×16×Pipeline).

### 5.3 T5 사전학습  
```bash
bash examples/t5/train_t5_220m_distributed.sh
```
- `--kv-channels`, `--ffn-hidden-size`, Encoder/Decoder 시퀀스 길이 지정.

### 5.4 분산 환경 설정  
- PyTorch `torchrun`/`torch.distributed` 사용  
- `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`, `--sequence-parallel`  
- Interleaved Pipeline: `--num-layers-per-virtual-pipeline-stage`.

## 6. 평가 및 후처리

### 6.1 체크포인트 병합  
```bash
python tools/checkpoint/convert.py \
  --model-type GPT \
  --load-dir ckpt/tp4_pp4 \
  --save-dir ckpt/tp2_pp2 \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 2
```

### 6.2 텍스트 생성 서버  
```bash
python tools/run_text_generation_server.py --load ckpt/gpt2_345m
curl -X PUT localhost:5000/api -d '{"prompts":["Hello"],"tokens_to_generate":10}'
```

### 6.3 평가 스크립트  
- **WikiText-103 Perplexity**  
- **LAMBADA Cloze Accuracy**  
- **RACE / MNLI / QQP** 등 downstream 태스크.

## 7. 모델 최적화 및 배포

### 7.1 양자화 및 TensorRT-LLM  
- 8비트·4비트 양자화  
- TensorRT-LLM 백엔드 지원으로 추론 성능 극대화.

### 7.2 MoE 배포  
- Expert Routing, Load Balancing, GroupedGEMM 활용  
- 대규모 MoE 모델 효율적 서비스화 지원.

## 8. 데이터셋 수집 및 재현성

- **Wikipedia**: WikiExtractor→JSON → 전처리  
- **WebText**: Common Crawl 기반  
- **재현성 모드**: `--deterministic-perf-mode`로 비트단위 재현성 확보  

## 9. 생태계 및 활용 사례

- Colossal-AI, Hugging Face Accelerate, NVIDIA NeMo 등이 Megatron-Core 위에 구축  
- Megatron-LM 기반 연구∙서비스: Google T5, Meta LLaVA, Mamba Models, Retro/InstructRetro 재현  
- GitHub 스타 2.4만+, 수십 개 기업·연구기관 채택.

## 결론  
NVIDIA Megatron-LM은 **GPU 하드웨어 최적화**와 **모델·데이터 병렬화**를 결합해 초대형 LLM을 대규모 클러스터에서 효율적으로 학습하는 데 특화된 플랫폼이다. Megatron-Core의 모듈형 GPU 최적화 기법, 다양한 병렬화 전략, ZeRO·FlashAttention·MoE 통합으로 전례 없는 확장성과 성능을 제공하며, Docker·PyPI·소스 설치 옵션과 풍부한 예제 스크립트로 **연구자와 엔지니어의 대규모 LLM 개발 흐름**을 강력히 지원한다.

**핵심 키워드**: 모델 병렬화, 텐서 병렬화, 시퀀스 병렬화, 파이프라인 병렬화, ZeRO, FlashAttention, MoE, 분산학습, GPU 최적화
