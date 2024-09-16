import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# CUDA 장치의 주요 버전과 부 버전을 가져옵니다.
major_version, minor_version = torch.cuda.get_device_capability()
major_version, minor_version

max_seq_length = 4096  # 최대 시퀀스 길이를 설정합니다. 내부적으로 RoPE 스케일링을 자동으로 지원합니다!
# 자동 감지를 위해 None을 사용합니다. Tesla T4, V100은 Float16, Ampere+는 Bfloat16을 사용하세요.
dtype = None
# 메모리 사용량을 줄이기 위해 4bit 양자화를 사용합니다. False일 수도 있습니다.
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/llama-3-8b-bnb-4bit",
    model_name="beomi/Llama-3-Open-Ko-8B-Instruct-preview",  # 모델 이름을 설정합니다.
    max_seq_length=max_seq_length,  # 최대 시퀀스 길이를 설정합니다.
    dtype=dtype,  # 데이터 타입을 설정합니다.
    load_in_4bit=load_in_4bit,  # 4bit 양자화 로드 여부를 설정합니다.
    # token = "hf_...", # 게이트된 모델을 사용하는 경우 토큰을 사용하세요. 예: meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 0보다 큰 어떤 숫자도 선택 가능! 8, 16, 32, 64, 128이 권장됩니다.
    lora_alpha=32,  # LoRA 알파 값을 설정합니다.
    lora_dropout=0.05,  # 드롭아웃을 지원합니다.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # 타겟 모듈을 지정합니다.
    bias="none",  # 바이어스를 지원합니다.
    # True 또는 "unsloth"를 사용하여 매우 긴 컨텍스트에 대해 VRAM을 30% 덜 사용하고, 2배 더 큰 배치 크기를 지원합니다.
    use_gradient_checkpointing="unsloth",
    random_state=123,  # 난수 상태를 설정합니다.
    use_rslora=False,  # 순위 안정화 LoRA를 지원합니다.
    loftq_config=None,  # LoftQ를 지원합니다.
)

# EOS_TOKEN은 문장의 끝을 나타내는 토큰입니다. 이 토큰을 추가해야 합니다.
EOS_TOKEN = tokenizer.eos_token

# AlpacaPrompt를 사용하여 지시사항을 포맷팅하는 함수입니다.
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


# 주어진 예시들을 포맷팅하는 함수입니다.
def formatting_prompts_func(examples):
    instructions = examples["instruction"]  # 지시사항을 가져옵니다.
    outputs = examples["output"]  # 출력값을 가져옵니다.
    texts = []  # 포맷팅된 텍스트를 저장할 리스트입니다.
    for instruction, output in zip(instructions, outputs):
        # EOS_TOKEN을 추가해야 합니다. 그렇지 않으면 생성이 무한히 진행될 수 있습니다.
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,  # 포맷팅된 텍스트를 반환합니다.
    }


# "teddylee777/QA-Dataset-mini" 데이터셋을 불러옵니다. 훈련 데이터만 사용합니다.
dataset = load_dataset("AIPrintOrcl/QA-Dataset-mini", split="train")

# 데이터셋에 formatting_prompts_func 함수를 적용합니다. 배치 처리를 활성화합니다.
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

tokenizer.padding_side = "right"  # 토크나이저의 패딩을 오른쪽으로 설정합니다.

# SFTTrainer를 사용하여 모델 학습 설정
trainer = SFTTrainer(
    model=model,  # 학습할 모델
    tokenizer=tokenizer,  # 토크나이저
    train_dataset=dataset,  # 학습 데이터셋
    eval_dataset=dataset,
    dataset_text_field="text",  # 데이터셋에서 텍스트 필드의 이름
    max_seq_length=max_seq_length,  # 최대 시퀀스 길이
    dataset_num_proc=2,  # 데이터 처리에 사용할 프로세스 수
    packing=False,  # 짧은 시퀀스에 대한 학습 속도를 5배 빠르게 할 수 있음
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 각 디바이스당 훈련 배치 크기
        gradient_accumulation_steps=4,  # 그래디언트 누적 단계
        warmup_steps=5,  # 웜업 스텝 수
        num_train_epochs=3,  # 훈련 에폭 수
        max_steps=100,  # 최대 스텝 수
        do_eval=True,
        evaluation_strategy="steps",
        logging_steps=1,  # logging 스텝 수
        learning_rate=2e-4,  # 학습률
        fp16=not torch.cuda.is_bf16_supported(),  # fp16 사용 여부, bf16이 지원되지 않는 경우에만 사용
        bf16=torch.cuda.is_bf16_supported(),  # bf16 사용 여부, bf16이 지원되는 경우에만 사용
        optim="adamw_8bit",  # 최적화 알고리즘
        weight_decay=0.01,  # 가중치 감소
        lr_scheduler_type="cosine",  # 학습률 스케줄러 유형
        seed=123,  # 랜덤 시드
        output_dir="outputs",  # 출력 디렉토리
    ),
)

trainer_stats = trainer.train()  # 모델을 훈련시키고 통계를 반환합니다.