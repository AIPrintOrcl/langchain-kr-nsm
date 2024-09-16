import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# 1. 필요한 라이브러리 설치
# !pip install transformers datasets torch accelerate peft

# 2. 데이터 준비
dataset = load_dataset("AIPrintOrcl/QA-Dataset-mini")  # 여기에 실제 데이터셋 이름을 넣으세요

# 3. 모델과 토크나이저 로드 (CPU 버전)
model_name = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"  # LLaMA 3가 아직 공개되지 않았으므로 LLaMA 2를 사용합니다
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 4. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 모델 준비
model = get_peft_model(model, lora_config)

# 6. 데이터 전처리
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 7. 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 8. 훈련 인자 설정 (CPU 버전)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 메모리 사용량 줄이기
    gradient_accumulation_steps=16,  # 배치 크기 조정
    learning_rate=2e-4,
    save_steps=100,
    logging_steps=10,
)

# 9. 트레이너 초기화 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# 10. 모델 저장
model.save_pretrained("./llama3_lora_finetuned")

# 11. 어댑터 사용 예시 (CPU 버전)
from peft import PeftModel, PeftConfig

# 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA 어댑터 로드
peft_model = PeftModel.from_pretrained(base_model, "./llama3_lora_finetuned")

# 추론
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = peft_model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))