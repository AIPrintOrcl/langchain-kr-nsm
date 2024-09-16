import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import json

# 1. 데이터 준비
def prepare_data(file_path):
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = [json.loads(line) for line in f]
    
    # dataset = [{
    #     'text': f"### 질문: {item['instruction']}\n\n### 답변: {item['output']}"
    # } for item in data]
    
    # return load_dataset('json', data=dataset)
    return load_dataset('json', data_files=file_path)

# 2. 모델과 토크나이저 로드
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return model, tokenizer

# 3. LoRA 설정
def setup_lora(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Llama 특화 레이어
    )
    model = get_peft_model(model, peft_config)
    return model

# 4. 데이터 전처리
# def preprocess_function(examples, tokenizer):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
def preprocess_function(examples, tokenizer):
    print("examples : ")
    print(examples)
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# 5. 트레이닝 설정
def setup_training(model, dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./lora_adapter",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=3e-4,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    'labels': torch.stack([f['input_ids'] for f in data])},
    )

    return trainer

# 메인 함수
def main():
    # 데이터 로드
    dataset = prepare_data("qa_pair.jsonl")
    
    # 모델과 토크나이저 로드 (Llama 2를 사용, Llama 3와 호환)
    model, tokenizer = load_model_and_tokenizer("beomi/Llama-3-Open-Ko-8B-Instruct-preview")
    
    # LoRA 설정
    model = setup_lora(model)
    
    # 데이터 전처리
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 트레이닝 설정 및 실행
    trainer = setup_training(model, tokenized_dataset, tokenizer)
    trainer.train()
    
    # LoRA 어댑터 저장
    model.save_pretrained("./lora_adapter")

if __name__ == "__main__":
    main()