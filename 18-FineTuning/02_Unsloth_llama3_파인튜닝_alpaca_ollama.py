import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import json
import requests

# 1. 데이터 준비
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"### 질문: {item['instruction']}\n\n### 답변: {item['output']}"
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

# 2. Ollama 모델 래퍼
class OllamaWrapper(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, input_ids, attention_mask=None):
        # Ollama API를 통해 모델 출력 얻기
        response = requests.post('http://localhost:11434/api/generate', 
                                 json={'model': self.model_name, 'prompt': input_ids})
        # 응답 처리 및 적절한 형태로 변환
        # 이 부분은 Ollama API의 실제 응답 형식에 따라 조정해야 합니다
        output = response.json()['response']
        # 출력을 적절한 텐서 형태로 변환
        return torch.tensor(output)

# 3. LoRA 설정
def setup_lora(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Ollama 모델에 맞게 조정 필요 # ["query_key_value"], ["q_proj", "v_proj"], ["attention"]
    )
    model = get_peft_model(model, peft_config)
    return model

# 4. 트레이닝 설정
def setup_training(model, dataset):
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
        train_dataset=dataset,
        data_collator=lambda data: {key: torch.stack([d[key] for d in data]) for key in data[0]},
    )

    return trainer

# 메인 함수
def main():
    # Ollama 모델 이름 (예: 'llama2')
    model_name = "llama3"
    
    # Tokenizer 설정 (Ollama 모델에 맞는 토크나이저 사용 필요)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M")
    
    # 데이터셋 준비
    dataset = QADataset("qa_pair.jsonl", tokenizer)
    
    # Ollama 모델 래퍼 생성
    model = OllamaWrapper(model_name)

    print("model 정보 : ")
    print(model)
    
    # LoRA 설정
    model = setup_lora(model)
    
    # 트레이닝 설정 및 실행
    trainer = setup_training(model, dataset)
    trainer.train()
    
    # LoRA 어댑터 저장
    model.save_pretrained("./lora_adapter")

if __name__ == "__main__":
    main()