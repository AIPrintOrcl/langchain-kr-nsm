import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Llama 구성 클래스 정의
class LlamaConfig:
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = 1e-5

# Llama 어텐션 메커니즘
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x):
        batch_size, seq_length, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        attention_scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention_probs = mx.softmax(attention_scores, axis=-1)
        
        output = mx.matmul(attention_probs, v).reshape(batch_size, seq_length, -1)
        return self.o_proj(output)

# Llama MLP
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

# Llama 레이어
class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x

# MLX에 최적화된 Llama 모델
class LlamaMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

# 데이터 로드 및 전처리
def load_and_preprocess_data():
    dataset = load_dataset("AIPrintOrcl/QA-Dataset-mini", split="train")
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B-Instruct-preview")
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    def preprocess_function(examples):
        inputs = [f"### Instruction:\n{instruction}\n\n### Response:\n{output}" 
                  for instruction, output in zip(examples["instruction"], examples["output"])]
        return tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer

# 모델 초기화
def initialize_model(tokenizer):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1024,  # 크기를 줄임
        num_hidden_layers=12,  # 레이어 수를 줄임
        num_attention_heads=16,
        intermediate_size=2816  # hidden_size의 2.75배
    )
    return LlamaMLX(config)

# 훈련 루프
def train_step(model, inputs, targets, optimizer):
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    
    loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
    optimizer.update(model, grads)
    return loss

def train_model(model, tokenized_dataset, num_epochs=3, batch_size=2):
    optimizer = optim.Adam(learning_rate=1e-4)

    for epoch in range(num_epochs):
        for i in range(0, len(tokenized_dataset), batch_size):
            batch = tokenized_dataset[i:i+batch_size]
            inputs = mx.array(batch["input_ids"])
            targets = mx.array(batch["labels"])
            loss = train_step(model, inputs, targets, optimizer)
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

    mx.save("llama3_finetuned_mlx.npz", model.parameters())
    print("Fine-tuning completed and model saved.")

# 메인 실행 함수
def main():
    tokenized_dataset, tokenizer = load_and_preprocess_data()
    model = initialize_model(tokenizer)
    train_model(model, tokenized_dataset)

if __name__ == "__main__":
    main()