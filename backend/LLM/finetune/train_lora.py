# 文件名：train_lora.py
# 作用：使用 LoRA 对 Qwen1.5-1.8B 做 SFT 指令微调

import json
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler  # 修正导入

# 配置
MODEL_PATH = "D:/Models/Qwen1.5-1.8B"   # 使用本地路径
DATA_PATH = "backend/LLM/finetune/sft_dataset.json"
OUTPUT_DIR = "backend/LLM/finetune/output"

MAX_LEN = 256  # 减少长度以节省显存
EPOCHS = 2     # 减少轮数
LR = 1e-5      # 适当降低学习率
BATCH_SIZE = 1  # 小批量处理
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建 LoRA 配置
def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # 减少目标模块，只选择关键模块
    )

def load_dataset():
    """加载训练数据"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 数据验证
    print(f"📊 加载 {len(data)} 条训练数据")
    for i, item in enumerate(data[:2]):  # 预览前2条
        print(f"样本 {i+1}:")
        print(f"  指令: {item['instruction'][:50]}...")
        print(f"  输入: {item['input'][:50]}...")
        print(f"  输出: {item['output'][:50]}...")
        print()
    
    return data

def format_prompt(example):
    """格式化训练样本"""
    return f"### 指令：\n{example['instruction']}\n\n### 输入：\n{example['input']}\n\n### 输出：\n{example['output']}"

def main():
    print("🚀 开始 LoRA 微调训练")
    print(f"📌 使用设备: {DEVICE}")
    print(f"📁 模型路径: {MODEL_PATH}")
    
    # 检查显存情况
    if DEVICE == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        free_memory = total_memory - allocated_memory
        print(f"💾 GPU显存: 总共 {total_memory:.2f}GB, 已用 {allocated_memory:.2f}GB, 可用 {free_memory:.2f}GB")
    
    # 1. 加载 tokenizer
    print("\n⬇️ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True  # 从本地加载
    )
    
    # 设置填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    print("⬇️ 加载模型...")
    try:
        # 选择合适的数据类型
        if DEVICE == "cuda":
            # 检查显存是否足够
            if free_memory > 3.0:  # 如果可用显存大于3GB，使用float16
                torch_dtype = torch.float16
                print("✅ 使用 float16 精度 (GPU显存充足)")
            else:
                torch_dtype = torch.float32
                print("⚠️  可用显存较少，使用 float32 精度")
        else:
            torch_dtype = torch.float32
            print("✅ 使用 float32 精度 (CPU)")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=True
        )
        
        # 关键修改：将模型移动到设备
        model = model.to(DEVICE)
        print(f"✅ 模型已移动到 {DEVICE}")
        
        # 应用 LoRA
        print("🔧 应用 LoRA 配置...")
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数数量
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 尝试使用更保守的加载方式...")
        # 回退方案：使用CPU和float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True
        )
        model = model.to("cpu")
        DEVICE = "cpu"
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model.config.use_cache = False
    model.train()
    
    # 3. 加载并预处理数据
    print("\n📊 加载训练数据...")
    dataset = load_dataset()
    
    # 数据预处理
    def tokenize_function(example):
        text = format_prompt(example)
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建 labels
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # 简化：整个序列都作为 labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略填充位置
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    tokenized_data = [tokenize_function(d) for d in dataset]
    
    # 创建 DataLoader
    dataloader = DataLoader(
        tokenized_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # 4. 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01
    )
    
    # 5. 训练循环 - 简化版，不使用混合精度
    print("\n🏋️ 开始训练...")
    
    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            # 移动到设备
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 检查 NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  步骤 {step}: 检测到 NaN/Inf 损失，跳过此批次")
                optimizer.zero_grad()
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步进
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 1 == 0:  # 每个批次都打印
                print(f"  步骤 {step}/{len(dataloader)} | 损失: {loss.item():.4f}")
        
        # 每个 epoch 的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f}")
        
        # 每个epoch后保存检查点
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"💾 检查点保存到: {checkpoint_dir}")
    
    # 6. 保存最终模型
    print(f"\n💾 保存最终模型到: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存 LoRA 权重
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存完整配置
    config = {
        "base_model": MODEL_PATH,
        "lora_config": lora_config.to_dict(),
        "training_args": {
            "epochs": EPOCHS,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LEN,
            "device": DEVICE
        }
    }
    
    config_path = os.path.join(OUTPUT_DIR, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 训练完成！模型已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()