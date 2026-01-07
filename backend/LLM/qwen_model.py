import os  #导入python的操作系统接口模块
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QWenGenerator:
    """
    本地 QWen 模型封装模块
    直接从本地路径加载模型
    """

    def __init__(self, local_model_path="D:/Models/Qwen1.5-1.8B"):
        """
        初始化模型加载器
        
        Args:
            local_model_path: 本地模型文件路径
        """
        # 自动选择设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📌 当前使用设备: {self.device}")
        print(f"📁 尝试从本地路径加载模型: {local_model_path}")

        try:
            # 从本地加载 tokenizer
            print("⬇️ 加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 从本地加载模型
            print("⬇️ 加载模型...")
            
            # 设置合适的dtype
            if self.device == "cuda":
                torch_dtype = torch.float16
                print("✅ 使用 float16 精度 (GPU)")
            else:
                torch_dtype = torch.float32
                print("✅ 使用 float32 精度 (CPU)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            # 如果没有自动分配到设备，则手动分配
            if self.device == "cuda" and self.model.device.type != "cuda":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            print("✅ QWen 模型加载完成！")
            
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            print("\n🔧 解决方案:")
            print("1. 确保模型已下载到 D:/Models/Qwen1.5-1.8B/ 目录")
            print("2. 下载地址: https://huggingface.co/Qwen/Qwen1.5-1.8B")
            print("3. 或使用以下命令下载:")
            print("   huggingface-cli download Qwen/Qwen1.5-1.8B --local-dir D:/Models/Qwen1.5-1.8B")
            print("\n💡 注意: 如果网络有问题，可以:")
            print("   - 使用VPN或代理")
            print("   - 使用镜像源: 设置 HF_ENDPOINT=https://hf-mirror.com")
            print("   - 手动从浏览器下载")
            raise

    def generate_greeting(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
        """
        基于输入 prompt 生成文本
        
        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数（0.0-1.0）
            top_p: 核采样参数
            
        Returns:
            生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 只返回生成部分，去掉原始prompt
            if text.startswith(prompt):
                return text[len(prompt):].strip()
            else:
                return text.strip()
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "生成失败，请检查模型配置。"