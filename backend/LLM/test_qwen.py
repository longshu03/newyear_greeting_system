from qwen_model import QWenGenerator

if __name__ == "__main__":
    # 初始化模型
    llm = QWenGenerator()

    # 测试生成
    prompt = "请写一句正式而简短的新春祝福语："
    result = llm.generate_greeting(prompt)
    print("\n====== 测试输出 ======")
    print(result)
