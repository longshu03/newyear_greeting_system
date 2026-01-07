"""
一键生成新春贺词脚本（单文件处理）
使用方法：
1. 将员工年度总结放在 data/reports/
2. 将贺词模板放在 data/templates/
3. 修改 FILE_NAME 为待处理的报告文件名
4. 设置 USE_LLM=True/False 切换是否使用 QWen
5. 运行 python run_greeting.py
"""

import os
from pipeline.greeting_pipeline import GreetingPipeline

# =========================
# 配置区域
# =========================
FILE_NAME = "4.docx"        # 待处理报告文件
FEATURE_TYPE = "sbert"      # 可选: "tfidf" / "doc2vec" / "sbert"
OUTPUT_DIR = "data/generated_greetings"
USE_LLM = True              # False = 使用模板, True = 使用 QWen LLM

# =========================
# 执行流程
# =========================
def main():
    pipeline = GreetingPipeline(
        feature_type=FEATURE_TYPE,
        use_llm=USE_LLM
    )

    result = pipeline.run_file(FILE_NAME, output_dir=OUTPUT_DIR)

    print("\n==============================")
    print(f"文件: {FILE_NAME}")

    # ⭐ 核心：根据模式区分输出
    if result["use_llm"]:
        print("生成模式: LLM 模式（QWen）")
        print("模板使用情况: 未直接使用模板（仅用于语义参考）")
    else:
        print("生成模式: 模板匹配模式")
        print(f"使用模板文件: {result['template_name']}")

    print(f"员工姓名: {result['employee_name']}")
    print(f"姓名来源: {'映射表' if result['employee_name_source']=='map' else '文件名'}")

    print("\n生成的贺词内容：\n")
    print(result["greeting"])

    print("\n==============================")
    print(f"使用特征模型: {result['feature_type']}")
    print(f"是否使用LLM: {result['use_llm']}")
    print(f"生成文件路径: {os.path.abspath(result['output_file'])}")
    print("==============================\n")


if __name__ == "__main__":
    main()
